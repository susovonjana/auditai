"""
Document reader — read ONE uploaded audit-file document and turn it into a
structured, auditor-ready insight (the engine behind the All-documents
"AI summary" and the ``document_extraction`` agent).

Pipeline (deterministic parts in Python, understanding by the LLM):

  fetch_document_meta   grant-scoped 1audit-be call → presigned URL + metadata
  content_key_for       fingerprint (document_id:size:updated_at) → cache key
  download_document     GET the presigned URL (size-capped)
  extract_content       route by type:
                          PDF   → PyMuPDF text per page; pages with no text layer
                                  are RENDERED to PNG for vision (scans / photos)
                          image → downscaled JPEG/PNG for vision
                          DOCX/XLSX/XLS → parser.py (python-docx / openpyxl)
                          CSV/TXT → decoded text
                        caps: pages, image edge, chars (head+tail sample)
  analyze_document      ONE multimodal structured call (text + images) → DocumentInsight
  run_checks            pure-Python arithmetic / VAT-rate / period / completeness
                        checks over the EXTRACTED figures (the model never adds up)
  build_insight         orchestrates the above with a progress callback

Numbers rule: the model may only EXTRACT figures printed on the document (that
IS the source); every computed quantity (sums, rates, deltas) comes from
``run_checks``. Nothing here writes to 1audit — the insight is cached in
auditai's own Postgres by ``document_insight_store``.
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from pydantic import BaseModel, Field

import config
from copilot_tools import CopilotContext
from prompts import document_reader as prompts
from structured import generate_structured_from_blocks

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, str], None]  # (stage, human message)


class DocumentReaderError(Exception):
    """A clean, user-facing failure (bad type, too big, download failed, empty)."""


# ---------------------------------------------------------------------------
# Structured output schema (what the LLM returns)
# ---------------------------------------------------------------------------
class Party(BaseModel):
    role: str = Field(description="issuer / customer / supplier / bank / lessor / lessee / employee / signatory / other")
    name: str = Field(description="name exactly as printed")
    identifiers: List[str] = Field(default_factory=list, description="VAT no., CR no., IBAN, ID no. … as printed, each prefixed with its label")
    address: Optional[str] = None


class DatedItem(BaseModel):
    label: str = Field(description="what the date is: issue date / due date / period start / signed on …")
    value: str = Field(description="ISO yyyy-mm-dd when unambiguous, else exactly as printed")


class Amount(BaseModel):
    label: str = Field(description="subtotal / vat / total / discount / paid / balance / monthly rent … as the document names it")
    value: float = Field(description="plain number as printed (no separators); negative for credits if shown so")
    currency: Optional[str] = Field(default=None, description="currency code or symbol as printed, e.g. SAR")


class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = Field(default=None, description="the line total as printed")
    vat: Optional[float] = Field(default=None, description="line VAT if printed")


class Reference(BaseModel):
    label: str = Field(description="invoice no. / PO no. / contract no. / account no. / IBAN / case no. …")
    value: str


class AuditRelevance(BaseModel):
    audit_areas: List[str] = Field(default_factory=list, description="e.g. Revenue, Trade receivables, Cash and bank, PPE, Payroll, VAT")
    assertions: List[str] = Field(default_factory=list, description="FS assertions this evidence supports: occurrence, completeness, accuracy, cut-off, existence, rights & obligations, valuation, classification, presentation")
    suggested_working_papers: List[str] = Field(default_factory=list, description="ONLY from the working papers list given; reference or name as listed")
    evidence_quality: Optional[str] = Field(default=None, description="original / copy / scan / photo / system-generated / unsigned / draft … plus a short note")
    notes: Optional[str] = Field(default=None, description="how an auditor would use this document (one or two sentences)")


class DocumentInsight(BaseModel):
    doc_type: str = Field(description="one of the taxonomy values, or a short free-text type when none fits")
    title: str = Field(description="a human title for the document, e.g. 'Tax invoice INV-2026-0142 from Al Noor Trading'")
    language_detected: Optional[str] = Field(default=None, description="ar / en / ar+en / other")
    summary_short: str = Field(description="1–2 sentences for the document list")
    summary: str = Field(description="the full summary, 4–10 concise bullet-style sentences separated by newlines")
    parties: List[Party] = Field(default_factory=list)
    dates: List[DatedItem] = Field(default_factory=list)
    amounts: List[Amount] = Field(default_factory=list, description="totals and key figures; NOT every line (those go in line_items)")
    currency: Optional[str] = Field(default=None, description="the document's main currency, as printed")
    line_items: List[LineItem] = Field(default_factory=list, description="itemised rows if the document has them (max 60)")
    references: List[Reference] = Field(default_factory=list)
    key_facts: List[str] = Field(default_factory=list, description="other important facts/terms not captured above (payment terms, warranty, approvals, signatures, stamps, period covered …)")
    audit_relevance: AuditRelevance = Field(default_factory=AuditRelevance)
    red_flags: List[str] = Field(default_factory=list)
    data_gaps: List[str] = Field(default_factory=list, description="what could not be read / is missing from the document")
    confidence: float = Field(default=0.0, description="0..1 — how legible/complete the reading is")


# ---------------------------------------------------------------------------
# Fetch (grant-scoped) + fingerprint
# ---------------------------------------------------------------------------
@dataclass
class ExtractedContent:
    text: str = ""
    images: List[Dict[str, Any]] = field(default_factory=list)  # {"media_type", "data", "page"}
    pages: int = 0
    read_method: str = "native_text"  # native_text | vision | mixed | word | spreadsheet | text
    truncated: bool = False
    chars: int = 0
    notes: List[str] = field(default_factory=list)


def fetch_document_meta(
    ctx: CopilotContext, *, document_id: Optional[int] = None, reference: Optional[str] = None
) -> Dict[str, Any]:
    """Resolve a document to its presigned URL + metadata via 1audit-be (grant-
    scoped, so a document of another file can never be read). Raises
    DocumentReaderError with a clean message on any failure."""
    if document_id is not None:
        meta = ctx.get(f"documents/by_id/{int(document_id)}/content")
    elif reference:
        meta = ctx.get(f"documents/{quote(str(reference), safe='')}/content")
    else:
        raise DocumentReaderError("document_id or reference is required")
    if not isinstance(meta, dict) or meta.get("error"):
        detail = meta.get("error") if isinstance(meta, dict) else "content endpoint failed"
        raise DocumentReaderError(f"Could not resolve the document in 1audit: {detail}")
    if not meta.get("download_url"):
        raise DocumentReaderError("1audit returned no download link for this document.")
    return meta


def content_key_for(meta: Dict[str, Any]) -> str:
    """Fingerprint the SOURCE without downloading it: a re-upload changes
    updated_at and (almost always) size. Falls back to the reference when ids
    are missing so the key is never empty."""
    raw = f"{meta.get('document_id') or meta.get('reference')}:{meta.get('size')}:{meta.get('updated_at')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _fetch_url(url: str, limit: int) -> bytes:
    resp = requests.get(url, timeout=(5, config.DOC_READER_DOWNLOAD_TIMEOUT_SEC), stream=True)
    resp.raise_for_status()
    buf = io.BytesIO()
    for chunk in resp.iter_content(chunk_size=1024 * 256):
        if not chunk:
            continue
        buf.write(chunk)
        if buf.tell() > limit:
            raise DocumentReaderError(
                f"This file is larger than {config.DOC_READER_MAX_FILE_MB} MB; the AI reader is capped to keep costs bounded."
            )
    return buf.getvalue()


def download_document(meta: Dict[str, Any], ctx: Optional[CopilotContext] = None) -> bytes:
    """Fetch the bytes: the attachment URL first, then the inline/preview URL
    (some CDN configurations reject one variant), then — when a context is
    given — straight through 1audit-be (``documents/by_id/{id}/bytes``, grant-
    scoped; be reads S3 with its own credentials, so no CDN dependency). Each
    path is size-capped. A DEV-ONLY static file (DOC_READER_DEV_STATIC_FILE)
    short-circuits everything so the UI flow can be exercised offline."""
    limit = int(config.DOC_READER_MAX_FILE_MB) * 1024 * 1024
    static = config.DOC_READER_DEV_STATIC_FILE
    if static:
        if not os.path.isfile(static):
            raise DocumentReaderError(f"DOC_READER_DEV_STATIC_FILE does not exist: {static}")
        logger.warning("document_reader: DEV static file in use instead of the document download: %s", static)
        with open(static, "rb") as fh:
            return fh.read(limit + 1)[:limit]
    size = meta.get("size")
    try:
        if size is not None and int(size) > limit:
            raise DocumentReaderError(
                f"This file is larger than {config.DOC_READER_MAX_FILE_MB} MB; the AI reader is capped to keep costs bounded."
            )
    except (TypeError, ValueError):
        pass
    urls = [u for u in (meta.get("download_url"), meta.get("preview_url")) if u]
    if not urls:
        raise DocumentReaderError("1audit returned no download link for this document.")
    last_exc: Optional[Exception] = None
    data = b""
    for i, url in enumerate(dict.fromkeys(urls)):  # de-dup, keep order
        try:
            data = _fetch_url(url, limit)
            break
        except DocumentReaderError:
            raise
        except requests.RequestException as exc:
            last_exc = exc
            logger.info("document_reader: download attempt %d failed: %s", i + 1, exc)
            continue
    if not data and ctx is not None and meta.get("document_id") is not None:
        try:
            data = ctx.get_bytes(
                f"documents/by_id/{int(meta['document_id'])}/bytes",
                max_bytes=limit, read_timeout=config.DOC_READER_DOWNLOAD_TIMEOUT_SEC,
            )
            if data:
                logger.info("document_reader: fetched document %s through 1audit-be (CDN URLs unusable)", meta.get("document_id"))
        except ValueError:
            raise DocumentReaderError(
                f"This file is larger than {config.DOC_READER_MAX_FILE_MB} MB; the AI reader is capped to keep costs bounded."
            )
        except requests.RequestException as exc:
            last_exc = last_exc or exc
            logger.info("document_reader: be bytes fallback failed: %s", exc)
    if not data and last_exc is not None:
        raise DocumentReaderError(f"Could not download the document: {last_exc}") from last_exc
    if not data:
        raise DocumentReaderError("The document is empty (0 bytes).")
    return data


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
_IMAGE_MIMES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif", "image/tiff", "image/bmp"}
_WORD_EXTS = {".docx"}
_SHEET_EXTS = {".xlsx", ".xls", ".xlsm"}
_TEXT_EXTS = {".txt", ".csv", ".md", ".json", ".xml", ".html", ".htm"}


def _ext_of(name: Optional[str], mime: Optional[str]) -> str:
    if name and "." in os.path.basename(name):
        return os.path.splitext(name)[1].lower()
    m = (mime or "").lower()
    if "pdf" in m:
        return ".pdf"
    if "wordprocessingml" in m or m == "application/msword":
        return ".docx"
    if "spreadsheetml" in m or "ms-excel" in m:
        return ".xlsx"
    if m.startswith("image/"):
        return "." + m.split("/", 1)[1].replace("jpeg", "jpg")
    if m.startswith("text/"):
        return ".txt"
    return ""


def classify_source(name: Optional[str], mime: Optional[str]) -> str:
    """pdf | image | word | spreadsheet | text | unsupported — what extract_content will do."""
    ext = _ext_of(name, mime)
    m = (mime or "").lower()
    if ext == ".pdf" or m == "application/pdf":
        return "pdf"
    if m in _IMAGE_MIMES or ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff", ".bmp"}:
        return "image"
    if ext in _WORD_EXTS:
        return "word"
    if ext in _SHEET_EXTS:
        return "spreadsheet"
    if ext in _TEXT_EXTS or m.startswith("text/"):
        return "text"
    return "unsupported"


def _encode_image(pil_img, max_edge: int) -> Tuple[str, str]:
    """Downscale to max_edge (Claude's sweet spot ~1568px) and return
    (media_type, base64). JPEG for opaque images, PNG when there is alpha."""
    from PIL import Image  # local import: Pillow is a hard dep but keep import cost lazy

    img = pil_img
    if getattr(img, "n_frames", 1) > 1:  # multi-frame TIFF/GIF → first frame
        img.seek(0)
    w, h = img.size
    scale = min(1.0, float(max_edge) / float(max(w, h) or 1))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    out = io.BytesIO()
    if has_alpha:
        img.convert("RGBA").save(out, format="PNG", optimize=True)
        media = "image/png"
    else:
        img.convert("RGB").save(out, format="JPEG", quality=85, optimize=True)
        media = "image/jpeg"
    return media, base64.b64encode(out.getvalue()).decode("ascii")


def _sample_text(text: str, limit: int) -> Tuple[str, bool]:
    """Head + tail sample when text exceeds the cap (the end of a contract /
    statement carries totals and signatures, so never drop it blindly)."""
    text = text.strip()
    if len(text) <= limit:
        return text, False
    head = int(limit * 0.72)
    tail = limit - head
    return (
        text[:head]
        + "\n\n[… middle of the document omitted — content truncated for length …]\n\n"
        + text[-tail:],
        True,
    )


def _extract_pdf(data: bytes, out: ExtractedContent) -> None:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - fitz is pinned in requirements
        raise DocumentReaderError("PDF support is not installed on this server (PyMuPDF).") from exc

    max_pages = int(config.DOC_READER_MAX_PAGES)
    min_chars = int(config.DOC_READER_MIN_TEXT_CHARS_PER_PAGE)
    vision = bool(config.DOC_READER_VISION_ENABLED)
    text_parts: List[str] = []
    scanned_pages: List[int] = []
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentReaderError(f"Could not open the PDF: {type(exc).__name__}") from exc
    try:
        if doc.needs_pass:
            raise DocumentReaderError("The PDF is password-protected and cannot be read.")
        total = doc.page_count
        out.pages = min(total, max_pages)
        if total > max_pages:
            out.notes.append(f"only the first {max_pages} of {total} pages were read")
            out.truncated = True
        for i in range(out.pages):
            page = doc[i]
            try:
                txt = (page.get_text("text") or "").strip()
            except Exception:
                txt = ""
            if len(txt) >= min_chars:
                text_parts.append(f"[Page {i + 1}]\n{txt}")
                continue
            scanned_pages.append(i)
            if not vision:
                continue
            try:
                # zoom so the longest edge lands near the cap (72 dpi base)
                rect = page.rect
                longest = max(rect.width, rect.height) or 1.0
                zoom = min(3.0, max(1.0, float(config.DOC_READER_MAX_IMAGE_EDGE) / longest))
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                from PIL import Image
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                media, b64 = _encode_image(img, int(config.DOC_READER_MAX_IMAGE_EDGE))
                out.images.append({"media_type": media, "data": b64, "page": i + 1})
            except Exception as exc:  # a bad page never kills the whole read
                logger.warning("document_reader: page %d render failed: %s", i + 1, exc)
                out.notes.append(f"page {i + 1} could not be rendered")
    finally:
        try:
            doc.close()
        except Exception:
            pass

    if scanned_pages and not vision:
        # Text-only mode: fall back to the Tesseract path in parser.py for scans.
        ocr_text = _ocr_fallback(data, ".pdf")
        if ocr_text:
            text_parts.append(ocr_text)
            out.notes.append("scanned pages were OCRed with Tesseract (vision disabled)")
    joined = "\n\n".join(text_parts)
    out.text, trunc = _sample_text(joined, int(config.DOC_READER_MAX_TEXT_CHARS))
    out.truncated = out.truncated or trunc
    out.chars = len(joined)
    if out.images and out.text:
        out.read_method = "mixed"
    elif out.images:
        out.read_method = "vision"
    else:
        out.read_method = "native_text"
    if scanned_pages:
        out.notes.append(f"{len(scanned_pages)} page(s) had no text layer (scanned) and were read visually")


def _ocr_fallback(data: bytes, suffix: str) -> str:
    """Legacy Tesseract path (parser.py) — only used when vision is disabled."""
    tmp = None
    try:
        from parser import extract_text
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(data)
            tmp = fh.name
        text, _kind = extract_text(tmp)
        return (text or "").strip()
    except Exception as exc:
        logger.info("document_reader: OCR fallback unavailable/failed: %s", exc)
        return ""
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _extract_image(data: bytes, out: ExtractedContent) -> None:
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise DocumentReaderError(f"Could not open the image: {type(exc).__name__}") from exc
    out.pages = 1
    if config.DOC_READER_VISION_ENABLED:
        media, b64 = _encode_image(img, int(config.DOC_READER_MAX_IMAGE_EDGE))
        out.images.append({"media_type": media, "data": b64, "page": 1})
        out.read_method = "vision"
    else:
        ext = ".png" if (img.format or "").upper() == "PNG" else ".jpg"
        text = _ocr_fallback(data, ext)
        if not text:
            raise DocumentReaderError("Could not read any text from this image (OCR unavailable).")
        out.text, out.truncated = _sample_text(text, int(config.DOC_READER_MAX_TEXT_CHARS))
        out.chars = len(text)
        out.read_method = "native_text"
        out.notes.append("image OCRed with Tesseract (vision disabled)")


def _extract_office(data: bytes, suffix: str, out: ExtractedContent, method: str) -> None:
    tmp = None
    try:
        from parser import extract_blocks
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(data)
            tmp = fh.name
        blocks, _kind = extract_blocks(tmp)
    except DocumentReaderError:
        raise
    except Exception as exc:
        msg = str(exc) or type(exc).__name__
        if suffix == ".xls":
            msg = "Legacy .xls workbooks are not supported — save as .xlsx and re-upload."
        raise DocumentReaderError(f"Could not read the file: {msg}") from exc
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    parts: List[str] = []
    sheets = set()
    for b in blocks:
        if b.section_heading and b.section_heading.startswith("Sheet:"):
            sheets.add(b.section_heading)
        if b.block_type == "heading":
            parts.append(f"## {b.content}")
        elif b.block_type == "table" and b.section_heading:
            parts.append(f"[{b.section_heading}]\n{b.content}")
        else:
            parts.append(b.content)
    joined = "\n\n".join(parts)
    out.text, out.truncated = _sample_text(joined, int(config.DOC_READER_MAX_TEXT_CHARS))
    out.chars = len(joined)
    out.pages = max(1, len(sheets)) if method == "spreadsheet" else 1
    out.read_method = method


def _extract_text_file(data: bytes, out: ExtractedContent) -> None:
    text = data.decode("utf-8", errors="replace")
    if "�" in text[:2000]:  # likely a non-UTF8 legacy encoding → try cp1256 (Arabic Windows)
        try:
            text = data.decode("cp1256")
        except Exception:
            pass
    out.text, out.truncated = _sample_text(text, int(config.DOC_READER_MAX_TEXT_CHARS))
    out.chars = len(text)
    out.pages = 1
    out.read_method = "text"


def extract_content(data: bytes, name: Optional[str], mime_type: Optional[str]) -> ExtractedContent:
    """Turn the raw bytes into text and/or page images for the model. Raises
    DocumentReaderError for unsupported / unreadable files."""
    kind = classify_source(name, mime_type)
    out = ExtractedContent()
    if kind == "pdf":
        _extract_pdf(data, out)
    elif kind == "image":
        _extract_image(data, out)
    elif kind == "word":
        _extract_office(data, ".docx", out, "word")
    elif kind == "spreadsheet":
        ext = _ext_of(name, mime_type) or ".xlsx"
        _extract_office(data, ext if ext in _SHEET_EXTS else ".xlsx", out, "spreadsheet")
    elif kind == "text":
        _extract_text_file(data, out)
    else:
        raise DocumentReaderError(
            "This file type is not supported by the AI reader yet (PDF, images, Word, Excel and text files are)."
        )
    if not out.text.strip() and not out.images:
        raise DocumentReaderError(
            "No readable content was found in this document (empty, corrupted, or a very low-quality scan)."
        )
    return out


# ---------------------------------------------------------------------------
# Analysis (the one LLM call)
# ---------------------------------------------------------------------------
def analyze_document(
    extracted: ExtractedContent,
    *,
    name: Optional[str],
    mime_type: Optional[str],
    language: str = "en",
    file_context: Optional[Dict[str, Any]] = None,
    working_papers: Optional[List[Dict[str, Any]]] = None,
    usage_out: Optional[dict] = None,
) -> DocumentInsight:
    blocks: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": prompts.build_intro_text(
                name=name, mime_type=mime_type, read_method=extracted.read_method,
                pages=extracted.pages, truncated=extracted.truncated,
                file_context=file_context, working_papers=working_papers, language=language,
            ),
        }
    ]
    for img in extracted.images:
        blocks.append({"type": "text", "text": f"[Page image {img.get('page', '?')}]"})
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": img["media_type"], "data": img["data"]},
        })
    if extracted.text.strip():
        blocks.append({"type": "text", "text": f"EXTRACTED TEXT:\n<document_text>\n{extracted.text}\n</document_text>"})
    blocks.append({"type": "text", "text": prompts.build_closing_text(language)})

    from agent.types import language_directive  # prose language = the reader's UI language

    system = prompts.SYSTEM_PROMPT + language_directive(language)
    return generate_structured_from_blocks(
        blocks,
        DocumentInsight,
        system=system,
        max_output_tokens=int(config.DOC_READER_MAX_OUTPUT_TOKENS),
        usage_out=usage_out,
    )


# ---------------------------------------------------------------------------
# Deterministic checks (pure Python — the only place numbers are computed)
# ---------------------------------------------------------------------------
_SUBTOTAL_RE = re.compile(r"sub\s*-?\s*total|net|before\s*(vat|tax)|taxable|المجموع الفرعي|قبل الضريبة|الإجمالي قبل", re.I)
_VAT_RE = re.compile(r"\bvat\b|value\s*added|\btax\b(?!\s*invoice)|ضريبة|القيمة المضافة", re.I)
_TOTAL_RE = re.compile(r"grand\s*total|total\s*(due|amount|payable|incl)|^total$|amount\s*due|الإجمالي|المجموع|المبلغ المستحق", re.I)
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_KSA_VAT_RATE = 0.15


def _pick(amounts: List[Dict[str, Any]], pattern: re.Pattern, *, exclude: Optional[re.Pattern] = None) -> Optional[float]:
    for a in amounts:
        label = str(a.get("label") or "")
        if pattern.search(label) and not (exclude and exclude.search(label)):
            try:
                return float(a.get("value"))
            except (TypeError, ValueError):
                continue
    return None


def _parse_iso(value: Any) -> Optional[date]:
    if isinstance(value, (date, datetime)):
        return value if isinstance(value, date) else value.date()
    m = _DATE_RE.match(str(value or "").strip())
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def run_checks(
    insight: Dict[str, Any],
    *,
    period_start: Any = None,
    period_end: Any = None,
) -> List[Dict[str, str]]:
    """Recompute what can be recomputed from the EXTRACTED figures and compare
    with what the document states. Each check: {check, status: ok|warning|info, detail}."""
    checks: List[Dict[str, str]] = []
    amounts = [a for a in (insight.get("amounts") or []) if isinstance(a, dict)]
    items = [li for li in (insight.get("line_items") or []) if isinstance(li, dict)]
    doc_type = str(insight.get("doc_type") or "").lower()

    subtotal = _pick(amounts, _SUBTOTAL_RE)
    vat = _pick(amounts, _VAT_RE, exclude=re.compile(r"before|excl|net|pre|قبل", re.I))
    total = _pick(amounts, _TOTAL_RE, exclude=re.compile(r"sub|net|before|excl|vat|tax|ضريبة|قبل|الفرعي|فرعي", re.I))

    # 1. subtotal + VAT = total
    if subtotal is not None and vat is not None and total is not None:
        diff = round(subtotal + vat - total, 2)
        if abs(diff) <= 0.05:
            checks.append({"check": "totals_arithmetic", "status": "ok",
                           "detail": f"Subtotal {subtotal:,.2f} + VAT {vat:,.2f} = Total {total:,.2f}."})
        else:
            checks.append({"check": "totals_arithmetic", "status": "warning",
                           "detail": f"Subtotal {subtotal:,.2f} + VAT {vat:,.2f} ≠ Total {total:,.2f} (difference {diff:,.2f})."})

    # 2. VAT rate vs KSA 15%
    if subtotal and vat is not None and subtotal != 0:
        rate = vat / subtotal
        if abs(rate - _KSA_VAT_RATE) <= 0.005:
            checks.append({"check": "vat_rate", "status": "ok", "detail": "VAT is 15% of the taxable amount (KSA standard rate)."})
        elif abs(rate) < 0.0005:
            checks.append({"check": "vat_rate", "status": "info", "detail": "VAT shown as zero — check exemption / zero-rating basis."})
        else:
            checks.append({"check": "vat_rate", "status": "warning",
                           "detail": f"VAT is {rate * 100:.2f}% of the taxable amount, not the 15% standard rate."})

    # 3. line items sum vs subtotal (or total when no VAT lines)
    line_amounts = [li.get("amount") for li in items if isinstance(li.get("amount"), (int, float))]
    if len(line_amounts) >= 2 and (subtotal is not None or total is not None):
        s = round(sum(float(x) for x in line_amounts), 2)
        target = subtotal if subtotal is not None else total
        label = "subtotal" if subtotal is not None else "total"
        # items normally add to the subtotal; when only a grand total is stated the
        # items may be VAT-inclusive (s == total) or exclusive (s + VAT == total)
        matches = abs(s - float(target)) <= 0.05 or (
            subtotal is None and vat is not None and abs(s + vat - float(target)) <= 0.05
        )
        if matches:
            checks.append({"check": "line_items_sum", "status": "ok",
                           "detail": f"{len(line_amounts)} line items add up to the stated {label} ({s:,.2f})."})
        else:
            checks.append({"check": "line_items_sum", "status": "warning",
                           "detail": f"{len(line_amounts)} line items sum to {s:,.2f}, but the stated {label} is {float(target):,.2f}."})

    # 4. dates vs the audit period
    ps, pe = _parse_iso(period_start), _parse_iso(period_end)
    if ps and pe:
        outside: List[str] = []
        inside = 0
        for d in insight.get("dates") or []:
            if not isinstance(d, dict):
                continue
            dv = _parse_iso(d.get("value"))
            if dv is None:
                continue
            if dv < ps or dv > pe:
                outside.append(f"{d.get('label') or 'date'} {dv.isoformat()}")
            else:
                inside += 1
        if outside:
            checks.append({"check": "period", "status": "warning",
                           "detail": f"Outside the audit period {ps.isoformat()}–{pe.isoformat()}: " + "; ".join(outside[:6]) + "."})
        elif inside:
            checks.append({"check": "period", "status": "ok",
                           "detail": f"All {inside} dated item(s) fall within the audit period {ps.isoformat()}–{pe.isoformat()}."})

    # 5. tax-invoice completeness (ZATCA): VAT registration number present
    if doc_type in ("invoice", "credit_note") or "invoice" in doc_type:
        has_vat_no = any(
            re.search(r"vat|tax\s*(reg|no|number)|ضريب|الرقم الضريبي", str(i), re.I)
            for p in (insight.get("parties") or []) if isinstance(p, dict)
            for i in (p.get("identifiers") or [])
        ) or any(
            re.search(r"vat|tax\s*(reg|no|number)|ضريب|الرقم الضريبي", str(r.get("label") or ""), re.I)
            for r in (insight.get("references") or []) if isinstance(r, dict)
        )
        checks.append({
            "check": "vat_registration_number",
            "status": "ok" if has_vat_no else "warning",
            "detail": ("A VAT registration number is shown." if has_vat_no
                       else "No VAT registration number was found — a KSA tax invoice must show the supplier's VAT number."),
        })

    return checks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _compact_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "document_id": meta.get("document_id"),
        "reference": meta.get("reference"),
        "name": meta.get("name"),
        "mime_type": meta.get("mime_type"),
        "size": meta.get("size"),
        "updated_at": meta.get("updated_at"),
        "confidential": bool(meta.get("confidential")),
    }


def build_insight(
    ctx: CopilotContext,
    *,
    document_id: Optional[int] = None,
    reference: Optional[str] = None,
    language: str = "en",
    file_context: Optional[Dict[str, Any]] = None,
    working_papers: Optional[List[Dict[str, Any]]] = None,
    progress: Optional[ProgressFn] = None,
    usage_out: Optional[dict] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Read one document end-to-end. Synchronous (LLM + HTTP are blocking) — async
    callers wrap it in ``asyncio.to_thread``. Returns a JSON-safe dict:
      {document, content_key, read_method, pages, insight (incl. checks, notes),
       extracted_text, elapsed_ms}"""
    def _p(stage: str, msg: str) -> None:
        if progress:
            try:
                progress(stage, msg)
            except Exception:  # a UI callback must never break the read
                pass

    t0 = time.perf_counter()
    if meta is None:
        _p("fetching", "Locating the document in 1audit…")
        meta = fetch_document_meta(ctx, document_id=document_id, reference=reference)
    key = content_key_for(meta)
    _p("fetching", "Downloading the file…")
    data = download_document(meta, ctx)
    name, mime = meta.get("name"), meta.get("mime_type")
    if config.DOC_READER_DEV_STATIC_FILE:
        # DEV static file: route extraction by the STATIC file's own type, not the
        # document's (a PNG document read from a PDF sample must parse as PDF)
        name, mime = os.path.basename(config.DOC_READER_DEV_STATIC_FILE), None
    _p("reading", "Reading the document (text + scanned pages)…")
    extracted = extract_content(data, name, mime)
    _p("analyzing", "Understanding the content with AI…")
    insight = analyze_document(
        extracted,
        name=meta.get("name"), mime_type=meta.get("mime_type"), language=language,
        file_context=file_context, working_papers=working_papers, usage_out=usage_out,
    )
    insight_dict = insight.model_dump(mode="json")
    fc = file_context or {}
    insight_dict["checks"] = run_checks(
        insight_dict, period_start=fc.get("period_start"), period_end=fc.get("period_end")
    )
    insight_dict["read_notes"] = list(extracted.notes)
    insight_dict["pages_read"] = extracted.pages
    insight_dict["read_method"] = extracted.read_method
    insight_dict["truncated"] = extracted.truncated
    stored_text = (extracted.text or "")[: int(config.DOC_READER_TEXT_STORE_CHARS)]
    return {
        "document": _compact_meta(meta),
        "content_key": key,
        "read_method": extracted.read_method,
        "pages": extracted.pages,
        "insight": insight_dict,
        "extracted_text": stored_text,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
    }
