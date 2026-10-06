"""Unit tests for the document reader engine (offline — no Bedrock, no S3, no 1audit).

Covers the deterministic parts: source classification, content fingerprinting,
text sampling, PDF/image/text extraction on synthetic files, the arithmetic/VAT/
period/completeness checks, and ``build_insight`` end-to-end with the LLM call
and the download stubbed out.
"""
from __future__ import annotations

import base64
import io

import pytest

import config
import document_reader as dr


# ---------------------------------------------------------------------------
# classification + fingerprint + sampling
# ---------------------------------------------------------------------------
def test_classify_source_routes_by_extension_and_mime():
    assert dr.classify_source("inv.PDF", None) == "pdf"
    assert dr.classify_source(None, "application/pdf") == "pdf"
    assert dr.classify_source("scan.jpeg", "image/jpeg") == "image"
    assert dr.classify_source("photo", "image/png") == "image"
    assert dr.classify_source("contract.docx", None) == "word"
    assert dr.classify_source("tb.xlsx", None) == "spreadsheet"
    assert dr.classify_source("notes.csv", "text/csv") == "text"
    assert dr.classify_source("deck.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation") == "unsupported"


def test_content_key_changes_when_source_changes():
    a = dr.content_key_for({"document_id": 7, "size": 100, "updated_at": "2026-01-01T00:00:00Z"})
    b = dr.content_key_for({"document_id": 7, "size": 100, "updated_at": "2026-02-01T00:00:00Z"})
    c = dr.content_key_for({"document_id": 7, "size": 100, "updated_at": "2026-01-01T00:00:00Z"})
    assert a != b and a == c and len(a) == 40


def test_sample_text_keeps_head_and_tail():
    text = "H" * 800 + "M" * 1000 + "T" * 300
    out, truncated = dr._sample_text(text, 500)
    assert truncated is True
    assert out.startswith("H") and out.endswith("T")
    assert "omitted" in out
    short, t2 = dr._sample_text("abc", 500)
    assert (short, t2) == ("abc", False)


# ---------------------------------------------------------------------------
# extraction on synthetic files
# ---------------------------------------------------------------------------
def _pdf_with_text(pages_text):
    import fitz
    doc = fitz.open()
    for t in pages_text:
        page = doc.new_page()
        if t:
            page.insert_text((72, 72), t, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def test_extract_pdf_native_text_pages():
    data = _pdf_with_text(["Tax Invoice INV-001 Total 1150.00 SAR " * 3, "Page two terms and conditions " * 3])
    out = dr.extract_content(data, "inv.pdf", "application/pdf")
    assert out.read_method == "native_text"
    assert out.pages == 2
    assert "[Page 1]" in out.text and "[Page 2]" in out.text
    assert "INV-001" in out.text
    assert out.images == []


def test_extract_pdf_scanned_pages_rendered_for_vision(monkeypatch):
    monkeypatch.setattr(config, "DOC_READER_VISION_ENABLED", True)
    data = _pdf_with_text(["", ""])  # no text layer → "scanned"
    out = dr.extract_content(data, "scan.pdf", "application/pdf")
    assert out.read_method == "vision"
    assert len(out.images) == 2
    assert out.images[0]["media_type"] in ("image/jpeg", "image/png")
    assert base64.b64decode(out.images[0]["data"])  # valid base64
    assert any("scanned" in n for n in out.notes)


def test_extract_pdf_page_cap(monkeypatch):
    monkeypatch.setattr(config, "DOC_READER_MAX_PAGES", 2)
    data = _pdf_with_text(["one " * 20, "two " * 20, "three " * 20])
    out = dr.extract_content(data, "long.pdf", "application/pdf")
    assert out.pages == 2 and out.truncated is True
    assert "[Page 3]" not in out.text


def test_extract_image_downscales_and_encodes(monkeypatch):
    from PIL import Image
    monkeypatch.setattr(config, "DOC_READER_VISION_ENABLED", True)
    monkeypatch.setattr(config, "DOC_READER_MAX_IMAGE_EDGE", 400)
    img = Image.new("RGB", (2000, 1000), (255, 255, 255))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    out = dr.extract_content(buf.getvalue(), "photo.png", "image/png")
    assert out.read_method == "vision" and len(out.images) == 1
    back = Image.open(io.BytesIO(base64.b64decode(out.images[0]["data"])))
    assert max(back.size) == 400  # longest edge capped


def test_extract_text_file_and_unsupported():
    out = dr.extract_content("a,b\n1,2\n".encode(), "x.csv", "text/csv")
    assert out.read_method == "text" and "a,b" in out.text
    with pytest.raises(dr.DocumentReaderError):
        dr.extract_content(b"PK...", "deck.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation")


# ---------------------------------------------------------------------------
# checks (pure python)
# ---------------------------------------------------------------------------
def test_run_checks_invoice_ok():
    insight = {
        "doc_type": "invoice",
        "amounts": [
            {"label": "Subtotal", "value": 1000.0, "currency": "SAR"},
            {"label": "VAT 15%", "value": 150.0, "currency": "SAR"},
            {"label": "Total", "value": 1150.0, "currency": "SAR"},
        ],
        "line_items": [{"description": "A", "amount": 600.0}, {"description": "B", "amount": 400.0}],
        "dates": [{"label": "Issue date", "value": "2026-03-10"}],
        "parties": [{"role": "issuer", "name": "X", "identifiers": ["VAT No: 300000000000003"]}],
    }
    checks = {c["check"]: c for c in dr.run_checks(insight, period_start="2026-01-01", period_end="2026-12-31")}
    assert checks["totals_arithmetic"]["status"] == "ok"
    assert checks["vat_rate"]["status"] == "ok"
    assert checks["line_items_sum"]["status"] == "ok"
    assert checks["period"]["status"] == "ok"
    assert checks["vat_registration_number"]["status"] == "ok"


def test_run_checks_flags_mismatch_rate_period_and_missing_vat_no():
    insight = {
        "doc_type": "invoice",
        "amounts": [
            {"label": "Subtotal", "value": 1000.0},
            {"label": "VAT", "value": 100.0},
            {"label": "Total", "value": 1150.0},
        ],
        "line_items": [{"description": "A", "amount": 600.0}, {"description": "B", "amount": 300.0}],
        "dates": [{"label": "Issue date", "value": "2025-03-10"}],
        "parties": [{"role": "issuer", "name": "X", "identifiers": []}],
        "references": [{"label": "Invoice no.", "value": "1"}],
    }
    checks = {c["check"]: c for c in dr.run_checks(insight, period_start="2026-01-01", period_end="2026-12-31")}
    assert checks["totals_arithmetic"]["status"] == "warning"
    assert checks["vat_rate"]["status"] == "warning" and "10.00%" in checks["vat_rate"]["detail"]
    assert checks["line_items_sum"]["status"] == "warning"
    assert checks["period"]["status"] == "warning" and "2025-03-10" in checks["period"]["detail"]
    assert checks["vat_registration_number"]["status"] == "warning"


def test_run_checks_arabic_labels_and_no_period():
    insight = {
        "doc_type": "receipt",
        "amounts": [
            {"label": "المجموع الفرعي", "value": 200.0},
            {"label": "ضريبة القيمة المضافة", "value": 30.0},
            {"label": "الإجمالي", "value": 230.0},
        ],
        "dates": [{"label": "date", "value": "10/03/2026"}],  # non-ISO → ignored
    }
    checks = {c["check"]: c for c in dr.run_checks(insight)}
    assert checks["totals_arithmetic"]["status"] == "ok"
    assert checks["vat_rate"]["status"] == "ok"
    assert "period" not in checks and "vat_registration_number" not in checks


# ---------------------------------------------------------------------------
# build_insight end-to-end with LLM + download stubbed
# ---------------------------------------------------------------------------
class _Ctx:
    """CopilotContext stand-in: only .get(endpoint) is used by the fetch step."""
    def __init__(self, meta):
        self.meta = meta
        self.calls = []

    def get(self, endpoint, params=None):
        self.calls.append(endpoint)
        return self.meta


def test_build_insight_pipeline(monkeypatch):
    meta = {"document_id": 42, "reference": "D-7", "name": "inv.pdf", "mime_type": "application/pdf",
            "size": 123, "updated_at": "2026-05-01T10:00:00Z", "download_url": "https://s3/x"}
    pdf = _pdf_with_text(["Tax Invoice INV-77 Subtotal 1000 VAT 150 Total 1150 " * 2])
    monkeypatch.setattr(dr, "download_document", lambda m, ctx=None: pdf)

    captured = {}

    def fake_generate(blocks, schema, *, system=None, max_output_tokens=0, usage_out=None, **_):
        captured["blocks"] = blocks
        captured["system"] = system
        if usage_out is not None:
            usage_out.update(input=1200, output=300, model="fake-sonnet")
        return schema(
            doc_type="invoice", title="Tax invoice INV-77", summary_short="An invoice.", summary="- line",
            amounts=[{"label": "Subtotal", "value": 1000}, {"label": "VAT", "value": 150}, {"label": "Total", "value": 1150}],
            dates=[{"label": "Issue date", "value": "2026-02-02"}],
            parties=[{"role": "issuer", "name": "ACME", "identifiers": ["VAT No 3000"]}],
            confidence=0.9,
        )

    monkeypatch.setattr(dr, "generate_structured_from_blocks", fake_generate)
    ctx = _Ctx(meta)
    stages = []
    usage = {}
    out = dr.build_insight(
        ctx, document_id=42, language="ar",
        file_context={"client": "ACME", "period_start": "2026-01-01", "period_end": "2026-12-31"},
        working_papers=[{"reference": "C-1", "name": "Revenue"}],
        progress=lambda s, m: stages.append(s), usage_out=usage,
    )
    assert ctx.calls == ["documents/by_id/42/content"]
    assert out["document"]["reference"] == "D-7" and out["content_key"] == dr.content_key_for(meta)
    assert out["read_method"] == "native_text" and out["pages"] == 1
    assert "INV-77" in out["extracted_text"]
    ins = out["insight"]
    assert ins["doc_type"] == "invoice"
    assert {c["check"]: c["status"] for c in ins["checks"]}["totals_arithmetic"] == "ok"
    assert ins["read_method"] == "native_text"
    assert stages == ["fetching", "fetching", "reading", "analyzing"]  # locate, download, read, analyse
    assert usage["model"] == "fake-sonnet"
    # prompt plumbing: WP list + engagement context + Arabic directive reached the model
    texts = " ".join(b["text"] for b in captured["blocks"] if b.get("type") == "text")
    assert "C-1" in texts and "ACME" in texts and "<document_text>" in texts
    assert "Arabic" in captured["system"]


def test_build_insight_vision_blocks_order(monkeypatch):
    monkeypatch.setattr(config, "DOC_READER_VISION_ENABLED", True)
    meta = {"document_id": 1, "reference": "D-1", "name": "scan.pdf", "mime_type": "application/pdf",
            "size": 1, "updated_at": "x", "download_url": "u"}
    monkeypatch.setattr(dr, "download_document", lambda m, ctx=None: _pdf_with_text([""]))
    captured = {}

    def fake_generate(blocks, schema, **kw):
        captured["blocks"] = blocks
        return schema(doc_type="other", title="t", summary_short="s", summary="s")

    monkeypatch.setattr(dr, "generate_structured_from_blocks", fake_generate)
    out = dr.build_insight(_Ctx(meta), document_id=1, meta=meta)
    types = [b["type"] for b in captured["blocks"]]
    assert "image" in types and types[0] == "text" and types[-1] == "text"
    assert out["read_method"] == "vision"


def test_build_insight_surfaces_clean_errors(monkeypatch):
    ctx = _Ctx({"error": "1audit returned HTTP 404"})
    with pytest.raises(dr.DocumentReaderError):
        dr.build_insight(ctx, document_id=9)
    meta = {"document_id": 9, "size": 10 ** 9, "updated_at": "x", "download_url": "u", "name": "big.pdf"}
    with pytest.raises(dr.DocumentReaderError):
        dr.download_document(meta)


def test_download_falls_back_to_preview_url(monkeypatch):
    """The attachment URL may be rejected by the CDN; the inline/preview URL is tried next."""
    import requests as rq
    calls = []

    def fake_fetch(url, limit):
        calls.append(url)
        if "attachment" in url:
            raise rq.RequestException("400 Client Error")
        return b"%PDF-1.4 ok"

    monkeypatch.setattr(dr, "_fetch_url", fake_fetch)
    monkeypatch.setattr(config, "DOC_READER_DEV_STATIC_FILE", "")
    data = dr.download_document({"download_url": "https://cdn/x.pdf?attachment", "preview_url": "https://cdn/x.pdf", "size": 10})
    assert data.startswith(b"%PDF") and calls == ["https://cdn/x.pdf?attachment", "https://cdn/x.pdf"]
    with pytest.raises(dr.DocumentReaderError):
        dr.download_document({"download_url": "https://cdn/y.pdf?attachment", "size": 10})  # both fail → clean error


def test_dev_static_file_short_circuits_download(monkeypatch, tmp_path):
    f = tmp_path / "s.pdf"; f.write_bytes(b"%PDF-static")
    monkeypatch.setattr(config, "DOC_READER_DEV_STATIC_FILE", str(f))
    assert dr.download_document({"download_url": "https://cdn/never-called"}) == b"%PDF-static"


def test_download_falls_back_to_be_bytes(monkeypatch):
    """CDN URLs rejected → the bytes come through 1audit-be (grant-scoped)."""
    import requests as rq

    def fail(url, limit):
        raise rq.RequestException("400 Client Error")

    class _CtxBytes:
        def get_bytes(self, endpoint, *, max_bytes, read_timeout=None):
            assert endpoint == "documents/by_id/48701/bytes"
            return b"\x89PNG from be"

    monkeypatch.setattr(dr, "_fetch_url", fail)
    monkeypatch.setattr(config, "DOC_READER_DEV_STATIC_FILE", "")
    data = dr.download_document({"document_id": 48701, "download_url": "https://cdn/a", "preview_url": "https://cdn/b", "size": 10}, _CtxBytes())
    assert data.startswith(b"\x89PNG")
