"""
File text extraction for PDF, Excel, and Image inputs.

  - PyMuPDF (fitz) for .pdf
  - openpyxl for .xlsx / .xls
  - pytesseract for .png / .jpg / .jpeg

Each parser returns a single plain-text string. Caller is responsible
for chunking and embedding the text downstream.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


class UnsupportedFileTypeError(Exception):
    """Raised when the uploaded file's extension is not allowed."""


class TextExtractionError(Exception):
    """Raised when no extractable text was found in the uploaded file."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def extract_text(file_path: str | Path) -> Tuple[str, str]:
    """
    Extract text from a file.

    Returns:
        (extracted_text, file_type_label)

    Raises:
        UnsupportedFileTypeError, TextExtractionError
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        text = _extract_pdf(path)
        file_type = "pdf"
    elif ext in {".xlsx", ".xls"}:
        text = _extract_excel(path)
        file_type = "excel"
    elif ext in {".png", ".jpg", ".jpeg"}:
        text = _extract_image(path)
        file_type = "image"
    else:
        raise UnsupportedFileTypeError(
            "This file type is not supported. Please upload PDF, Excel, or image files."
        )

    text = (text or "").strip()
    if not text:
        raise TextExtractionError(
            "Could not extract text from this file. Please verify the file is not "
            "corrupted or password-protected."
        )

    return text, file_type


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _extract_pdf(path: Path) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF (fitz) is not installed") from exc

    pieces: list[str] = []
    with fitz.open(str(path)) as doc:
        for page_index, page in enumerate(doc, start=1):
            page_text = page.get_text("text") or ""
            page_text = page_text.strip()
            if page_text:
                pieces.append(f"[Page {page_index}]\n{page_text}")
    return "\n\n".join(pieces)


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------
def _extract_excel(path: Path) -> str:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openpyxl is not installed") from exc

    pieces: list[str] = []
    # data_only=True returns computed cell values rather than formulas
    wb = load_workbook(filename=str(path), data_only=True, read_only=True)
    try:
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            sheet_lines: list[str] = [f"[Sheet: {sheet_name}]"]
            for row in sheet.iter_rows(values_only=True):
                values = [
                    str(cell).strip()
                    for cell in row
                    if cell is not None and str(cell).strip() != ""
                ]
                if values:
                    sheet_lines.append(" | ".join(values))
            if len(sheet_lines) > 1:
                pieces.append("\n".join(sheet_lines))
    finally:
        wb.close()

    return "\n\n".join(pieces)


# ---------------------------------------------------------------------------
# Image (OCR)
# ---------------------------------------------------------------------------
def _extract_image(path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pytesseract / Pillow are not installed") from exc

    try:
        with Image.open(str(path)) as img:
            # Use English by default; can be extended via config later.
            return pytesseract.image_to_string(img) or ""
    except Exception as exc:
        logger.error("OCR failed for %s: %s", path, exc)
        return ""
