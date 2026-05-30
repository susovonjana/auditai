"""
Magic-byte file type validation.

Extension alone is trivial to spoof. This module reads the actual leading
bytes of the file (the "magic number") and verifies they match the declared
extension. Uses the pure-Python `filetype` library — no native deps.

PDF, DOCX (zip), XLSX (zip), XLS (CFB), PNG, JPEG all have well-known
magic numbers. Plain text is treated as suspicious — none of our supported
formats are plain text.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import filetype


# Map our allowed extensions to the MIME types `filetype` will detect.
# Multiple MIME types per extension because OOXML formats sometimes
# detect as plain zip.
_ALLOWED_MIMES: dict[str, set[str]] = {
    ".pdf":  {"application/pdf"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",                       # DOCX is a zip — sometimes detected as such
        "application/octet-stream",
    },
    ".xlsx": {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/zip",
        "application/octet-stream",
    },
    ".xls":  {
        "application/vnd.ms-excel",
        "application/x-ole-storage",             # CFB / OLE
        "application/octet-stream",
    },
    ".png":  {"image/png"},
    ".jpg":  {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
}


class FileValidationError(Exception):
    """Raised when a file's contents do not match its declared extension."""


def detect_kind(file_path: str | Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Read magic bytes from disk. Returns (mime, extension) where either may
    be None if the type couldn't be determined.
    """
    kind = filetype.guess(str(file_path))
    if kind is None:
        return None, None
    return kind.mime, "." + kind.extension


def detect_kind_bytes(blob: bytes) -> Tuple[Optional[str], Optional[str]]:
    """Same as detect_kind but on an in-memory byte slice."""
    kind = filetype.guess(blob)
    if kind is None:
        return None, None
    return kind.mime, "." + kind.extension


def validate_file(file_path: str | Path, declared_ext: str) -> None:
    """
    Raise FileValidationError if the file at `file_path` does not match
    the declared extension based on its magic bytes.

    DOCX/XLSX files are ZIP archives so they may be detected only as
    'application/zip' — we accept that for those extensions, but reject
    a `.docx` file that actually contains a PDF magic header.
    """
    declared_ext = declared_ext.lower()
    if declared_ext not in _ALLOWED_MIMES:
        raise FileValidationError(
            f"Extension {declared_ext!r} is not on the allowed list."
        )

    detected_mime, detected_ext = detect_kind(file_path)

    # If we can't detect anything at all, treat as suspicious for binary
    # formats. (Some valid text inputs would fail filetype detection,
    # but none of our supported formats are plain text.)
    if detected_mime is None:
        raise FileValidationError(
            "Could not determine the file type from its contents. "
            "It may be empty, corrupted, or an unsupported format."
        )

    allowed_for_ext = _ALLOWED_MIMES[declared_ext]
    if detected_mime not in allowed_for_ext:
        raise FileValidationError(
            f"This file claims to be {declared_ext} but its contents look like "
            f"{detected_mime}. The upload has been rejected for safety."
        )

    # Cross-extension sanity: if declared is .docx but content reads as
    # .pdf, that's not allowed even though both would individually pass.
    if detected_ext and detected_ext.lower() != declared_ext:
        # Special case: OOXML and zip detection overlap is fine.
        if not (
            declared_ext in {".docx", ".xlsx"}
            and detected_ext.lower() in {".zip", ".docx", ".xlsx"}
        ):
            raise FileValidationError(
                f"Extension {declared_ext} does not match detected type "
                f"{detected_ext}. The upload has been rejected for safety."
            )
