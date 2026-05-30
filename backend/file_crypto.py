"""
Encryption at rest for uploaded files (Fernet — AES-128-CBC + HMAC-SHA256).

Why this matters:
  - If the server filesystem is ever compromised, encrypted uploads
    are useless to the attacker without the FILE_ENCRYPTION_KEY.
  - The key is held in environment variables, NOT on disk next to the
    files. A typical production setup puts the key in a secret manager
    (AWS Secrets Manager, Vault, systemd EnvironmentFile chmod 600).

Flow:
  - On upload: read the original file bytes → encrypt → write to disk
    with a ".enc" suffix.
  - On parse: decrypt to a temporary file → parse → temp file is auto-
    deleted when the context manager exits.
  - On delete: just remove the encrypted file from disk.

If FILE_ENCRYPTION_KEY is empty or ENCRYPT_UPLOADS is false, this module
falls back to plaintext (useful for local development).
"""
from __future__ import annotations

import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from config import ENCRYPT_UPLOADS, FILE_ENCRYPTION_KEY

logger = logging.getLogger(__name__)


_fernet: Optional[object] = None


def _get_fernet():
    """Return a cached Fernet instance, or None if encryption is disabled."""
    global _fernet
    if not ENCRYPT_UPLOADS:
        return None
    if _fernet is not None:
        return _fernet
    if not FILE_ENCRYPTION_KEY:
        logger.warning(
            "ENCRYPT_UPLOADS=true but FILE_ENCRYPTION_KEY is empty. "
            "Falling back to plaintext storage. Generate a key with: "
            "python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\""
        )
        return None
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "cryptography is not installed. Run: pip install -r requirements.txt"
        ) from exc
    try:
        _fernet = Fernet(FILE_ENCRYPTION_KEY.encode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "FILE_ENCRYPTION_KEY is not a valid Fernet key. "
            "Generate a new one with: python -c \"from cryptography.fernet "
            "import Fernet; print(Fernet.generate_key().decode())\""
        ) from exc
    return _fernet


def is_encryption_enabled() -> bool:
    """True if a usable encryption key is configured."""
    return _get_fernet() is not None


def encrypted_path(stored_path: str | Path) -> Path:
    """Append a `.enc` suffix when encryption is on."""
    p = Path(stored_path)
    if is_encryption_enabled():
        return p.with_suffix(p.suffix + ".enc")
    return p


def encrypt_and_write(plaintext_bytes: bytes, target_path: str | Path) -> Path:
    """
    Encrypt `plaintext_bytes` and write to `target_path` (with `.enc`
    appended if encryption is on). Returns the actual path written.
    """
    final_path = encrypted_path(target_path)
    fernet = _get_fernet()
    if fernet is None:
        final_path.write_bytes(plaintext_bytes)
    else:
        final_path.write_bytes(fernet.encrypt(plaintext_bytes))
    return final_path


@contextmanager
def open_decrypted(stored_path: str | Path) -> Iterator[Path]:
    """
    Context manager that yields a Path to a plaintext copy of the file.
    The plaintext copy is deleted automatically when the context exits.

    If encryption is disabled, this just yields the existing path
    unchanged — no temp file is created or deleted.
    """
    p = Path(stored_path)
    fernet = _get_fernet()

    if fernet is None:
        yield p
        return

    # If caller passed the un-suffixed path, fix it up
    if not p.exists() and p.with_suffix(p.suffix + ".enc").exists():
        p = p.with_suffix(p.suffix + ".enc")

    ciphertext = p.read_bytes()
    try:
        plaintext = fernet.decrypt(ciphertext)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to decrypt {p}. Has FILE_ENCRYPTION_KEY changed?"
        ) from exc

    # Preserve the original extension WITHOUT the .enc suffix so the
    # downstream parser routes by extension correctly.
    suffix = "".join(p.suffixes[:-1]) if p.suffix == ".enc" else p.suffix
    fd, tmp_name = tempfile.mkstemp(suffix=suffix, prefix="auditai_dec_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(plaintext)
        yield Path(tmp_name)
    finally:
        try:
            os.remove(tmp_name)
        except OSError:
            pass


def remove_encrypted(stored_path: str | Path) -> None:
    """Delete the stored file (encrypted or plain), best-effort."""
    p = Path(stored_path)
    candidates = [p]
    if not p.suffix.endswith(".enc"):
        candidates.append(p.with_suffix(p.suffix + ".enc"))
    for c in candidates:
        try:
            if c.exists():
                c.unlink()
        except OSError:
            pass
