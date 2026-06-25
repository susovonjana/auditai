#!/usr/bin/env python3
"""Bulk-ingest a folder of documents into the auditai knowledge base.

Logs in as admin, uploads every supported file in a folder via POST
/admin/upload, then polls each document to `active`. Stdlib only — no pip
install needed, so it runs from the host straight against the container.

Supported file types (must match config.ALLOWED_EXTENSIONS):
    .pdf .docx .xlsx .xls .png .jpg .jpeg

Usage:
    python ingest_kb.py <folder> \
        [--base http://localhost:8000] \
        [--user admin] [--password ****] \
        [--category auditing-standards]

Credentials default to ADMIN_USERNAME / ADMIN_PASSWORD from the environment
(the same values the container was started with). Example:

    ADMIN_USERNAME=admin ADMIN_PASSWORD=123456 \
        python ingest_kb.py ~/standards --category auditing-standards
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import uuid
from pathlib import Path
from urllib import error, request

# Keep in sync with config.ALLOWED_EXTENSIONS.
ALLOWED = {".pdf", ".docx", ".xlsx", ".xls", ".png", ".jpg", ".jpeg"}


def _call(url, *, data=None, headers=None, method=None, timeout=180):
    req = request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except error.HTTPError as exc:
        return exc.code, exc.read()
    except error.URLError as exc:
        return 0, str(exc).encode()


def login(base: str, user: str, password: str) -> str:
    body = json.dumps({"username": user, "password": password}).encode()
    st, raw = _call(f"{base}/admin/login", data=body,
                    headers={"Content-Type": "application/json"}, method="POST")
    if st != 200:
        sys.exit(f"admin login failed ({st}): {raw[:200]!r}")
    return json.loads(raw)["access_token"]


def kb_status(base: str, token: str) -> dict:
    st, raw = _call(f"{base}/admin/status",
                    headers={"Authorization": f"Bearer {token}"})
    return json.loads(raw) if st == 200 else {}


def upload(base: str, token: str, path: Path, category: str | None):
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    pre = b""
    if category:
        pre = (f"--{boundary}\r\n"
               'Content-Disposition: form-data; name="category"\r\n\r\n'
               f"{category}\r\n").encode()
    head = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n").encode()
    body = pre + head + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return _call(f"{base}/admin/upload", data=body,
                 headers={"Authorization": f"Bearer {token}",
                          "Content-Type": f"multipart/form-data; boundary={boundary}"},
                 method="POST", timeout=300)


def doc_status(base: str, token: str, doc_id: str) -> dict | None:
    st, raw = _call(f"{base}/admin/documents/{doc_id}",
                    headers={"Authorization": f"Bearer {token}"})
    return json.loads(raw) if st == 200 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", help="folder of documents to ingest")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--user", default=os.getenv("ADMIN_USERNAME", "admin"))
    ap.add_argument("--password", default=os.getenv("ADMIN_PASSWORD", ""))
    ap.add_argument("--category", default="auditing-standards")
    ap.add_argument("--poll-timeout", type=int, default=1200,
                    help="seconds to wait per document for embedding to finish")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        sys.exit(f"not a folder: {folder}")
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in ALLOWED)
    if not files:
        sys.exit(f"no supported files in {folder} (allowed: {sorted(ALLOWED)})")

    token = login(args.base, args.user, args.password)
    before = kb_status(args.base, token)
    print(f"Logged in. KB has {before.get('total_documents', '?')} docs / "
          f"{before.get('total_chunks', '?')} chunks.")
    print(f"Ingesting {len(files)} file(s) with category={args.category!r}.\n")

    results = []
    for path in files:
        st, raw = upload(args.base, token, path, args.category)
        if st not in (200, 201):
            print(f"  x {path.name}: upload failed ({st}) {raw[:160]!r}")
            results.append((path.name, "upload_failed"))
            continue
        doc_id = json.loads(raw).get("id")
        print(f"  > {path.name}: uploaded (id={doc_id}); processing…")
        deadline = time.time() + args.poll_timeout
        final = "timeout (still processing)"
        while time.time() < deadline:
            time.sleep(3)
            info = doc_status(args.base, token, doc_id)
            if not info:
                continue
            status = info.get("status")
            if status == "active":
                final = f"active — {info.get('total_chunks', '?')} chunks"
                break
            if status == "error":
                final = f"error: {(info.get('error_message') or '')[:140]}"
                break
        print(f"    {path.name}: {final}")
        results.append((path.name, final))

    after = kb_status(args.base, token)
    print("\n=== summary ===")
    for name, outcome in results:
        print(f"  {name}: {outcome}")
    print(f"\nKB now has {after.get('total_documents', '?')} docs / "
          f"{after.get('total_chunks', '?')} chunks "
          f"(was {before.get('total_documents', '?')} / {before.get('total_chunks', '?')}).")


if __name__ == "__main__":
    main()
