from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def load_or_create_key(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()[:32].ljust(32, b"\0")
    key = AESGCM.generate_key(bit_length=256)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


class SecretBox:
    """AES-256-GCM，把敏感配置加密后写进 SQLite。"""

    def __init__(self, key: bytes):
        self._aes = AESGCM(key)

    @classmethod
    def from_path(cls, path: Path) -> "SecretBox":
        return cls(load_or_create_key(path))

    def encrypt(self, plaintext: str) -> str:
        if not plaintext:
            return ""
        nonce = os.urandom(12)
        blob = self._aes.encrypt(nonce, plaintext.encode("utf-8"), b"issueforge")
        return base64.urlsafe_b64encode(nonce + blob).decode("ascii")

    def decrypt(self, token: str) -> str:
        if not token:
            return ""
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
        nonce, blob = raw[:12], raw[12:]
        return self._aes.decrypt(nonce, blob, b"issueforge").decode("utf-8")


def sha256_text(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update((part or "").encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()
