from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from issueforge_server.settings_store import RuntimeConfig

log = logging.getLogger("issueforge_server.embed")

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass
class Embedder:
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        return self.name


class HashEmbedder(Embedder):
    """无模型、无网络的确定性嵌入。用于测试、离线演示和降级。"""

    def __init__(self, dim: int = 384):
        super().__init__(name=f"hash-{dim}", dim=dim)

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN_RE.findall((text or "").lower())
            if not tokens:
                continue
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, index] += sign
                # 字符三元组，缓解中文整句被当成一个 token 的问题
                for n in range(max(1, len(token) - 2)):
                    gram = token[n : n + 3]
                    d = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                    i = int.from_bytes(d[:4], "little") % self.dim
                    out[row, i] += 0.35 * (1.0 if d[4] & 1 else -1.0)
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


class BgeM3Embedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        super().__init__(name=model_name, dim=1024)
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class OpenAIEmbedder(Embedder):
    def __init__(self, base_url: str, api_key: str, model: str = "text-embedding-3-small"):
        super().__init__(name=model, dim=1536)
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key
        self.model = model

    def encode(self, texts: list[str]) -> np.ndarray:
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()["data"]
        vectors = [item["embedding"] for item in data]
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9)


_EMBEDDER_CACHE: dict[tuple[str, str, str], Embedder] = {}


def get_embedder(config: RuntimeConfig) -> Embedder:
    backend = (config.embedding_backend or "auto").lower()
    key = (backend, config.embedding_model or "", config.embedding_api_base or "")
    if key in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[key]
    embedder = _build_embedder(config, backend)
    _EMBEDDER_CACHE[key] = embedder
    return embedder


def _build_embedder(config: RuntimeConfig, backend: str) -> Embedder:
    if backend in {"bge-m3", "bge_m3", "bge"}:
        try:
            return BgeM3Embedder(config.embedding_model or "BAAI/bge-m3")
        except Exception as exc:  # noqa: BLE001
            log.warning("bge-m3 加载失败，降级为 hash 嵌入：%s", exc)
            return HashEmbedder()
    if backend == "openai" and config.embedding_api_key:
        return OpenAIEmbedder(
            config.embedding_api_base or "https://api.openai.com/v1",
            config.embedding_api_key,
            config.embedding_model or "text-embedding-3-small",
        )
    if backend == "auto":
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except Exception:
            return HashEmbedder()
        if _bge_cached_locally(config.embedding_model or "BAAI/bge-m3"):
            try:
                return BgeM3Embedder(config.embedding_model or "BAAI/bge-m3")
            except Exception as exc:  # noqa: BLE001
                log.warning("bge-m3 加载失败，降级为 hash 嵌入：%s", exc)
        else:
            log.info("本地没有 bge-m3 缓存，auto 模式使用 hash 嵌入（可设置 ISSUEFORGE_ALLOW_MODEL_DOWNLOAD=1 允许下载）")
        return HashEmbedder()
    return HashEmbedder()


def _bge_cached_locally(model: str) -> bool:
    if os.environ.get("ISSUEFORGE_ALLOW_MODEL_DOWNLOAD") == "1":
        return True
    hub = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    slug = "models--" + model.replace("/", "--")
    return (hub / slug).exists()


def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob: bytes | None, dim: int | None = None) -> np.ndarray | None:
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    if dim and arr.size != dim:
        return None
    return arr


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def approx_tokens(text: str) -> int:
    return max(1, int(len(text or "") / 4) + 1)


def _unused(value: float) -> float:
    # 保留一个纯数学辅助，方便未来扩展 BM25 归一化
    return math.tanh(value)
