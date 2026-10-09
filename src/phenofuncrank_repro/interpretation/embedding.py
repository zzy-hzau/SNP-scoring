"Embedding service abstraction (SOP §4.2.4)."

from __future__ import annotations

import hashlib
import logging
import os
from typing import Protocol

from phenofuncrank_repro.interpretation.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService(Protocol):
    def available(self) -> bool: ...

    def embed_texts(self, texts: list[str], *, batch_size: int = 256) -> list[list[float]] | None: ...


class LocalEmbeddingService:
    """sentence-transformers embedder, lazily loaded."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._load_error: str | None = None

    def available(self) -> bool:
        return self._ensure_model() is not None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            return self._model
        except Exception as exc:  
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Local embedding unavailable (%s); dense retrieval disabled", self._load_error)
            return None

    def embed_texts(self, texts: list[str], *, batch_size: int = 256) -> list[list[float]] | None:
        model = self._ensure_model()
        if model is None:
            return None
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vectors.extend(model.encode(batch).tolist())
        return vectors


class HashEmbeddingService:
    """Deterministic fixed-dimension fallback (tests / explicit opt-in only)."""

    DIM = 64

    def available(self) -> bool:
        return True

    def embed_texts(self, texts: list[str], *, batch_size: int = 256) -> list[list[float]] | None:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.DIM
            for token in (text or "").lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = digest[0] % self.DIM
                vec[idx] += 1.0
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            out.append(vec)
        return out


class DisabledEmbeddingService:
    "BM25-only mode for retrieval experiments."

    def available(self) -> bool:
        return False

    def embed_texts(self, texts: list[str], *, batch_size: int = 256) -> list[list[float]] | None:
        return None


_SERVICE: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _SERVICE
    if _SERVICE is None:
        backend = os.environ.get("RICE_PH_AGENT_EMBEDDING_BACKEND", "local").lower()
        if backend in {"disabled", "none", "off"}:
            _SERVICE = DisabledEmbeddingService()
        elif backend == "hash":
            _SERVICE = HashEmbeddingService()
        else:
            _SERVICE = LocalEmbeddingService()
    return _SERVICE
