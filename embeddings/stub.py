import hashlib
from typing import Any
from vectorstore.services.embeddings.base import BaseEmbeddingFunction


class StubEmbedder(BaseEmbeddingFunction):
    """Deterministic, dependency-free embedder for tests / local dev."""
    def __init__(self, model: str, embedding_config: dict[str, Any]):
        super().__init__(model, embedding_config)
        self._dimension = 16

    async def _resolve_dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return "stub_model"

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        out = []
        for t in documents:
            digest = hashlib.sha256(t.encode('utf-8')).digest()
            vec = [b / 255.0 for b in digest[: self._dimension]]
            out.append(vec)
        return out
