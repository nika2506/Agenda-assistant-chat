import httpx
import asyncio
from typing import Any
import logging
from embeddings.base import BaseEmbeddingFunction

LOG = logging.getLogger(__name__)


model_to_dimension = {
        "nomic-embed-text:latest": 768,
        "nomic-embed-text-v2-moe:latest": 768
    }


class LocalOllamaEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(self, model: str, embedding_config: dict[str, Any]):
        super().__init__(model, embedding_config)
        self._url = embedding_config['url']
        self._timeout = float(embedding_config.get('timeout', 120))
        self._client: httpx.AsyncClient | None = None

    async def _resolve_dimension(self) -> int:
        known = model_to_dimension.get(self.model)
        if known is not None:
            return known
        if self._client is None:
            await self.connect()
        return await self._calculate_dimension()

    @property
    def name(self) -> str:
        return f"local:{self.model}"

    async def connect(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._url, timeout=self._timeout)

    async def embed_documents(self, documents: list[str], retries: int = 3) -> list[list[float]]:
        if self._client is None:
            await self.connect()

        assert self._client is not None

        for attempt in range(retries):
            try:
                response = await self._client.post(
                    "/api/embed",
                    json={
                        "model": self.model,
                        "input": documents,
                    },
                )
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("embeddings")
                if not isinstance(embeddings, list) or not embeddings or not all(
                    isinstance(inner, list)
                    and all(isinstance(value, (int, float)) for value in inner)
                    for inner in embeddings
                ):
                    if attempt < retries - 1:
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
                    raise ValueError(f"Bad embedding response")

                return embeddings

            except httpx.HTTPError:
                if attempt < retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
                raise

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
