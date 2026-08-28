from openai import AsyncOpenAI
from typing import Any
import asyncio
import httpx
from vectorstore.services.embeddings.base import BaseEmbeddingFunction


model_to_dimension = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072
    }


class OpenAIEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(self,  model: str, embedding_config: dict[str, Any]):
        super().__init__(model, embedding_config)
        self._client = AsyncOpenAI(api_key=embedding_config['api_key'])

    async def _resolve_dimension(self) -> int:
        known = model_to_dimension.get(self.model)
        if known is not None:
            return known
        return await self._calculate_dimension()

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    async def embed_documents(self, documents: list[str], retries: int = 3) -> list[list[float]]:
        # TODO: no retries fir some errors (example: "Error code: 429 You exceeded your current quota")
        for attempt in range(retries):
            try:
                response = await self._client.embeddings.create(
                    model=self.model,
                    input=documents,
                )
                embeddings = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
                if (
                        not embeddings
                        or not isinstance(embeddings, list) and all(
                            isinstance(inner, list) and all(isinstance(x, float) for x in inner)
                            for inner in embeddings
                        )
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
