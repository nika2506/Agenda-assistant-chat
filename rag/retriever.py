from rag.base import BaseRetriever, DocumentChunk
from typing import Any
from collections import OrderedDict
import asyncio


class InMemoryRetriever(BaseRetriever):
    def __init__(self, chunks: list[dict[str, Any]], embeddings: list[list[float]],
                 embedding_function, #query_cache_size: int=256
                 ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks and embeddings must match"
            )
        # if query_cache_size <= 0:
        #     raise ValueError(
        #         "query_cache_size must be greater than zero"
        #     )
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedding_function = embedding_function
        # self._query_cache_size = query_cache_size
        # self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        # self._cache_lock = asyncio.Lock()

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(query.strip().split())

    @staticmethod
    def _cosine_similarity(
            first: list[float],
            second: list[float],
    ) -> float:
        dot_product = sum(
            left * right
            for left, right in zip(first, second)
        )
        first_norm = sum(value * value for value in first) ** 0.5
        second_norm = sum(value * value for value in second) ** 0.5
        if not first_norm or not second_norm:
            return 0.0
        return dot_product / (first_norm * second_norm)

    async def get_query_embedding(
        self,
        query: str,
    ) -> list[float]:
        # cache_key = self._normalize_query(query)
        # async with self._cache_lock:
        #     cached_embedding = self._query_cache.get(cache_key)
        #
        #     if cached_embedding is not None:
        #         self._query_cache.move_to_end(cache_key)
        #         return cached_embedding

        query_embeddings = (
            await self.embedding_function.embed_documents([self._normalize_query(query)])
        )

        if (
            len(query_embeddings) != 1
            or not query_embeddings[0]
        ):
            raise ValueError(
                "Failed to create query embedding"
            )

        query_embedding = query_embeddings[0]

        # async with self._cache_lock:
        #     self._query_cache[cache_key] = query_embedding
        #     self._query_cache.move_to_end(cache_key)
        #
        #     while len(self._query_cache) > self._query_cache_size:
        #         self._query_cache.popitem(last=False)

        return query_embedding

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        thr: float = 0.5,
        query_embedding: list[float] | None = None,
    ) -> list[DocumentChunk]:

        if query_embedding is None:
            query_embedding = await self.get_query_embedding(query)

        results = []
        for chunk, embedding in zip(self.chunks, self.embeddings):
            metadata = dict(chunk["metadata"])

            score = self._cosine_similarity(
                query_embedding,
                embedding,
            )

            if thr is not None and score < thr:
                continue

            results.append(DocumentChunk(
                id=chunk["id"],
                content=chunk["text"],
                metadata=metadata,
                score=score,
            ))

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:limit]
