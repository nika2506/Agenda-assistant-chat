import asyncio
import re
from collections import OrderedDict
from rag.base import CacheEntry, RAGResponse


class SemanticResponseCache:
    def __init__(
        self,
        max_size: int = 100,
        similarity_threshold: float = 0.97,
    ) -> None:
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

        self._entries: OrderedDict[str, CacheEntry] = (
            OrderedDict()
        )
        self._lock = asyncio.Lock()

    @staticmethod
    def normalize_query(query: str) -> str:
        query = query.casefold()
        query = re.sub(r"[^\w\s]", " ", query)
        return " ".join(query.split())

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

    async def get_exact(
        self,
        query: str,
    ) -> RAGResponse | None:
        key = self.normalize_query(query)

        async with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                return None

            self._entries.move_to_end(key)
            return entry.response

    async def get_similar(
        self,
        query_embedding: list[float],
    ) -> RAGResponse | None:
        async with self._lock:
            best_key = None
            best_score = -1.0

            for key, entry in self._entries.items():
                score = self._cosine_similarity(
                    query_embedding,
                    entry.embedding,
                )
                if score > best_score:
                    best_key = key
                    best_score = score

            if best_key is None or best_score < self.similarity_threshold:
                return None

            self._entries.move_to_end(best_key)
            return self._entries[best_key].response

    async def put(
        self,
        query: str,
        query_embedding: list[float],
        response: RAGResponse,
    ) -> None:
        key = self.normalize_query(query)

        async with self._lock:
            self._entries[key] = CacheEntry(
                query=query,
                embedding=query_embedding,
                response=response,
            )
            self._entries.move_to_end(key)

            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)