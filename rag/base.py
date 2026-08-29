from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    query: str
    embedding: list[float]
    response: RAGResponse

@dataclass
class DocumentChunk:
    id: str
    content: str
    metadata: dict[str, Any]
    score: float | None = None

@dataclass
class RAGResponse:
    answer: str
    chunks: list[DocumentChunk]
    prompt: str | None = None


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        query_embedding: list[float] | None = None,
    ) -> list[DocumentChunk]:
        raise NotImplementedError

    async def get_query_embedding(self, query: str) -> list[float]:
        raise NotImplementedError

class BaseChatModel(ABC):
    @abstractmethod
    async def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError