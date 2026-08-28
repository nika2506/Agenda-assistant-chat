from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RAGModelConfig:
    model_name: str
    url: str

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
    ) -> list[DocumentChunk]:
        raise NotImplementedError

class BaseChatModel(ABC):
    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        ...