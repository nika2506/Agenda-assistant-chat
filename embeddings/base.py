from abc import abstractmethod
from typing import Any, Protocol
import logging

LOG = logging.getLogger(__name__)


class BaseEmbeddingFunction(Protocol):
    def __init__(self, model: str, embedding_config: dict[str, Any]):
        self.model = model
        self._dimension: int | None = None

    async def _calculate_dimension(self) -> int:
        try:
            embeddings = await self.embed_documents(['test_query'])
            return len(embeddings[0])
        except Exception as e:
            LOG.error(f'Calculate dimension for {self.model} failed {e}')
            raise e

    async def get_dimension(self) -> int:
        if self._dimension is None:
            self._dimension = await self._resolve_dimension()
        return self._dimension

    @abstractmethod
    async def _resolve_dimension(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        raise NotImplementedError
