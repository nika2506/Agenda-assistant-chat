"""Async backend protocol + in-memory backend for tests."""
from __future__ import annotations

import math
from typing import Protocol
from enum import Enum
from pydantic import BaseModel


class VectorDBType(Enum):
    CHROMA = "chroma"

class VectorConfig(BaseModel):
    db_type: VectorDBType
    host: str
    port: int

class VectorBackend(Protocol):
    async def create_collection(self, name: str, metric: str, dim: int, top_k: int, thr: float) -> None: ...

    async def delete_collection(self, name: str) -> None: ...

    async def upsert(self, name: str, ids: list[str],
                     embeddings: list[list[float]],
                     documents: list[str],
                     metadatas: list[dict]) -> None: ...

    async def query(self, name: str, embedding: list[float],
                    top_k: int, thr: float) -> list[dict]: ...

    async def list_documents(self, name: str) -> list[dict]: ...

    async def delete_documents(self, name: str,
                               document_names: list[str]) -> None: ...

    async def close(self) -> None: ...


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _doc_name(item: dict) -> str:
    meta_name = (item.get('metadata') or {}).get('document_name')
    if meta_name:
        return meta_name
    return item['id'].split('::', 1)[0]

