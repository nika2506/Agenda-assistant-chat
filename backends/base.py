"""Async backend protocol + in-memory backend for tests."""
from __future__ import annotations

import math
from typing import Protocol


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


class InMemoryBackend(VectorBackend):
    """Dependency-free async backend used for tests."""

    def __init__(self):
        self._collections: dict[str, dict] = {}
        self._metric = None

    async def create_collection(self, name, metric, dim, top_k, thr):
        self._metric = metric
        self._collections.setdefault(
            name, {'metric': self._metric, 'dim': dim, 'items': []})

    async def delete_collection(self, name):
        self._collections.pop(name, None)

    def _coll(self, name):
        coll = self._collections.get(name)
        if coll is None:
            raise KeyError(f'collection {name!r} not found')
        return coll

    async def upsert(self, name, ids, embeddings, documents, metadatas):
        coll = self._coll(name)
        existing = {it['id']: i for i, it in enumerate(coll['items'])}
        for i, _id in enumerate(ids):
            row = {
                'id': _id,
                'embedding': embeddings[i],
                'document': documents[i],
                'metadata': metadatas[i] if i < len(metadatas) else {},
            }
            if _id in existing:
                coll['items'][existing[_id]] = row
            else:
                coll['items'].append(row)

    async def query(self, name, embedding, top_k, thr):
        coll = self._coll(name)
        metric = coll.get('metric', 'cosine')
        scored = []
        for it in coll['items']:
            if metric == 'cosine':
                score = 1.0 - _cosine(embedding, it['embedding'])
            else:
                score = _l2(embedding, it['embedding'])
            scored.append({
                'document_name': _doc_name(it),
                'chunk': it['document'],
                'score': score,
            })
        scored.sort(key=lambda x: x['score'])
        return scored[:top_k]

    async def list_documents(self, name):
        coll = self._coll(name)
        names = {_doc_name(it) for it in coll['items']}
        return [{'document_name': n} for n in names if n]

    async def delete_documents(self, name, document_names):
        coll = self._coll(name)
        target = set(document_names)
        coll['items'] = [
            it for it in coll['items']
            if _doc_name(it) not in target
        ]

    async def close(self) -> None:
        return None
