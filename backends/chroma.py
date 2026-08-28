"""Async ChromaDB backend.

Uses chromadb's AsyncHttpClient. Each "collection" maps 1:1 to a Chroma
collection. Document-level deletion uses metadata filtering on
`document_name` so the REST API can drop all chunks belonging to a file.
"""
from __future__ import annotations

import logging
from typing import Any

import chromadb
import math

from vectorstore.services.backends.base import VectorBackend

LOG = logging.getLogger(__name__)


class ChromaBackend(VectorBackend):
    """Async wrapper around the Chroma HTTP client."""

    def __init__(self, db_config: dict[str, Any]):
        host = db_config.get('host')
        port = db_config.get('port')
        if not host or port is None:
            raise ValueError(
                f'chroma db_config missing host/port: {db_config!r}')
        self._host = host
        self._port = int(port)
        self._client = None
        self._metric = None
        self._thr = None
        self._top_k = None

    async def _ensure_client(self):
        if self._client is None:
            self._client = await chromadb.AsyncHttpClient(
                host=self._host, port=self._port)
        return self._client

    async def create_collection(self, name: str,  metric: str, dim: int,
                                top_k: int, thr: float) -> None:  # pylint: disable=unused-argument
        # Chroma infers dim from the first upsert; argument kept for the
        # backend protocol parity with pgvector.
        LOG.info(f'creating Chroma collection with parameters: '
                 f'metric: {metric}, dim: {dim}, top_k: {top_k}, thr: {thr}')
        self._metric = metric
        self._thr = thr
        self._top_k = top_k
        client = await self._ensure_client()
        await client.get_or_create_collection(
            name=name,
            metadata={'hnsw:space': self._metric})

    async def delete_collection(self, name: str) -> None:
        client = await self._ensure_client()
        try:
            await client.delete_collection(name=name)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOG.warning('chroma delete_collection: %s', exc)

    async def upsert(self, name, ids, embeddings, documents, metadatas):
        client = await self._ensure_client()
        coll = await client.get_collection(name=name)
        # TODO: drop placeholder once real metadata (e.g. document_name)
        # is populated upstream. Chroma rejects empty-dict metadata with
        # "Expected metadata to be a non-empty dict".
        await coll.upsert(
            ids=ids, embeddings=embeddings,
            documents=documents, metadatas=metadatas)

    async def query(self, name, embedding, top_k=None, thr=None):
        client = await self._ensure_client()
        coll = await client.get_collection(name=name)
        res = await coll.query(
            query_embeddings=[embedding], n_results=top_k or self._top_k)
        if self._metric == 'l2':
            res["distances"][0] = [math.sqrt(dist) for dist in res["distances"][0]]
        if self._metric == 'ip':
            res["distances"][0] = [dist-1 for dist in res["distances"][0]]
        hits = []
        thr = thr or self._thr
        for chunk, meta, dist in zip(
                res['documents'][0], res['metadatas'][0],
                res['distances'][0]):
            if not thr or dist <= thr:
                hits.append({
                    'document_name': (meta or {}).get('document_name', ''),
                    'chunk': chunk,
                    'score': float(dist),
                })
        return hits

    async def list_documents(self, name):
        client = await self._ensure_client()
        coll = await client.get_collection(name=name)
        res = await coll.get(include=['metadatas'])
        names = {(m or {}).get('document_name') for m in res['metadatas']}
        return [{'document_name': n} for n in names if n]

    async def delete_documents(self, name, document_names):
        client = await self._ensure_client()
        coll = await client.get_collection(name=name)
        await coll.delete(
            where={'document_name': {'$in': list(document_names)}})

    async def close(self) -> None:
        return None
