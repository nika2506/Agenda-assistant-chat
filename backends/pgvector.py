"""Async PGVector backend.

One physical table per collection (`vs_<name>`) with columns
``(id text primary key, document_name text, chunk text, metadata jsonb,
embedding vector(dim))``. Distance operator chosen from the configured
metric: ``<=>`` for cosine, ``<->`` for L2.

Uses ``psycopg_pool.AsyncConnectionPool`` so concurrent upserts/queries
each get their own connection.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from psycopg_pool import AsyncConnectionPool

from vectorstore.services.backends.base import VectorBackend

LOG = logging.getLogger(__name__)


class PGVectorBackend(VectorBackend):
    def __init__(self, db_config: dict[str, Any]):
        self._metric_for: dict[str, str] = {}
        conninfo = (
            f"host={db_config.get('host', 'vector-pg')} "
            f"port={int(db_config.get('port', 5432))} "
            f"user={db_config.get('user', 'postgres')} "
            f"password={db_config.get('password', '')} "
            f"dbname={db_config.get('database', 'vectorstore')}"
        )
        max_size = int(db_config.get('pool_max_size', 10))
        self._pool = AsyncConnectionPool(
            conninfo=conninfo, max_size=max_size,
            kwargs={'autocommit': True}, open=False)
        self._opened = False
        self._thr = None
        self._top_k = None

    async def _ensure_open(self) -> None:
        if not self._opened:
            await self._pool.open()
            self._opened = True

    @staticmethod
    def _table(name: str) -> str:
        return f'vs_{"".join(c for c in name if c.isalnum() or c == "_")}'

    @staticmethod
    def _op(metric: str) -> str:
        if metric == 'cosine':
            return '<=>'
        elif metric == 'l2':
            return '<->'
        elif metric == 'ip':
            return '<#>'
        else:
            raise ValueError(f'Unknown metric: {metric}')

    async def create_collection(self, name: str,  metric: str, dim: int,
                                top_k: int, thr: float):
        LOG.info(f'creating PGVector collection with parameters: '
                 f'metric: {metric}, dim: {dim}, top_k: {top_k}, thr: {thr}')
        await self._ensure_open()
        t = self._table(name)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
                await cur.execute(
                    f'CREATE TABLE IF NOT EXISTS {t} ('
                    ' id text PRIMARY KEY,'
                    ' document_name text,'
                    ' chunk text,'
                    ' metadata jsonb,'
                    f' embedding vector({int(dim)})'
                    ')')
        self._metric_for[name] = metric
        self._thr = thr
        self._top_k = top_k

    async def delete_collection(self, name: str):
        await self._ensure_open()
        t = self._table(name)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f'DROP TABLE IF EXISTS {t}')
        self._metric_for.pop(name, None)

    async def upsert(self, name, ids, embeddings, documents, metadatas):
        await self._ensure_open()
        t = self._table(name)
        rows = []
        for i, _id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            rows.append((_id, meta.get('document_name'), documents[i],
                         json.dumps(meta), embeddings[i]))
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    f'INSERT INTO {t} (id, document_name, chunk, metadata, '
                    'embedding) VALUES (%s, %s, %s, %s, %s) '
                    'ON CONFLICT (id) DO UPDATE SET '
                    'document_name=EXCLUDED.document_name, '
                    'chunk=EXCLUDED.chunk, '
                    'metadata=EXCLUDED.metadata, '
                    'embedding=EXCLUDED.embedding',
                    rows)

    async def query(self, name, embedding, top_k=None, thr=None):
        await self._ensure_open()
        t = self._table(name)
        op = self._op(self._metric_for.get(name, 'cosine'))
        thr = thr or self._thr
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f'SELECT document_name, chunk, '
                    f'embedding {op} %s::vector AS score '
                    f'FROM {t} ORDER BY score ASC LIMIT %s',
                    (embedding, int(top_k or self._top_k)))
                rows = await cur.fetchall()
        return [{'document_name': r[0], 'chunk': r[1], 'score': float(r[2])}
                for r in rows if (not thr or float(r[2]) <= thr)]

    async def list_documents(self, name):
        await self._ensure_open()
        t = self._table(name)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f'SELECT DISTINCT document_name FROM {t}')
                rows = await cur.fetchall()
        return [{'document_name': r[0]} for r in rows if r[0]]

    async def delete_documents(self, name, document_names):
        await self._ensure_open()
        t = self._table(name)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f'DELETE FROM {t} WHERE document_name = ANY(%s)',
                    (list(document_names),))

    async def close(self) -> None:
        if self._opened:
            await self._pool.close()
            self._opened = False
