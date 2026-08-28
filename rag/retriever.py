from rag.base import BaseRetriever, DocumentChunk
from vectorstores.chroma import ChromaBackend
from pathlib import Path
from embeddings.local import LocalOllamaEmbeddingFunction
from chunking.load_from_file import load_agenda_chunks
from typing import Any


class InMemoryRetriever(BaseRetriever):
    def __init__(self, chunks: list[dict[str, Any]], embeddings: list[list[float]], embedding_function) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks and embeddings must match"
            )
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedding_function = embedding_function

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

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        thr: float = 0.5,
    ) -> list[DocumentChunk]:
        query_embeddings = (
            await self.embedding_function.embed_documents([query])
        )
        if not query_embeddings:
            raise ValueError("Failed to create query embedding")
        query_embedding = query_embeddings[0]

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
