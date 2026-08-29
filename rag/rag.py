from rag.base import BaseChatModel, RAGResponse, BaseRetriever, DocumentChunk
from rag.semantic_cache import SemanticResponseCache


class RAGService:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseChatModel,
    ) -> None:
        self.retriever = retriever
        self.llm = llm

        self.cache = SemanticResponseCache(
            max_size=100,
            similarity_threshold=0.97,
        )

    async def get_response(
        self,
        question: str,
        limit: int = 5,
    ) -> RAGResponse:
        # Exact match: we don't even call the embedding model.
        cached_response = await self.cache.get_exact(question)
        if cached_response is not None:
            return cached_response

        # We need to get the request embedding for semantic cache.
        query_embedding = (
            await self.retriever.get_query_embedding(question)
        )

        # Similar question: embedding is called, but LLM is not called anymore.
        cached_response = await self.cache.get_similar(
            query_embedding
        )

        if cached_response is not None:
            return cached_response

        chunks = await self.retriever.retrieve(
            question,
            limit=limit,
            query_embedding=query_embedding,
        )

        prompt = self._build_prompt(question, chunks)

        await self.llm.connect()

        try:
            answer = await self.llm.generate(prompt)
        finally:
            await self.llm.close()

        response = RAGResponse(
            answer=answer,
            chunks=chunks,
            prompt=prompt,
        )

        await self.cache.put(
            query=question,
            query_embedding=query_embedding,
            response=response,
        )

        return response

    @staticmethod
    def _build_prompt(question: str, chunks: list[DocumentChunk]) -> str:
        context = "\n\n".join(
            f"[Source: {chunk.id}]\n{chunk.content}"
            for chunk in chunks
        )

        return (
            "Answer the question using only context below.\n"
            "If there is no answer in context, just say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
