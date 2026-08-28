from rag.base import BaseChatModel, RAGResponse, BaseRetriever, DocumentChunk

class RAGService:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseChatModel,
    ):
        self.retriever = retriever
        self.llm = llm

    async def get_response(
            self,
            question: str,
            limit: int = 5,
    ) -> RAGResponse:
        chunks = await self.retriever.retrieve(
            question,
            limit=limit,
        )

        if not chunks:
            return RAGResponse(
                answer="I couldn't find that in the provided agenda data.",
                chunks=[],
                prompt="",
            )

        prompt = self._build_prompt(question, chunks)

        await self.llm.connect()

        try:
            answer = await self.llm.generate(prompt)
        finally:
            await self.llm.close()

        return RAGResponse(
            answer=answer,
            chunks=chunks,
            prompt=prompt,
        )

    async def get_response2(self, question: str, limit: int = 5) -> RAGResponse:
        chunks = await self.retriever.retrieve(question, limit=limit)
        prompt = self._build_prompt(question, chunks)
        print('prompt: \n', prompt)
        await self.llm.connect()
        answer = await self.llm.generate(prompt)
        await self.llm.close()

        return RAGResponse(
            answer=answer,
            chunks=chunks,
            prompt=prompt,
        )

    async def answer(self, question: str, limit: int = 5):
        rag_response = await self.get_response(question, limit)
        filenames = list(dict.fromkeys([chunk.id.rsplit("_", 1)[0] for chunk in rag_response.chunks]))

        answer = (f"Answer:\n{rag_response.answer}\n"
                  f"For more detailed information, please see the files:\n"
                  f"{filenames}")
        return answer

    def _build_prompt(
        self,
        question: str,
        chunks: list[DocumentChunk],
    ) -> str:
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