from rag.base import BaseRetriever, DocumentChunk
from vectorstores.chroma import ChromaBackend

class VectorStoreRetriever(BaseRetriever):
    def __init__(self, vector_store: ChromaBackend):
        self.vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[DocumentChunk]:
        results = await self.vector_store.search(query_text=query, n_results=limit)
        #print('results: \n', results)
        return [
            DocumentChunk(
                id=res["id"],
                content=res["document"],
                source=res["source"],
                #metadata=item.get("metadata", {}),
                score=res["score"],
            )
            for res in results
        ]

