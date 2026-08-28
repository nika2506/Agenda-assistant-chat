from chunking.load_from_file import load_agenda_chunks
from vectorstores.chroma import ChromaBackend
from embeddings.local import LocalOllamaEmbeddingFunction
import asyncio


async def test_vector_search():
    ids, documents, metadatas = load_agenda_chunks("data/sigma_agenda.json")
    print(ids)
    print(documents)
    print(metadatas)



    db_config = {'host': "localhost",
                 'port': 8000}
    config = {
        "collection": "sigma_agenda1",
        "metric": "cosine",
        "top_k": 5,
        "thr": None
    }
    embedding_config = {'url': 'http://localhost:11434'}
    embedder = LocalOllamaEmbeddingFunction(model='nomic-embed-text', embedding_config=embedding_config)
    embeddings = await embedder.embed_documents(documents)
    print(embeddings)

    vector_client = ChromaBackend(db_config)

    await vector_client.create_collection(
        name=config.get('collection'), metric=config.get('metric'), dim=await embedder.get_dimension(),
        top_k=config.get('top_k'), thr=config.get('thr'))

    await vector_client.upsert(name=config.get('collection'),
                               ids=ids,
                               documents=documents,
                               embeddings=embeddings,
                               metadatas=metadatas)

    # Search
    query_text = "SiGMA Malta 2026"
    embedding = await embedder.embed_documents([query_text])
    print('searching....')
    results = await vector_client.query(
        name=config.get('collection'),
        embedding=embedding[0],
        top_k=config.get('top_k'),
        thr=config.get('thr'),
    )
    print(f'search results for {query_text}: {results}')




if __name__ == "__main__":
    asyncio.run(test_vector_search())




