# SiGMA Agenda Assistant

A grounded question-answering interface for the supplied fictional SiGMA Malta 2026 agenda dataset. It answers questions about sessions, speakers, venues, and exhibitors without relying on any paid service.

## Setup and run

Prerequisites: Python 3.11+ and [Ollama](https://ollama.com/).

## Windows

Install Ollama using the installer from the [Ollama](https://ollama.com/) download page.

Then run the following commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

ollama pull llama3.2:3b
ollama pull nomic-embed-text

uvicorn app:app --reload
```

The Ollama Windows application normally starts the Ollama server automatically. If it is already running, do not run ollama serve separately.

Open `http://127.0.0.1:8000`. Ollama must be running locally. The default chat model is `llama3.2:3b`; The default embedding model is `nomic-embed-text`; configure another pulled model with `OLLAMA_MODEL` and `EMBEDDING_OLLAMA_MODEL`. Set `OLLAMA_URL` when Ollama is not at `http://localhost:11434`.

## Linux

Install Ollama:

```powershell
curl -fsSL https://ollama.com/install.sh | sh
```

Create and activate the Python virtual environment:

```powershell
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the required Ollama models:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

If the Ollama service is not already running, start it in a separate terminal:

```powershell
ollama serve
```

Then start the FastAPI application:

```powershell
source .venv/bin/activate
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. Ollama must be running locally. The default chat model is `llama3.2:3b`; The default embedding model is `nomic-embed-text`; configure another pulled model with `OLLAMA_MODEL` and `EMBEDDING_OLLAMA_MODEL`. Set `OLLAMA_URL` when Ollama is not at `http://localhost:11434`.

## Architecture and AI design

The FastAPI server loads `data/sigma_agenda.json` and creates one semantic chunk for the event overview, each session, and each exhibitor. During application startup, `LocalOllamaEmbeddingFunction` generates embeddings for these chunks in batches of 16. Both the chunks and their embeddings are kept in memory, so the application does not require a vector database.

For each question, `InMemoryRetriever` generates a query embedding and compares it with the stored chunk embeddings using cosine similarity. Results below the configured similarity threshold are discarded, and the five highest-scoring chunks are passed to `RAGService`.

RAGService builds a grounded prompt containing only the retrieved chunks and the user’s question. It then sends the prompt to the local Ollama chat model. The prompt requires the model to answer exclusively from the supplied context and to state when the available agenda data does not contain an answer.

The application uses response caching to reduce redundant model calls. Normalized exact-match questions can reuse a cached response without calling either model again. Near-duplicate questions are detected by comparing query embeddings with cached query embeddings. When their similarity exceeds a high threshold, the cached response is reused, avoiding another chat-model call.

This is retrieval-augmented generation rather than full-dataset stuffing. It reduces the amount of context sent to the chat model, provides relevant grounding sources for the UI, and separates retrieval from answer generation. In-memory cosine search is appropriate for this small dataset while keeping the implementation independent of an external vector database.

BaseEmbeddingFunction, BaseRetriever, and BaseChatModel define the main provider interfaces. `LocalOllamaEmbeddingFunction`, `InMemoryRetriever`, and `LlamaLocalChatModel` provide the current local implementations. Other embedding models, retrievers, vector stores, or chat providers can be added by implementing the corresponding interfaces.

FastAPI’s lifespan handler initializes the embedder, generates the agenda embeddings, constructs the retriever, and stores the resulting `RAGService` in application state. If Ollama is unavailable during initialization, the static browser UI remains accessible and /api/ask returns a service-unavailable error instead of preventing the entire application from starting.

The browser UI includes an empty state, a request-in-progress message, expandable grounding sources, and an error state for unavailable models, initialization failures, or other backend errors.

## Known limitations

- Agenda embeddings are stored only in memory and must be regenerated whenever the application restarts.
- In-memory retrieval performs a linear comparison against every chunk. This is appropriate for the current dataset but will not scale as efficiently as a vector database for large collections.
- The application depends on a running Ollama instance with the configured embedding and chat models already downloaded.
- Prompting strongly constrains the model, but an LLM remains probabilistic. Retrieved source records are displayed so that users can verify generated answers.
- Responses are returned only after generation completes rather than being token-streamed.
- There is no conversational memory. Every question is evaluated independently, except when a cached answer is reused.
- If initialization fails because Ollama is unavailable, the application does not automatically rebuild the RAG service after Ollama becomes available; the server must currently be restarted.

## With more time

- Add hybrid retrieval that combines semantic similarity with structured filters for dates, tracks, rooms, speakers, session IDs, and exhibitor categories.
- Build a retrieval evaluation set and measure metrics such as Recall@K, MRR, answer correctness, faithfulness, and citation accuracy.
- Persist precomputed agenda embeddings and invalidate them using dataset and embedding-model hashes, avoiding regeneration on every application restart.
- Improve semantic caching with TTL and size limits, cache hit telemetry, and safeguards against reusing answers for similar questions containing different dates, times, names, or session IDs.
- Add background initialization retries and readiness reporting so the RAG service can recover automatically if Ollama becomes available after application startup.
- Stream Ollama tokens to the interface and support request cancellation and generation timeouts.
- Add automated unit, API, integration, and browser tests covering retrieval, caching, Ollama failures, response validation, and application lifecycle behavior.
- Add observability for retrieval scores, cache hit rates, embedding and generation latency, error rates, and model usage while avoiding the logging of sensitive query content.
