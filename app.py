"""Grounded agenda question-answering web application."""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from chunking.load_from_file import load_agenda_chunks
from rag.models import LlamaLocalChatModel
from rag.retriever import InMemoryRetriever
from rag.rag import RAGService
from rag.base import DocumentChunk
from embeddings.local import LocalOllamaEmbeddingFunction

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sigma_agenda.json"
ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434")
embedding_model_name = os.getenv("EMBEDDING_OLLAMA_MODEL", "nomic-embed-text")
retriever_model_name=os.getenv("OLLAMA_MODEL", "llama3.2:3b")


@dataclass
class Source:
    id: str
    label: str
    content: str
    keywords: set[str]


def terms(value: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9]+", value.lower())
        if len(word) > 1 and word not in STOP_WORDS
    }


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class SourceResponse(BaseModel):
    id: str
    label: str
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


async def make_chunks_and_embeddings(embedder):
    chunks = load_agenda_chunks(DATA_PATH)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = []
    for start in range(0, len(texts), 16):
        embeddings.extend(
            await embedder.embed_documents(texts[start:start + 16], retries=1)
        )
    return chunks, embeddings


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = LocalOllamaEmbeddingFunction(
        model=embedding_model_name,
        embedding_config={
            "url": ollama_url,
            "timeout": 60,
        },
    )

    try:
        # Keep the UI available if Ollama is unavailable; /api/ask reports a
        # service error instead of making the entire web app fail at startup.
        app.state.source_count = len(load_agenda_chunks(DATA_PATH))
        app.state.model_name = retriever_model_name
        app.state.rag_service = None
        app.state.initialization_error = None

        try:
            chunks, embeddings = await make_chunks_and_embeddings(
                embedder
            )
            retriever = InMemoryRetriever(
                chunks=chunks,
                embeddings=embeddings,
                embedding_function=embedder,
            )
            model = LlamaLocalChatModel(
                url=ollama_url,
                model_name=retriever_model_name,
                timeout=120,
            )
            app.state.rag_service = RAGService(
                retriever=retriever,
                llm=model,
            )
        except (httpx.HTTPError, RuntimeError, ValueError):
            app.state.initialization_error = (
                "Could not initialize local agenda search. Start Ollama and pull "
                f"the configured embedding model ({embedding_model_name})."
            )
        yield
    finally:
        await embedder.close()


#app = FastAPI(lifespan=lifespan)


app = FastAPI(title="SiGMA Agenda Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")

@app.get("/api/health")
async def health(request: Request) -> dict[str, str | int]:
    return {
        "status": "ok",
        "sources": request.app.state.source_count,
        "model": request.app.state.model_name,
    }


def get_chunk_label(chunk: DocumentChunk) -> str:
    return str(
        chunk.metadata.get("title")
        or chunk.metadata.get("name")
        or chunk.id
    )

@app.post("/api/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    request: Request,
) -> AskResponse:
    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Enter a question before sending it.",
        )

    rag_service: RAGService | None = request.app.state.rag_service
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail=request.app.state.initialization_error or "Agenda search is unavailable.",
        )

    try:
        response = await rag_service.get_response(
            question,
            limit=5,
        )
    except (ConnectionError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return AskResponse(
        answer=response.answer,
        sources=[
            SourceResponse(
                id=chunk.id,
                label=get_chunk_label(chunk),
                content=chunk.content,
            )
            for chunk in response.chunks
        ],
    )
