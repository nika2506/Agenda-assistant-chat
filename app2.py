"""Grounded agenda question-answering web application."""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from chunking.load_from_file import load_agenda_chunks
from rag.models import LlamaLocalChatModel
from embeddings.local import LocalOllamaEmbeddingFunction

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sigma_agenda.json"
STOP_WORDS = {
    "a", "an", "and", "are", "at", "can", "do", "for", "from", "how", "i",
    "in", "is", "me", "of", "on", "or", "the", "to", "what", "when", "where",
    "which", "who", "with", "will", "about", "does", "there", "any", "tell",
}
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


def load_sources(path: Path) -> list[Source]:
    data = json.loads(path.read_text(encoding="utf-8"))
    event = data["event"]
    sources = [
        Source(
            id="event",
            label="Event overview",
            content=(
                f"Event: {event['name']}\nVenue: {event['venue']}\nDates: {event['dates']}\n"
                f"Note: {event['note']}"
            ),
            keywords=terms(" ".join(str(value) for value in event.values())),
        )
    ]
    for session in data.get("sessions", []):
        content = (
            f"Session ID: {session['id']}\nTitle: {session['title']}\nTrack: {session['track']}\n"
            f"Date: {session['day']}\nTime: {session['start']}-{session['end']}\nRoom: {session['room']}\n"
            f"Speakers: {', '.join(session.get('speakers', [])) or 'Not specified'}\n"
            f"Description: {session['abstract']}"
        )
        sources.append(Source(session["id"], session["title"], content, terms(content)))
    for exhibitor in data.get("exhibitors", []):
        content = (
            f"Exhibitor ID: {exhibitor['id']}\nName: {exhibitor['name']}\n"
            f"Category: {exhibitor['category']}\nStand: {exhibitor['stand']}\n"
            f"Description: {exhibitor['description']}"
        )
        sources.append(Source(exhibitor["id"], exhibitor["name"], content, terms(content)))
    return sources


class AgendaRetriever:
    def __init__(self, sources: list[Source]) -> None:
        self.sources = sources

    def retrieve(self, question: str, limit: int = 5) -> list[Source]:
        query_terms = terms(question)
        scored = [
            (len(query_terms & source.keywords), source)
            for source in self.sources
        ]
        return [source for score, source in sorted(scored, key=lambda item: item[0], reverse=True)[:limit] if score]


def build_prompt(question: str, sources: list[Source]) -> str:
    context = "\n\n".join(f"[{source.id}: {source.label}]\n{source.content}" for source in sources)
    return f"""You are the SiGMA Malta 2026 agenda assistant. Answer only from the supplied dataset excerpts.
Do not invent or infer sessions, speakers, exhibitors, dates, times, rooms, companies, or facts.
If the excerpts do not explicitly answer the question, reply exactly: "I couldn't find that in the provided agenda data."
Keep the answer concise. Mention source IDs in square brackets for every factual claim, for example [S002].

Dataset excerpts:
{context}

Question: {question}
Answer:"""


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class SourceResponse(BaseModel):
    id: str
    label: str
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


#sources = load_sources(DATA_PATH)
async def make_chunks_and_embeddings():
    embedder = LocalOllamaEmbeddingFunction(model=embedding_model_name, embedding_config={'url': ollama_url})
    chunks = load_agenda_chunks(DATA_PATH)
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    embeddings = await embedder.embed_documents(texts)
    return

#sources = load_agenda_chunks(DATA_PATH)
#retriever = AgendaRetriever(sources)
#model = OllamaChatModel(
#    base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
#    model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
model = LlamaLocalChatModel(
    url=ollama_url,
    model_name=retriever_model_name,
)

app = FastAPI(title="SiGMA Agenda Assistant")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str | int]:
    return {"status": "ok", "sources": len(sources), "model": model.model_name}


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Enter a question before sending it.")

    matches = retriever.retrieve(question)
    if not matches:
        return AskResponse(
            answer="I couldn't find that in the provided agenda data.",
            sources=[],
        )

    try:
        answer = await model.generate(build_prompt(question, matches))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AskResponse(
        answer=answer,
        sources=[SourceResponse(id=source.id, label=source.label, content=source.content) for source in matches],
    )
