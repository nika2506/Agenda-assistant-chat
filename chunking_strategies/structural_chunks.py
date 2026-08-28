from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Optional
import tiktoken
from vectorstore.services.chunking_strategies.base import ChunkingStrategy, clean_text
from typing import Tuple, Any


@dataclass
class Section:
    title: str
    level: int
    content: str
    start_pos: int
    end_pos: int


@dataclass
class Chunk:
    chunk_id: str
    section_title: str
    section_level: int
    chunk_index_in_section: int
    token_count: int
    text: str
    start_pos: int
    end_pos: int


class StructuralChunkingStrategy(ChunkingStrategy):
    HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

    def __init__(self, chunking_config: dict[str, Any]) -> None:
        self.max_tokens = chunking_config['max_tokens']
        if self.max_tokens <= 0:
            raise ValueError('max_tokens must be positive')  # TODO: move ValueErrors to pydantic
        self.overlap_tokens = chunking_config['overlap_tokens']
        if self.overlap_tokens < 0 or self.overlap_tokens >= self.max_tokens:
            raise ValueError('overlap_tokens must be from 0 to max_tokens - 1')
        self.min_chunk_tokens = chunking_config['min_chunk_tokens']
        if self.min_chunk_tokens <= 0 or self.min_chunk_tokens >= self.max_tokens:
            raise ValueError('min_chunk_tokens must be positive and less or equal than max_tokens')
        try:
            self.encoding = tiktoken.encoding_for_model(chunking_config['model_name'])
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def split_markdown_sections(self, text: str) -> List[Section]:

        matches = list(self.HEADER_RE.finditer(text))

        if not matches:
            return [
                Section(
                    title="root",
                    level=0,
                    content=text.strip(),
                    start_pos=0,
                    end_pos=len(text),
                )
            ]

        sections: List[Section] = []

        first = matches[0]
        if first.start() > 0:
            preamble = text[: first.start()].strip()
            if preamble:
                sections.append(
                    Section(
                        title="preamble",
                        level=0,
                        content=preamble,
                        start_pos=0,
                        end_pos=first.start(),
                    )
                )

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()

            content_start = match.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[content_start:content_end].strip()

            sections.append(
                Section(
                    title=title,
                    level=level,
                    content=content,
                    start_pos=match.start(),
                    end_pos=content_end,
                )
            )

        return sections

    def split_into_paragraphs(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return paragraphs

    def split_large_paragraph_by_sentences(self, paragraph: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9])", paragraph.strip())
        return [s.strip() for s in sentences if s.strip()]

    def split_large_text_by_tokens(self, text: str) -> List[str]:
        token_ids = self.encoding.encode(text)
        result: List[str] = []

        if len(token_ids) <= self.max_tokens:
            return [text]

        step = max(1, self.max_tokens - self.overlap_tokens)
        for start in range(0, len(token_ids), step):
            end = start + self.max_tokens
            piece_ids = token_ids[start:end]
            piece_text = self.encoding.decode(piece_ids).strip()
            if piece_text:
                result.append(piece_text)
            if end >= len(token_ids):
                break

        return result

    def build_chunks_from_section(self, section: Section, doc_id: str) -> List[Chunk]:
        if not section.content.strip():
            return []

        paragraphs = self.split_into_paragraphs(section.content)
        if not paragraphs:
            paragraphs = [section.content.strip()]

        chunk_texts: List[str] = []
        current_parts: List[str] = []
        current_tokens = 0

        def flush_current() -> None:
            nonlocal current_parts, current_tokens
            if current_parts:
                joined = "\n\n".join(current_parts).strip()
                if joined:
                    chunk_texts.append(joined)
            current_parts = []
            current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            if para_tokens > self.max_tokens:
                flush_current()

                sentences = self.split_large_paragraph_by_sentences(para)
                if len(sentences) <= 1:
                    chunk_texts.extend(self.split_large_text_by_tokens(para))
                    continue

                sentence_buf: List[str] = []
                sentence_buf_tokens = 0

                def flush_sentence_buf() -> None:
                    nonlocal sentence_buf, sentence_buf_tokens
                    if sentence_buf:
                        text_piece = " ".join(sentence_buf).strip()
                        if self.count_tokens(text_piece) > self.max_tokens:
                            chunk_texts.extend(self.split_large_text_by_tokens(text_piece))
                        else:
                            chunk_texts.append(text_piece)
                    sentence_buf = []
                    sentence_buf_tokens = 0

                for sent in sentences:
                    sent_tokens = self.count_tokens(sent)

                    if sent_tokens > self.max_tokens:
                        flush_sentence_buf()
                        chunk_texts.extend(self.split_large_text_by_tokens(sent))
                        continue

                    if sentence_buf_tokens + sent_tokens <= self.max_tokens:
                        sentence_buf.append(sent)
                        sentence_buf_tokens += sent_tokens
                    else:
                        flush_sentence_buf()
                        sentence_buf = [sent]
                        sentence_buf_tokens = sent_tokens

                flush_sentence_buf()
                continue

            if current_tokens + para_tokens <= self.max_tokens:
                current_parts.append(para)
                current_tokens += para_tokens
            else:
                flush_current()
                current_parts = [para]
                current_tokens = para_tokens

        flush_current()

        chunk_texts = self.apply_overlap(chunk_texts)

        chunks: List[Chunk] = []
        running_pos = section.start_pos

        for idx, chunk_text in enumerate(chunk_texts):
            token_count = self.count_tokens(chunk_text)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{self.slugify(section.title)}:{idx}",
                    section_title=section.title,
                    section_level=section.level,
                    chunk_index_in_section=idx,
                    token_count=token_count,
                    text=chunk_text,
                    start_pos=running_pos,
                    end_pos=running_pos + len(chunk_text),
                )
            )
            running_pos += len(chunk_text)

        return chunks

    def apply_overlap(self, chunk_texts: List[str]) -> List[str]:
        if not chunk_texts or self.overlap_tokens <= 0:
            return chunk_texts

        overlapped: List[str] = [chunk_texts[0]]

        for i in range(1, len(chunk_texts)):
            prev = chunk_texts[i - 1]
            curr = chunk_texts[i]

            prev_ids = self.encoding.encode(prev)
            overlap_ids = prev_ids[-self.overlap_tokens:] if prev_ids else []
            overlap_text = self.encoding.decode(overlap_ids).strip()

            merged = f"{overlap_text}\n\n{curr}".strip() if overlap_text else curr
            overlapped.append(merged)

        return overlapped

    def slugify(self, value: str) -> str:
        value = value.lower()
        value = re.sub(r"[^a-zа-яё0-9]+", "-", value, flags=re.IGNORECASE)
        return value.strip("-") or "section"

    def chunk_document(self, text: str, doc_id: str = "doc") -> List[Chunk]:
        sections = self.split_markdown_sections(text)
        all_chunks: List[Chunk] = []

        for section in sections:
            if not section.content.strip():
                continue
            section_chunks = self.build_chunks_from_section(section, doc_id=doc_id)
            all_chunks.extend(section_chunks)

        all_chunks = self.merge_tiny_chunks(all_chunks, doc_id)
        return all_chunks

    def make_chunks(self, filenames: list[str], documents: list[list[str]],
                    delimiter: str) -> Tuple[list[str], list[str]]:
        sentences: list[str] = []
        indices: list[str] = []
        for document, filename in zip(documents, filenames):
            chunk_id = 1
            if len(document) > 1:
                document = [" ".join(document)]
            chunks = self.chunk_document(clean_text(document[0], for_structural_chunking=True))
            for ch in chunks:
                sentences.append(ch.section_title + " \n" + ch.text)
                indices.append(f"{filename}{delimiter}{chunk_id}")
                chunk_id += 1
        return indices, sentences

    def merge_tiny_chunks(self, chunks: List[Chunk], doc_id: str) -> List[Chunk]:
        if not chunks:
            return chunks

        merged: List[Chunk] = []
        buffer: Optional[Chunk] = None

        for chunk in chunks:
            if buffer is None:
                buffer = chunk
                continue

            if buffer.token_count < self.min_chunk_tokens:
                candidate = f"{buffer.text}\n\n{chunk.text}".strip()
                candidate_tokens = self.count_tokens(candidate)

                if candidate_tokens <= self.max_tokens:
                    buffer = Chunk(
                        chunk_id=buffer.chunk_id,
                        section_title=buffer.section_title,
                        section_level=buffer.section_level,
                        chunk_index_in_section=buffer.chunk_index_in_section,
                        token_count=candidate_tokens,
                        text=candidate,
                        start_pos=buffer.start_pos,
                        end_pos=chunk.end_pos,
                    )
                    continue
                else:
                    merged.append(buffer)
                    buffer = chunk
            else:
                merged.append(buffer)
                buffer = chunk

        if buffer is not None:
            merged.append(buffer)

        reindexed: List[Chunk] = []
        for i, ch in enumerate(merged):
            reindexed.append(
                Chunk(
                    chunk_id=f"{doc_id}:chunk:{i}",
                    section_title=ch.section_title,
                    section_level=ch.section_level,
                    chunk_index_in_section=i,
                    token_count=ch.token_count,
                    text=ch.text,
                    start_pos=ch.start_pos,
                    end_pos=ch.end_pos,
                )
            )
        return reindexed
