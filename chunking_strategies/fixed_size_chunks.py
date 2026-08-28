from typing import Tuple, Any
import tiktoken
from vectorstore.services.chunking_strategies.base import ChunkingStrategy, clean_text


class FixedSizeChunkingStrategy(ChunkingStrategy):
    def __init__(self, chunking_config: dict[str, Any]) -> None:
        self.model_name = chunking_config['model_name']
        self.num_tokens = chunking_config['num_tokens']
        if self.num_tokens <= 0:
            raise ValueError('num_tokens must be positive')
        self.overlap_tokens = chunking_config['overlap_tokens']
        if self.overlap_tokens < 0 or self.overlap_tokens >= self.num_tokens:
            raise ValueError('overlap_tokens must be from 0 to num_tokens - 1')
        try:
            self.encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def split_large_text_by_tokens(self, text: str) -> list[str]:
        token_ids = self.encoding.encode(text)
        result: list[str] = []

        if len(token_ids) <= self.num_tokens:
            return [text]

        step = max(1, self.num_tokens - self.overlap_tokens)
        for start in range(0, len(token_ids), step):
            end = start + self.num_tokens
            piece_ids = token_ids[start:end]
            piece_text = self.encoding.decode(piece_ids).strip()
            if piece_text:
                result.append(piece_text)
            if end >= len(token_ids):
                break

        return result

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def make_chunks(self, filenames: list[str], documents: list[list[str]], delimiter: str) \
            -> Tuple[list[str], list[str]]:
        sentences: list[str] = []
        indices: list[str] = []
        for document, filename in zip(documents, filenames):
            if len(document) > 1:
                document = [" ".join(document)]

            chunked_doc = self.split_large_text_by_tokens(clean_text(document[0]))
            indices += [f'{filename}{delimiter}{chunk_id + 1}' for chunk_id in range(len(chunked_doc))]
            sentences += chunked_doc
        return indices, sentences
