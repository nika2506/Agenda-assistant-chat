from vectorstore.services.chunking_strategies.base import ChunkingStrategy, clean_text
from typing import Tuple, Any


class PageChunkingStrategy(ChunkingStrategy):
    def __init__(self, chunking_config: dict[str, Any]) -> None:
        pass

    def make_chunks(self, filenames: list[str], documents: list[list[str]], delimiter: str) \
            -> Tuple[list[str], list[str]]:
        sentences: list[str] = []
        indices: list[str] = []
        for document, filename in zip(documents, filenames):
            indices += [f'{filename}{delimiter}{chunk_id + 1}' for chunk_id in range(len(document))]
            sentences += [clean_text(page) for page in document]
        return indices, sentences
