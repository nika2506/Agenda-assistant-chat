from abc import ABC, abstractmethod
from typing import Tuple, Any
import re


def clean_text(text: str, for_structural_chunking: bool = False) -> str:
    text = text.lower()
    text = re.sub(r"<!--.*?-->", "", text)
    if not for_structural_chunking:
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\x00", "")
    return text.strip()


class ChunkingStrategy(ABC):
    @abstractmethod
    def __init__(self, chunking_config: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def make_chunks(self, filenames: list[str],
                    documents: list[list[str]],
                    delimiter: str) -> Tuple[list[str], list[str]]: ...
