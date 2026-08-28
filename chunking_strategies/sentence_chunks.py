from vectorstore.services.chunking_strategies.base import ChunkingStrategy, clean_text
from typing import Tuple, Any


class SentenceChunkingStrategy(ChunkingStrategy):
    def __init__(self, chunking_config: dict[str, Any]) -> None:
        self.max_chunk_len = chunking_config['max_chunk_len']
        self.merge_short_sentences = chunking_config['merge_short_sentences']

    def merge_short_sentences_func(self, documents: list[list[str]], sep: str) -> list[list[str]]:
        new_documents: list[list[str]] = []
        for document in documents:
            new_document = []
            prev_line = ""
            for line in document:
                new_line = prev_line + sep + line
                if len(new_line) <= self.max_chunk_len:
                    prev_line = new_line
                else:
                    new_document.append(prev_line)
                    prev_line = line
            if prev_line:
                new_document.append(prev_line)
            new_documents.append(new_document)
        return new_documents

    def crop_sentence(self, sentence: str, limit: int) -> str:
        max_range_to_dot = 100
        cropped_sentence = ''
        end_symbols = ['.', '!', '?']
        comma = False
        for i, symbol in enumerate(reversed(sentence[:limit])):
            if symbol in end_symbols:
                cropped_sentence = sentence[:(limit - i)]
                break
            elif (symbol == ',' or symbol == ';' or symbol == ':') and not comma:
                cropped_sentence = sentence[:(limit - i)]
                comma = True
            elif symbol == ' ' and cropped_sentence == '':
                cropped_sentence = sentence[:(limit - i)]
            if i >= max_range_to_dot:
                break
        if cropped_sentence == '':
            cropped_sentence = sentence[:limit]
        return cropped_sentence

    def make_chunks(self, filenames: list[str], documents: list[list[str]], delimiter: str) \
            -> Tuple[list[str], list[str]]:
        if self.merge_short_sentences:
            documents = self.merge_short_sentences_func(documents, sep=" ")
        sentences: list[str] = []
        indices: list[str] = []
        for document, filename in zip(documents, filenames):
            i = 1
            for line in document:
                line = clean_text(line)
                if not line:
                    continue
                if len(line) > self.max_chunk_len:
                    end_idx = 0
                    while end_idx < len(line):
                        cropped_sentence = self.crop_sentence(line[end_idx:], self.max_chunk_len)
                        sentences.append(cropped_sentence.lstrip())
                        end_idx += len(cropped_sentence)
                        indices.append(f'{filename}{delimiter}{i}')
                        i += 1
                else:
                    sentences.append(line)
                    indices.append(f'{filename}{delimiter}{i}')
                    i += 1
        return indices, sentences
