from typing import Any
from vectorstore.services.embeddings.base import BaseEmbeddingFunction
from vectorstore.services.embeddings.openai import OpenAIEmbeddingFunction
from vectorstore.services.embeddings.local import LocalOllamaEmbeddingFunction
from vectorstore.services.embeddings.stub import StubEmbedder
from vectorstore.schemas import EmbeddingProviderType


provider_to_class = {
    EmbeddingProviderType.OPENAI: OpenAIEmbeddingFunction,
    EmbeddingProviderType.OLLAMA: LocalOllamaEmbeddingFunction,
    EmbeddingProviderType.STUB: StubEmbedder,
}


def get_embedder(provider: EmbeddingProviderType, model: str, config: dict[str, Any]) -> BaseEmbeddingFunction:
    return provider_to_class[provider](model, config)
