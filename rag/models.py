from rag.base import BaseChatModel
import httpx


class LlamaLocalChatModel(BaseChatModel):
    def __init__(self, url: str, model_name: str, timeout: float = 100.0):
        self.url = url
        self.model_name = model_name
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.url,
                timeout=self.timeout,
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate(self, prompt: str) -> str:
        if self._client is None:
            await self.connect()

        assert self._client is not None

        try:
            response = await self._client.post(
                "/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            answer = response.json().get("response", "").strip()
            if not answer:
                raise RuntimeError("The local model returned an empty response.")
            return answer
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Could not reach the local Ollama model. Start Ollama and pull the configured model."
            ) from exc
