from rag.base import BaseChatModel
import requests
import httpx

class OpenAIChatModel(BaseChatModel): #TODO: do it later
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

    async def generate(self, prompt: str) -> str:
        try:
            response = await self.client.responses.create(
                model=self.model_name,
                input=prompt,
            )
            return response.output_text

        except Exception as e:
            return f"Error: {e}"

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
            data = response.json()
            return data.get("response", "").strip()

        except httpx.HTTPError as e:
            return f"Error: {e}"
