# How to Run a Chroma Server

## Option 1: CLI (simplest)

Requires `chromadb` installed in your Python environment.

```bash
pip install chromadb
```

Start the server:

```bash
chroma run --host localhost --port 8000
```

By default data is stored in the current directory. To specify a persistent path:

```bash
chroma run --host localhost --port 8000 --path ./chroma_data
```

## Option 2: Docker

```bash
docker run -d \
  --name chromadb \
  -p 8000:8000 \
  -v ./chroma_data:/chroma/chroma \
  chromadb/chroma
```

- `-d` — run in background
- `-v` — persist data to `./chroma_data` on the host
- Remove `-v` if you don't need persistence

Stop / remove:

```bash
docker stop chromadb
docker rm chromadb
```

## Option 3: Docker Compose

Create a `docker-compose.yml`:

```yaml
services:
  chromadb:
    image: chromadb/chroma
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - ANONYMIZED_TELEMETRY=False       # optional: disable telemetry
      # - CHROMA_SERVER_AUTHN_PROVIDER=chromadb.auth.token_authn.TokenAuthenticationServerProvider
      # - CHROMA_SERVER_AUTHN_CREDENTIALS=my-secret-token

volumes:
  chroma_data:
```

Run:

```bash
docker compose up -d
```

Stop:

```bash
docker compose down
```

## Option 4: Chroma Cloud

No self-hosting required. Sign up at [trychroma.com](https://www.trychroma.com/) and connect with:

```python
import chromadb

client = await chromadb.AsyncHttpClient(
    host="your-instance.trychroma.com",
    port=443,
    ssl=True,
    headers={"Authorization": "Bearer YOUR_API_KEY"},
)
```

## Verify the Server is Running

```bash
curl http://localhost:8000/api/v1/heartbeat
```

Expected response:

```json
{"nanosecond heartbeat":1234567890}
```

## Authentication (optional)

To enable token-based auth, set these environment variables on the server:

```bash
export CHROMA_SERVER_AUTHN_PROVIDER=chromadb.auth.token_authn.TokenAuthenticationServerProvider
export CHROMA_SERVER_AUTHN_CREDENTIALS=my-secret-token
```

Then connect with:

```python
client = await chromadb.AsyncHttpClient(
    host="localhost",
    port=8000,
    headers={"Authorization": "Bearer my-secret-token"},
)
```
