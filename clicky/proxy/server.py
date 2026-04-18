"""FastAPI proxy server for Clicky.

Mirrors the Cloudflare Worker from the original macOS app.
Routes:
  POST /chat             -> Anthropic Messages API (streaming)
  POST /tts              -> ElevenLabs TTS API
  POST /transcribe-token -> AssemblyAI streaming token

All API keys are held server-side so they never ship in the client.
"""

import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, Response
import httpx
from typing import Optional

app = FastAPI(title="Clicky Proxy")

# Load keys from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")

# HTTP client (reuse connections)
_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
    return _http_client


@app.post("/chat")
async def proxy_chat(request: Request):
    """Proxy to Anthropic Messages API with streaming SSE."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    body = await request.json()
    client = await get_http_client()

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "accept": "text/event-stream",
    }

    # Ensure streaming is enabled
    body["stream"] = True

    async def stream_response():
        async with client.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            json=body,
            headers=headers,
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                yield f"event: error\ndata: {error_body.decode()}\n\n"
                return
            async for line in response.aiter_lines():
                yield line + "\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "cache-control": "no-cache",
            "connection": "keep-alive",
        },
    )


@app.post("/tts")
async def proxy_tts(request: Request):
    """Proxy to ElevenLabs TTS API. Returns audio/mpeg."""
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY not configured")

    body = await request.json()
    text = body.get("text", "")
    voice_id = body.get("voice_id", ELEVENLABS_VOICE_ID)
    model_id = body.get("model_id", "eleven_monolingual_v1")

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' field")
    if not voice_id:
        raise HTTPException(status_code=400, detail="Missing voice_id and ELEVENLABS_VOICE_ID not set")

    client = await get_http_client()

    response = await client.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        json={"text": text, "model_id": model_id},
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "content-type": "application/json",
            "accept": "audio/mpeg",
        },
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"ElevenLabs error: {response.text}",
        )

    return Response(
        content=response.content,
        media_type="audio/mpeg",
    )


@app.post("/transcribe-token")
async def proxy_transcribe_token(request: Request):
    """Fetch a short-lived AssemblyAI streaming token."""
    if not ASSEMBLYAI_API_KEY:
        raise HTTPException(status_code=500, detail="ASSEMBLYAI_API_KEY not configured")

    client = await get_http_client()

    response = await client.post(
        "https://streaming.assemblyai.com/v3/token",
        headers={
            "authorization": ASSEMBLYAI_API_KEY,
            "content-type": "application/json",
        },
        json={"expires_in": 480},
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"AssemblyAI error: {response.text}",
        )

    return Response(
        content=response.content,
        media_type="application/json",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


def run_proxy():
    """Entry point for `clicky-proxy` command."""
    import uvicorn

    port = int(os.getenv("PROXY_PORT", "8787"))
    print(f"Starting Clicky proxy on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run_proxy()