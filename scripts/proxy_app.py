import json
import os
import time
import uuid
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask


UPSTREAM_URL = os.environ.get("LITELLM_UPSTREAM_URL", "http://127.0.0.1:7861").rstrip("/")
GENLABS_BASE_URL = os.environ.get("GENLABS_API_BASE", "https://api.genlabs.dev/deca/v1").rstrip("/")
GENLABS_MODELS = (
    "genlabs/deca-2.5-mini",
    "genlabs/deca-2.5-pro",
    "genlabs/deca-2.5-ultra",
)
NONSTREAM_UPSTREAM_MODELS = {
    model.strip()
    for model in os.environ.get("NONSTREAM_UPSTREAM_MODELS", "voidai/gpt-oss-120b").split(",")
    if model.strip()
}
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))


@app.on_event("shutdown")
async def shutdown() -> None:
    await client.aclose()


def clean_headers(headers: Any) -> dict[str, str]:
    return {
        key: value
        for key, value in dict(headers).items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def genlabs_api_key() -> str | None:
    value = os.environ.get("GENLABS_API_KEY", "").strip()
    return value or None


def is_genlabs_chat_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("model") in GENLABS_MODELS


def is_nonstream_upstream_payload(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and bool(payload.get("stream"))
        and str(payload.get("model", "")) in NONSTREAM_UPSTREAM_MODELS
    )


def genlabs_payload(payload: dict[str, Any], *, stream: bool) -> dict[str, Any]:
    out = dict(payload)
    out["model"] = str(payload["model"]).split("/", 1)[1]
    out["stream"] = stream
    return out


def sse_payload(data: dict[str, Any] | str) -> bytes:
    if isinstance(data, str):
        return f"data: {data}\n\n".encode("utf-8")
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n".encode("utf-8")


def completion_to_sse(payload: dict[str, Any]) -> Any:
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)

    response_id = payload.get("id") or f"chatcmpl-{uuid.uuid4().hex}"
    created = payload.get("created") or int(time.time())
    model = payload.get("model") or "unknown"
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    finish_reason = finish_reason or "stop"

    def event(
        delta: dict[str, Any],
        *,
        finish: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish,
                }
            ],
        }
        if usage is not None:
            out["usage"] = usage
        return out

    yield sse_payload(event({"role": "assistant"}))
    if content:
        yield sse_payload(event({"content": content}))
    yield sse_payload(event({}, finish=finish_reason))

    usage = payload.get("usage")
    if isinstance(usage, dict):
        yield sse_payload(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage,
            }
        )
    yield sse_payload("[DONE]")


async def nonstream_upstream_chat(request: Request, payload: dict[str, Any]) -> Response:
    upstream_payload = dict(payload)
    upstream_payload["stream"] = False
    upstream_response = await client.post(
        f"{UPSTREAM_URL}/v1/chat/completions",
        params=request.query_params,
        headers=clean_headers(request.headers),
        json=upstream_payload,
    )

    if upstream_response.status_code < 200 or upstream_response.status_code >= 300:
        return Response(
            upstream_response.content,
            status_code=upstream_response.status_code,
            headers=clean_headers(upstream_response.headers),
            media_type=upstream_response.headers.get("content-type"),
        )

    try:
        response_payload = upstream_response.json()
    except json.JSONDecodeError:
        return Response(
            upstream_response.content,
            status_code=upstream_response.status_code,
            headers=clean_headers(upstream_response.headers),
            media_type=upstream_response.headers.get("content-type"),
        )

    return StreamingResponse(
        completion_to_sse(response_payload),
        status_code=upstream_response.status_code,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def genlabs_stream(payload: dict[str, Any]) -> Response:
    api_key = genlabs_api_key()
    if not api_key:
        return JSONResponse({"error": "GENLABS_API_KEY is not configured"}, status_code=503)

    request = client.build_request(
        "POST",
        f"{GENLABS_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        json=genlabs_payload(payload, stream=True),
    )
    response = await client.send(request, stream=True)
    return StreamingResponse(
        response.aiter_raw(),
        status_code=response.status_code,
        headers=clean_headers(response.headers),
        background=BackgroundTask(response.aclose),
    )


async def genlabs_completion(payload: dict[str, Any]) -> Response:
    api_key = genlabs_api_key()
    if not api_key:
        return JSONResponse({"error": "GENLABS_API_KEY is not configured"}, status_code=503)

    model_name = str(payload["model"])
    content_parts: list[str] = []
    finish_reason = "stop"
    usage: dict[str, Any] | None = None
    response_id = f"chatcmpl-{uuid.uuid4().hex}"

    async with client.stream(
        "POST",
        f"{GENLABS_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        json=genlabs_payload(payload, stream=True),
    ) as response:
        if response.status_code < 200 or response.status_code >= 300:
            body = await response.aread()
            return Response(
                body,
                status_code=response.status_code,
                headers=clean_headers(response.headers),
                media_type=response.headers.get("content-type"),
            )

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            response_id = chunk.get("id") or response_id
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])

    return JSONResponse(
        {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "".join(content_parts),
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage
            or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    )


async def genlabs_chat_completions(request: Request) -> Response:
    payload = await request.json()
    if bool(payload.get("stream")):
        return await genlabs_stream(payload)
    return await genlabs_completion(payload)


async def models_response(request: Request) -> Response:
    upstream = await client.get(
        f"{UPSTREAM_URL}/v1/models",
        params=request.query_params,
        headers=clean_headers(request.headers),
    )
    try:
        payload = upstream.json()
    except json.JSONDecodeError:
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            headers=clean_headers(upstream.headers),
            media_type=upstream.headers.get("content-type"),
        )

    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        existing = {item.get("id") for item in payload["data"] if isinstance(item, dict)}
        for model in GENLABS_MODELS:
            if model not in existing:
                payload["data"].append(
                    {
                        "id": model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "genlabs",
                    }
                )
    return JSONResponse(payload, status_code=upstream.status_code)


async def proxy_request(path: str, request: Request) -> Response:
    url = f"{UPSTREAM_URL}/{path}"
    body = await request.body()
    upstream_request = client.build_request(
        request.method,
        url,
        params=request.query_params,
        headers=clean_headers(request.headers),
        content=body,
    )
    upstream_response = await client.send(upstream_request, stream=True)
    return StreamingResponse(
        upstream_response.aiter_raw(),
        status_code=upstream_response.status_code,
        headers=clean_headers(upstream_response.headers),
        background=BackgroundTask(upstream_response.aclose),
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def route(path: str, request: Request) -> Response:
    normalized = f"/{path}"
    if request.method == "GET" and normalized == "/v1/models":
        return await models_response(request)
    if request.method == "POST" and normalized == "/v1/chat/completions":
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        if is_genlabs_chat_payload(payload):
            if bool(payload.get("stream")):
                return await genlabs_stream(payload)
            return await genlabs_completion(payload)
        if is_nonstream_upstream_payload(payload):
            return await nonstream_upstream_chat(request, payload)

    return await proxy_request(path, request)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "7860")),
    )
