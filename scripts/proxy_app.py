import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask


UPSTREAM_URL = os.environ.get("LITELLM_UPSTREAM_URL", "http://127.0.0.1:7861").rstrip("/")
MODEL_CATALOG_PATH = Path(os.environ.get("MODEL_CATALOG_PATH", "/app/config/model-catalog.json"))
USABLE_MODELS_PATH = Path(os.environ.get("USABLE_MODELS_PATH", "/app/config/usable-models.json"))
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
CLOUDFLARE_IMAGE_MODELS = {
    model.strip()
    for model in os.environ.get(
        "CLOUDFLARE_IMAGE_MODELS",
        ",".join(
            [
                "cloudflare/@cf/black-forest-labs/flux-1-schnell",
                "cloudflare/@cf/black-forest-labs/flux-2-dev",
                "cloudflare/@cf/black-forest-labs/flux-2-klein-4b",
                "cloudflare/@cf/black-forest-labs/flux-2-klein-9b",
                "cloudflare/@cf/bytedance/stable-diffusion-xl-lightning",
                "cloudflare/@cf/leonardo/lucid-origin",
                "cloudflare/@cf/leonardo/phoenix-1.0",
                "cloudflare/@cf/lykon/dreamshaper-8-lcm",
                "cloudflare/@cf/runwayml/stable-diffusion-v1-5-img2img",
                "cloudflare/@cf/runwayml/stable-diffusion-v1-5-inpainting",
                "cloudflare/@cf/stabilityai/stable-diffusion-xl-base-1.0",
            ]
        ),
    ).split(",")
    if model.strip()
}
CLOUDFLARE_IMAGE_MAX_N = max(1, int(os.environ.get("CLOUDFLARE_IMAGE_MAX_N", "1")))
POLLINATIONS_IMAGE_MODELS = {
    model.strip()
    for model in os.environ.get(
        "POLLINATIONS_IMAGE_MODELS",
        "pollinations/flux,pollinations/zimage,pollinations/gptimage",
    ).split(",")
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
catalog_metadata_cache: dict[str, dict[str, Any]] | None = None
usable_metadata_cache: dict[str, dict[str, Any]] | None = None


@app.on_event("shutdown")
async def shutdown() -> None:
    await client.aclose()


def clean_headers(headers: Any) -> dict[str, str]:
    return {
        key: value
        for key, value in dict(headers).items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def model_looks_free(model_id: str, name: str = "") -> bool:
    text = f"{model_id} {name}".lower()
    return (
        model_id.lower().endswith(":free")
        or ":free" in model_id.lower()
        or " free" in text
        or text.startswith("free ")
        or text.endswith("(free)")
    )


def pricing_is_free(pricing: Any) -> bool:
    if not isinstance(pricing, dict) or not pricing:
        return False
    numeric_prices = []
    for value in pricing.values():
        try:
            numeric_prices.append(float(value))
        except (TypeError, ValueError):
            pass
    return bool(numeric_prices) and all(price == 0 for price in numeric_prices)


def capability_from_mode(mode: str | None) -> list[str]:
    if not mode:
        return []
    normalized = mode.strip().lower().replace("-", "_")
    if normalized in {"image", "image_generation", "text_to_image"}:
        return ["image"]
    if normalized in {"audio_transcription", "transcription"}:
        return ["audio", "transcription"]
    if normalized in {"audio_speech", "speech", "text_to_speech"}:
        return ["audio", "speech"]
    if normalized in {"embedding", "embeddings"}:
        return ["embedding"]
    if normalized in {"rerank", "reranking"}:
        return ["rerank"]
    if normalized in {"chat", "completion", "responses"}:
        return ["text"]
    return []


def unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        label = value.strip().lower().replace("-", "_")
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def suffix_parts(suffix: Any) -> tuple[str, str, dict[str, Any]]:
    if isinstance(suffix, dict):
        alias = str(suffix.get("alias") or suffix.get("id") or suffix.get("model") or "")
        model = str(suffix.get("model") or alias)
        return alias, model, suffix
    value = str(suffix)
    return value, value, {}


def explicit_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1", "free"}:
            return True
        if lowered in {"false", "no", "n", "0", "paid"}:
            return False
    return None


def catalog_entry_metadata(group: dict[str, Any], suffix: Any) -> tuple[str, dict[str, Any]] | None:
    alias_prefix = str(group.get("alias_prefix") or "").strip()
    model_prefix = str(group.get("model_prefix") or "").strip()
    alias_suffix, model_suffix, suffix_meta = suffix_parts(suffix)
    if not alias_prefix or not alias_suffix:
        return None

    model_id = f"{alias_prefix}/{alias_suffix}"
    upstream_model = f"{model_prefix}/{model_suffix}" if model_prefix else model_suffix
    group_info = group.get("model_info") if isinstance(group.get("model_info"), dict) else {}
    suffix_info = suffix_meta.get("model_info") if isinstance(suffix_meta.get("model_info"), dict) else {}
    model_info = {**group_info, **suffix_info}
    mode = (
        suffix_meta.get("mode")
        or group.get("mode")
        or model_info.get("mode")
        or suffix_meta.get("task")
        or group.get("task")
    )
    mode = str(mode).strip() if mode else None
    raw_capabilities = []
    for source in (group, suffix_meta, model_info):
        capabilities = source.get("capabilities") if isinstance(source, dict) else None
        if isinstance(capabilities, list):
            raw_capabilities.extend(capabilities)
        elif isinstance(capabilities, dict):
            raw_capabilities.extend(
                key for key, enabled in capabilities.items() if enabled
            )

    pricing = suffix_meta.get("pricing") or group.get("pricing") or model_info.get("pricing")
    free_value = (
        explicit_bool(suffix_meta.get("free"))
        if "free" in suffix_meta
        else explicit_bool(suffix_meta.get("is_free"))
    )
    if free_value is None:
        free_value = (
            explicit_bool(group.get("free"))
            if "free" in group
            else explicit_bool(group.get("is_free"))
        )
    if free_value is None:
        free_value = pricing_is_free(pricing) or model_looks_free(model_id, str(model_info.get("name") or ""))

    capabilities = unique_strings([*capability_from_mode(mode), *raw_capabilities])
    provider = str(group.get("provider") or alias_prefix).strip()
    metadata: dict[str, Any] = {
        "id": model_id,
        "provider": provider,
        "source_model": upstream_model,
        "free": bool(free_value),
        "is_free": bool(free_value),
        "capabilities": capabilities,
        "catalog_source": "config/model-catalog.json",
    }
    if mode:
        metadata["task"] = mode
        metadata["mode"] = mode
    if isinstance(pricing, dict):
        metadata["pricing"] = pricing
    if model_info:
        metadata["model_info"] = {
            **model_info,
            "mode": mode or model_info.get("mode"),
            "capabilities": capabilities or model_info.get("capabilities", []),
            "free": bool(free_value),
            "is_free": bool(free_value),
        }
    else:
        metadata["model_info"] = {
            "mode": mode,
            "capabilities": capabilities,
            "free": bool(free_value),
            "is_free": bool(free_value),
        }
    return model_id, metadata


def merge_catalog_metadata(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if key == "model_info":
            current_info = merged.get("model_info")
            merged_info = dict(current_info) if isinstance(current_info, dict) else {}
            if isinstance(value, dict):
                for info_key, info_value in value.items():
                    current_value = merged_info.get(info_key)
                    if current_value in (None, [], {}) and info_value not in (None, [], {}):
                        merged_info[info_key] = info_value
                    elif (
                        info_key == "capabilities"
                        and isinstance(info_value, list)
                        and isinstance(current_value, list)
                        and len(info_value) > len(current_value)
                    ):
                        merged_info[info_key] = info_value
            if merged_info:
                merged["model_info"] = merged_info
            continue

        current = merged.get(key)
        if current in (None, [], {}) and value not in (None, [], {}):
            merged[key] = value
        elif (
            key == "capabilities"
            and isinstance(value, list)
            and isinstance(current, list)
            and len(value) > len(current)
        ):
            merged[key] = value

    return merged


def load_catalog_metadata() -> dict[str, dict[str, Any]]:
    global catalog_metadata_cache
    if catalog_metadata_cache is not None:
        return catalog_metadata_cache

    metadata: dict[str, dict[str, Any]] = {}
    try:
        catalog = json.loads(MODEL_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        catalog_metadata_cache = metadata
        return metadata

    if not isinstance(catalog, dict):
        catalog_metadata_cache = metadata
        return metadata

    for group in catalog.get("groups", []):
        if not isinstance(group, dict):
            continue
        for suffix in group.get("suffixes", []):
            entry = catalog_entry_metadata(group, suffix)
            if entry:
                model_id, model_metadata = entry
                if model_id in metadata:
                    metadata[model_id] = merge_catalog_metadata(metadata[model_id], model_metadata)
                else:
                    metadata[model_id] = model_metadata

    for model in GENLABS_MODELS:
        metadata.setdefault(
            model,
            {
                "id": model,
                "provider": "genlabs",
                "source_model": model.split("/", 1)[1],
                "free": False,
                "is_free": False,
                "capabilities": ["text"],
                "task": "chat",
                "mode": "chat",
                "catalog_source": "wrapper",
                "model_info": {
                    "mode": "chat",
                    "capabilities": ["text"],
                    "free": False,
                    "is_free": False,
                },
            },
        )
    for model in sorted(CLOUDFLARE_IMAGE_MODELS | POLLINATIONS_IMAGE_MODELS):
        provider = model.split("/", 1)[0] if "/" in model else "image"
        metadata.setdefault(
            model,
            {
                "id": model,
                "provider": provider,
                "source_model": model.split("/", 1)[1] if "/" in model else model,
                "free": False,
                "is_free": False,
                "capabilities": ["image"],
                "task": "image_generation",
                "mode": "image_generation",
                "catalog_source": "wrapper",
                "model_info": {
                    "mode": "image_generation",
                    "capabilities": ["image"],
                    "free": False,
                    "is_free": False,
                },
            },
        )

    catalog_metadata_cache = metadata
    return metadata


def load_usable_metadata() -> dict[str, dict[str, Any]]:
    global usable_metadata_cache
    if usable_metadata_cache is not None:
        return usable_metadata_cache

    metadata: dict[str, dict[str, Any]] = {}
    try:
        payload = json.loads(USABLE_MODELS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        usable_metadata_cache = metadata
        return metadata

    if not isinstance(payload, dict):
        usable_metadata_cache = metadata
        return metadata

    checked_at = payload.get("checked_at")
    for model_id in payload.get("usable_model_ids", []):
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        metadata[model_id.strip()] = {
            "usable": True,
            "verified_usable": True,
            "usable_checked_at": checked_at,
            "usable_source": "config/usable-models.json",
        }

    for model in payload.get("models", []):
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        key = model_id.strip()
        item = metadata.setdefault(
            key,
            {
                "usable": True,
                "verified_usable": True,
                "usable_checked_at": checked_at,
                "usable_source": "config/usable-models.json",
            },
        )
        item["usable_latency_ms"] = model.get("latency_ms")
        item["usable_http_status"] = model.get("http_status")

    usable_metadata_cache = metadata
    return metadata


def model_metadata(model_id: str) -> dict[str, Any]:
    return {
        **load_catalog_metadata().get(model_id, {}),
        **load_usable_metadata().get(model_id, {}),
    }


def runtime_extra_model_ids() -> set[str]:
    model_ids: set[str] = set()
    if genlabs_api_key():
        model_ids.update(GENLABS_MODELS)
    if cloudflare_api_token() and cloudflare_account_id():
        model_ids.update(CLOUDFLARE_IMAGE_MODELS)
    if pollinations_api_key():
        model_ids.update(POLLINATIONS_IMAGE_MODELS)
    return model_ids


def merge_model_metadata(raw_model: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(raw_model)
    for key in (
        "provider",
        "free",
        "is_free",
        "pricing",
        "task",
        "mode",
        "capabilities",
        "catalog_source",
        "usable",
        "verified_usable",
        "usable_checked_at",
        "usable_source",
        "usable_latency_ms",
        "usable_http_status",
    ):
        if key in metadata and key not in enriched:
            enriched[key] = metadata[key]

    raw_info = enriched.get("model_info")
    merged_info = dict(raw_info) if isinstance(raw_info, dict) else {}
    metadata_info = metadata.get("model_info")
    if isinstance(metadata_info, dict):
        for key, value in metadata_info.items():
            if key not in merged_info or merged_info.get(key) in (None, [], {}):
                merged_info[key] = value
    if merged_info:
        enriched["model_info"] = merged_info
    return enriched


def genlabs_api_key() -> str | None:
    value = os.environ.get("GENLABS_API_KEY", "").strip()
    return value or None


def cloudflare_api_token() -> str | None:
    value = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    return value or None


def cloudflare_account_id() -> str | None:
    value = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    return value or None


def pollinations_api_key() -> str | None:
    for name in ("POLLINATIONS_API_KEY", "POLLINATIONS_API_KEY_1"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def is_genlabs_chat_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("model") in GENLABS_MODELS


def is_nonstream_upstream_payload(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and bool(payload.get("stream"))
        and str(payload.get("model", "")) in NONSTREAM_UPSTREAM_MODELS
    )


def cloudflare_image_model_id(model: str) -> str | None:
    if model in CLOUDFLARE_IMAGE_MODELS:
        return model.split("/", 1)[1]
    if model.startswith("@cf/") and f"cloudflare/{model}" in CLOUDFLARE_IMAGE_MODELS:
        return model
    return None


def pollinations_image_model_name(model: str) -> str | None:
    if model in POLLINATIONS_IMAGE_MODELS:
        return model.split("/", 1)[1]
    return None


def is_image_only_model(model: str) -> bool:
    return bool(cloudflare_image_model_id(model) or pollinations_image_model_name(model))


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


def openai_image_data(image_base64: str, prompt: str, media_type: str = "image/jpeg") -> dict[str, str]:
    return {
        "b64_json": image_base64,
        "url": f"data:{media_type};base64,{image_base64}",
        "revised_prompt": prompt,
    }


def cloudflare_image_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"prompt": payload.get("prompt")}
    for key in ("steps", "seed"):
        if payload.get(key) is not None:
            out[key] = payload[key]
    return out


async def cloudflare_images_generations(payload: dict[str, Any]) -> Response:
    model = str(payload.get("model") or "")
    model_id = cloudflare_image_model_id(model)
    if not model_id:
        return JSONResponse(
            {"error": f"Unsupported Cloudflare image model: {model or '(missing)'}"},
            status_code=400,
        )

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return JSONResponse({"error": "Image generation prompt is required"}, status_code=400)

    api_token = cloudflare_api_token()
    account_id = cloudflare_account_id()
    if not api_token or not account_id:
        return JSONResponse(
            {"error": "CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID must be configured"},
            status_code=503,
        )

    requested_n = payload.get("n") if isinstance(payload.get("n"), int) else 1
    image_count = min(max(1, requested_n), CLOUDFLARE_IMAGE_MAX_N)
    images: list[dict[str, str]] = []
    for _ in range(image_count):
        response = await client.post(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_id}",
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            },
            json=cloudflare_image_payload(payload),
        )

        content_type = response.headers.get("content-type", "")
        if response.status_code < 200 or response.status_code >= 300:
            return Response(
                response.content,
                status_code=response.status_code,
                headers=clean_headers(response.headers),
                media_type=content_type,
            )

        if content_type.startswith("image/"):
            import base64

            media_type = content_type.split(";", 1)[0]
            images.append(openai_image_data(base64.b64encode(response.content).decode("ascii"), prompt, media_type))
            continue

        try:
            response_payload = response.json()
        except json.JSONDecodeError:
            return Response(
                response.content,
                status_code=response.status_code,
                headers=clean_headers(response.headers),
                media_type=content_type,
            )

        result = response_payload.get("result") if isinstance(response_payload, dict) else None
        image_base64 = result.get("image") if isinstance(result, dict) else None
        if not isinstance(image_base64, str) or not image_base64:
            return JSONResponse(
                {
                    "error": "Cloudflare image response did not include result.image",
                    "provider_response": response_payload,
                },
                status_code=502,
            )
        images.append(openai_image_data(image_base64, prompt))

    return JSONResponse(
        {
            "created": int(time.time()),
            "data": images,
        }
    )


def pollinations_image_payload(payload: dict[str, Any], model_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "prompt": payload.get("prompt"),
        "model": model_name,
    }
    for key in ("n", "size", "quality", "response_format", "user", "image", "safe"):
        if payload.get(key) is not None:
            out[key] = payload[key]
    return out


async def pollinations_images_generations(payload: dict[str, Any]) -> Response:
    model = str(payload.get("model") or "")
    model_name = pollinations_image_model_name(model)
    if not model_name:
        return JSONResponse(
            {"error": f"Unsupported Pollinations image model: {model or '(missing)'}"},
            status_code=400,
        )

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return JSONResponse({"error": "Image generation prompt is required"}, status_code=400)

    api_key = pollinations_api_key()
    if not api_key:
        return JSONResponse({"error": "POLLINATIONS_API_KEY must be configured"}, status_code=503)

    response = await client.post(
        "https://gen.pollinations.ai/v1/images/generations",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=pollinations_image_payload(payload, model_name),
    )
    return Response(
        response.content,
        status_code=response.status_code,
        headers=clean_headers(response.headers),
        media_type=response.headers.get("content-type"),
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
        enriched_data: list[Any] = []
        existing = set()
        for item in payload["data"]:
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str):
                    existing.add(model_id)
                    metadata = model_metadata(model_id)
                    enriched_data.append(merge_model_metadata(item, metadata) if metadata else item)
                    continue
            enriched_data.append(item)
        payload["data"] = enriched_data

        for model in sorted(runtime_extra_model_ids()):
            if model not in existing:
                metadata = model_metadata(model)
                payload["data"].append(
                    {
                        "id": model,
                        "object": "model",
                        "created": 0,
                        "owned_by": metadata.get("provider") or model.split("/", 1)[0],
                        **{key: value for key, value in metadata.items() if key != "id"},
                    }
                )
    return JSONResponse(payload, status_code=upstream.status_code)


def model_catalog_response() -> Response:
    metadata = {
        **load_catalog_metadata(),
    }
    for model_id, usable in load_usable_metadata().items():
        metadata[model_id] = {**metadata.get(model_id, {"id": model_id}), **usable}
    return JSONResponse(
        {
            "object": "model_catalog",
            "sources": [str(MODEL_CATALOG_PATH), str(USABLE_MODELS_PATH)],
            "count": len(metadata),
            "data": [metadata[key] for key in sorted(metadata)],
        }
    )


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
    if request.method == "GET" and normalized in {"/v1/model-catalog", "/model-catalog"}:
        return model_catalog_response()
    if request.method == "GET" and normalized == "/v1/models":
        return await models_response(request)
    if request.method == "POST" and normalized == "/v1/chat/completions":
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        if is_image_only_model(str(payload.get("model") or "")):
            return JSONResponse(
                {
                    "error": {
                        "message": (
                            f"{payload.get('model')} is an image-only route in this wrapper. "
                            "Use /v1/images/generations for image generation and select a chat model "
                            "for normal Open WebUI conversations."
                        ),
                        "type": "invalid_request_error",
                        "code": "image_model_used_for_chat",
                    }
                },
                status_code=400,
            )
        if is_genlabs_chat_payload(payload):
            if bool(payload.get("stream")):
                return await genlabs_stream(payload)
            return await genlabs_completion(payload)
        if is_nonstream_upstream_payload(payload):
            return await nonstream_upstream_chat(request, payload)
    if request.method == "POST" and normalized in {"/v1/images/generations", "/images/generations"}:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        if cloudflare_image_model_id(str(payload.get("model") or "")):
            return await cloudflare_images_generations(payload)
        if pollinations_image_model_name(str(payload.get("model") or "")):
            return await pollinations_images_generations(payload)

    return await proxy_request(path, request)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "7860")),
    )
