import argparse
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OUTPUT = Path("config/usable-models.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    if not base_url:
        raise ValueError("LiteLLM base URL is required.")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def request_json(
    method: str,
    url: str,
    *,
    api_key: str | None,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
        if not body:
            return response.status, None
        content_type = response.headers.get("Content-Type", "")
        if "json" not in content_type.lower():
            return response.status, {
                "_raw_content_type": content_type,
                "_raw_bytes": len(body),
            }
        return response.status, json.loads(body.decode("utf-8"))


def fetch_model_ids(base_url: str, api_key: str | None, timeout: float) -> list[str]:
    return [entry["id"] for entry in fetch_model_entries(base_url, api_key, timeout)]


def fetch_model_entries(
    base_url: str,
    api_key: str | None,
    timeout: float,
) -> list[dict[str, Any]]:
    _, payload = request_json(
        "GET",
        f"{base_url}/models",
        api_key=api_key,
        timeout=timeout,
    )
    raw_models = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(raw_models, list):
        return []

    entries: dict[str, dict[str, Any]] = {}
    for model in raw_models:
        if isinstance(model, str):
            model_id = model
        elif isinstance(model, dict):
            model_id = model.get("id") or model.get("model") or model.get("name")
        else:
            continue
        if isinstance(model_id, str) and model_id.strip():
            key = model_id.strip()
            raw = dict(model) if isinstance(model, dict) else {"id": key, "name": key}
            raw["id"] = key
            entries.setdefault(key, raw)

    return [entries[key] for key in sorted(entries)]


def short_error(exc: BaseException) -> tuple[str, int | None]:
    if isinstance(exc, HTTPError):
        try:
            body = exc.read(500).decode("utf-8", errors="replace")
        except Exception:
            body = ""
        message = f"HTTP {exc.code}"
        if body:
            message = f"{message}: {body[:220]}"
        return message, exc.code
    if isinstance(exc, URLError):
        return f"{type(exc.reason).__name__}: {exc.reason}", None
    return f"{type(exc).__name__}: {exc}", None


def is_transient_error(error: str | None) -> bool:
    if not error:
        return False
    needles = (
        "getaddrinfo failed",
        "ConnectionResetError",
        "timed out",
        "temporarily unavailable",
        "forcibly closed",
        "unreachable host",
    )
    return any(needle.lower() in error.lower() for needle in needles)


def is_retryable_result(result: dict[str, Any]) -> bool:
    status = result.get("http_status")
    if isinstance(status, int) and status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    return is_transient_error(str(result.get("error") or ""))


def ping_model_once(
    model_id: str,
    *,
    base_url: str,
    api_key: str | None,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    try:
        status, response = request_json(
            "POST",
            f"{base_url}/chat/completions",
            api_key=api_key,
            payload=payload,
            timeout=timeout,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        choices = response.get("choices") if isinstance(response, dict) else None
        ok = 200 <= status < 300 and isinstance(choices, list)
        return {
            "id": model_id,
            "ok": ok,
            "http_status": status,
            "latency_ms": elapsed_ms,
            "error": None if ok else "missing choices in successful response",
        }
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        message, status = short_error(exc)
        return {
            "id": model_id,
            "ok": False,
            "http_status": status,
            "latency_ms": elapsed_ms,
            "error": message,
        }


def ping_model(
    model_id: str,
    *,
    base_url: str,
    api_key: str | None,
    prompt: str,
    max_tokens: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    attempts = max(1, retries + 1)
    last_result: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        result = ping_model_once(
            model_id,
            base_url=base_url,
            api_key=api_key,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        result["attempts"] = attempt
        if result.get("ok"):
            return result
        last_result = result
        if attempt < attempts and is_retryable_result(result):
            time.sleep(max(0, retry_sleep))
            continue
        return result
    return last_result or {"id": model_id, "ok": False, "error": "unknown"}


def raw_model_dicts(raw_model: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[int] = set()

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 4 or not isinstance(value, dict):
            return
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        result.append(value)
        for key in ("info", "meta", "model_info"):
            visit(value.get(key), depth + 1)

    visit(raw_model)
    return result


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def image_capability_from_task(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    normalized = normalize_label(value)
    return normalized in {"texttoimage", "imagegeneration", "text2image", "t2i"}


def raw_image_metadata(raw_model: dict[str, Any]) -> bool:
    for mapping in raw_model_dicts(raw_model):
        for key in ("task", "task_name", "type", "mode", "sample_spec"):
            value = mapping.get(key)
            if isinstance(value, dict):
                value = value.get("name")
            if image_capability_from_task(value):
                return True

        capabilities = mapping.get("capabilities")
        if isinstance(capabilities, list):
            if any(normalize_label(str(item)) in {"image", "imagegeneration", "texttoimage", "generatesimages"} for item in capabilities):
                return True
        elif isinstance(capabilities, dict):
            for capability, capability_enabled in capabilities.items():
                if enabled(capability_enabled) and normalize_label(str(capability)) in {
                    "image",
                    "imagegeneration",
                    "texttoimage",
                    "generatesimages",
                }:
                    return True

        for key in (
            "image_generation",
            "supports_image_generation",
            "supports_image_output",
        ):
            if enabled(mapping.get(key)):
                return True

        output_modalities = mapping.get("output_modalities")
        if isinstance(output_modalities, list) and any(
            str(item).strip().lower() == "image" for item in output_modalities
        ):
            return True

    return False


IMAGE_GENERATION_TERMS = (
    "black-forest-labs",
    "dall-e",
    "dalle",
    "flux",
    "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview",
    "gpt-image",
    "hunyuan-image",
    "image",
    "image-generation",
    "image_generation",
    "image-preview",
    "imagen",
    "ideogram",
    "kandinsky",
    "kolors",
    "midjourney",
    "nano-banana",
    "playground-v",
    "qwen-image",
    "recraft",
    "sdxl",
    "seedream",
    "stable-diffusion",
    "text-to-image",
    "z-image",
    "zimage",
)

IMAGE_GENERATION_LABELS = (
    "t2i",
    "texttoimage",
    "text2image",
    "imagegeneration",
    "generatesimages",
)

IMAGE_NON_GENERATION_TERMS = (
    "animate",
    "depth",
    "edit",
    "first-last-image-to-video",
    "i2v",
    "image-to-image",
    "image-to-video",
    "img2img",
    "inpaint",
    "multi-image-to-video",
    "outpaint",
    "pose",
    "reframe",
    "remove-background",
    "replace",
    "r2v",
    "text-to-video",
    "upscale",
    "video",
    "vto",
)


def raw_model_text(raw_model: dict[str, Any]) -> str:
    values: list[str] = []
    for mapping in raw_model_dicts(raw_model):
        for key in ("id", "model", "name", "title", "task", "task_name", "type", "mode"):
            value = mapping.get(key)
            if isinstance(value, dict):
                value = value.get("name")
            if value is not None:
                values.append(str(value))
    return " ".join(values).lower()


def image_candidate_kind(raw_model: dict[str, Any]) -> str | None:
    text = raw_model_text(raw_model)
    normalized = normalize_label(text)
    has_generation_metadata = raw_image_metadata(raw_model)
    has_generation_term = any(term in text for term in IMAGE_GENERATION_TERMS)
    has_generation_label = any(label in normalized for label in IMAGE_GENERATION_LABELS)

    if not has_generation_metadata and not has_generation_term and not has_generation_label:
        return None
    if any(term in text for term in IMAGE_NON_GENERATION_TERMS):
        return "other-image"
    return "generation"


def is_image_model_entry(raw_model: dict[str, Any], *, scope: str = "generation") -> bool:
    kind = image_candidate_kind(raw_model)
    if scope == "all":
        return kind is not None
    return kind == "generation"


def classify_failure(result: dict[str, Any]) -> str | None:
    if result.get("ok"):
        return None
    status = result.get("http_status")
    error = str(result.get("error") or "").lower()
    if status == 402:
        return "payment_required"
    if status == 403 and any(term in error for term in ("funds", "balance", "payment method", "top up")):
        return "provider_funds_or_billing"
    if status == 403:
        return "provider_forbidden"
    if status == 404:
        return "model_not_found"
    if status == 429:
        return "rate_limited"
    if status in {500, 502, 503, 504}:
        return "provider_server_error"
    if status == 400 and any(term in error for term in ("invalid payload", "required", "multipart", "image")):
        return "wrong_image_payload_or_endpoint"
    if status == 400:
        return "bad_request"
    if is_transient_error(error):
        return "network_transient"
    if result.get("error") == "missing image data in successful response":
        return "unexpected_response_shape"
    return "unknown"


def sanitize_failure_error(error: Any) -> str | None:
    if error is None:
        return None
    text = str(error).replace("\n", " ")
    text = re.sub(r"\$[0-9]+(?:\.[0-9]+)?", "$[amount]", text)
    text = re.sub(
        r"(?i)(available balance(?: is| of)? )[-+]?[0-9]+(?:\.[0-9]+)?",
        r"\1[amount]",
        text,
    )
    text = re.sub(
        r"(?i)(balance(?: of)? )[-+]?[0-9]+(?:\.[0-9]+)?",
        r"\1[amount]",
        text,
    )
    text = re.sub(
        r"(?i)(costs? ~?)[-+]?[0-9]+(?:\.[0-9]+)?",
        r"\1[amount]",
        text,
    )
    if len(text) > 260:
        return f"{text[:257]}..."
    return text


def compact_failure(result: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "id": result.get("id"),
        "ok": False,
        "http_status": result.get("http_status"),
        "latency_ms": result.get("latency_ms"),
        "attempts": result.get("attempts"),
        "error_class": classify_failure(result),
        "error": sanitize_failure_error(result.get("error")),
    }
    return {key: value for key, value in compact.items() if value is not None}


def error_class_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for result in results:
        failure_class = classify_failure(result)
        if failure_class:
            counts[failure_class] = counts.get(failure_class, 0) + 1
    return [
        {"error_class": error_class, "count": count}
        for error_class, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


IMAGE_CARRYABLE_FAILURE_CLASSES = {
    "network_transient",
    "provider_server_error",
    "rate_limited",
}


def carry_previous_image_models(
    existing: dict[str, Any],
    image_payload: dict[str, Any],
) -> dict[str, Any]:
    current_models = [
        model for model in image_payload.get("image_models", []) if isinstance(model, dict)
    ]
    current_ids = {str(model.get("id")) for model in current_models if model.get("id")}
    failures = {
        str(model.get("id")): model
        for model in image_payload.get("image_failure_models", [])
        if isinstance(model, dict) and model.get("id")
    }

    carried: list[dict[str, Any]] = []
    for previous in existing.get("image_models", []):
        if not isinstance(previous, dict) or not previous.get("id"):
            continue
        model_id = str(previous["id"])
        if model_id in current_ids:
            continue
        failure = failures.get(model_id)
        if not failure:
            continue
        failure_class = str(failure.get("error_class") or "")
        if failure_class not in IMAGE_CARRYABLE_FAILURE_CLASSES:
            continue
        carried_model = dict(previous)
        carried_model["carried_from_previous"] = True
        carried_model["last_probe_error_class"] = failure_class
        carried_model["last_probe_http_status"] = failure.get("http_status")
        carried_model["last_probe_checked_at"] = image_payload.get("image_checked_at")
        carried.append(carried_model)

    combined = sorted(
        current_models + carried,
        key=lambda item: str(item.get("id") or ""),
    )
    updated = dict(image_payload)
    updated["image_current_usable_count"] = len(current_models)
    updated["image_carried_usable_count"] = len(carried)
    updated["image_usable_count"] = len(combined)
    updated["image_usable_model_ids"] = [str(model["id"]) for model in combined]
    updated["image_models"] = combined
    return updated


def ping_image_model_once(
    model_id: str,
    *,
    base_url: str,
    api_key: str | None,
    prompt: str,
    image_size: str | None,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "model": model_id,
        "prompt": prompt,
        "n": 1,
    }
    if image_size:
        payload["size"] = image_size

    try:
        status, response = request_json(
            "POST",
            f"{base_url}/images/generations",
            api_key=api_key,
            payload=payload,
            timeout=timeout,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        ok = False
        if isinstance(response, dict):
            data = response.get("data")
            raw_content_type = str(response.get("_raw_content_type") or "")
            ok = (
                isinstance(data, list)
                and bool(data)
                or raw_content_type.lower().startswith("image/")
            )
        return {
            "id": model_id,
            "ok": 200 <= status < 300 and ok,
            "http_status": status,
            "latency_ms": elapsed_ms,
            "error": None if ok else "missing image data in successful response",
        }
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        message, status = short_error(exc)
        return {
            "id": model_id,
            "ok": False,
            "http_status": status,
            "latency_ms": elapsed_ms,
            "error": message,
        }


def ping_image_model(
    model_id: str,
    *,
    base_url: str,
    api_key: str | None,
    prompt: str,
    image_size: str | None,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    attempts = max(1, retries + 1)
    last_result: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        result = ping_image_model_once(
            model_id,
            base_url=base_url,
            api_key=api_key,
            prompt=prompt,
            image_size=image_size,
            timeout=timeout,
        )
        result["attempts"] = attempt
        if result.get("ok"):
            return result
        last_result = result
        if attempt < attempts and is_retryable_result(result):
            time.sleep(max(0, retry_sleep))
            continue
        return result
    return last_result or {"id": model_id, "ok": False, "error": "unknown"}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(path)


def error_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for result in results:
        if result.get("ok"):
            continue
        error = str(result.get("error") or "unknown")
        if error.startswith("HTTP "):
            error = error.split(":", 1)[0]
        counts[error] = counts.get(error, 0) + 1
    return [
        {"error": error, "count": count}
        for error, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def output_payload(
    *,
    checked_at: str,
    base_url: str,
    prompt: str,
    max_tokens: int,
    total_models: int,
    targets: list[str],
    results: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    usable = sorted(
        [
            {
                "id": str(result["id"]),
                "http_status": result.get("http_status"),
                "latency_ms": result.get("latency_ms"),
                "attempts": result.get("attempts"),
            }
            for result in results
            if result.get("ok")
        ],
        key=lambda item: item["id"],
    )
    failures = [result for result in results if not result.get("ok")]
    return {
        "version": 1,
        "checked_at": checked_at,
        "base_url": base_url,
        "prompt": prompt,
        "max_tokens": max(1, max_tokens),
        "complete": complete,
        "total_models": total_models,
        "tested_models": len(results),
        "target_models": len(targets),
        "usable_count": len(usable),
        "failure_count": len(failures),
        "usable_model_ids": [model["id"] for model in usable],
        "models": usable,
        "failure_summary": error_summary(results),
        "failure_samples": sorted(failures, key=lambda item: str(item["id"]))[:50],
    }


def image_output_payload(
    *,
    checked_at: str,
    base_url: str,
    prompt: str,
    image_size: str | None,
    total_models: int,
    targets: list[str],
    results: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    usable = sorted(
        [
            {
                "id": str(result["id"]),
                "http_status": result.get("http_status"),
                "latency_ms": result.get("latency_ms"),
                "attempts": result.get("attempts"),
            }
            for result in results
            if result.get("ok")
        ],
        key=lambda item: item["id"],
    )
    failures = [result for result in results if not result.get("ok")]
    compact_failures = [compact_failure(result) for result in sorted(failures, key=lambda item: str(item["id"]))]
    return {
        "image_checked_at": checked_at,
        "image_base_url": base_url,
        "image_prompt": prompt,
        "image_size": image_size,
        "image_complete": complete,
        "image_total_models": total_models,
        "image_tested_models": len(results),
        "image_target_models": len(targets),
        "image_usable_count": len(usable),
        "image_failure_count": len(failures),
        "image_usable_model_ids": [model["id"] for model in usable],
        "image_models": usable,
        "image_failure_summary": error_summary(results),
        "image_failure_class_summary": error_class_summary(results),
        "image_failure_models": compact_failures,
        "image_failure_samples": compact_failures[:50],
    }


def merge_image_output(path: Path, image_payload: dict[str, Any]) -> dict[str, Any]:
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    merged = dict(existing)
    image_payload = carry_previous_image_models(existing, image_payload)
    merged.update(image_payload)
    return merged


def filtered_models(
    model_ids: list[str],
    *,
    include_pattern: str | None,
    exclude_patterns: list[str],
    max_models: int | None,
) -> list[str]:
    filtered = model_ids
    if include_pattern:
        include_re = re.compile(include_pattern, re.IGNORECASE)
        filtered = [model_id for model_id in filtered if include_re.search(model_id)]
    for pattern in exclude_patterns:
        exclude_re = re.compile(pattern, re.IGNORECASE)
        filtered = [model_id for model_id in filtered if not exclude_re.search(model_id)]
    if max_models:
        filtered = filtered[: max(0, max_models)]
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify which LiteLLM models can answer a tiny chat or image request."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LITELLM_BASE_URL") or os.environ.get("LITELLM_API_BASE_URL") or "http://127.0.0.1:7860",
        help="LiteLLM gateway base URL. May include or omit /v1.",
    )
    parser.add_argument("--api-key", default=os.environ.get("LITELLM_API_KEY"))
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--probe",
        choices=("chat", "image"),
        default="chat",
        help="Probe chat completions or image generations.",
    )
    parser.add_argument("--prompt", default="hi")
    parser.add_argument(
        "--image-prompt",
        default="a tiny blue square on a white background",
    )
    parser.add_argument(
        "--image-size",
        default=None,
        help="Optional image size to send. Omit by default for provider compatibility.",
    )
    parser.add_argument(
        "--image-scope",
        choices=("generation", "all"),
        default="generation",
        help="Image candidate scope. generation keeps prompt-to-image models; all also includes edit/video/upscale-like image entries for diagnosis.",
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=25)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--include-pattern", default=None)
    parser.add_argument("--exclude-pattern", action="append", default=[])
    parser.add_argument("--yes", action="store_true", help="Actually ping models. Without this, only prints the plan.")
    args = parser.parse_args()

    api_key = args.api_key
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
    base_url = normalize_base_url(args.base_url)
    model_entries = fetch_model_entries(base_url, api_key, args.timeout)
    if args.probe == "image":
        model_entries = [
            entry
            for entry in model_entries
            if is_image_model_entry(entry, scope=args.image_scope)
        ]
    model_ids = [entry["id"] for entry in model_entries]
    targets = filtered_models(
        model_ids,
        include_pattern=args.include_pattern,
        exclude_patterns=args.exclude_pattern,
        max_models=args.max_models,
    )
    print(
        f"Fetched {len(model_entries)} {args.probe} candidate models "
        f"from {base_url}; selected {len(targets)} targets."
    )

    if not args.yes:
        print("Dry run only. Re-run with --yes to ping models.")
        return 0

    if not targets:
        raise SystemExit("No models selected.")

    checked_at = utc_now()
    results: list[dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(
                ping_image_model if args.probe == "image" else ping_model,
                model_id,
                base_url=base_url,
                api_key=api_key,
                prompt=args.image_prompt if args.probe == "image" else args.prompt,
                **(
                    {"image_size": args.image_size}
                    if args.probe == "image"
                    else {"max_tokens": max(1, args.max_tokens)}
                ),
                timeout=args.timeout,
                retries=max(0, args.retries),
                retry_sleep=args.retry_sleep,
            )
            for model_id in targets
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            completed += 1
            if result.get("ok"):
                print(f"[{index}/{len(targets)}] ok {result['id']}")
            elif index == 1 or index % 25 == 0:
                print(f"[{index}/{len(targets)}] failures so far: {index - sum(1 for item in results if item.get('ok'))}")
            if args.checkpoint_every and completed % max(1, args.checkpoint_every) == 0:
                if args.probe == "image":
                    checkpoint_payload = merge_image_output(
                        args.output,
                        image_output_payload(
                            checked_at=checked_at,
                            base_url=base_url,
                            prompt=args.image_prompt,
                            image_size=args.image_size,
                            total_models=len(model_entries),
                            targets=targets,
                            results=results,
                            complete=False,
                        ),
                    )
                else:
                    checkpoint_payload = output_payload(
                        checked_at=checked_at,
                        base_url=base_url,
                        prompt=args.prompt,
                        max_tokens=args.max_tokens,
                        total_models=len(model_entries),
                        targets=targets,
                        results=results,
                        complete=False,
                    )
                write_json_atomic(args.output, checkpoint_payload)

    if args.probe == "image":
        output = merge_image_output(
            args.output,
            image_output_payload(
                checked_at=checked_at,
                base_url=base_url,
                prompt=args.image_prompt,
                image_size=args.image_size,
                total_models=len(model_entries),
                targets=targets,
                results=results,
                complete=True,
            ),
        )
    else:
        output = output_payload(
            checked_at=checked_at,
            base_url=base_url,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            total_models=len(model_entries),
            targets=targets,
            results=results,
            complete=True,
        )
    write_json_atomic(args.output, output)
    if args.probe == "image":
        print(f"Wrote {args.output} with {output['image_usable_count']} image-usable models.")
    else:
        print(f"Wrote {args.output} with {output['usable_count']} usable models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
