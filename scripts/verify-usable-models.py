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
        if attempt < attempts and is_transient_error(str(result.get("error") or "")):
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
    return normalized in {"texttoimage", "imagegeneration"}


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


def is_image_model_entry(raw_model: dict[str, Any]) -> bool:
    if raw_image_metadata(raw_model):
        return True

    text = " ".join(
        str(raw_model.get(key) or "")
        for key in ("id", "model", "name", "title", "task", "mode")
    ).lower()
    image_terms = (
        "black-forest-labs",
        "dall-e",
        "dalle",
        "flux",
        "gpt-image",
        "image-generation",
        "image_generation",
        "image-preview",
        "imagen",
        "ideogram",
        "kandinsky",
        "kolors",
        "midjourney",
        "playground-v",
        "recraft",
        "sdxl",
        "seedream",
        "stable-diffusion",
        "text-to-image",
    )
    return any(term in text for term in image_terms)


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
        if attempt < attempts and is_transient_error(str(result.get("error") or "")):
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
        "image_failure_samples": sorted(failures, key=lambda item: str(item["id"]))[:50],
    }


def merge_image_output(path: Path, image_payload: dict[str, Any]) -> dict[str, Any]:
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    merged = dict(existing)
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
        model_entries = [entry for entry in model_entries if is_image_model_entry(entry)]
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
