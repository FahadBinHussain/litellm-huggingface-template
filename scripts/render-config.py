import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ENV_REF_RE = re.compile(r"os\.environ/([A-Za-z0-9_]+)")
OPTIONAL_ENV_REFS = {
    "CLOUDFLARE_ACCOUNT_ID",
    "DATABASE_URL",
    "LITELLM_MASTER_KEY",
    "MODAL_API_BASE",
}
API_PROVIDER_ENVS = {
    "AIMLAPI_API_KEY",
    "ASSEMBLYAI_API_KEY",
    "CEREBRAS_API_KEY",
    "CLOUDFLARE_API_TOKEN",
    "COHERE_API_KEY",
    "DEEPGRAM_API_KEY",
    "DISCORD_TOKEN",
    "EDENAI_API_KEY",
    "ELECTRONHUB_API_KEY",
    "ELEVENLABS_API_KEY",
    "EXA_API_KEY",
    "GEMINI_API_KEY",
    "GENLABS_API_KEY",
    "GITHUB_API_KEY",
    "GROQ_API_KEY",
    "HUGGINGFACE_API_KEY",
    "HUGGINGFACE_API_KEY_1",
    "HUGGINGFACE_API_KEY_2",
    "INFERENCE_SH_API_KEY",
    "JINA_AI_API_KEY",
    "MAPLEFLOW_API_KEY",
    "MISTRAL_API_KEY",
    "MODAL_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY_1",
    "OPENROUTER_API_KEY_2",
    "OPENROUTER_API_KEY_3",
    "OPENROUTER_API_KEY_4",
    "POLLINATIONS_API_KEY",
    "POLLINATIONS_API_KEY_1",
    "SAMBANOVA_API_KEY",
    "STABLEHORDE_API_KEY",
    "TAVILY_API_KEY",
    "TWELVELABS_API_KEY",
    "VERCEL_AI_GATEWAY_API_KEY",
    "VOIDAI_API_KEY",
    "WORLDLABS_API_KEY",
    "XIAOMI_MIMO_API_KEY",
    "XIAOMI_MIMO_PLAN_TOKEN",
    "YOU_API_KEY",
}

PASS_THROUGH_ENDPOINTS: list[dict[str, Any]] = [
    {
        "path": "/genlabs",
        "target": "https://api.genlabs.dev/deca/v1",
        "include_subpath": True,
        "headers": {"Authorization": "Bearer os.environ/GENLABS_API_KEY"},
    },
    {
        "path": "/inference-sh",
        "target": "https://api.inference.sh",
        "include_subpath": True,
        "headers": {"Authorization": "Bearer os.environ/INFERENCE_SH_API_KEY"},
    },
    {
        "path": "/tavily",
        "target": "https://api.tavily.com",
        "include_subpath": True,
        "headers": {"Authorization": "Bearer os.environ/TAVILY_API_KEY"},
    },
    {
        "path": "/exa",
        "target": "https://api.exa.ai",
        "include_subpath": True,
        "headers": {"x-api-key": "os.environ/EXA_API_KEY"},
    },
    {
        "path": "/discord",
        "target": "https://api.zukijourney.com/v1",
        "include_subpath": True,
        "headers": {"Authorization": "Bearer os.environ/DISCORD_TOKEN"},
    },
    {
        "path": "/worldlabs",
        "target": "https://api.worldlabs.ai",
        "include_subpath": True,
        "headers": {"WLT-Api-Key": "os.environ/WORLDLABS_API_KEY"},
    },
    {
        "path": "/twelvelabs",
        "target": "https://api.twelvelabs.io",
        "include_subpath": True,
        "headers": {"x-api-key": "os.environ/TWELVELABS_API_KEY"},
    },
    {
        "path": "/stablehorde",
        "target": "https://aihorde.net/api",
        "include_subpath": True,
        "headers": {"apikey": "os.environ/STABLEHORDE_API_KEY"},
    },
    {
        "path": "/you",
        "target": "https://ydc-index.io",
        "include_subpath": True,
        "headers": {"X-API-Key": "os.environ/YOU_API_KEY"},
    },
    {
        "path": "/modal",
        "target": "os.environ/MODAL_API_BASE",
        "include_subpath": True,
        "headers": {"Authorization": "Bearer os.environ/MODAL_API_KEY"},
    },
]


def env(name: str) -> str:
    return os.environ.get(name, "").strip()


def env_names(base: str) -> list[str]:
    names: list[str] = []
    if env(base):
        names.append(base)
    for index in range(1, 11):
        name = f"{base}_{index}"
        if env(name):
            names.append(name)
    return names


def load_secrets(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")

    loaded = 0
    for name, value in payload.items():
        if not isinstance(name, str) or not isinstance(value, str) or not value:
            continue
        os.environ[name] = value
        loaded += 1
    return loaded


def yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    return yaml_quote(str(value))


def render_yaml(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(render_yaml(child, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {render_scalar(child)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if isinstance(item, dict):
                if not item:
                    lines.append(f"{prefix}- {{}}")
                    continue
                first = True
                for key, child in item.items():
                    marker = "- " if first else "  "
                    if isinstance(child, (dict, list)):
                        lines.append(f"{prefix}{marker}{key}:")
                        lines.extend(render_yaml(child, indent + 4))
                    else:
                        lines.append(f"{prefix}{marker}{key}: {render_scalar(child)}")
                    first = False
            elif isinstance(item, list):
                lines.append(f"{prefix}-")
                lines.extend(render_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {render_scalar(item)}")
        return lines
    return [f"{prefix}{render_scalar(value)}"]


def add_model(
    models: list[dict[str, Any]],
    alias: str,
    model: str,
    model_info: dict[str, Any] | None = None,
    **params: str,
) -> None:
    litellm_params = {"model": model}
    litellm_params.update({key: value for key, value in params.items() if value is not None})
    entry: dict[str, Any] = {"model_name": alias, "litellm_params": litellm_params}
    if model_info:
        entry["model_info"] = model_info
    models.append(entry)


def suffixed(alias: str, index: int, total: int) -> str:
    return alias if total == 1 else f"{alias}-{index}"


def build_legacy_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []

    gemini_keys = env_names("GEMINI_API_KEY")
    for index, key_name in enumerate(gemini_keys, start=1):
        suffix_total = len(gemini_keys)
        add_model(
            models,
            suffixed("gemini-flash", index, suffix_total),
            "gemini/gemini-2.5-flash",
            api_key=f"os.environ/{key_name}",
        )
        add_model(
            models,
            suffixed("gemini-pro", index, suffix_total),
            "gemini/gemini-2.5-pro",
            api_key=f"os.environ/{key_name}",
        )

    openrouter_base = env("OPENROUTER_API_BASE_URL") or "https://openrouter.ai/api/v1"
    openrouter_keys = env_names("OPENROUTER_API_KEY")
    for index, key_name in enumerate(openrouter_keys, start=1):
        add_model(
            models,
            suffixed("openrouter-auto", index, len(openrouter_keys)),
            "openrouter/auto",
            api_key=f"os.environ/{key_name}",
            api_base=openrouter_base,
        )

    openai_keys = env_names("OPENAI_API_KEY")
    for index, key_name in enumerate(openai_keys, start=1):
        add_model(
            models,
            suffixed("openai-fast", index, len(openai_keys)),
            "openai/gpt-4o-mini",
            api_key=f"os.environ/{key_name}",
        )

    anthropic_keys = env_names("ANTHROPIC_API_KEY")
    for index, key_name in enumerate(anthropic_keys, start=1):
        add_model(
            models,
            suffixed("claude-haiku", index, len(anthropic_keys)),
            "anthropic/claude-3-5-haiku-latest",
            api_key=f"os.environ/{key_name}",
        )

    custom_base = env("CUSTOM_OPENAI_API_BASE")
    if custom_base:
        custom_alias = env("CUSTOM_OPENAI_ALIAS") or "custom-openai"
        custom_model = env("CUSTOM_OPENAI_MODEL") or "gpt-4o-mini"
        params = {"api_base": custom_base}
        if env("CUSTOM_OPENAI_API_KEY"):
            params["api_key"] = "os.environ/CUSTOM_OPENAI_API_KEY"
        add_model(models, custom_alias, f"openai/{custom_model}", **params)

    return models


def default_catalog_path(template_path: Path) -> Path:
    return template_path.resolve().parent / "model-catalog.json"


def load_model_catalog(path: Path) -> list[dict[str, Any]]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog.get("version") != 1:
        raise ValueError(f"Unsupported model catalog version in {path}")

    models: list[dict[str, Any]] = []
    for group in catalog.get("groups", []):
        params = dict(group.get("params") or {})
        api_key_env = group.get("api_key_env")
        literal_api_key = group.get("literal_api_key")
        if api_key_env:
            params["api_key"] = f"os.environ/{api_key_env}"
        elif literal_api_key is not None:
            params["api_key"] = literal_api_key

        for suffix in group.get("suffixes", []):
            if isinstance(suffix, dict):
                alias_suffix = suffix["alias"]
                model_suffix = suffix["model"]
            else:
                alias_suffix = str(suffix)
                model_suffix = str(suffix)

            add_model(
                models,
                f"{group['alias_prefix']}/{alias_suffix}",
                f"{group['model_prefix']}/{model_suffix}",
                model_info=group.get("model_info"),
                **params,
            )

    return models


def render_models(models: list[dict[str, Any]]) -> str:
    if not models:
        return "  []"
    return "\n".join(render_yaml(models, indent=2))


def render_general_settings() -> str:
    settings: dict[str, Any] = {
        "pass_through_endpoints": PASS_THROUGH_ENDPOINTS,
    }
    if env("LITELLM_MASTER_KEY"):
        settings = {"master_key": "os.environ/LITELLM_MASTER_KEY", **settings}
    if env("DATABASE_URL"):
        settings["database_url"] = "os.environ/DATABASE_URL"

    return "\n".join(render_yaml(settings, indent=2))


def render_template(template: str, models: list[dict[str, Any]]) -> str:
    rendered = template
    if "__AUTO_MODEL_LIST__" in rendered:
        rendered = rendered.replace("__AUTO_MODEL_LIST__", render_models(models))
    if "__GENERAL_SETTINGS__" in rendered:
        rendered = rendered.replace("__GENERAL_SETTINGS__", render_general_settings())
    return rendered


def env_refs(text: str) -> set[str]:
    return set(ENV_REF_RE.findall(text))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--catalog",
        type=Path,
        help="Model catalog JSON to expand into model_list. Defaults to config/model-catalog.json beside the template.",
    )
    parser.add_argument(
        "--include-legacy-aliases",
        action="store_true",
        help="Also add the old short aliases such as gemini-flash and openai-fast.",
    )
    parser.add_argument(
        "--secrets",
        type=Path,
        default=Path(os.environ["LITELLM_SECRETS_FILE"]) if os.environ.get("LITELLM_SECRETS_FILE") else None,
        help="Optional local JSON secret file. Values are loaded only into this process.",
    )
    parser.add_argument("--strict-env", action="store_true")
    parser.add_argument("--summary-json", action="store_true")
    args = parser.parse_args()

    secrets_loaded = 0
    if args.secrets:
        secrets_loaded = load_secrets(args.secrets)

    catalog_path = args.catalog or default_catalog_path(args.template)
    models = load_model_catalog(catalog_path)
    if args.include_legacy_aliases:
        models.extend(build_legacy_models())

    template = args.template.read_text(encoding="utf-8")
    rendered = render_template(template, models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")

    refs = env_refs(rendered)
    present = {name for name in refs if env(name)}
    missing_required = sorted(refs - present - OPTIONAL_ENV_REFS)
    missing_optional = sorted((refs - present) & OPTIONAL_ENV_REFS)
    api_refs = sorted(refs & API_PROVIDER_ENVS)
    api_present = sorted(set(api_refs) & present)
    missing_api_refs = sorted(set(api_refs) - present)
    summary = {
        "template": str(args.template),
        "catalog": str(catalog_path),
        "output": str(args.output),
        "secretsLoaded": secrets_loaded,
        "models": len(models),
        "apiProviderEnvRefs": len(api_refs),
        "apiProviderEnvRefsPresent": len(api_present),
        "missingApiProviderEnvRefs": missing_api_refs,
        "envRefs": len(refs),
        "envRefsPresent": len(present),
        "missingRequired": missing_required,
        "missingOptional": missing_optional,
    }

    if args.summary_json:
        print(json.dumps(summary, indent=2), file=sys.stderr)
    else:
        print(
            "Rendered LiteLLM config "
            f"({len(models)} models, {len(api_refs)} API provider env refs, "
            f"{len(api_present)} API provider env refs present, {len(refs)} total env refs, "
            f"{len(missing_required)} required missing).",
            file=sys.stderr,
        )

    if args.strict_env and missing_required:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
