---
title: LiteLLM Hugging Face Template
sdk: docker
app_port: 7860
license: mit
---

# LiteLLM Hugging Face Template

One wrapper for the same LiteLLM setup everywhere:

- A Google Sheet can be used as the inventory source.
- `config/local.secrets.json` is the local secret store, generated from the sheet and never committed.
- `config/config.yaml` is a tiny no-secret template.
- `config/model-catalog.json` is the wrapper-owned 5k model catalog.
- The optional sheet sync can populate API provider env entries; some are model routes and some are pass-through API routes.
- Local and Hugging Face both render a full ignored LiteLLM config from the same template, catalog, and configured env names.
- Upstream LiteLLM stays untouched.

## Local

Sync sheet keys into the local secret store:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync-local-secrets.ps1 -Email user@example.com -SpreadsheetId YOUR_SPREADSHEET_ID
```

Start local uv LiteLLM through this wrapper:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-local.ps1
```

No global LiteLLM install is required. The script uses `litellm` if it is already on PATH, or falls back to `uvx --from litellm[proxy] litellm`. The generated local config is `config/local.generated.yaml`; it contains the expanded model list and is not committed.

## Hugging Face

Upload this folder to the Space, then sync the same secret names:

```powershell
hf upload YOUR_USERNAME/litellm-huggingface-template . --repo-type space
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync-hf-secrets.ps1 -SpaceId YOUR_USERNAME/litellm-huggingface-template
```

The Space uses the same `config/config.yaml` and `config/model-catalog.json`. At container startup, `scripts/render-config.py` writes the full LiteLLM config to `/tmp/litellm-config.yaml`, then starts the official LiteLLM proxy with that rendered file.

## Notes

The model catalog exposes OpenAI-compatible providers as LiteLLM model routes. Only configured provider env slots are rendered into the runtime model list, and base slots also accept account-suffixed variants such as `OPENROUTER_API_KEY_WORK` or `HUGGINGFACE_API_KEY_WORK` as additional deployments. Tool APIs that do not provide chat/model-compatible responses stay available through pass-through endpoints instead, for example `/tavily`, `/you`, `/twelvelabs`, `/worldlabs`, `/inference-sh`, `/exa`, and `/modal`.

Capability metadata in the catalog is intentionally conservative. Confirmed provider modes such as chat, embeddings, speech, transcription, and image generation are kept in `model_info`; filename/model-name hints are treated as heuristics until a provider endpoint or official docs confirm them. Paid, usage-based, trial-gated, or payment-method-gated models should stay labeled as such and should not become defaults.

The public container runs a tiny wrapper in front of upstream LiteLLM. The wrapper forwards almost everything to LiteLLM, but adapts GenLabs Deca so `genlabs/deca-2.5-mini`, `genlabs/deca-2.5-pro`, and `genlabs/deca-2.5-ultra` work through `/v1/chat/completions` even though Deca returns streaming SSE responses.

`MODAL_API_KEY` is synced as a secret, but `/modal` also needs `MODAL_API_BASE` set to a deployed Modal endpoint before that route is callable.
