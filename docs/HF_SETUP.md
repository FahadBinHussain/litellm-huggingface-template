# Hugging Face Setup

This Space is a wrapper around upstream LiteLLM. It does not store provider keys or rendered runtime config in git.

`config/config.yaml` stays small. `config/model-catalog.json` stores the grouped model catalog, and `scripts/render-config.py` expands it into `/tmp/litellm-config.yaml` every time the Space starts.

The optional sheet sync can produce API provider env entries. The rendered config includes those as model-backed providers where LiteLLM can route models, and as pass-through endpoints for API providers that are not chat/model catalogs.

## Deploy Code

```powershell
hf repos create YOUR_USERNAME/litellm-huggingface-template --repo-type space --space-sdk docker --exist-ok
hf upload YOUR_USERNAME/litellm-huggingface-template . --repo-type space
```

## Sync Secrets

First refresh the local secret store from the sheet:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync-local-secrets.ps1 -Email user@example.com -SpreadsheetId YOUR_SPREADSHEET_ID
```

Then send the same names to the Space:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync-hf-secrets.ps1 -SpaceId YOUR_USERNAME/litellm-huggingface-template
```

The script uses `hf spaces secrets add --secrets-file` with a temporary file that is deleted after the CLI returns.

## Optional Values

These are not sheet `[ai apis]` keys, but the config can use them:

```text
LITELLM_MASTER_KEY
CLOUDFLARE_ACCOUNT_ID
MODAL_API_BASE
DATABASE_URL
```

`MODAL_API_BASE` must point at your deployed Modal endpoint for `/modal` to work.
