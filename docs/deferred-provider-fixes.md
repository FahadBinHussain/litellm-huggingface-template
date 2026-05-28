# Deferred provider fixes

These provider issues were investigated and intentionally deferred. Recheck this file before rediscovering them from scratch.

## Cloudflare

- Keep the full Cloudflare catalog for now.
- Some entries work with the current key, and some return `invalid model` or fail because they are not chat models.
- Do not delete or label the broader catalog yet; account tier, model availability, or route support may change later.

Verified working examples:

- `cloudflare/@cf/meta/llama-3.2-3b-instruct`
- `cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast`

Known currently failing examples:

- `cloudflare/@cf/meta/llama-3.1-8b-instruct`
- `cloudflare/@cf/qwen/qwen1.5-14b-chat-awq`

## AssemblyAI

- `assemblyai/best` and `assemblyai/nano` currently fail through LiteLLM.
- The current account key is accepted by AssemblyAI, but the account does not have LLM Gateway access.
- Do not pursue paid upgrade paths without explicit user approval.

## Modal

- `/modal/*` is only a reserved pass-through slot until `MODAL_API_BASE` points to a real deployed Modal endpoint.
- `MODAL_API_KEY` alone is not enough; the wrapper also needs the base URL to forward requests to.

## Inference.sh

- Correct endpoint through the wrapper: `POST /inference-sh/apps/run`.
- The old root probe `/inference-sh/` returns `404` because it is not a real API endpoint.
- `infsh/shell` confirmed the route shape, but the current account returned `402 payment_required` / insufficient balance.
- The docs sample `infsh/echo` returned `500 Failed to run app` both through the wrapper and direct to Inference.sh, so do not use it as the health check.

## WorldLabs

- Correct endpoint family through the wrapper: `/worldlabs/marble/v1/...`.
- The old probe `/worldlabs/v1/tasks` is from the wrong API family and returns `404`.
- Safe verified endpoint: `POST /worldlabs/marble/v1/worlds:list` with `{ "page_size": 1 }`, which returned `200`.
- Do not run `worlds:generate` without explicit approval because it can spend credits.
