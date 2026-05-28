#!/usr/bin/env sh
set -eu

: "${HOST:=0.0.0.0}"
: "${PORT:=7860}"
: "${LITELLM_INTERNAL_PORT:=7861}"
: "${LITELLM_CONFIG_TEMPLATE:=/app/config/config.yaml}"
: "${LITELLM_RENDERED_CONFIG:=/tmp/litellm-config.yaml}"

render_args="$LITELLM_CONFIG_TEMPLATE $LITELLM_RENDERED_CONFIG"
if [ "${LITELLM_STRICT_ENV:-}" = "1" ]; then
  render_args="$render_args --strict-env"
fi

python /app/scripts/render-config.py $render_args

litellm --config "$LITELLM_RENDERED_CONFIG" --host 127.0.0.1 --port "$LITELLM_INTERNAL_PORT" &
litellm_pid="$!"

cleanup() {
  kill "$litellm_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

export LITELLM_UPSTREAM_URL="http://127.0.0.1:$LITELLM_INTERNAL_PORT"
python /app/scripts/proxy_app.py
