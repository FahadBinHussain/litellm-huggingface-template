FROM ghcr.io/berriai/litellm:main-latest

WORKDIR /app

USER root

COPY config/config.yaml /app/config/config.yaml
COPY config/model-catalog.json /app/config/model-catalog.json
COPY config/usable-models.json /app/config/usable-models.json
COPY scripts/proxy_app.py /app/scripts/proxy_app.py
COPY scripts/render-config.py /app/scripts/render-config.py
COPY scripts/start-litellm.sh /app/scripts/start-litellm.sh

RUN chmod +x /app/scripts/start-litellm.sh

ENV HOST=0.0.0.0 \
    PORT=7860 \
    LITELLM_INTERNAL_PORT=7861 \
    LITELLM_CONFIG_TEMPLATE=/app/config/config.yaml \
    LITELLM_RENDERED_CONFIG=/tmp/litellm-config.yaml

EXPOSE 7860

ENTRYPOINT []
CMD ["/app/scripts/start-litellm.sh"]
