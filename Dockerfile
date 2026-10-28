# syntax=docker/dockerfile:1
# One Dockerfile, three images (see docker-compose.yml), selected by SYNC_ARGS:
#   api        (default)                 core: FastAPI + LightGBM + SQLAlchemy, no training code
#   dashboard  --only-group dashboard    Streamlit + httpx only, talks to the API over HTTP
#   bootstrap  --group train             + data build, training, MLflow; runs once and exits
#
# Two stages: uv resolves and installs into /app/.venv in the builder; the runtime stage
# copies only the venv and the code (no uv, no build cache). Behind a TLS-intercepting
# proxy, pass the proxy's CA bundle as a BuildKit secret (see README): it is mounted only
# while dependencies download and never stored in any layer.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never

ARG SYNC_ARGS="--no-default-groups"

# Dependencies first so code changes don't invalidate the (large) dependency layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=secret,id=ca_bundle \
    if [ -s /run/secrets/ca_bundle ]; then export SSL_CERT_FILE=/run/secrets/ca_bundle; fi; \
    uv sync --locked $SYNC_ARGS --no-install-project --no-editable

COPY src ./src
RUN --mount=type=secret,id=ca_bundle \
    if [ -s /run/secrets/ca_bundle ]; then export SSL_CERT_FILE=/run/secrets/ca_bundle; fi; \
    uv sync --locked $SYNC_ARGS --no-editable


FROM python:3.12-slim

# LightGBM needs the OpenMP runtime; it is the only system package added.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
EXPOSE 8000
# Default: the API, serving models/serving/ (written by `crisk export` or the bootstrap).
CMD ["uvicorn", "crisk.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
