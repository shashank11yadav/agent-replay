# One Python image for the builder and runtime stages, so the venv runs on the
# interpreter it was built with.
ARG PYTHON_IMAGE=python:3.12.16-slim

# Stage 1: build the trace viewer.
FROM node:22.23.4-slim AS viewer
WORKDIR /build/viewer
COPY viewer/package.json viewer/package-lock.json ./
RUN npm ci
COPY viewer/ ./
RUN npm run build

# Stage 2: install the project and its dependencies into /app/.venv.
FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-editable --no-dev
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-dev

# Stage 3: runtime image, non-root.
FROM ${PYTHON_IMAGE}
RUN useradd --system --uid 10001 --create-home --home-dir /home/app app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=viewer /build/viewer/dist /app/viewer/dist
COPY evals/ /app/evals/
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    AGENT_REPLAY_VIEWER_DIST=/app/viewer/dist
USER app
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "agent_replay.api.asgi:app", "--host", "0.0.0.0", "--port", "8000"]
