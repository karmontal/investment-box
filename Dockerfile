# syntax=docker/dockerfile:1
# The syntax directive and the cache mounts below require BuildKit, which is
# the default for `docker build` when buildx is installed. Homebrew's docker
# formula does not include buildx: install docker-buildx too, or the build
# falls back to the legacy builder and fails on the --mount flags.
# Multi-stage so the runtime image carries no build tooling.
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Dependencies first, so a source change does not invalidate the layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra broker

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra broker


FROM python:3.11-slim AS runtime

# Run as a non-root user. A trading process has credentials in its
# environment; it does not also need root in its container.
RUN groupadd --system --gid 1001 investment \
    && useradd --system --uid 1001 --gid investment --create-home investment

WORKDIR /app

COPY --from=builder --chown=investment:investment /app/.venv /app/.venv
COPY --chown=investment:investment src/ ./src/
COPY --chown=investment:investment config/ ./config/
COPY --chown=investment:investment scripts/ ./scripts/
COPY --chown=investment:investment alembic.ini ./

# Optional build stamp, shown in the dashboard sidebar so a stale page is
# visible at a glance. .git is excluded from the build context, so the commit
# has to be passed in; it defaults to "unknown" and nothing depends on it.
ARG GIT_COMMIT=unknown

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    IB__DATA_DIR=/data \
    IB_BUILD_COMMIT=$GIT_COMMIT

# The database, the parquet cache and the logs live on a volume, so a rebuild
# never destroys trade history.
RUN mkdir -p /data && chown investment:investment /data
VOLUME ["/data"]

USER investment

# Deliberately no default CMD that trades. The compose file names the service
# explicitly, so `docker run` on this image cannot start an engine by accident.
CMD ["python", "scripts/health_check.py"]
