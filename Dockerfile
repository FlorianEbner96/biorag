# syntax=docker/dockerfile:1

# Two stages: the builder installs everything into a virtualenv and downloads
# the embedding model; the runtime stage copies only those results onto a clean
# base, so compilers, pip caches and build leftovers never ship.
#
# Layers are ordered from slowest-changing to fastest-changing. Torch and the
# model rarely change, so a typical code edit rebuilds only the last few layers.

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    HF_HOME=/opt/hf
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv "$VIRTUAL_ENV"

# CPU-only torch. The default PyPI wheel bundles ~4 GB of CUDA libraries that a
# CPU-only MiniLM service never uses; installing it first means
# sentence-transformers finds torch already satisfied and doesn't pull CUDA.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ".[api,generation]"

# Bake the embedding model into the image so containers start without network
# access and without an 80 MB download on every cold start. The model name is
# read from config so the image can never disagree with the code.
RUN python -c "from biorag.config import DEFAULT; \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer(DEFAULT.embedding.model_name)"


FROM python:${PYTHON_VERSION}-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    BIORAG_DATA_DIR=/app/data \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Unprivileged user: a compromised process can't modify the image's files.
RUN useradd --create-home --uid 10001 biorag

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf /opt/hf

# The default index. Mount a volume over /app/data/index to serve a different
# corpus without rebuilding.
COPY data/index /app/data/index

USER biorag
WORKDIR /app
EXPOSE 8000

# /health returns 200 even with no index (so it's always reachable for
# debugging); the check additionally requires status "ok" so an image missing
# its index reports unhealthy instead of silently serving 503s.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import json, urllib.request as u; \
r = json.load(u.urlopen('http://127.0.0.1:8000/health', timeout=4)); \
raise SystemExit(r['status'] != 'ok')"

CMD ["uvicorn", "biorag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
