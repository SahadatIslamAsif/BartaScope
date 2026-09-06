# Bartascope serving image -- ONNX Runtime only, no torch.
#
# Build:
#   docker build -t bartascope .
# Run:
#   docker run --rm -p 8080:8080 --cpus 2 bartascope
#
# Dockerfile.pytorch is the baseline-benchmark image, not a second serving
# image, and exists so the before/after comparison stays reproducible. Do not
# merge them: keeping torch out of this file is the entire point.

# ---------------------------------------------------------------------------
# Stage 1: build the dependency tree in a throwaway layer.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Own virtualenv rather than --user or the system site-packages, so stage 2
# copies exactly one self-contained directory and nothing else follows it.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Strip test suites and bytecode from site-packages. Mostly this is numpy's
# and onnxruntime's own tests, which are tens of MB of files that can never
# run here.
RUN find /opt/venv -type d -name "tests" -prune -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -name "*.pyc" -delete 2>/dev/null || true

# ---------------------------------------------------------------------------
# Stage 2: runtime.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# PYTHONDONTWRITEBYTECODE keeps the filesystem clean; PYTHONUNBUFFERED makes
# logs appear in `docker logs` immediately rather than whenever the buffer
# happens to flush, which matters when timing cold starts.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080

ENV BARTASCOPE_MODEL_PATH=/app/onnx_out/model.onnx \
    BARTASCOPE_TOKENIZER_DIR=/app/onnx_out \
    BARTASCOPE_MAX_LENGTH=64

# Left empty on purpose: the entrypoint fills it from the cgroup CPU quota.
# A number here would bake in the build machine's shape instead.
ENV BARTASCOPE_THREADS=""

# transformers must never reach the network at runtime. The tokenizer is baked
# into the image; a hub lookup would turn a cold start into a DNS timeout.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY --from=builder /opt/venv /opt/venv

# Create the user BEFORE copying anything into /app. A `chown -R /app` after
# the fact rewrites every file into a fresh layer, and since /app holds a
# 489 MB model that silently doubles the image. Measured: it added exactly
# 489 MB. COPY --chown writes the payload once, with the right owner.
RUN useradd --create-home --uid 10001 bartascope

WORKDIR /app

COPY --chown=bartascope:bartascope app/ /app/app/

# Both files are required and they are not interchangeable. The dynamo
# exporter puts the graph in model.onnx (~1.5 MB) and every weight in
# model.onnx.data (~470 MB). Shipping only the graph produces an image that
# builds, starts, and fails at session creation.
COPY --chown=bartascope:bartascope onnx_out/model.onnx onnx_out/model.onnx.data /app/onnx_out/
COPY --chown=bartascope:bartascope onnx_out/tokenizer.json onnx_out/tokenizer_config.json /app/onnx_out/

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Non-root. Nothing here needs to write to disk at runtime.
USER bartascope

EXPOSE 8080

# Liveness only. /readyz is the one that reports a failed model load, and it
# is deliberately not used here: restarting a container whose model file is
# missing just produces a crash loop with no diagnosable endpoint.
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import os,sys,urllib.request; url='http://127.0.0.1:'+os.environ.get('PORT','8080')+'/healthz'; sys.exit(0 if urllib.request.urlopen(url,timeout=2).status==200 else 1)"

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
