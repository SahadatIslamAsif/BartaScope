"""
Bartascope API -- SMS message type classification.

    POST /v1/classify         one message
    POST /v1/classify/batch   up to 64 messages
    GET  /healthz             liveness: is the process up
    GET  /readyz              readiness: is the model loaded
    GET  /metrics             Prometheus exposition

The ONNX session is created once in the lifespan handler and reused. Model
selection is deliberate and recorded in CLAUDE.md: fp32 is served rather
than INT8, because INT8 tripled phishing false negatives into `normal`
(4 -> 13) for a 1.48-point macro-F1 loss. Speed was not worth that on a
safety classifier.

Logging rule, non-negotiable: message text is never logged. Users paste real
SMS containing their own transaction details. Length and predicted class
only.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from pathlib import Path

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from app.metrics import metrics
from app.model import MODEL_VERSION, Classifier
from app.schemas import (BatchClassifyRequest, BatchClassifyResponse,
                         ClassifyRequest, ClassifyResponse, HealthResponse,
                         ReadyResponse)

logging.basicConfig(
    level=os.getenv("BARTASCOPE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("bartascope")

MODEL_PATH = os.getenv("BARTASCOPE_MODEL_PATH", "onnx_out/model.onnx")
TOKENIZER_DIR = os.getenv("BARTASCOPE_TOKENIZER_DIR", "onnx_out")
MAX_LENGTH = int(os.getenv("BARTASCOPE_MAX_LENGTH", "64"))
# Should equal the container CPU limit. See the note in app/model.py about
# oversubscription making the optimised build slower than the baseline.
THREADS = int(os.getenv("BARTASCOPE_THREADS", "0"))

classifier = Classifier(MODEL_PATH, TOKENIZER_DIR, MAX_LENGTH, THREADS)

# ---------------------------------------------------------------------------
# Load shedding, sized from benchmarks/latency.csv.
#
# The load test found the service holds 90 rps at p95 21.63 ms, ceilings at
# ~130 rps of actual throughput, and above that degrades purely by queueing:
# at 240 rps offered it still answered every request correctly, after making
# the caller wait 42 seconds. Nothing shed. A 42-second success is worse than
# an immediate 503 -- the caller has long since given up, the work is wasted,
# and on Cloud Run the request would have been killed by the platform anyway.
#
# So: bound the number of in-flight inferences, and refuse rather than queue
# once that bound is saturated for longer than a caller would plausibly wait.
#
# MAX_CONCURRENCY is small on purpose. One request occupies all
# intra_op_num_threads, so requests do not really run in parallel; allowing
# many at once just splits the same CPU into slower slices and inflates every
# latency instead of failing the excess fast. Little's law on the measured
# numbers agrees: 90 rps at ~10 ms of service time is under one request in
# flight on average.
MAX_CONCURRENCY = int(os.getenv("BARTASCOPE_MAX_CONCURRENCY", "0"))
# How long a request may wait for a slot before it is shed. Chosen well under
# the point where measured p95 exploded (1331 ms at 120 rps) so the queue can
# never grow into that regime.
QUEUE_TIMEOUT_S = float(os.getenv("BARTASCOPE_QUEUE_TIMEOUT_MS", "200")) / 1000
# Hard ceiling on a single inference. Nothing measured comes close -- the
# slowest single batch-1 request seen was ~165 ms -- so tripping this means
# something is wrong, not merely busy.
INFERENCE_TIMEOUT_S = float(
    os.getenv("BARTASCOPE_INFERENCE_TIMEOUT_MS", "2000")) / 1000

_slots: asyncio.Semaphore | None = None


class Shed(Exception):
    """Raised to turn away work rather than queue it."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail


async def run_inference(texts: list, endpoint: str):
    """Acquire a slot, run inference off the event loop, or shed.

    Inference runs in a worker thread rather than inline. Inline it would
    block the event loop for the duration, which means the server could not
    accept, time out, or reject anything while a request was being served --
    a timeout that cannot fire while the thing it guards is running is not a
    timeout. ONNX Runtime releases the GIL inside Run, so a thread is a real
    thread here.
    """
    try:
        await asyncio.wait_for(_slots.acquire(), QUEUE_TIMEOUT_S)
    except (asyncio.TimeoutError, TimeoutError):
        raise Shed("overloaded",
                   "Server is at capacity. Retry shortly.") from None

    try:
        return await asyncio.wait_for(
            run_in_threadpool(classifier.predict, texts), INFERENCE_TIMEOUT_S)
    except (asyncio.TimeoutError, TimeoutError):
        # The worker thread cannot be cancelled and will finish on its own;
        # the slot is released below either way. This path means something is
        # pathological, so freeing the caller matters more than the orphan.
        raise Shed("timeout", "Inference timed out.") from None
    finally:
        _slots.release()


def shed_response(exc: Shed, endpoint: str) -> JSONResponse:
    metrics.record_shed(endpoint, exc.reason)
    log.warning("shed endpoint=%s reason=%s", endpoint, exc.reason)
    return JSONResponse(
        status_code=503,
        content={"detail": exc.detail, "reason": exc.reason},
        # Tells a well-behaved client to back off instead of hot-looping,
        # and lets a load balancer drain rather than hammer.
        headers={"Retry-After": "1"},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _slots, MAX_CONCURRENCY
    try:
        classifier.load()
        log.info("model loaded path=%s version=%s threads=%d load_s=%.3f",
                 classifier.model_path, classifier.model_version,
                 classifier.threads, classifier.load_seconds)
    except Exception:
        # Start anyway so /healthz answers and the platform can report a
        # useful readiness failure instead of a crash loop with no endpoint.
        log.exception("model failed to load; /readyz will report not ready")

    # Default to the thread count, i.e. the container's CPU limit. One
    # request already uses all of them, so this permits one full-width
    # inference plus one waiting to start, and nothing more.
    if MAX_CONCURRENCY <= 0:
        MAX_CONCURRENCY = max(2, classifier.threads)
    _slots = asyncio.Semaphore(MAX_CONCURRENCY)

    # Cap Starlette's worker pool to match. Left at its default of 40 it
    # would happily run 40 concurrent inferences, each spawning a full-width
    # ORT thread pool -- exactly the oversubscription that made the optimised
    # build measure slower than the PyTorch baseline during benchmarking.
    anyio.to_thread.current_default_thread_limiter().total_tokens = \
        MAX_CONCURRENCY

    log.info("load shedding: max_concurrency=%d queue_timeout_ms=%.0f "
             "inference_timeout_ms=%.0f",
             MAX_CONCURRENCY, QUEUE_TIMEOUT_S * 1000,
             INFERENCE_TIMEOUT_S * 1000)
    yield


app = FastAPI(
    title="Bartascope",
    description="SMS message type classifier. Advisory only.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    response = await call_next(request)
    metrics.record_request(request.url.path, response.status_code)
    return response


INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
async def index():
    """The frontend, served from this same app.

    One origin for the page and the API, so the browser treats /v1/classify
    as same-origin and no CORS configuration exists to get wrong. A single
    FileResponse rather than a StaticFiles mount: there is exactly one file,
    and mounting a directory would also expose anything that later lands in
    it.
    """
    if not INDEX_HTML.exists():
        return JSONResponse(status_code=404,
                            content={"detail": "frontend not installed"})
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/healthz", response_model=HealthResponse)
async def healthz():
    """Liveness. Deliberately does not touch the model: a live process with
    a broken model should be reported by /readyz, not restarted in a loop."""
    return HealthResponse(status="ok", model_version=MODEL_VERSION)


@app.get("/readyz", response_model=ReadyResponse)
async def readyz():
    if not classifier.ready:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "model_version": MODEL_VERSION,
                     "model_path": str(classifier.model_path),
                     "threads": classifier.threads,
                     "max_length": classifier.max_length,
                     "load_seconds": None},
        )
    return ReadyResponse(
        status="ready", model_version=MODEL_VERSION,
        model_path=str(classifier.model_path), threads=classifier.threads,
        max_length=classifier.max_length, load_seconds=classifier.load_seconds,
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    return PlainTextResponse(metrics.render(),
                             media_type="text/plain; version=0.0.4")


def _not_ready() -> JSONResponse:
    return JSONResponse(status_code=503,
                        content={"detail": "model not loaded"})


@app.post("/v1/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    if not classifier.ready:
        metrics.record_error("/v1/classify", "not_ready")
        return _not_ready()

    t0 = time.perf_counter()
    try:
        results, inference_ms = await run_inference([req.text], "/v1/classify")
    except Shed as exc:
        return shed_response(exc, "/v1/classify")
    metrics.record_latency(time.perf_counter() - t0)

    result = results[0]
    metrics.record_prediction(result["label"], result["language"])
    # Length and class only. Never the text.
    log.info("classify chars=%d label=%s lang=%s ms=%.2f",
             result["chars"], result["label"], result["language"],
             inference_ms)
    return ClassifyResponse(inference_ms=round(inference_ms, 3), **result)


@app.post("/v1/classify/batch", response_model=BatchClassifyResponse)
async def classify_batch(req: BatchClassifyRequest):
    if not classifier.ready:
        metrics.record_error("/v1/classify/batch", "not_ready")
        return _not_ready()

    t0 = time.perf_counter()
    try:
        results, inference_ms = await run_inference(req.texts,
                                                    "/v1/classify/batch")
    except Shed as exc:
        return shed_response(exc, "/v1/classify/batch")
    metrics.record_latency(time.perf_counter() - t0)

    for result in results:
        metrics.record_prediction(result["label"], result["language"])
    log.info("classify_batch n=%d chars_total=%d ms=%.2f",
             len(results), sum(r["chars"] for r in results), inference_ms)
    return BatchClassifyResponse(
        results=results, inference_ms=round(inference_ms, 3),
        count=len(results),
    )
