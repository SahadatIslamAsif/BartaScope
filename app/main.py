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

import logging
import os
import time
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        classifier.load()
        log.info("model loaded path=%s version=%s threads=%d load_s=%.3f",
                 classifier.model_path, classifier.model_version,
                 classifier.threads, classifier.load_seconds)
    except Exception:
        # Start anyway so /healthz answers and the platform can report a
        # useful readiness failure instead of a crash loop with no endpoint.
        log.exception("model failed to load; /readyz will report not ready")
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
    results, inference_ms = classifier.predict([req.text])
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
    results, inference_ms = classifier.predict(req.texts)
    metrics.record_latency(time.perf_counter() - t0)

    for result in results:
        metrics.record_prediction(result["label"], result["language"])
    log.info("classify_batch n=%d chars_total=%d ms=%.2f",
             len(results), sum(r["chars"] for r in results), inference_ms)
    return BatchClassifyResponse(
        results=results, inference_ms=round(inference_ms, 3),
        count=len(results),
    )
