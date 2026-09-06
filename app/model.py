"""
ONNX Runtime classifier wrapper.

One session, created once at startup and reused for every request. Creating
a session per request would dominate the latency budget entirely -- the
whole point of the ONNX work is that inference costs single-digit
milliseconds, and session creation costs hundreds.

Thread configuration is deliberate. intra_op_num_threads is set from
BARTASCOPE_THREADS, which should equal the container's CPU limit. Running
several uvicorn workers that each spawn a full-width ORT thread pool
oversubscribes the CPU, and ORT's pools spin-wait rather than sleeping when
idle, so the oversubscribed build can measure *slower* than the PyTorch
baseline it was meant to beat. That is not hypothetical: it happened in this
project's own benchmark run when two sessions shared a process.
"""

import os
import time
from pathlib import Path

import numpy as np

from app.textproc import LABELS, detect_lang, normalise

MODEL_VERSION = "bartascope-minilm-v1.0.0"


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class Classifier:
    """Loads the ONNX graph and its tokenizer, and runs predictions."""

    def __init__(self, model_path: str, tokenizer_dir: str,
                 max_length: int = 64, threads: int = 0):
        self.model_path = Path(model_path)
        self.tokenizer_dir = tokenizer_dir
        self.max_length = max_length
        self.threads = threads or os.cpu_count() or 1
        self.model_version = MODEL_VERSION
        self.loaded_at = None
        self.load_seconds = None
        self._session = None
        self._tokenizer = None
        self._input_names = None

    def load(self) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        t0 = time.perf_counter()
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = \
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self.model_path), opts, providers=["CPUExecutionProvider"])
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)

        self.load_seconds = round(time.perf_counter() - t0, 3)
        self.loaded_at = time.time()

    @property
    def ready(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    def predict(self, texts: list) -> tuple:
        """Classify a list of messages. Returns (results, inference_ms).

        inference_ms covers tokenisation plus the ORT run, because that is
        what the caller actually waits for. The benchmark CSV measures the
        ORT run alone, so the two numbers are not interchangeable and this
        one will always be a little larger.
        """
        # Same normalisation the model was fine-tuned on. Skipping it is a
        # silent failure: the model still returns confident probabilities,
        # they are just drawn from a distribution it never saw.
        cleaned = [normalise(t) for t in texts]

        t0 = time.perf_counter()
        enc = self._tokenizer(
            cleaned, return_tensors="np", padding="longest",
            truncation=True, max_length=self.max_length)
        feed = {n: enc[n] for n in self._input_names}
        logits = self._session.run(["logits"], feed)[0]
        inference_ms = (time.perf_counter() - t0) * 1000.0

        probs = softmax(logits)
        results = []
        for text, cleaned_text, row in zip(texts, cleaned, probs):
            idx = int(np.argmax(row))
            lang, needs_review = detect_lang(cleaned_text)
            results.append({
                "label": LABELS[idx],
                "confidence": float(row[idx]),
                "probabilities": {
                    label: float(p) for label, p in zip(LABELS, row)
                },
                "language": lang,
                "language_uncertain": needs_review,
                "chars": len(text),
                "model_version": self.model_version,
            })
        return results, inference_ms
