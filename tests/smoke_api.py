"""
End-to-end smoke test for the Bartascope API.

Runs the real app against the real ONNX graph through Starlette's TestClient,
so the lifespan handler executes and the model actually loads. No mocks: the
thing most likely to break here is model loading and preprocessing, and a
mock would hide exactly that.

Usage:
    python tests/smoke_api.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.textproc import LABELS  # noqa: E402

# Deliberately mixed: English transactional, a phishing attempt with a URL,
# Bangla, and a banglish personal message.
SAMPLES = [
    "Your OTP is 123456. Do not share it with anyone.",
    "Congratulations! Your bKash account is suspended. "
    "Verify now at http://bkash-verify-bd.info/login to restore access.",
    "আগামীকাল বিকেলে দেখা হবে, ভালো থেকো।",
    "Bhai ami ekhon office e achi, pore call dibo.",
]

failures = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  PASS  {message}")
    else:
        print(f"  FAIL  {message}")
        failures.append(message)


def main() -> None:
    with TestClient(app) as client:
        print("\n-- health and readiness --")
        r = client.get("/healthz")
        check(r.status_code == 200, f"/healthz -> {r.status_code}")

        r = client.get("/readyz")
        check(r.status_code == 200, f"/readyz -> {r.status_code}")
        ready = r.json()
        check(ready.get("model_version") == "bartascope-minilm-v1.0.0",
              f"model_version is {ready.get('model_version')}")
        print(f"        threads={ready.get('threads')} "
              f"load_seconds={ready.get('load_seconds')} "
              f"max_length={ready.get('max_length')}")

        print("\n-- single classify --")
        for text in SAMPLES:
            r = client.post("/v1/classify", json={"text": text})
            check(r.status_code == 200, f"POST /v1/classify -> {r.status_code}")
            if r.status_code != 200:
                continue
            body = r.json()
            probs = body["probabilities"]
            check(body["label"] in LABELS, f"label {body['label']} is valid")
            check(len(probs) == 4, "four probabilities returned")
            check(abs(sum(probs.values()) - 1.0) < 1e-4,
                  f"probabilities sum to {sum(probs.values()):.6f}")
            check(body["inference_ms"] > 0, "inference_ms populated")
            print(f"        {body['label']:<14} conf {body['confidence']:.3f}"
                  f"  lang {body['language']:<9}"
                  f"  {body['inference_ms']:.2f} ms")

        print("\n-- batch classify --")
        r = client.post("/v1/classify/batch", json={"texts": SAMPLES})
        check(r.status_code == 200,
              f"POST /v1/classify/batch -> {r.status_code}")
        if r.status_code == 200:
            body = r.json()
            check(body["count"] == len(SAMPLES),
                  f"count is {body['count']}, expected {len(SAMPLES)}")
            check(len(body["results"]) == len(SAMPLES),
                  "one result per input")

        print("\n-- batch and single agree --")
        # The same message must classify identically alone and in a batch.
        # Dynamic padding pads to the batch maximum, so a batch feeds the
        # model longer sequences; if attention masking were wrong, this is
        # where it would show.
        single_labels = []
        for text in SAMPLES:
            single_labels.append(
                client.post("/v1/classify", json={"text": text}).json()["label"])
        batch_labels = [r["label"] for r in
                        client.post("/v1/classify/batch",
                                    json={"texts": SAMPLES}).json()["results"]]
        check(single_labels == batch_labels,
              f"single {single_labels} == batch {batch_labels}")

        print("\n-- validation --")
        r = client.post("/v1/classify", json={"text": ""})
        check(r.status_code == 422, f"empty text rejected -> {r.status_code}")
        r = client.post("/v1/classify/batch", json={"texts": []})
        check(r.status_code == 422, f"empty batch rejected -> {r.status_code}")
        r = client.post("/v1/classify/batch", json={"texts": ["hi"] * 65})
        check(r.status_code == 422, f"oversized batch rejected -> {r.status_code}")

        print("\n-- metrics --")
        r = client.get("/metrics")
        check(r.status_code == 200, f"/metrics -> {r.status_code}")
        body = r.text
        check("bartascope_predictions_total" in body,
              "predictions counter present")
        check("bartascope_inference_seconds_count" in body,
              "latency histogram present")
        # The one thing that must never appear.
        leaked = [s for s in SAMPLES if s[:20] in body]
        check(not leaked, "no message text present in /metrics")

    print(f"\n{'=' * 60}")
    if failures:
        print(f"{len(failures)} FAILED")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
