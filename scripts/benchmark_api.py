"""
End-to-end API latency, measured over HTTP against a running service.

This is a different number from every other row in benchmarks/latency.csv and
must not be compared with them casually. The `onnx-fp32` rows time the ORT
`session.run` call alone, on pre-tokenised tensors. This times what a caller
actually experiences:

    HTTP request -> uvicorn -> pydantic validation -> normalise() ->
    tokenise -> ORT run -> softmax -> pydantic serialisation -> HTTP response

so it includes the network round trip, the ASGI stack, JSON encoding and the
preprocessing that the model-only benchmark deliberately hoisted out of the
timing loop. It will always be larger, and the gap is the service overhead --
which is the point of measuring it.

It lands under backend `api-http-e2e` so it cannot be mistaken for the
model-only figure.

Usage:
    python scripts/benchmark_api.py --base-url http://127.0.0.1:8080
"""

import argparse
import csv
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import CSV_FIELDS, chunk_for, summarise  # noqa: E402


def header(text: str) -> None:
    print(f"\n{'=' * 68}\n{text}\n{'=' * 68}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--data", default="data/merged.csv")
    ap.add_argument("--out", default="benchmarks/latency.csv")
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-measured", type=int, default=500)
    ap.add_argument("--batch-sizes", default="1",
                    help="1 uses /v1/classify; larger uses /v1/classify/batch")
    ap.add_argument("--threads", type=int, default=0,
                    help="record the server's thread count; 0 asks /readyz")
    ap.add_argument("--notes", default="end-to-end over HTTP")
    args = ap.parse_args()

    import httpx

    client = httpx.Client(base_url=args.base_url, timeout=60.0)

    ready = client.get("/readyz")
    if ready.status_code != 200:
        raise SystemExit(f"server not ready: {ready.status_code} {ready.text}")
    info = ready.json()
    threads = args.threads or int(info.get("threads") or 0)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    machine = f"{platform.processor() or platform.machine()} / {platform.system()}"

    header("CONFIGURATION")
    print(f"  base_url       : {args.base_url}")
    print(f"  run_id         : {run_id}")
    print(f"  server threads : {threads}")
    print(f"  model_version  : {info.get('model_version')}")
    print(f"  load_seconds   : {info.get('load_seconds')}")
    print(f"  warmup/measured: {args.n_warmup}/{args.n_measured} requests")

    df = pd.read_csv(args.data)
    texts = df[df.split == "test"].text.astype(str).tolist()
    # Same seeded permutation as benchmark.py, so the HTTP row and the
    # model-only rows are measured on identical messages in identical order.
    order = np.random.default_rng(0).permutation(len(texts))
    texts = [texts[i] for i in order]
    print(f"  test messages  : {len(texts)} (seeded permutation, seed 0)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    fh = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()

    server_ms_all = []

    for batch_size in [int(b) for b in args.batch_sizes.split(",")]:
        if batch_size == 1:
            path = "/v1/classify"
            def call(chunk):
                return client.post(path, json={"text": chunk[0]})
        else:
            path = "/v1/classify/batch"
            def call(chunk):
                return client.post(path, json={"texts": chunk})

        header(f"batch={batch_size}  endpoint={path}")

        for i in range(args.n_warmup):
            call(chunk_for(texts, i, batch_size))

        times, server_ms, failures = [], [], 0
        for i in range(args.n_measured):
            chunk = chunk_for(texts, i, batch_size)
            t0 = time.perf_counter()
            response = call(chunk)
            elapsed = (time.perf_counter() - t0) * 1000.0
            if response.status_code != 200:
                failures += 1
                continue
            times.append(elapsed)
            server_ms.append(response.json()["inference_ms"])

        if failures:
            print(f"  WARNING: {failures} non-200 responses, excluded")
        if not times:
            raise SystemExit("every request failed")

        stats = summarise(times, batch_size)
        server = np.array(server_ms)
        server_ms_all.append(server)

        overhead_p50 = np.percentile(times, 50) - np.percentile(server, 50)
        print(f"  end-to-end   p50 {stats['p50_ms']:>8.2f}  "
              f"p95 {stats['p95_ms']:>8.2f}  p99 {stats['p99_ms']:>8.2f} ms")
        print(f"  server-side  p50 {np.percentile(server, 50):>8.2f}  "
              f"p95 {np.percentile(server, 95):>8.2f}  "
              f"p99 {np.percentile(server, 99):>8.2f} ms   "
              f"(inference_ms as reported by the service)")
        print(f"  overhead     p50 {overhead_p50:>8.2f} ms  "
              f"(HTTP + ASGI + validation + JSON)")

        writer.writerow({
            "run_id": run_id, "backend": "api-http-e2e",
            "batch_size": batch_size, "padding": "dynamic-to-batch-max",
            "threads": threads, "n_warmup": args.n_warmup,
            "n_measured": len(times), "n_messages": batch_size * len(times),
            "mean_seq_len": "", "max_seq_len": "", "machine": machine,
            "timestamp": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "notes": f"{args.notes}; server-side p50 "
                     f"{np.percentile(server, 50):.2f} ms",
            **stats,
        })
        fh.flush()

    fh.close()
    print(f"\n  appended -> {out_path}")
    print("  backend is 'api-http-e2e'; do NOT compare it directly with the "
          "onnx-fp32\n  rows, which time session.run alone on pre-tokenised "
          "tensors.\n")


if __name__ == "__main__":
    main()
