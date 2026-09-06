"""
Step 4: latency benchmark harness.

Every configuration runs in the same process, on the same machine, over
the *same* pre-tokenised input sequence. Tokenisation happens once, up
front, and is deliberately outside the timing loop: the question is how
fast the model runs, and letting tokeniser variance leak into the numbers
would make the PyTorch/ONNX comparison meaningless.

Sweeps backend x batch size x padding strategy. The padding axis matters
more than it looks: max_length=64 with a p95 of 32 tokens means fixed-64
padding wastes roughly half of every batch on <pad>, and most of the
available speedup is in not doing that.

Results are appended to a CSV so INT8 can be measured later and land in
the same table as fp32 without re-running anything.

Usage:
    python benchmark.py --data data/merged.csv --model-dir models/minilm \
        --onnx-dir onnx_out --out benchmarks/latency.csv
"""

import argparse
import csv
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CSV_FIELDS = [
    "run_id", "backend", "batch_size", "padding", "threads",
    "n_warmup", "n_measured", "n_messages",
    "p50_ms", "p95_ms", "p99_ms", "mean_ms", "min_ms",
    "per_msg_p50_ms", "per_msg_p95_ms", "throughput_msg_s",
    "mean_seq_len", "max_seq_len", "machine", "timestamp", "notes",
]

PADDINGS = ["fixed-64", "dynamic-to-batch-max"]


def header(text: str) -> None:
    print(f"\n{'=' * 68}\n{text}\n{'=' * 68}")


def chunk_for(texts, i, batch_size):
    """The i-th batch of messages, wrapping around the pool.

    One definition, used by every backend, so a TF-IDF row and an ONNX row
    are timed on exactly the same messages in the same order.
    """
    pool = len(texts)
    start = (i * batch_size) % pool
    return [texts[(start + j) % pool] for j in range(batch_size)]


def build_text_batches(texts, batch_size, n_batches):
    """Raw text batches, for backends that do their own vectorising."""
    return [chunk_for(texts, i, batch_size) for i in range(n_batches)]


def build_batches(tokenizer, texts, batch_size, padding, max_length,
                  n_batches):
    """Pre-tokenise every batch once.

    Identical text slices for both padding strategies, so the only thing
    that differs between a fixed-64 row and a dynamic row is the shape of
    the tensor, never its content.
    """
    batches = []
    for i in range(n_batches):
        chunk = chunk_for(texts, i, batch_size)
        pad = "max_length" if padding == "fixed-64" else "longest"
        enc = tokenizer(chunk, return_tensors="np", padding=pad,
                        truncation=True, max_length=max_length)
        batches.append(dict(enc))
    return batches


def summarise(times_ms, batch_size):
    t = np.array(times_ms)
    return {
        "p50_ms": round(float(np.percentile(t, 50)), 3),
        "p95_ms": round(float(np.percentile(t, 95)), 3),
        "p99_ms": round(float(np.percentile(t, 99)), 3),
        "mean_ms": round(float(t.mean()), 3),
        "min_ms": round(float(t.min()), 3),
        "per_msg_p50_ms": round(float(np.percentile(t, 50)) / batch_size, 3),
        "per_msg_p95_ms": round(float(np.percentile(t, 95)) / batch_size, 3),
        "throughput_msg_s": round(batch_size * len(t) / (t.sum() / 1000.0), 1),
    }


def run_timed(run_one, batches, n_warmup, n_measured):
    for i in range(n_warmup):
        run_one(batches[i % len(batches)])
    times = []
    for i in range(n_measured):
        feed = batches[i % len(batches)]
        t0 = time.perf_counter()
        run_one(feed)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/merged.csv")
    ap.add_argument("--model-dir", default="models/minilm")
    ap.add_argument("--onnx-dir", default="onnx_out")
    ap.add_argument("--int8-path", default="onnx_out/model.int8.onnx")
    ap.add_argument("--tfidf-model", default="models/baseline/model.joblib")
    ap.add_argument("--out", default="benchmarks/latency.csv")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-measured", type=int, default=500)
    ap.add_argument("--n-batches", type=int, default=64,
                    help="distinct pre-tokenised batches, cycled over")
    ap.add_argument("--batch-sizes", default="1,8,32")
    ap.add_argument("--threads", type=int, default=0,
                    help="0 = torch's default; applied to ORT identically")
    ap.add_argument("--backends",
                    default="pytorch-fp32,onnx-fp32,onnx-int8,tfidf-logreg")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    import torch
    import onnxruntime as ort
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    threads = args.threads or torch.get_num_threads()
    torch.set_num_threads(threads)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    machine = f"{platform.processor() or platform.machine()} / {platform.system()}"

    header("CONFIGURATION")
    print(f"  run_id         : {run_id}")
    print(f"  machine        : {machine}")
    print(f"  threads        : {threads} (torch and ORT intra_op, identical)")
    print(f"  warmup/measured: {args.n_warmup}/{args.n_measured} iterations")
    print(f"  max_length     : {args.max_length}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    df = pd.read_csv(args.data)
    texts = df[df.split == "test"].text.astype(str).tolist()
    # Fixed permutation: every backend sees the identical message order,
    # and a run on another day sees it too.
    order = np.random.default_rng(0).permutation(len(texts))
    texts = [texts[i] for i in order]
    print(f"  test messages  : {len(texts)} (seeded permutation, seed 0)")

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    wanted = [b.strip() for b in args.backends.split(",")]

    # ---- backends -------------------------------------------------------
    runners = {}

    if "pytorch-fp32" in wanted:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir).eval()

        def pt_runner(feed):
            with torch.inference_mode():
                model(input_ids=torch.from_numpy(feed["input_ids"]),
                      attention_mask=torch.from_numpy(feed["attention_mask"]))
        runners["pytorch-fp32"] = pt_runner

    def make_ort_runner(path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(path), so,
                                    providers=["CPUExecutionProvider"])
        names = [i.name for i in sess.get_inputs()]

        def runner(feed):
            sess.run(["logits"], {n: feed[n] for n in names})
        return runner

    onnx_fp32 = Path(args.onnx_dir) / "model.onnx"
    if "onnx-fp32" in wanted:
        if onnx_fp32.exists():
            runners["onnx-fp32"] = make_ort_runner(onnx_fp32)
        else:
            print(f"  SKIP onnx-fp32 : {onnx_fp32} not found")

    int8_path = Path(args.int8_path)
    if "onnx-int8" in wanted:
        if int8_path.exists():
            runners["onnx-int8"] = make_ort_runner(int8_path)
        else:
            print(f"  SKIP onnx-int8 : {int8_path} not found (quantise first, "
                  f"then re-run with --backends onnx-int8 to append)")

    # The classical baseline. It has no tokeniser and no padding, so it is
    # timed once per batch size rather than once per padding strategy, and
    # its rows carry padding "n/a". Same messages, same order, same process
    # as the transformers -- that is the whole point of measuring it here
    # rather than trusting the number baseline_tfidf.py prints.
    tfidf_path = Path(args.tfidf_model)
    if "tfidf-logreg" in wanted:
        if tfidf_path.exists():
            import joblib
            pipe = joblib.load(tfidf_path)

            def tfidf_runner(chunk):
                pipe.predict(chunk)
            runners["tfidf-logreg"] = tfidf_runner
        else:
            print(f"  SKIP tfidf-logreg: {tfidf_path} not found (run "
                  f"baseline_tfidf.py --outdir models/baseline first)")

    if not runners:
        raise SystemExit("no backends available to benchmark")

    # ---- sweep ----------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    fh = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()

    def record(backend, batch_size, padding, times, seq_lens):
        stats = summarise(times, batch_size)
        row = {
            "run_id": run_id, "backend": backend,
            "batch_size": batch_size, "padding": padding,
            "threads": threads, "n_warmup": args.n_warmup,
            "n_measured": args.n_measured,
            "n_messages": batch_size * args.n_measured,
            "mean_seq_len": round(float(np.mean(seq_lens)), 1) if seq_lens else "",
            "max_seq_len": max(seq_lens) if seq_lens else "",
            "machine": machine,
            "timestamp": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "notes": args.notes, **stats,
        }
        writer.writerow(row)
        fh.flush()
        rows.append(row)
        print(f"  {backend:<14} p50 {stats['p50_ms']:>8.2f}  "
              f"p95 {stats['p95_ms']:>8.2f}  "
              f"p99 {stats['p99_ms']:>8.2f} ms   "
              f"{stats['throughput_msg_s']:>8.1f} msg/s")

    rows = []
    for batch_size in batch_sizes:
        if "tfidf-logreg" in runners:
            text_batches = build_text_batches(texts, batch_size,
                                              args.n_batches)
            header(f"batch={batch_size}  padding=n/a  (no tokeniser)")
            times = run_timed(runners["tfidf-logreg"], text_batches,
                              args.n_warmup, args.n_measured)
            record("tfidf-logreg", batch_size, "n/a", times, None)

        token_backends = [b for b in runners if b != "tfidf-logreg"]
        for padding in PADDINGS if token_backends else []:
            batches = build_batches(tokenizer, texts, batch_size, padding,
                                    args.max_length, args.n_batches)
            seq_lens = [b["input_ids"].shape[1] for b in batches]
            header(f"batch={batch_size}  padding={padding}  "
                   f"seq_len mean {np.mean(seq_lens):.1f} max {max(seq_lens)}")
            for backend, runner in runners.items():
                if backend == "tfidf-logreg":
                    continue  # timed above, padding does not apply
                times = run_timed(runner, batches, args.n_warmup,
                                  args.n_measured)
                record(backend, batch_size, padding, times, seq_lens)
    fh.close()

    # ---- headline -------------------------------------------------------
    header("SINGLE-MESSAGE COMPARISON (batch=1)")
    for r in [r for r in rows if r["batch_size"] == 1 and r["padding"] == "n/a"]:
        print(f"\n  padding = n/a\n    {r['backend']:<14} "
              f"p50 {r['p50_ms']:>7.2f}  p95 {r['p95_ms']:>7.2f} ms")
    for padding in PADDINGS:
        single = [r for r in rows if r["batch_size"] == 1
                  and r["padding"] == padding]
        base = next((r for r in single if r["backend"] == "pytorch-fp32"), None)
        print(f"\n  padding = {padding}")
        for r in single:
            if base and r["backend"] != "pytorch-fp32":
                speedup = f"   {base['p95_ms'] / r['p95_ms']:.2f}x vs PyTorch"
            else:
                speedup = "   (baseline)" if base else ""
            print(f"    {r['backend']:<14} p50 {r['p50_ms']:>7.2f}  "
                  f"p95 {r['p95_ms']:>7.2f} ms{speedup}")

    print(f"\n  appended {len(rows)} rows -> {out_path}\n")


if __name__ == "__main__":
    main()
