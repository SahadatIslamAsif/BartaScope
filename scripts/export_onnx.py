"""
Step 3: export the fine-tuned model to ONNX and verify it.

No quantisation here. The only goal is an ONNX graph that produces the
same logits as PyTorch. If parity fails, every latency number measured
afterwards describes a model that gives different answers than the one
you evaluated.

Usage:
    python export_onnx.py --model-dir minilm_out/model --data merged.csv
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["normal", "transactional", "promotional", "phishing"]
PARITY_TOLERANCE = 1e-4


def header(text: str) -> None:
    print(f"\n{'=' * 62}\n{text}\n{'=' * 62}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="minilm_out/model")
    ap.add_argument("--data", default="merged.csv")
    ap.add_argument("--outdir", default="onnx_out")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--n-parity", type=int, default=200)
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    onnx_path = outdir / "model.onnx"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).eval()

    # Which inputs does this architecture actually take? MiniLM is a BERT
    # underneath and wants token_type_ids; XLM-R does not. Guessing wrong
    # produces a graph that runs and returns plausible nonsense.
    sample = tokenizer("test message", return_tensors="pt")
    input_names = [n for n in ("input_ids", "attention_mask", "token_type_ids")
                   if n in sample]
    print(f"model inputs: {input_names}")

    header("EXPORT")
    # Batch of 2 and length 16 so neither dimension can be mistaken for a
    # constant during tracing.
    dummy = tokenizer(["first sample message", "second one"],
                      return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_length)
    dummy_inputs = tuple(dummy[n] for n in input_names)

    dynamic_axes = {n: {0: "batch", 1: "sequence"} for n in input_names}
    dynamic_axes["logits"] = {0: "batch"}

    torch.onnx.export(
        model,
        dummy_inputs,
        str(onnx_path),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    # The dynamo exporter writes weights to model.onnx.data alongside the
    # graph. The graph file alone is ~1.5 MB; reporting that as the model
    # size would be a lie. Count every artifact the session needs.
    size_mb = sum(p.stat().st_size for p in outdir.glob("model.onnx*")) / 1e6
    print(f"  wrote {onnx_path} ({size_mb:.1f} MB)")

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    print("  onnx.checker: passed")

    # Confirm the axes really are dynamic. A frozen sequence dimension is
    # the single most common silent export bug, and it would cost you the
    # padding win that is most of the speedup.
    for inp in graph.graph.input:
        dims = [d.dim_param or d.dim_value
                for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name:<18} shape {dims}")
        if not all(isinstance(d, str) for d in dims[:2]):
            print("    WARNING: this axis is frozen, not dynamic")

    header("PARITY CHECK")
    df = pd.read_csv(args.data)
    texts = df[df.split == "test"].text.astype(str).tolist()[:args.n_parity]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_diff = 0.0
    disagreements = 0

    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=args.max_length)
        with torch.no_grad():
            torch_logits = model(**{n: enc[n] for n in input_names}).logits.numpy()
        onnx_logits = sess.run(
            ["logits"], {n: enc[n].numpy() for n in input_names})[0]

        max_diff = max(max_diff, float(np.abs(torch_logits - onnx_logits).max()))
        if np.argmax(torch_logits) != np.argmax(onnx_logits):
            disagreements += 1

    print(f"  samples compared        : {len(texts)}")
    print(f"  max abs logit difference: {max_diff:.2e}")
    print(f"  predicted-label mismatch: {disagreements}")

    passed = max_diff < PARITY_TOLERANCE and disagreements == 0
    print(f"\n  {'PASS' if passed else 'FAIL'} "
          f"(tolerance {PARITY_TOLERANCE:.0e})")
    if not passed:
        print("  do not proceed to quantisation until this passes")

    header("VARIABLE-LENGTH CHECK")
    # If the sequence axis were frozen, one of these would throw.
    for n_tokens in (8, 32, 64):
        enc = tokenizer("word " * n_tokens, return_tensors="pt",
                        truncation=True, max_length=args.max_length)
        out = sess.run(["logits"], {n: enc[n].numpy() for n in input_names})[0]
        print(f"  {enc['input_ids'].shape[1]:>3} tokens -> logits {out.shape}  ok")

    header("LATENCY (ONNX fp32, CPU, single message)")
    for _ in range(20):
        enc = tokenizer(texts[0], return_tensors="pt", truncation=True,
                        max_length=args.max_length)
        sess.run(["logits"], {n: enc[n].numpy() for n in input_names})

    times = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=args.max_length)
        feed = {n: enc[n].numpy() for n in input_names}
        t0 = time.perf_counter()
        sess.run(["logits"], feed)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    for p in (50, 95, 99):
        print(f"  p{p} {np.percentile(times, p):.2f} ms")

    tokenizer.save_pretrained(str(outdir))
    json.dump(
        {"source_model": args.model_dir,
         "input_names": input_names,
         "labels": LABELS,
         "max_length": args.max_length,
         "opset": args.opset,
         "size_mb": round(size_mb, 1),
         "parity_max_diff": max_diff,
         "parity_passed": passed,
         "onnx_fp32_p50_ms": round(float(np.percentile(times, 50)), 2),
         "onnx_fp32_p95_ms": round(float(np.percentile(times, 95)), 2)},
        open(outdir / "export_info.json", "w"), indent=2)

    print(f"\n  model + tokenizer -> {outdir}/")
    print(f"  export info       -> {outdir}/export_info.json\n")


if __name__ == "__main__":
    main()
