"""
Step 5: dynamic INT8 quantisation of the exported ONNX graph.

Dynamic rather than static: weights are quantised ahead of time, activation
ranges are computed per-inference. No calibration set, no risk of the
calibration distribution not matching production traffic. The cost is that
activation quantisation happens at runtime, so the speedup is smaller than
static INT8 would give.

Note what dominates this model's size. Vocab is 250,037 x 384 = 96M of the
118M parameters, so the embedding table is most of the file. Quantising the
Gather that reads it is where the size win lives -- and also where accuracy
loss lives, if it appears. --op-types lets you exclude it and re-measure
rather than guess.

Nothing here is trustworthy until evaluate_onnx.py has been run on the
result. Quantisation is the one step in this pipeline that can silently
change predictions.

Usage:
    python quantize_onnx.py --model onnx_out/model.onnx \
        --out onnx_out/model.int8.onnx
"""

import argparse
import json
from pathlib import Path


def header(text: str) -> None:
    print(f"\n{'=' * 68}\n{text}\n{'=' * 68}")


def artifact_mb(path: Path) -> float:
    """Graph plus any external-data sidecar it depends on."""
    return sum(p.stat().st_size
               for p in path.parent.glob(path.name + "*")) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="onnx_out/model.onnx")
    ap.add_argument("--out", default="onnx_out/model.int8.onnx")
    ap.add_argument("--op-types", default="",
                    help="comma-separated op types to quantise; empty means "
                         "onnxruntime's default set (MatMul and Gather among "
                         "them). Try 'MatMul' to spare the embedding table.")
    ap.add_argument("--no-preprocess", action="store_true",
                    help="skip symbolic shape inference. ORT warns loudly if "
                         "you quantise without it; the graph usually still "
                         "works, but transformer shapes are exactly where it "
                         "matters.")
    ap.add_argument("--strip-value-info", action="store_true",
                    help="clear intermediate shape annotations and let onnx "
                         "re-infer them. The dynamo exporter emits value_info "
                         "that onnx's own inferencer contradicts, and both "
                         "quant_pre_process and quantize_dynamic refuse to "
                         "run on it. Inputs and outputs keep their shapes; "
                         "only intermediates are recomputed.")
    args = ap.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    src = Path(args.model)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)

    header("SOURCE")
    print(f"  {src}  ({artifact_mb(src):.1f} MB)")

    model_in = src
    if args.strip_value_info:
        header("NORMALISE (drop stale intermediate shape annotations)")
        import onnx
        graph = onnx.load(str(src), load_external_data=True)
        n_dropped = len(graph.graph.value_info)
        del graph.graph.value_info[:]
        graph = onnx.shape_inference.infer_shapes(graph, strict_mode=False)
        normalised = dst.parent / "model.normalized.onnx"
        # Under 2 GB once the weights are inlined, so a single file is fine
        # and keeps the quantiser from chasing an external-data sidecar.
        onnx.save_model(graph, str(normalised))
        print(f"  dropped {n_dropped} value_info entries, re-inferred")
        print(f"  wrote {normalised}  ({artifact_mb(normalised):.1f} MB)")
        model_in = normalised

    if not args.no_preprocess:
        header("PREPROCESS (symbolic shape inference)")
        prepped = dst.parent / "model.preprocessed.onnx"
        quant_pre_process(str(src), str(prepped), skip_symbolic_shape=False)
        print(f"  wrote {prepped}  ({artifact_mb(prepped):.1f} MB)")
        model_in = prepped

    header("QUANTISE (dynamic, INT8 weights)")
    op_types = [t.strip() for t in args.op_types.split(",") if t.strip()]
    kwargs = {"weight_type": QuantType.QInt8}
    if op_types:
        kwargs["op_types_to_quantize"] = op_types
        print(f"  op_types_to_quantize: {op_types}")
    else:
        print("  op_types_to_quantize: onnxruntime default")

    quantize_dynamic(str(model_in), str(dst), **kwargs)

    src_mb = artifact_mb(src)
    dst_mb = artifact_mb(dst)
    print(f"  wrote {dst}  ({dst_mb:.1f} MB)")

    header("RESULT")
    print(f"  fp32  {src_mb:>8.1f} MB")
    print(f"  int8  {dst_mb:>8.1f} MB   {src_mb / dst_mb:.2f}x smaller")
    print("\n  NOT VALIDATED YET. Run:")
    print(f"    python scripts/evaluate_onnx.py --model {dst}")
    print("  and compare macro-F1 against the fp32 graph before believing "
          "any latency number this file produces.\n")

    info = {
        "source": str(src),
        "output": str(dst),
        "op_types_to_quantize": op_types or "onnxruntime default",
        "weight_type": "QInt8",
        "preprocessed": not args.no_preprocess,
        "fp32_mb": round(src_mb, 1),
        "int8_mb": round(dst_mb, 1),
        "size_ratio": round(src_mb / dst_mb, 2),
    }
    out_json = dst.parent / "quantize_info.json"
    out_json.write_text(json.dumps(info, indent=2))
    print(f"  quantisation info -> {out_json}\n")


if __name__ == "__main__":
    main()
