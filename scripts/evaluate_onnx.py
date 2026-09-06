"""
Full four-class evaluation of an ONNX graph on the test split.

Runs against whatever .onnx file you point it at, so fp32 and INT8 go
through identical code, identical batching and identical decoding. The
only thing that differs between two runs is the graph.

Reports macro-F1 and never accuracy: train is 27:1 imbalanced, so accuracy
mostly measures how often the model says "normal". Per-language and
per-source breakdowns are printed every time, because the source datasets
are trivially separable and an aggregate number hides that.

Usage:
    python evaluate_onnx.py --model onnx_out/model.onnx
    python evaluate_onnx.py --model onnx_out/model.int8.onnx
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

LABELS = ["normal", "transactional", "promotional", "phishing"]


def header(text: str) -> None:
    print(f"\n{'=' * 68}\n{text}\n{'=' * 68}")


def subset_macro_f1(y_true, y_pred):
    """Macro-F1 over the classes actually present in this subset's truth.

    Averaging over all four labels on a subset that contains only three is
    not comparable between models: a single prediction of the absent class
    adds an F1 of 0.0 to the average and divides by four instead of three.
    mendeley has no transactional rows, so one stray prediction there costs
    about 25 points of apparent score. That is a property of the metric, not
    of the model, so the stray predictions are counted and shown separately
    instead of being buried in the average.
    """
    present = sorted(set(y_true))
    score = f1_score(y_true, y_pred, average="macro", labels=present,
                     zero_division=0)
    stray = int(sum(p not in present for p in y_pred))
    return float(score), present, stray


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="onnx_out/model.onnx")
    ap.add_argument("--tokenizer-dir", default="onnx_out")
    ap.add_argument("--data", default="data/merged.csv")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="",
                    help="write metrics JSON here (default: alongside the "
                         "model, as <model stem>.eval.json)")
    args = ap.parse_args()

    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_path = Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(str(model_path), so,
                                providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]

    df = pd.read_csv(args.data)
    test = df[df.split == "test"].reset_index(drop=True)

    header("MODEL")
    print(f"  graph        : {model_path}")
    print(f"  size         : "
          f"{sum(p.stat().st_size for p in model_path.parent.glob(model_path.name + '*')) / 1e6:.1f} MB")
    print(f"  inputs       : {input_names}")
    print(f"  test rows    : {len(test)}")

    texts = test.text.astype(str).tolist()
    preds = []
    for i in range(0, len(texts), args.batch_size):
        chunk = texts[i:i + args.batch_size]
        enc = tokenizer(chunk, return_tensors="np", padding="longest",
                        truncation=True, max_length=args.max_length)
        logits = sess.run(["logits"],
                          {n: enc[n] for n in input_names})[0]
        preds.extend(np.argmax(logits, axis=-1).tolist())

    pred = np.array([LABELS[i] for i in preds])
    y = test.label.to_numpy()
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)

    header("HEADLINE")
    print(f"  test macro-F1 : {macro_f1:.4f}")
    print(f"  test accuracy : {(pred == y).mean():.4f}   <- ignore this, "
          f"it is inflated by class imbalance")

    header("PER-CLASS (test)")
    print(classification_report(y, pred, labels=LABELS, digits=3,
                                zero_division=0))

    header("CONFUSION MATRIX (test)")
    cm = confusion_matrix(y, pred, labels=LABELS)
    print(f"{'true \\ pred':<16}" + "".join(f"{l[:12]:>14}" for l in LABELS))
    for name, row in zip(LABELS, cm):
        print(f"{name:<16}" + "".join(f"{v:>14}" for v in row))

    per_class = {
        label: round(float(f1_score(y == label, pred == label,
                                    zero_division=0)), 4)
        for label in LABELS
    }

    header("PER-LANGUAGE macro-F1 (test)")
    print("  averaged over the classes present in each subset; 'stray' counts "
          "predictions of a class\n  that subset's ground truth never contains")
    per_lang = {}
    for lang in ["bangla", "banglish", "codemix", "english"]:
        m = (test.lang == lang).to_numpy()
        if m.sum() == 0:
            continue
        score, present, stray = subset_macro_f1(y[m], pred[m])
        per_lang[lang] = {"macro_f1": round(score, 4), "n": int(m.sum()),
                          "classes": len(present), "stray_preds": stray}
        print(f"  {lang:<10} n={m.sum():>5}   macro-F1 {score:.3f}   "
              f"over {len(present)} classes"
              + (f"   stray {stray}" if stray else ""))

    header("PER-SOURCE macro-F1 (test)")
    print("  a large gap here means the model learned source style, "
          "not message type")
    per_source = {}
    for src in sorted(test.source.unique()):
        m = (test.source == src).to_numpy()
        score, present, stray = subset_macro_f1(y[m], pred[m])
        per_source[src] = {"macro_f1": round(score, 4), "n": int(m.sum()),
                           "classes": len(present), "stray_preds": stray}
        print(f"  {src:<20} n={m.sum():>5}   macro-F1 {score:.3f}   "
              f"over {len(present)} classes"
              + (f"   stray {stray}" if stray else ""))

    out_path = Path(args.out) if args.out else \
        model_path.parent / f"{model_path.stem}.eval.json"
    out_path.write_text(json.dumps({
        "model": str(model_path),
        "test_macro_f1": round(float(macro_f1), 4),
        "test_n": int(len(test)),
        "per_class_f1": per_class,
        "per_language_macro_f1": per_lang,
        "per_source_macro_f1": per_source,
        "max_length": args.max_length,
    }, indent=2))
    print(f"\n  metrics -> {out_path}\n")


if __name__ == "__main__":
    main()
