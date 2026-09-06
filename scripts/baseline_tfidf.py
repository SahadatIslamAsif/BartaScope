"""
Step 1 baseline: character n-gram TF-IDF + logistic regression.

This is the floor. Every transformer you train afterwards has to justify
its cost against these numbers - both the macro-F1 and the latency.

Usage:
    python baseline_tfidf.py --data merged.csv
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

LABELS = ["normal", "transactional", "promotional", "phishing"]


def header(text: str) -> None:
    print(f"\n{'=' * 62}\n{text}\n{'=' * 62}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="merged.csv")
    ap.add_argument("--outdir", default="baseline_out")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    train = df[df.split == "train"]
    val = df[df.split == "val"]
    test = df[df.split == "test"]
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")

    # char_wb keeps n-grams inside word boundaries. Works for Bangla script,
    # Latin script and the mix, without needing a tokeniser for either.
    model = make_pipeline(
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=2,
            max_features=200_000,
            sublinear_tf=True,
        ),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",  # transactional has 186 rows vs normal's 5083
            C=4.0,
            n_jobs=-1,
        ),
    )

    t0 = time.perf_counter()
    model.fit(train.text, train.label)
    print(f"trained in {time.perf_counter() - t0:.1f}s")

    val_f1 = f1_score(val.label, model.predict(val.text), average="macro")
    pred = model.predict(test.text)
    test_f1 = f1_score(test.label, pred, average="macro")

    header("HEADLINE")
    print(f"  val  macro-F1 : {val_f1:.4f}")
    print(f"  test macro-F1 : {test_f1:.4f}")
    print(f"  test accuracy : {(pred == test.label).mean():.4f}   <- ignore this, "
          f"it is inflated by class imbalance")

    header("PER-CLASS (test)")
    print(classification_report(test.label, pred, labels=LABELS, digits=3, zero_division=0))

    header("CONFUSION MATRIX (test)")
    cm = confusion_matrix(test.label, pred, labels=LABELS)
    print(f"{'true \\ pred':<16}" + "".join(f"{l[:12]:>14}" for l in LABELS))
    for name, row in zip(LABELS, cm):
        print(f"{name:<16}" + "".join(f"{v:>14}" for v in row))

    header("PER-LANGUAGE macro-F1 (test)")
    for lang in ["bangla", "banglish", "codemix", "english"]:
        m = test.lang == lang
        if m.sum() == 0:
            continue
        print(f"  {lang:<10} n={m.sum():>5}   macro-F1 {f1_score(test.label[m], pred[m], average='macro', zero_division=0):.3f}")

    header("PER-SOURCE macro-F1 (test)")
    print("  a large gap here means the model learned source style, not message type")
    for src in sorted(test.source.unique()):
        m = test.source == src
        print(f"  {src:<20} n={m.sum():>5}   macro-F1 "
              f"{f1_score(test.label[m], pred[m], average='macro', zero_division=0):.3f}")

    header("LATENCY (single message, CPU)")
    sample = test.text.iloc[:200].tolist()
    for _ in range(20):  # warm up
        model.predict([sample[0]])
    times = []
    for s in sample:
        t = time.perf_counter()
        model.predict([s])
        times.append((time.perf_counter() - t) * 1000)
    times = np.array(times)
    print(f"  p50 {np.percentile(times, 50):.3f} ms")
    print(f"  p95 {np.percentile(times, 95):.3f} ms")
    print(f"  p99 {np.percentile(times, 99):.3f} ms")
    print("  this is the number XLM-R has to justify itself against")

    # Save the errors - this is what you actually learn from
    err = test[pred != test.label].copy()
    err["predicted"] = pred[pred != test.label]
    err[["text", "label", "predicted", "lang", "source"]].to_csv(
        outdir / "errors.csv", index=False)

    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(outdir / "confusion_matrix.csv")

    # Persist the fitted pipeline so benchmark.py can time it under the same
    # protocol as the transformers. The latency printed above uses this
    # script's own 200-iteration loop; the CSV row that the model-selection
    # table quotes comes from the shared harness instead.
    joblib.dump(model, outdir / "model.joblib")

    print(f"\n  {len(err)} misclassified rows -> {outdir}/errors.csv")
    print(f"  confusion matrix         -> {outdir}/confusion_matrix.csv")
    print(f"  fitted pipeline          -> {outdir}/model.joblib\n")


if __name__ == "__main__":
    main()
