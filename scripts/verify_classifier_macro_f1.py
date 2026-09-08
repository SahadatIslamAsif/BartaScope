"""Confirms app.model.Classifier (tokenizers-based, post transformers removal)
reproduces the recorded test macro-F1, on the full test split rather than the
200-message sample verify_tokenizer_parity.py checked.

scripts/evaluate_onnx.py still uses transformers.AutoTokenizer directly (it is
a workstation script, not shipped in requirements-runtime.txt, so that is
fine) -- re-running it would only reconfirm the old path, not exercise the
code that changed. This script drives the actual production Classifier
instead, batch size 64 to match /v1/classify/batch, and compares against the
0.9695 recorded in CLAUDE.md and onnx_out/model.eval.json.

Usage: .venv/Scripts/python.exe scripts/verify_classifier_macro_f1.py
"""

import sys

import pandas as pd
from sklearn.metrics import f1_score

from app.model import Classifier

LABELS = ["normal", "transactional", "promotional", "phishing"]
BATCH_SIZE = 64
RECORDED_MACRO_F1 = 0.9695


def main() -> int:
    clf = Classifier("onnx_out/model.onnx", "onnx_out", max_length=64,
                     threads=0)
    clf.load()

    df = pd.read_csv("data/merged.csv")
    test = df[df.split == "test"].reset_index(drop=True)
    texts = test.text.astype(str).tolist()
    y = test.label.to_numpy()

    preds = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i:i + BATCH_SIZE]
        results, _ = clf.predict(chunk)
        preds.extend(r["label"] for r in results)

    macro_f1 = f1_score(y, preds, average="macro", labels=LABELS,
                        zero_division=0)
    print(f"test rows            : {len(test)}")
    print(f"classifier macro-F1  : {macro_f1:.4f}")
    print(f"recorded macro-F1    : {RECORDED_MACRO_F1:.4f}")
    diff = abs(macro_f1 - RECORDED_MACRO_F1)
    print(f"diff                 : {diff:.4f}")

    ok = diff < 1e-4
    print("RESULT:", "PASS -- unchanged" if ok else "FAIL -- macro-F1 moved")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
