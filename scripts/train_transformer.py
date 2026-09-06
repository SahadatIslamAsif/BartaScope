"""
Step 2: fine-tune a transformer on the four-class SMS task.

Prints the same tables as baseline_tfidf.py so the two are directly
comparable. The point is not to beat the baseline - it is to find out
whether a 278M-parameter model earns its cost over a 0.5 ms logistic
regression, and to produce a checkpoint to export and quantise later.

Usage:
    python train_transformer.py --data merged.csv
    python train_transformer.py --data merged.csv --model microsoft/Multilingual-MiniLM-L12-H384
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LABELS = ["normal", "transactional", "promotional", "phishing"]
L2I = {l: i for i, l in enumerate(LABELS)}


def header(text: str) -> None:
    print(f"\n{'=' * 62}\n{text}\n{'=' * 62}")


class SMSDataset(Dataset):
    """Tokenises lazily and pads per batch, not to a fixed width."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = [L2I[l] for l in labels]
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_length)
        enc["labels"] = self.labels[i]
        return enc


class WeightedTrainer(Trainer):
    """Cross-entropy weighted by inverse class frequency.

    Without this the model can score 0.93 accuracy by never predicting
    transactional, which has 186 training rows against normal's 5083.
    """

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))(
            outputs.logits.view(-1, len(LABELS)), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}


def measure_latency(model, tokenizer, texts, max_length, device):
    """Single-message latency on CPU - the number to compare against the
    baseline, and the number ONNX has to improve."""
    model = model.to("cpu").eval()
    for t in texts[:20]:  # warm up
        with torch.no_grad():
            model(**tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length))
    times = []
    for t in texts[:200]:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**enc)
        times.append((time.perf_counter() - t0) * 1000)
    model.to(device)
    return np.array(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="merged.csv")
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer name if it differs from the model. "
                         "Multilingual MiniLM needs xlm-roberta-base: its "
                         "config declares no tokenizer_class, so AutoTokenizer "
                         "fails on it.")
    ap.add_argument("--outdir", default="transformer_out")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | model: {args.model}")

    df = pd.read_csv(args.data)
    train = df[df.split == "train"]
    val = df[df.split == "val"]
    test = df[df.split == "test"]
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")

    tok_name = args.tokenizer or args.model
    if tok_name != args.model:
        print(f"tokenizer: {tok_name}")
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label={i: l for l, i in L2I.items()},
        label2id=L2I,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.1f}M")

    counts = train.label.value_counts()
    weights = torch.tensor(
        [len(train) / (len(LABELS) * counts[l]) for l in LABELS], dtype=torch.float
    )
    print("class weights:", {l: round(float(w), 2) for l, w in zip(LABELS, weights)})

    targs = TrainingArguments(
        output_dir=str(outdir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        seed=args.seed,
        fp16=(device == "cuda"),
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=targs,
        train_dataset=SMSDataset(train.text, train.label, tokenizer, args.max_length),
        eval_dataset=SMSDataset(val.text, val.label, tokenizer, args.max_length),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    t0 = time.perf_counter()
    trainer.train()
    print(f"\ntrained in {(time.perf_counter() - t0) / 60:.1f} min")

    val_f1 = trainer.evaluate()["eval_macro_f1"]
    test_ds = SMSDataset(test.text, test.label, tokenizer, args.max_length)
    logits = trainer.predict(test_ds).predictions
    pred = np.array([LABELS[i] for i in np.argmax(logits, axis=-1)])
    y = test.label.to_numpy()
    test_f1 = f1_score(y, pred, average="macro", zero_division=0)

    header("HEADLINE")
    print(f"  model         : {args.model} ({n_params / 1e6:.0f}M params)")
    print(f"  val  macro-F1 : {val_f1:.4f}")
    print(f"  test macro-F1 : {test_f1:.4f}")
    print(f"  test accuracy : {(pred == y).mean():.4f}   <- ignore this")
    print(f"  TF-IDF baseline was 0.9787 at p95 0.794 ms")

    header("PER-CLASS (test)")
    print(classification_report(y, pred, labels=LABELS, digits=3, zero_division=0))

    header("CONFUSION MATRIX (test)")
    cm = confusion_matrix(y, pred, labels=LABELS)
    print(f"{'true \\ pred':<16}" + "".join(f"{l[:12]:>14}" for l in LABELS))
    for name, row in zip(LABELS, cm):
        print(f"{name:<16}" + "".join(f"{v:>14}" for v in row))

    header("PER-LANGUAGE macro-F1 (test)")
    for lang in ["bangla", "banglish", "codemix", "english"]:
        m = (test.lang == lang).to_numpy()
        if m.sum():
            present = sorted(set(y[m]))
            print(f"  {lang:<10} n={m.sum():>5}   acc {(pred[m] == y[m]).mean():.3f}"
                  f"   macro-F1 {f1_score(y[m], pred[m], labels=present, average='macro', zero_division=0):.3f}")

    header("PER-SOURCE macro-F1 (test)")
    print("  averaged only over classes actually present in each subset")
    print("  a large gap here means the model learned source style, not message type")
    for src in sorted(test.source.unique()):
        m = (test.source == src).to_numpy()
        present = sorted(set(y[m]))
        print(f"  {src:<20} n={m.sum():>5}   acc {(pred[m] == y[m]).mean():.3f}"
              f"   macro-F1 {f1_score(y[m], pred[m], labels=present, average='macro', zero_division=0):.3f}")

    header("LATENCY (single message, CPU, PyTorch fp32)")
    times = measure_latency(model, tokenizer, test.text.tolist(), args.max_length, device)
    print(f"  p50 {np.percentile(times, 50):.2f} ms")
    print(f"  p95 {np.percentile(times, 95):.2f} ms")
    print(f"  p99 {np.percentile(times, 99):.2f} ms")
    print("  this is your ONNX baseline - record it")

    final = outdir / "model"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    err = test[pred != y].copy()
    err["predicted"] = pred[pred != y]
    err[["text", "label", "predicted", "lang", "source"]].to_csv(
        outdir / "errors.csv", index=False)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(outdir / "confusion_matrix.csv")
    json.dump(
        {"model": args.model, "params_m": round(n_params / 1e6, 1),
         "val_macro_f1": round(float(val_f1), 4),
         "test_macro_f1": round(float(test_f1), 4),
         "latency_p50_ms": round(float(np.percentile(times, 50)), 2),
         "latency_p95_ms": round(float(np.percentile(times, 95)), 2),
         "max_length": args.max_length, "epochs": args.epochs,
         "lr": args.lr},
        open(outdir / "results.json", "w"), indent=2)

    print(f"\n  model      -> {final}")
    print(f"  {len(err)} errors -> {outdir}/errors.csv")
    print(f"  metrics    -> {outdir}/results.json\n")


if __name__ == "__main__":
    main()
