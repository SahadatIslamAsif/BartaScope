# Bartascope — SMS message type classifier as a production microservice

Bartascope is a paste-an-SMS web app that returns the message's type.
The **engineering** is the deliverable: ONNX optimisation, a FastAPI
service, Docker, cloud deploy, and load testing. Accuracy is already near
this dataset's ceiling.

Naming, for consistency across the repo:

| Where | Value |
|---|---|
| repo / directory | `bartascope` |
| display name | Bartascope |
| service | `bartascope-api` |
| Docker image | `bartascope` |
| `model_version` in responses | `bartascope-minilm-v1.0.0` |

## Goals

1. Serve a fine-tuned transformer via ONNX Runtime
2. Cut p95 inference latency by **at least half** vs PyTorch fp32
3. Production-ready FastAPI + Docker microservice, deployed to cloud
4. Load test it and chart before/after

## Status

Done: dataset merged, three models trained and compared, deployment
model chosen.
Next: ONNX export → quantisation → benchmark → FastAPI → Docker →
deploy → load test → frontend.

## Layout

```
bartascope/
data/merged.csv           12,808 rows. Single artifact, has a `split` column.
models/minilm/            fine-tuned checkpoint (deployed model)
models/results.json       metrics
models/errors.csv         49 misclassified test rows
scripts/prepare_dataset.py  rebuilds merged.csv from raw sources
scripts/baseline_tfidf.py   classical baseline (runs in <1 min, CPU)
scripts/train_transformer.py  fine-tuning (needs a GPU — use Kaggle)
scripts/export_onnx.py      ONNX export + parity check (written, NOT yet run)
```

## Dataset

Four classes. `spam` was deliberately collapsed into `promotional`,
because one source dataset had no spam category and labelled such
messages `promo`; keeping them separate made the boundary inconsistent
by construction.

| label | meaning |
|---|---|
| `normal` | personal / conversational |
| `transactional` | genuine automated service notification (bank, MFS, OTP) |
| `promotional` | advertising, incl. what the source data called spam |
| `phishing` | deceptive attempt to obtain credentials, OTPs or money |

Columns: `label`, `source`, `has_url`, `has_phone`, `text`, `lang`, `split`.
`lang` ∈ bangla / banglish / codemix / english. `source` = provenance.

Splits: train 8,965 / val 1,920 / test 1,923. **Filter on `split`, never
re-split** — the existing split is stratified on `(label, lang)` jointly
AND keeps template families whole (templated SMS are identical once
digits are stripped; splitting them leaks train into test).

Test-set class counts: normal 1089, transactional 39, promotional 325,
phishing 470. Train is 27:1 imbalanced, so **always report macro-F1, never
accuracy**, and use class weights in the loss.

Token length: median 12, p95 32, max 171. **`max_length=64`.** Never pad
to 128 or 512 — most of the available speedup is in not doing that.

### Known data problems — do not paper over these

- The Bangla source dataset is almost certainly **synthetic**: near-perfect
  uniform language balance, 1,095 phone mentions from only 266 unique
  numbers, template families differing only in digits.
- A logistic regression predicts which source a message came from with
  **0.962 accuracy** (majority baseline 0.531). Sources are trivially
  separable.
- Consequence: models score ~0.98 on synthetic Bangla and ~0.89–0.92 on
  real English SMS. Report per-source and per-language metrics always.
- `transactional` is English-only, and that is a **fact about the world**,
  not a gap: Bangla in SMS forces UCS-2 encoding (70 chars vs 160), so
  Bangladeshi banks and MFS providers send alerts in English. Document
  this; do not fill it with synthetic data. The user has ruled out
  synthetic data entirely.
- Three templates carry contradictory labels in the source data
  (annotation errors upstream). Left in deliberately.
- A real-world evaluation set collected from actual phones is still
  missing. Test scores overstate real performance.

## Model selection (measured, single-message CPU p95)

| Model | Params | test macro-F1 | p95 | Size |
|---|---|---|---|---|
| TF-IDF char n-gram + LogReg | — | **0.9787** | **0.79 ms** | ~5 MB |
| **MiniLM L12-H384** (deployed) | 118M | 0.9695 | 35.82 ms | 471 MB |
| XLM-R base | 278M | 0.9645 | 93.29 ms | 1.1 GB |

The classical baseline wins on accuracy at 45× lower latency. That is a
headline finding, not an embarrassment — keep it in the README. MiniLM is
deployed because the project is about serving a transformer, and MiniLM
beat XLM-R on both axes.

**Run-to-run noise is about ±0.3 macro-F1** (GPU non-determinism, measured
across two identical runs). Treat sub-0.5-point differences as noise.

### Reproducing the three rows

`train_transformer.py` is model-agnostic — the checkpoint is a flag, so all
three rows came from the same code path, same data, same evaluation. That
is what makes the comparison trustworthy; do not fork it per model.

Deployed model (MiniLM), needs a GPU — ~4 min on a Kaggle T4:

```
python scripts/train_transformer.py --data data/merged.csv \
  --model microsoft/Multilingual-MiniLM-L12-H384 \
  --tokenizer xlm-roberta-base --epochs 6 --lr 5e-5 --outdir models/minilm
```

XLM-R comparison row (not deployed):

```
python scripts/train_transformer.py --data data/merged.csv \
  --epochs 6 --outdir models/xlmr
```

Classical baseline, CPU, under a minute. Not deployed, but its number is
what justifies choosing any transformer at all — keep the script:

```
python scripts/baseline_tfidf.py --data data/merged.csv
```

## Gotchas already discovered

- **MiniLM needs XLM-R's tokenizer.** Its config declares no
  `tokenizer_class`, so `AutoTokenizer` fails. Pass `xlm-roberta-base`.
- **MiniLM is BERT underneath**, so it takes `token_type_ids` as a third
  input. XLM-R does not. Getting this wrong yields a graph that runs and
  returns plausible nonsense.
- **`onnxscript` is required** for ONNX export on recent PyTorch, and is
  not pulled in by `onnx` or `onnxruntime`.
- **Verify dynamic axes after export.** A frozen sequence dimension is the
  most common silent export bug and costs the entire padding win.
- **Do not strip URLs, digits or punctuation** in preprocessing. They are
  the strongest phishing signals. Standard tutorial cleaning destroys them.

## Remaining work

1. **ONNX export + parity** — `scripts/export_onnx.py` exists and was
   tested against a small BERT (parity 1.4e-08). Not yet run on the real
   checkpoint. Must PASS at <1e-4 max logit diff before anything else.
2. **Dynamic INT8 quantisation** — re-run full eval after. Record macro-F1
   before/after; anything under 0.5 points is noise.
3. **Benchmark harness** — fixed hardware, 20 warm-up iterations, 500
   measured, report p50/p95/p99. Sweep batch 1/8/32 and padding strategy
   (fixed-64 vs dynamic-to-batch-max). Save every stage to CSV.
4. **FastAPI service** — `POST /v1/classify`, `/v1/classify/batch`,
   `GET /healthz`, `/readyz`, `/metrics`. ORT session loaded once in a
   `lifespan` handler. Response: label, all four probabilities, detected
   language, `model_version` (`bartascope-minilm-v1.0.0`), `inference_ms`.
5. **Docker** — image `bartascope`, multi-stage, **torch excluded from the
   runtime image**
   (onnxruntime + tokenizers + fastapi ≈ 350 MB vs 2 GB+). Keep a second
   Dockerfile with the PyTorch build so the before/after benchmark is
   reproducible.
6. **Deploy** — Cloud Run as `bartascope-api`, min instances 0. Measure
   cold start honestly and put the number in the README.
7. **Load test** — k6. Two profiles: constant-arrival-rate to find where
   p95 breaks, and idle-then-burst for the scale-to-zero story. Run
   against both containers.
8. **Frontend** — titled Bartascope. Single paste box, verdict card
   showing all four
   probabilities, detected language, and inference time. Safety line:
   advisory, not a guarantee; never share OTPs or PINs regardless.

## Rules

- Never log message text. Log length and predicted class only — users will
  paste real SMS containing their own transaction details.
- Set `intra_op_num_threads` to the container CPU limit. Do not run N
  uvicorn workers each spawning N ORT threads; that oversubscription can
  make the optimised build slower than the baseline. Measure both.
- Every latency claim compares against a recorded CSV baseline on the same
  hardware. No remembered numbers.
- Prefer being honest about a disappointing number over hiding it. The
  synthetic-data finding and the classical-baseline result are the most
  interesting content in this project.
