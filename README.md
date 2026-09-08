# Bartascope

Paste an SMS, find out what kind of message it is: personal, a genuine
service notification, advertising, or a phishing attempt. Bangla, English,
banglish and code-mixed messages all go through the same model.

The **engineering** is the deliverable — ONNX export and optimisation, a
FastAPI service, Docker, load testing — not the accuracy. Accuracy was
already near this dataset's ceiling before any of that started.

```bash
docker build -t bartascope .
docker run --rm -p 8080:8080 --cpus 2 bartascope
# then open http://localhost:8080
```

---

## What this project actually found

Three results here cut against the story the project set out to tell. They
are the most interesting things in it, so they go first rather than in an
appendix.

### 1. A TF-IDF baseline beats the deployed transformer, at 23× lower latency

| | test macro-F1 | p95 latency | size |
|---|---|---|---|
| **TF-IDF char n-gram + LogReg** | **0.9787** | **0.55 ms** | ~5 MB |
| MiniLM L12-H384 (deployed) | 0.9695 | 12.91 ms | 471 MB |

The classical baseline is more accurate *and* 23× faster. MiniLM ships
anyway, because the point of the project is to serve a transformer properly
— but on the merits of this dataset, the 5 MB model wins, and any honest
reading of the table says so.

### 2. The ONNX speedup target was missed

The goal was to halve p95 latency versus PyTorch fp32. Measured, same
machine, same process, batch 1:

| backend | p50 | p95 | speedup |
|---|---|---|---|
| PyTorch fp32 (baseline) | 8.71 | 12.28 ms | — |
| **ONNX fp32 (deployed)** | 5.09 | **9.14 ms** | **1.34×** |
| ONNX INT8 (rejected) | 3.07 | 5.55 ms | 2.21× |

**1.34×, not 2×.** INT8 would have cleared the bar at 2.21× and was rejected
for the reason below. The target is recorded as missed rather than quietly
restated to match what was achieved.

Two related findings from the same measurements:

- **Padding matters more than the runtime.** Dropping fixed-64 padding for
  dynamic-to-batch-max saves more p95 than switching PyTorch → ONNX does.
- **ONNX only wins at batch 1.** At batch 8 and 32 it is *slower* than
  PyTorch. Batching also hurts overall: batch 1 is the highest-throughput
  configuration measured, because one long message drags a whole batch up to
  its length.

### 3. INT8 was rejected — it hid a phishing regression inside a healthy F1

Dynamic INT8 quantisation is 3.96× smaller and 2.21× faster. It costs 1.48
macro-F1 points. That alone might be arguable. This is not:

| | ONNX fp32 | INT8 |
|---|---|---|
| test macro-F1 | 0.9695 | 0.9547 |
| phishing **recall** | 0.962 | 0.940 |
| phishing F1 | 0.963 | 0.959 |
| **phishing messages misread as `normal`** | **4** | **13** |

INT8 more than triples phishing false negatives into `normal` — the worst
available direction for a safety classifier to fail in. And it is nearly
invisible in the summary: phishing's **F1 barely moves**, because precision
rises as recall falls. Anyone reading macro-F1 or the per-class F1 column
would have shipped it.

MatMul-only quantisation (sparing the embedding table) was tested too and is
strictly worse: 407.8 MB *and* macro-F1 0.9507. The loss is in the MatMuls,
so there is no variant that buys size without buying error.

**fp32 ships.** Both INT8 graphs and their metrics are kept in
`benchmarks/` so the decision stays auditable.

---

## Model selection

Single-message CPU p95. **Read the hardware column before quoting these.**

| Model | Params | test macro-F1 | p95 | Hardware | Size |
|---|---|---|---|---|---|
| TF-IDF char n-gram + LogReg | — | **0.9787** | **0.55 ms** | local | ~5 MB |
| **MiniLM L12-H384** (deployed) | 118M | 0.9695 | 12.91 ms | local | 471 MB |
| XLM-R base | 278M | 0.9645 | 93.29 ms | **Kaggle — not comparable** | 1.1 GB |

The first two rows are measured on the same local machine. The XLM-R row is
the original Kaggle measurement and has never been re-measured locally,
because only the MiniLM checkpoint was kept and reproducing it needs a GPU.
Kaggle ran roughly 2.8× slower than this machine, so XLM-R's local p95 is
probably nearer 33 ms — that is an inference, not a measurement, and it is
not quoted as one. The XLM-R-vs-MiniLM *ordering* is still safe: both were
measured on the same box as each other.

MiniLM is deployed because the project exists to serve a transformer, and it
beat XLM-R on both accuracy and latency.

Run-to-run noise is about ±0.3 macro-F1 from GPU non-determinism. Treat
sub-0.5-point differences as noise. Quantisation, by contrast, is
deterministic — the 1.48-point INT8 drop is not noise.

---

## Data caveats

These are load-bearing. The headline accuracy numbers overstate real-world
performance, and the reasons are known.

**The Bangla source dataset is almost certainly synthetic.** Near-perfect
uniform language balance, 1,095 phone mentions drawn from only 266 unique
numbers, and template families that differ only in their digits.

**The sources are trivially separable.** A logistic regression predicts which
source dataset a message came from with **0.962 accuracy** against a 0.531
majority baseline. A model can score well by learning source style rather
than message type, which is why per-source metrics are reported every time:

| source | n | macro-F1 |
|---|---|---|
| bengali_smishing (synthetic) | 1013 | 0.978 |
| mendeley (real English SMS) | 871 | **0.890** |

Roughly nine points worse on the real data than the synthetic data. The
aggregate number is flattered by the easy half.

**`transactional` is English-only, and that is a fact about the world, not a
gap in the data.** Bangla text in SMS forces UCS-2 encoding — 70 characters
per message instead of 160 — so Bangladeshi banks and mobile-money providers
send their alerts in English. It has been left that way deliberately. No
synthetic data was generated to fill it.

**Three templates carry contradictory labels** in the source data, an
upstream annotation error, left in deliberately rather than quietly cleaned.

**There is no real-world evaluation set.** Nothing here was tested against
messages collected from actual phones beyond a 39-message sample. Every test
score on this page should be read with that in mind.

The four classes: `normal`, `transactional`, `promotional`, `phishing`. Spam
was deliberately folded into `promotional`, because one source dataset had no
spam category and labelled such messages `promo`; keeping them apart made the
boundary inconsistent by construction. Train is 27:1 imbalanced, so
**macro-F1 is reported everywhere and accuracy nowhere** — accuracy mostly
measures how often the model says `normal`.

---

## Service

FastAPI, ONNX Runtime, one session created at startup and reused.

| endpoint | purpose |
|---|---|
| `GET /` | the web page |
| `POST /v1/classify` | one message |
| `POST /v1/classify/batch` | up to 64 messages |
| `GET /healthz` | liveness |
| `GET /readyz` | readiness — reports a failed model load |
| `GET /metrics` | Prometheus exposition |

```bash
curl -X POST localhost:8080/v1/classify \
  -H 'Content-Type: application/json' \
  -d '{"text": "Your bKash account will be closed. Verify at http://bkash-verify-bd.info"}'
```

```json
{
  "label": "phishing",
  "confidence": 0.9719,
  "probabilities": {
    "normal": 0.0043, "transactional": 0.0167,
    "promotional": 0.0071, "phishing": 0.9719
  },
  "language": "english",
  "language_uncertain": false,
  "chars": 98,
  "model_version": "bartascope-minilm-v1.0.0",
  "inference_ms": 9.385
}
```

**Message text is never logged.** Users paste real SMS containing their own
transaction details; only length and predicted class are recorded, and
`/metrics` carries counts and labels only.

The service applies the same `normalise()` the model was fine-tuned on —
URLs reduced to scheme and domain, phone numbers to `<PHONE>`, OTP digits to
`<CODE>`. Serving raw text would be a silent failure: confident probabilities
drawn from a distribution the model never saw. That function lives in
`app/textproc.py` and is imported by both the dataset builder and the
service, verified byte-identical across all 12,808 rows, so the two paths
cannot drift.

---

## Container

| measurement | value | meaning |
|---|---|---|
| `docker images` | 1.33 GB disk / 511 MB content | uncompressed + attestation. Misleading, don't quote it. |
| `docker image inspect .Size` | **511 MB** | compressed — what a node pulls |
| `du -sm /` inside | 772 MB | disk while running |

Multi-stage, **torch excluded from the runtime image**. Torch is needed to
*produce* the ONNX graph, never to run it, and including it would add ~2 GB
for code that would never execute.

`transformers` is excluded too, as of this measurement. It was only ever
used to call `AutoTokenizer.from_pretrained()`; swapping in the raw
`tokenizers` package (`Tokenizer.from_file`, verified byte-identical
tokenisation first) dropped site-packages from 247 MB to **181 MB**. That's
a 66 MB uncompressed win, but only 12 MB of it survives compression (523 MB
→ 511 MB pulled) — `transformers` is mostly plain-Python source, which gzips
hard, so most of the saving doesn't show up in pull size even though it's
real on disk.

The model is 467 MB of the 511 MB, so dependency trimming has little
remaining headroom.

**Cold start: 4.30 s median** (`docker run` → `/readyz` 200, `--cpus 2`,
three runs: 4.04 / 4.30 / 4.48), of which ~1.8 s is loading the model. The
web page is built around this: the button shows an animated spinner, and
after 1.2 s a hint explains that the classifier is starting.

`BARTASCOPE_THREADS` is derived from the **cgroup CPU quota**, not
`os.cpu_count()` — inside a container that reports the host's core count and
ignores the limit entirely. On a 32-core node with a 1-CPU limit, ONNX
Runtime would otherwise spawn 32 spin-waiting threads to share one core.

---

## Capacity and load shedding

k6, two profiles, against the container on 2 vCPU. **Max sustained rate with
p95 under 50 ms: 90 rps.**

The first load test found something worth fixing. At every offered rate, up
to 240 rps, the service returned **zero errors** — it degraded purely by
queueing. At 240 rps it still answered every request correctly, after making
the caller wait **42 seconds**. Nothing shed. A 42-second success is worse
than an immediate 503: the caller has long since given up, the work is
wasted, and on Cloud Run the platform would have killed the request anyway.

So the service now bounds in-flight inference and refuses rather than queues.
Same profile, before and after:

| offered rps | | p50 | p95 | p99 | worst | shed |
|---|---|---|---|---|---|---|
| 90 | before | 9.04 | 21.63 | 56.27 | 164 ms | — |
| 90 | **after** | 8.85 | 17.64 | 49.88 | 93 ms | 0% |
| 120 | before | 329.53 | **1331.44** | 1987.65 | 3137 ms | — |
| 120 | **after** | 176.84 | **213.41** | 233.86 | 276 ms | 3.5% |
| 240 | before | 2362.82 | **42250.80** | 43558.54 | 44085 ms | — |
| 240 | **after** | 211.69 | **257.17** | 272.94 | 313 ms | 58.1% |

At 240 rps p95 drops from 42.3 seconds to 257 ms — **164× better** — by
refusing 58% of the traffic outright. Latency is now bounded by construction:
a request waits at most 200 ms for a slot before it is shed, so nothing can
grow into the multi-second regime. Sustainable load is unaffected: at 90 rps
nothing is shed and p95 slightly improves.

It also made the service *predictable*. Before shedding, the knee was bimodal
— 95 rps measured p95 45 ms in one run and 352 ms in another, so the limit
was a coin flip. After, two independent runs agree to within 3 ms at both
overload rates (120 rps: 213.41 and 213.13; 240 rps: 257.17 and 260.24).
Refusing work early removes the queue that was generating the variance.

Sizing comes from the measurements, not from taste. One request occupies all
`intra_op_num_threads`, so requests do not really run in parallel and
`max_concurrency` defaults to the thread count — permitting one full-width
inference plus one waiting. Little's law agrees: 90 rps at ~10 ms of service
time is under one request in flight on average.

Shed responses return `503` with `Retry-After: 1` and are counted separately
from errors in `/metrics` (`bartascope_shed_total`), because a shed request
is the service working correctly under overload, not a fault.

Other findings from the load test:

- **Throughput ceilings at ~130 rps** however much is offered. Requests
  serialise on the intra-op threads rather than parallelising.
- **Bursting from idle is free** at a sustainable rate: idle → 60 rps gave a
  burst p95 of 15.09 ms, *below* the idle p95 of 19.57, with only the single
  worst request (75.6 ms) showing the wake-up.
- **The container segfaulted once** (exit 139, not OOM) during sustained
  overload, and did not reproduce in two targeted attempts. Memory was flat
  throughout, so it is neither a leak nor the OOM killer. Recorded as one
  unexplained crash — not a known bug, and not evidence of stability either.

## Latency: API vs model

These are different numbers and the CSV keeps them under different names.

| | p50 | p95 | p99 |
|---|---|---|---|
| `onnx-fp32` — `session.run` only, 6 threads | 5.09 | 9.14 | 10.36 |
| `api-http-e2e` — full round trip, container, 2 threads | 9.09 | **16.14** | 18.66 |

HTTP, ASGI, validation and JSON account for only ~2 ms at p50; the rest is
preprocessing plus running on 2 threads instead of 6. **9.14 ms is not a
latency any caller ever sees** — quote 16.14 ms for the API.

---

## Reproducing

Every latency claim in this README comes from `benchmarks/latency.csv`, which
is append-only and records machine, thread count, batch size, padding
strategy and warm-up for every row. No remembered numbers.

```bash
python scripts/prepare_dataset.py --outdir data/     # rebuild merged.csv
python scripts/baseline_tfidf.py  --data data/merged.csv --outdir models/baseline
python scripts/train_transformer.py --data data/merged.csv \
    --model microsoft/Multilingual-MiniLM-L12-H384 \
    --tokenizer xlm-roberta-base --epochs 6 --lr 5e-5 --outdir models/minilm
python scripts/export_onnx.py   --model-dir models/minilm --data data/merged.csv
python scripts/evaluate_onnx.py --model onnx_out/model.onnx
python scripts/benchmark.py     --backends onnx-fp32     # one backend per process
python tests/smoke_api.py
```

Training needs a GPU (~4 minutes on a Kaggle T4). Everything else runs on
CPU. On Windows, `export_onnx.py` needs `PYTHONIOENCODING=utf-8` or torch's
own success message crashes cp1252 stdout mid-export.

**Benchmark one backend per process.** A combined run holding two ONNX
Runtime sessions plus torch inflated PyTorch's p95 from 15.57 ms to 92.67 ms,
because ORT thread pools spin-wait rather than sleeping when idle. That run
produced a flattering, entirely fake "7.28× vs PyTorch".

---

## Layout

```
app/                 FastAPI service, ONNX wrapper, shared preprocessing, web page
scripts/             dataset build, training, export, quantisation, eval, benchmarks
load/                k6 profiles
tests/smoke_api.py   end-to-end against the real model, no mocks
benchmarks/          latency.csv and the eval JSONs — the committed record
data/merged.csv      12,808 rows with a `split` column
```

`models/` and `onnx_out/` are build artifacts and are gitignored; regenerate
them with the commands above. Filter on the existing `split` column — never
re-split. The split is stratified on `(label, lang)` jointly and keeps
template families whole, because templated SMS are identical once digits are
stripped and splitting them leaks train into test.
