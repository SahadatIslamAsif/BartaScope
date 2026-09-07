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

Done: dataset merged, three models trained and compared, deployment model
chosen, ONNX export + parity, INT8 quantisation (**evaluated and
rejected**), benchmark harness, FastAPI service.
Next: Docker → deploy → load test → frontend.

**Goal 2 is not met by the deployed configuration, and that is a result,
not a todo.** ONNX fp32 cuts p95 by 1.34×, not the 2× the goal asks for.
INT8 reached 2.21× but was rejected on safety grounds (see below). Report
the 1.34× honestly; do not quietly restate the goal to match it.

## Layout

```
bartascope/
data/merged.csv           12,808 rows. Single artifact, has a `split` column.
models/minilm/            fine-tuned checkpoint (deployed model)
models/results.json       metrics (latency fields are LOCAL, see below)
models/errors.csv         49 misclassified test rows
models/baseline/          fitted TF-IDF pipeline + its errors
onnx_out/model.onnx       DEPLOYED graph, fp32, 472 MB (+ .onnx.data)
onnx_out/model.int8.onnx  rejected INT8 graph, kept for the record
benchmarks/latency.csv    every latency row ever measured, append-only
benchmarks/eval_*.json    fp32 / int8 / int8-matmul metrics, the decision record
benchmarks/export_info.json  parity result and export settings
app/textproc.py           normalise + detect_lang, shared with the dataset builder
app/model.py              ORT session wrapper, loaded once
app/main.py               FastAPI service
app/metrics.py            hand-rolled Prometheus exposition (no dependency)
tests/smoke_api.py        end-to-end, real model, no mocks
scripts/prepare_dataset.py  rebuilds merged.csv from raw sources
scripts/baseline_tfidf.py   classical baseline (runs in <1 min, CPU)
scripts/train_transformer.py  fine-tuning (needs a GPU — use Kaggle)
scripts/export_onnx.py      ONNX export + parity check (PASSED, 6.20e-06)
scripts/quantize_onnx.py    INT8 quantisation (ran; result rejected)
scripts/evaluate_onnx.py    full four-class eval of any ONNX graph
scripts/benchmark.py        the latency harness
```

`onnx_out/` and `models/` are gitignored — they are build artifacts.
Regenerate with `export_onnx.py`; the CSV and the eval JSONs are the
committed record.

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

| Model | Params | test macro-F1 | p95 | Hardware | Size |
|---|---|---|---|---|---|
| TF-IDF char n-gram + LogReg | — | **0.9787** | **0.55 ms** | local | ~5 MB |
| **MiniLM L12-H384** (deployed) | 118M | 0.9695 | 12.91 ms | local | 471 MB |
| XLM-R base | 278M | 0.9645 | 93.29 ms | **Kaggle — NOT comparable** | 1.1 GB |

The classical baseline wins on accuracy at 23× lower latency. That is a
headline finding, not an embarrassment — keep it in the README. MiniLM is
deployed because the project is about serving a transformer, and MiniLM
beat XLM-R on both axes.

**Read the hardware column before quoting any of these.** The first two
rows are from `benchmarks/latency.csv` on the local AMD box, batch 1,
dynamic padding, 20 warm-up / 500 measured. The XLM-R row is the original
Kaggle measurement and has never been re-measured locally, because only
`models/minilm/` was kept — reproducing it needs a GPU retrain. Every
Kaggle number measured on this project ran roughly 2.8× slower than local
(MiniLM was 35.82 ms there, 12.91 ms here), so XLM-R's real local p95 is
probably nearer 33 ms. **That is an inference, not a measurement — do not
put it in the README as a number.** The XLM-R-vs-MiniLM ordering is still
safe: both were measured on the same Kaggle box.

The earlier "45× lower latency" claim came from comparing a Kaggle MiniLM
number against a Kaggle TF-IDF number. Both are now local and the true
ratio is 23×. The gap narrowed because faster hardware helps a 118M-param
transformer far more than it helps a sparse dot product.

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

## Deployment decision: serve fp32, reject INT8

**Decided. Do not revisit without new evidence.** `onnx_out/model.onnx`
(fp32) is what the service loads. The INT8 graph is kept as the rejected
alternative, with its metrics, so the decision stays auditable.

| | ONNX fp32 (deployed) | INT8 (rejected) |
|---|---|---|
| artifact size | 472.1 MB | 119.1 MB (3.96× smaller) |
| p95, batch 1, dynamic | 9.14 ms | 5.55 ms |
| speedup vs PyTorch p95 | 1.34× | 2.21× |
| test macro-F1 | 0.9695 | 0.9547 (−1.48) |
| phishing recall | 0.962 | 0.940 |
| **phishing missed as `normal`** | **4** | **13** |
| transactional recall | 1.000 | 0.923 |

The disqualifying number is the last-but-one row. INT8 more than triples
phishing false negatives into `normal` — the single worst direction for a
safety classifier to fail in. It is invisible in the headline: phishing's
*F1* barely moves (0.963 → 0.959) because precision rises as recall falls.
Anyone reading only macro-F1 would have shipped this.

A 1.48-point macro-F1 drop is also well outside the ±0.3 run-to-run noise
band, and quantisation is deterministic, so it is not noise at all.

MatMul-only quantisation (sparing the embedding `Gather`) was tested and is
**strictly dominated**: 407.8 MB *and* macro-F1 0.9507. The accuracy loss
comes from the MatMuls, not from the embedding table, so there is no
variant that buys size without buying error. Do not re-try this.

## Benchmark results (local, isolated processes)

`benchmarks/latency.csv`, rows tagged `isolated process`. 20 warm-up / 500
measured, seeded message order, tokenisation outside the timing loop.

Single message (batch 1, dynamic padding), p95:

| backend | p50 | p95 | msg/s |
|---|---|---|---|
| tfidf-logreg | 0.37 | **0.48** | 2666 |
| onnx-int8 *(rejected)* | 3.07 | 5.55 | 307 |
| **onnx-fp32 (deployed)** | 5.09 | **9.14** | 182 |
| pytorch-fp32 (baseline) | 8.71 | 12.28 | 111 |

Two findings that should shape the service, not just the README:

- **ONNX only wins at batch 1.** At batch 8 and 32 it is *slower* than
  PyTorch (batch 8 dynamic: 60.37 vs 50.35 ms p50). Do not assume the ONNX
  win scales with batch size; it inverts.
- **Batching hurts, so do not add it for throughput.** Batch 1 dynamic is
  the highest-throughput configuration measured. One long message drags the
  whole batch to its length (mean seq len 27.4 at batch 1 vs 60.6 at batch
  32), and the threads already saturate on a single short sequence. If
  `/v1/classify/batch` ever needs to be fast, length-bucket first.

**Measure each backend in its own process.** A combined run holding two ORT
sessions plus torch on a 12-core box inflated PyTorch's p95 from 15.57 to
92.67 ms, because ORT thread pools spin-wait rather than sleeping when
idle. That run produced a flattering, entirely fake "7.28× vs PyTorch". The
CSV keeps it, tagged `combined single-process run` — ignore those rows.

## Container: measured, not estimated

Image `bartascope`, built from `Dockerfile`. Docker reports three different
sizes and they mean different things — quote the right one:

| measurement | value | meaning |
|---|---|---|
| `docker images` | 1.41 GB | uncompressed layers + attestation manifests. **Misleading, do not quote.** |
| `docker image inspect .Size` | **523 MB** | compressed layers — what a node pulls |
| `du -sm /` in the container | 838 MB | disk footprint while running |

Against the ~600 MB expectation: **523 MB pulled, under target; 838 MB on
disk, over it.** The pull size is what governs cold start on Cloud Run.

Composition: model payload 489 MB (`model.onnx.data` 470.6 + `tokenizer.json`
16.8 + graph 1.5), site-packages 247 MB, Debian slim + Python ~100 MB. The
model is 58% of the image, so dependency trimming has limited headroom —
dropping `transformers` in favour of raw `tokenizers` would save roughly
85 MB and is the only remaining lever worth considering.

torch is confirmed absent: the container logs
`[transformers] PyTorch was not found` on startup.

**The `chown -R` trap, since it will recur.** An early build was 928 MB
because `RUN useradd ... && chown -R bartascope:bartascope /app` came after
the COPY. `chown -R` rewrites every file into a new layer, so the 489 MB
model was stored twice. Create the user first and use
`COPY --chown=`. This cost exactly 489 MB — 405 MB of the final image's
savings came from that one line.

Cold start, `docker run` to `/readyz` 200, `--cpus 2`, three runs:
**median 4.30 s** (4.04 / 4.30 / 4.48), of which model load is ~1.8 s. The
rest is container start, Python import and uvicorn boot.

`BARTASCOPE_THREADS` is set by `docker-entrypoint.sh` from the cgroup CPU
quota, verified: `--cpus 2` produces `BARTASCOPE_THREADS=2`. Never let it
fall back to `os.cpu_count()`, which reports the host's cores and ignores the
quota entirely.

## API latency is not model latency

`benchmarks/latency.csv` now carries a `api-http-e2e` backend row. It is a
different measurement from the `onnx-fp32` rows and must not be compared
with them casually:

| | p50 | p95 | p99 |
|---|---|---|---|
| `onnx-fp32`, model only, 6 threads | 5.09 | 9.14 | 10.36 |
| `api-http-e2e`, container, 2 threads | 9.09 | **16.14** | 18.66 |
| ...of which server-side `inference_ms` | 7.12 | 13.67 | 15.77 |

The `onnx-fp32` rows time `session.run` alone on pre-tokenised tensors. The
API row times the full round trip: HTTP, ASGI, pydantic validation,
`normalise()`, tokenisation, ORT, softmax, JSON serialisation. HTTP and
framework overhead is only **~2 ms at p50**; the rest of the gap is
preprocessing plus running on 2 threads instead of 6.

**9.14 ms is not a number this service ever delivers to a caller.** Quote
16.14 ms p95 for the API and keep the model-only figure for the ONNX-vs-
PyTorch comparison it was measured for.

## Load test (k6, local container, 2 vCPU)

`load/constant_arrival.js` and `load/idle_burst.js`, driven by
`scripts/loadtest.py`. k6 runs in a container on `bartascope-net` and reaches
the service by container name, keeping the host loopback and Windows port
forwarding out of the measurement.

**Max sustained rate at p95 < 50 ms: 90 rps.** Confirmed across two runs
(p95 21.63 and 44.30 ms). Per message that is ~11 ms of budget on 2 vCPU.

| target rps | achieved | p50 | p95 | p99 | errors |
|---|---|---|---|---|---|
| 40 | 40.0 | 8.04 | 14.40 | 16.78 | 0% |
| 80 | 80.0 | 9.11 | 20.19 | 56.47 | 0% |
| 85 | 85.0 | 8.90 | 17.84 | 33.56 | 0% |
| **90** | **90.0** | **9.04** | **21.63** | 56.27 | 0% |
| 95 | 95.0 | 24.81 | 351.66 | 395.55 | 0% |
| 100 | 100.0 | 11.38 | 133.53 | 194.47 | 0% |
| 120 | 118.3 | 329.53 | 1331.44 | 1987.65 | 0% |
| 240 | 132.7 | 2362.82 | 42250.80 | 43558.54 | 0% |

Three things this table says that a single number would not:

- **95 rps is a knee, not a limit.** It measured 45.51 ms p95 in one run and
  351.66 ms in another. Right at saturation the service is bimodal, so 90 is
  the honest answer and anything above it is a coin flip.
- **Throughput ceilings at ~130-140 rps** no matter what is offered. Offering
  240 rps yields 132.7. One request occupies both intra-op threads, so
  requests serialise rather than parallelise.
- **Zero HTTP errors at every rate, including 240 rps.** It degrades by
  queueing, not by failing: at 240 rps callers wait 42 seconds and still get
  a correct answer. Nothing sheds load. A real deployment needs a timeout or
  a concurrency limit, because a 42-second success is worse than a fast 503.

### Load shedding (added after the first load test)

The finding above — zero errors, 42-second waits, nothing shed — was fixed.
`app/main.py` bounds in-flight inference with a semaphore, sheds with `503`
+ `Retry-After: 1` when a request waits longer than `QUEUE_TIMEOUT`, and caps
a single inference at `INFERENCE_TIMEOUT`.

| offered | | p50 | p95 | p99 | worst | shed |
|---|---|---|---|---|---|---|
| 90 | before | 9.04 | 21.63 | 56.27 | 164 ms | — |
| 90 | after | 8.85 | 17.64 | 49.88 | 93 ms | 0% |
| 120 | before | 329.53 | 1331.44 | 1987.65 | 3137 ms | — |
| 120 | after | 176.84 | 213.41 | 233.86 | 276 ms | 3.5% |
| 240 | before | 2362.82 | 42250.80 | 43558.54 | 44085 ms | — |
| 240 | after | 211.69 | 257.17 | 272.94 | 313 ms | 58.1% |

164× better p95 at 240 rps, bought by refusing 58% of it. Sustainable load is
untouched: 90 rps sheds nothing and p95 slightly improves.

**It also removed the bimodality.** Before shedding, 95 rps measured p95
45.51 ms in one run and 351.66 ms in another — the knee was a coin flip. After,
two independent runs agree to within 3 ms at both overload rates (120 rps:
213.41 / 213.13; 240 rps: 257.17 / 260.24). The unbounded queue was generating
the variance, so bounding it made overload behaviour reproducible. Both runs
are in the CSV; do not treat the second as a duplicate to be cleaned up.

**Inference had to move off the event loop for any of this to work.** It ran
inline in the `async def` handler, which blocks the loop for the duration —
so a timeout could never fire, because the loop that would fire it was
blocked by the call it was guarding. It now runs in a worker thread
(`run_in_threadpool`); ORT releases the GIL inside `Run`, so that is a real
thread. The anyio thread limiter is capped to `MAX_CONCURRENCY` as well: left
at its default of 40 it would run 40 concurrent inferences, each with a
full-width ORT pool, which is the oversubscription that corrupted the
benchmark tails.

Defaults, all overridable by env var: `MAX_CONCURRENCY` = thread count (2 in
the container), `QUEUE_TIMEOUT_MS` = 200, `INFERENCE_TIMEOUT_MS` = 2000.
Worst-case latency is therefore bounded at roughly queue timeout plus one
inference, which is what the 276-313 ms maxima show. **Tightening
`QUEUE_TIMEOUT_MS` buys a lower p95 under overload at the cost of shedding
more** — 200 ms was chosen to sit far below the 1331 ms regime while leaving
sustainable traffic untouched.

`shed` is counted apart from errors in `/metrics`
(`bartascope_shed_total{reason=...}`), and `constant_arrival.js` treats 503
as an expected status while recording latency in a separate `served_duration`
trend — blending fast 503s into `http_req_duration` would drag p95 down and
make an overloaded service look faster than a healthy one.

**Warm up before the first measured step.** Without it the first rate step
absorbs cold-process cost: 85 rps reported p95 64.12 ms cold and 17.84 ms
warm, which reads as a capacity cliff that does not exist. `constant_arrival.js`
now runs an untagged 15 s warm-up first. CSV rows from run_id
`20260906T233438Z` predate that fix and their 85 rps step is cold-contaminated.

Idle-then-burst, 60 s at 1 rps then a step change:

| burst rate | idle p95 | burst p50 | burst p95 | worst request |
|---|---|---|---|---|
| 60 rps (sustainable) | 19.57 | 8.62 | **15.09** | 75.6 ms |
| 150 rps (above capacity) | 15.30 | 1917.64 | 13327.56 | 20511 ms |

Bursting from idle to a rate the service can serve costs essentially nothing
— burst p95 came in *below* idle p95, and only the single worst request
(75.6 ms) shows the wake-up. The 150 rps row is not a burst measurement, it
is overload wearing a burst's clothing; do not quote it as burst cost. Note
this profile is only fully meaningful on Cloud Run with min-instances 0,
where the burst hits zero running instances. Locally nothing scales from
zero.

### The SIGSEGV

The container died once with **exit code 139 (SIGSEGV)**, `OOMKilled=false`,
during the first full load run — about 8 seconds into a 150 rps burst that
followed a complete 40→240 rps ramp. Every subsequent measurement in that
session was taken against a dead container and produced seven rows of zeros
at 100% error, which were removed from the CSV.

**It did not reproduce**, in two targeted attempts: 240 rps for 45 s, and the
same idle→150 rps burst after a prior overload. Memory was flat throughout
(522 → 529 MiB), so it is not a leak and not the OOM killer.

So: one unexplained crash under sustained overload, not reproducible on
demand. Do not describe it as a known crash bug, and do not describe the
service as stable under overload either. If it recurs, the first suspects are
concurrent `session.run` calls into one ORT session, and `PYTHONFAULTHANDLER=1`
plus a core dump would be the way to find out.

`scripts/loadtest.py` now checks container liveness before and after every
profile and refuses to write a row when more than half the requests failed.
That defect — silently recording a dead service as a measurement — is how the
seven zero rows got in.

## Frontend

`app/static/index.html` — one file, no framework, no build step, no CDN.
Served by the API itself at `GET /` via a single `FileResponse`, not a
`StaticFiles` mount: there is exactly one file, and mounting a directory
would publish whatever later lands in it. Same origin as `/v1/classify`, so
there is no CORS configuration to get wrong.

Verified in a real browser against the container, not just asserted:
phishing lure → red `phishing` badge at 97.0%; banglish personal message →
neutral `normal` at 98.5% with `Language: banglish`. All four probabilities
render as labelled bars in fixed order (not sorted by value) so positions
stay comparable between messages.

All three failure paths were exercised against a live browser:

| case | how it was triggered | result |
|---|---|---|
| over 2000 chars | 2500-char value | counter turns red, error shown, no request sent |
| network failure | `docker stop` mid-session | "Could not reach the classifier", button re-enabled |
| model not loaded | container run with a bad `BARTASCOPE_MODEL_PATH` | 503 → "The classifier is not loaded yet" |

**An error hides any previous verdict.** Found while testing: the stale
`normal` card stayed on screen underneath the new error, directly below the
message that had just failed, and the obvious reading is that the label
describes *that* message. On a phishing classifier the dangerous version of
that is a stale `normal`. `showError()` now hides the result card.

The page classifies on click only, never on keystroke — one request per
button press.

**Loading state is sized for the cold start, not the warm path.** With
min-instances 0 the first request pays the full ~4.3 s container start while
every later one answers in single-digit milliseconds, so the button shows an
animated spinner (not just a changed label — a static "Checking…" for four
seconds reads as a hung page), and after 1200 ms a hint appears explaining
that the classifier is starting up. Verified by delaying `fetch` in the page
to 4300 ms: spinner and hint both appear, then clear on resolution. At normal
speed the hint never renders at all — confirmed with a MutationObserver over
a 7.4 ms request — so the fast path does not flicker. The spinner has a
`prefers-reduced-motion` fallback that pulses instead of rotating.

Note `inference_ms` in the footer is the server's own measurement, so it
stays honest during a slow round trip: the delayed-fetch test still reported
9.5 ms while the request took 4.3 s.

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
- **The serving path must call `normalise()`.** `merged.csv` holds already
  normalised text (verified idempotent on all 12,808 rows), so the model was
  fine-tuned on text with `<PHONE>`, `<CODE>` and domain-only URLs. Serving
  raw text is a silent failure — confident probabilities from a distribution
  the model never saw. This is why `normalise` and `detect_lang` live in
  `app/textproc.py` and are imported by both `prepare_dataset.py` and the
  service, rather than existing twice.
- **ONNX's dynamo exporter breaks ORT quantisation tooling.**
  `quant_pre_process` fails with `unsupported broadcast between Min(512,
  batch) sequence`, and `quantize_dynamic` then fails with `Inferred shape
  and existing shape differ in dimension 0: (384) vs (4)`. The exporter
  emits `value_info` that onnx's own inferencer contradicts. Fix is
  `quantize_onnx.py --strip-value-info`, which drops the 814 intermediate
  annotations and re-infers.
- **`detect_lang` agrees with the dataset's `lang` column only 89.9% of the
  time**, because that column came from the source datasets, not from this
  heuristic. The API reports the heuristic's answer. It is context for the
  caller, never an input to the model — do not start treating it as ground
  truth.
- **Windows needs `PYTHONIOENCODING=utf-8`** to run `export_onnx.py`, or
  torch's own success message crashes cp1252 stdout mid-export.

## Remaining work

1. ~~**ONNX export + parity**~~ — DONE. 6.20e-06 max logit diff over 200
   test messages, zero label disagreements, both axes dynamic.
2. ~~**Dynamic INT8 quantisation**~~ — DONE and **rejected**. See the
   deployment decision above. Do not re-open without new evidence.
3. ~~**Benchmark harness**~~ — DONE. `scripts/benchmark.py`, results in
   `benchmarks/latency.csv`. Run one backend per process.
4. ~~**FastAPI service**~~ — DONE. `app/main.py`, all five endpoints,
   session loaded in `lifespan`, verified by `tests/smoke_api.py` (26
   checks, real model, no mocks).
5. ~~**Docker**~~ — DONE for the serving image. See below.
   `Dockerfile.pytorch` is written but **has never been built**.
6. **Docker (original text, for reference)** — image `bartascope`, multi-stage, **torch excluded from the
   runtime image**
   (onnxruntime + tokenizers + fastapi ≈ 350 MB vs 2 GB+). Keep a second
   Dockerfile with the PyTorch build so the before/after benchmark is
   reproducible.
6. **Deploy** — Cloud Run as `bartascope-api`, min instances 0. Measure
   cold start honestly and put the number in the README.
7. ~~**Load test**~~ — DONE against the local container; see above. Still
   outstanding: re-run against Cloud Run once item 6 lands, and against the
   PyTorch container, which has never been built.
8. ~~**Frontend**~~ — DONE. `app/static/index.html`, served at `GET /` by
   the same FastAPI app. See below.

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
