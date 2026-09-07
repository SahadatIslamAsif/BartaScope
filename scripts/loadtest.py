"""
Step 7: k6 load tests, run in Docker against the container.

Generates the message corpus, runs both k6 profiles, parses their summaries
and appends the results to benchmarks/latency.csv under their own backend
names so they cannot be confused with the single-client measurements.

k6 runs in a container on the same Docker network as the service, addressing
it by container name. That keeps the load generator off the host's loopback
stack and out of the port-forwarding path, which on Windows is its own
source of latency.

Usage:
    python scripts/loadtest.py --profile both
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import CSV_FIELDS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LOAD_DIR = REPO / "load"


def header(text: str) -> None:
    print(f"\n{'=' * 68}\n{text}\n{'=' * 68}")


def write_messages(data_path: Path, n: int) -> None:
    """Same seeded permutation as benchmark.py, so the load test and the
    single-client benchmarks exercise the same messages."""
    df = pd.read_csv(data_path)
    texts = df[df.split == "test"].text.astype(str).tolist()
    order = np.random.default_rng(0).permutation(len(texts))
    texts = [texts[i] for i in order][:n]
    out = LOAD_DIR / "messages.json"
    out.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {out} ({len(texts)} messages)")


def assert_healthy(container: str, health_url: str, when: str) -> None:
    """Refuse to measure a service that is not up.

    A load test against a dead container produces a full set of confident
    looking rows in which every number is zero and every request failed.
    That happened once here: the service took SIGSEGV during an overload
    step and the next run recorded seven rows of nothing. Check before, and
    check after -- a crash mid-run is exactly what a load test is for.
    """
    import httpx

    state = subprocess.run(
        ["docker", "inspect", container, "--format",
         "{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}"],
        text=True, capture_output=True)
    info = state.stdout.strip()
    if state.returncode != 0:
        raise SystemExit(f"[{when}] container {container} not found")
    status = info.split()[0] if info else "?"
    if status != "running":
        raise SystemExit(
            f"[{when}] container {container} is '{info}'. "
            f"Exit 139 means SIGSEGV. Restart it and investigate before "
            f"trusting any numbers.")

    try:
        if httpx.get(health_url, timeout=5.0).status_code != 200:
            raise SystemExit(f"[{when}] {health_url} did not return 200")
    except Exception as exc:
        raise SystemExit(f"[{when}] cannot reach {health_url}: {exc}")
    print(f"  health [{when}]: {container} running, {health_url} ok")


def run_k6(script: str, env: dict, network: str) -> None:
    cmd = ["docker", "run", "--rm", "--network", network,
           "-v", f"{LOAD_DIR.as_posix()}:/work", "-w", "/work"]
    for key, value in env.items():
        cmd += ["-e", f"{key}={value}"]
    cmd += ["grafana/k6", "run", f"/work/{script}"]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=True)
    # k6 exits non-zero when a threshold fails. Ours are set so they cannot,
    # but a crash should still be visible rather than swallowed.
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise SystemExit(f"k6 exited {result.returncode}")
    tail = [ln for ln in result.stdout.splitlines()
            if "dropped" in ln or "checks" in ln or "http_reqs" in ln]
    for line in tail[:6]:
        print("   ", line.strip())


def metric(summary: dict, name: str, field: str, default=0.0):
    entry = summary.get("metrics", {}).get(name)
    if not entry:
        return default
    values = entry.get("values", entry)
    return values.get(field, default)


def make_row(run_id, backend, machine, threads, notes, duration_s,
             durations: dict, iterations: float, failed_rate: float,
             dropped: float):
    # A row where everything failed is not a slow result, it is the
    # absence of a result. Recording it pollutes the CSV with zeros that
    # look like measurements.
    if failed_rate > 0.5:
        raise SystemExit(
            f"{backend} / {notes}: {failed_rate * 100:.0f}% of requests "
            f"failed. Refusing to write a row. Check the container.")
    p50 = durations.get("med", 0.0)
    p95 = durations.get("p(95)", 0.0)
    p99 = durations.get("p(99)", 0.0)
    achieved = iterations / duration_s if duration_s else 0.0
    return {
        "run_id": run_id, "backend": backend, "batch_size": 1,
        "padding": "dynamic-to-batch-max", "threads": threads,
        "n_warmup": 0, "n_measured": int(iterations),
        "n_messages": int(iterations),
        "p50_ms": round(p50, 3), "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3),
        "mean_ms": round(durations.get("avg", 0.0), 3),
        "min_ms": round(durations.get("min", 0.0), 3),
        "per_msg_p50_ms": round(p50, 3), "per_msg_p95_ms": round(p95, 3),
        "throughput_msg_s": round(achieved, 1),
        "mean_seq_len": "", "max_seq_len": "", "machine": machine,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": f"{notes}; achieved {achieved:.1f} rps; "
                 f"errors {failed_rate * 100:.2f}%; dropped {int(dropped)}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="both",
                    choices=["constant", "burst", "both"])
    ap.add_argument("--network", default="bartascope-net")
    ap.add_argument("--base-url", default="http://bartascope-api:8080")
    ap.add_argument("--data", default="data/merged.csv")
    ap.add_argument("--out", default="benchmarks/latency.csv")
    ap.add_argument("--container", default="bartascope-api",
                    help="container name, checked alive before and after")
    ap.add_argument("--health-url",
                    default="http://127.0.0.1:8099/healthz")
    ap.add_argument("--threads", type=int, default=2,
                    help="the service's thread count, recorded in the CSV")
    ap.add_argument("--n-messages", type=int, default=200)
    ap.add_argument("--rates", default="20,40,60,80,100,120,140,160")
    ap.add_argument("--stage-seconds", type=int, default=30)
    ap.add_argument("--gap-seconds", type=int, default=5)
    ap.add_argument("--idle-seconds", type=int, default=60)
    ap.add_argument("--burst-seconds", type=int, default=30)
    ap.add_argument("--burst-rate", type=int, default=100)
    ap.add_argument("--sla-ms", type=float, default=50.0,
                    help="p95 budget used to pick max sustained rate")
    args = ap.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    machine = f"{platform.processor() or platform.machine()} / {platform.system()}"

    header("SETUP")
    write_messages(REPO / args.data, args.n_messages)

    out_path = REPO / args.out
    new_file = not out_path.exists()
    fh = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()

    rows = []

    if args.profile in ("constant", "both"):
        header("PROFILE 1: constant arrival rate, stepped")
        assert_healthy(args.container, args.health_url, "before")
        run_k6("constant_arrival.js", {
            "BASE_URL": args.base_url, "RATES": args.rates,
            "STAGE_SECONDS": args.stage_seconds,
            "GAP_SECONDS": args.gap_seconds,
        }, args.network)
        assert_healthy(args.container, args.health_url, "after")

        summary = json.loads(
            (LOAD_DIR / "summary_constant.json").read_text(encoding="utf-8"))
        print(f"\n  {'rate':>6} {'achieved':>9} {'p50':>8} {'p95':>8} "
              f"{'p99':>8} {'max':>9} {'err%':>6} {'dropped':>8}")
        print("  " + "-" * 70)
        for rate in [int(r) for r in args.rates.split(",")]:
            durations = summary["metrics"].get(
                f"http_req_duration{{rate:{rate}}}", {}).get("values", {})
            iterations = metric(summary, f"iterations{{rate:{rate}}}", "count")
            dropped = metric(summary,
                             f"dropped_iterations{{rate:{rate}}}", "count")
            failed = metric(summary, f"http_req_failed{{rate:{rate}}}", "rate")
            row = make_row(run_id, "k6-constant-arrival", machine,
                           args.threads, f"target {rate} rps",
                           args.stage_seconds, durations, iterations, failed,
                           dropped)
            writer.writerow(row)
            fh.flush()
            rows.append((rate, row, durations.get("max", 0.0)))
            print(f"  {rate:>6} {row['throughput_msg_s']:>9.1f} "
                  f"{row['p50_ms']:>8.2f} {row['p95_ms']:>8.2f} "
                  f"{row['p99_ms']:>8.2f} {durations.get('max', 0):>9.1f} "
                  f"{failed * 100:>6.2f} {int(dropped):>8}")

        header(f"MAX SUSTAINED RATE AT p95 < {args.sla_ms:.0f} ms")
        passing = [(rate, row) for rate, row, _ in rows
                   if row["p95_ms"] < args.sla_ms
                   and row["throughput_msg_s"] >= rate * 0.95
                   and "errors 0.00%" in row["notes"]]
        if passing:
            best_rate, best = max(passing, key=lambda x: x[0])
            print(f"  {best_rate} rps   p95 {best['p95_ms']:.2f} ms   "
                  f"achieved {best['throughput_msg_s']:.1f} rps")
            print("  (requires: p95 under budget, achieved rate within 5% of "
                  "target, zero errors)")
        else:
            print(f"  none -- p95 exceeded {args.sla_ms:.0f} ms at every rate "
                  f"tested, or the service could not hold the offered rate")

    if args.profile in ("burst", "both"):
        header("PROFILE 2: idle, then burst")
        assert_healthy(args.container, args.health_url, "before")
        run_k6("idle_burst.js", {
            "BASE_URL": args.base_url, "IDLE_SECONDS": args.idle_seconds,
            "BURST_SECONDS": args.burst_seconds,
            "BURST_RATE": args.burst_rate,
        }, args.network)
        assert_healthy(args.container, args.health_url, "after")

        summary = json.loads(
            (LOAD_DIR / "summary_burst.json").read_text(encoding="utf-8"))
        for phase, duration in (("idle", args.idle_seconds),
                                ("burst", args.burst_seconds)):
            durations = summary["metrics"].get(
                f"http_req_duration{{phase:{phase}}}", {}).get("values", {})
            iterations = metric(summary, f"iterations{{phase:{phase}}}",
                                "count")
            dropped = metric(summary,
                             f"dropped_iterations{{phase:{phase}}}", "count")
            failed = metric(summary, f"http_req_failed{{phase:{phase}}}",
                            "rate")
            row = make_row(run_id, "k6-idle-burst", machine, args.threads,
                           f"phase {phase}", duration, durations, iterations,
                           failed, dropped)
            writer.writerow(row)
            fh.flush()
            print(f"  {phase:<6} p50 {row['p50_ms']:>7.2f}  "
                  f"p95 {row['p95_ms']:>7.2f}  p99 {row['p99_ms']:>7.2f}  "
                  f"max {durations.get('max', 0):>8.1f} ms   "
                  f"{row['throughput_msg_s']:.1f} rps")

    fh.close()
    print(f"\n  appended -> {out_path}\n")


if __name__ == "__main__":
    main()
