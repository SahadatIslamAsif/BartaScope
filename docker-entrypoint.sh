#!/bin/sh
# Set BARTASCOPE_THREADS from the container's CPU limit, not the host's core
# count.
#
# This matters more than it looks. os.cpu_count() inside a container reports
# the *host* core count, ignoring the cgroup quota entirely. On a 32-core
# node with a 1-CPU limit, ONNX Runtime would spawn 32 intra-op threads to
# share a single core's worth of quota. Those threads spin-wait rather than
# sleeping when idle, so the result is not merely no-gain, it is slower than
# a correctly sized pool -- the same oversubscription that inflated PyTorch's
# p95 from 15.57 to 92.67 ms during this project's own benchmarking.
#
# An explicitly passed BARTASCOPE_THREADS always wins; this only fills in a
# default.

set -e

detect_cpu_limit() {
    # cgroup v2: "$MAX $PERIOD", where $MAX is "max" when unlimited.
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r quota period < /sys/fs/cgroup/cpu.max || true
        if [ -n "$quota" ] && [ "$quota" != "max" ] && [ -n "$period" ] \
           && [ "$period" -gt 0 ] 2>/dev/null; then
            # Round up: a 1.5-CPU limit should get 2 threads, not 1.
            echo $(( (quota + period - 1) / period ))
            return
        fi
    fi

    # cgroup v1
    if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] \
       && [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
        quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        if [ "$quota" -gt 0 ] 2>/dev/null && [ "$period" -gt 0 ] 2>/dev/null; then
            echo $(( (quota + period - 1) / period ))
            return
        fi
    fi

    # No quota set (`docker run` with no --cpus). nproc respects cpuset
    # affinity even when no quota exists, so it beats os.cpu_count() here.
    nproc 2>/dev/null || echo 1
}

if [ -z "$BARTASCOPE_THREADS" ] || [ "$BARTASCOPE_THREADS" = "0" ]; then
    BARTASCOPE_THREADS=$(detect_cpu_limit)
    export BARTASCOPE_THREADS
    echo "entrypoint: BARTASCOPE_THREADS=$BARTASCOPE_THREADS (from CPU limit)"
else
    echo "entrypoint: BARTASCOPE_THREADS=$BARTASCOPE_THREADS (explicit)"
fi

# One worker by default, on purpose. N workers each spawning an N-wide ORT
# pool is the oversubscription this file exists to prevent. Scale with
# replicas, not workers, and if you must raise it, lower BARTASCOPE_THREADS
# to match and re-measure.
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8080}" \
    --workers "${BARTASCOPE_WORKERS:-1}" \
    --log-level "${UVICORN_LOG_LEVEL:-info}"
