"""
Minimal Prometheus exposition, hand-rolled.

prometheus_client would be the obvious dependency, but the runtime image is
supposed to stay near 350 MB and every avoided package helps. What is needed
here is three counters and one histogram, which is less code than the import
line justifies.

Nothing recorded here contains message text -- only counts, lengths and
predicted labels. Users paste real SMS containing their own transaction
details, so the metrics endpoint must stay safe to scrape and store.
"""

import threading
import time

# Seconds. Chosen around the measured p50/p95 of ~3-13 ms so the interesting
# region is not squashed into one bucket.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.requests_total = {}
        self.predictions_total = {}
        self.errors_total = {}
        # Deliberate refusals, counted apart from errors. A shed request is
        # the service working as designed under overload, not a fault, and
        # blending the two hides both.
        self.shed_total = {}
        self.messages_total = 0
        self.latency_buckets = {b: 0 for b in LATENCY_BUCKETS}
        self.latency_inf = 0
        self.latency_sum = 0.0
        self.latency_count = 0

    def record_request(self, endpoint: str, status: int) -> None:
        with self._lock:
            key = (endpoint, str(status))
            self.requests_total[key] = self.requests_total.get(key, 0) + 1

    def record_error(self, endpoint: str, kind: str) -> None:
        with self._lock:
            key = (endpoint, kind)
            self.errors_total[key] = self.errors_total.get(key, 0) + 1

    def record_shed(self, endpoint: str, reason: str) -> None:
        with self._lock:
            key = (endpoint, reason)
            self.shed_total[key] = self.shed_total.get(key, 0) + 1

    def record_prediction(self, label: str, language: str) -> None:
        with self._lock:
            key = (label, language)
            self.predictions_total[key] = self.predictions_total.get(key, 0) + 1
            self.messages_total += 1

    def record_latency(self, seconds: float) -> None:
        with self._lock:
            self.latency_sum += seconds
            self.latency_count += 1
            for bound in LATENCY_BUCKETS:
                if seconds <= bound:
                    self.latency_buckets[bound] += 1
            if seconds > LATENCY_BUCKETS[-1]:
                self.latency_inf += 1

    def render(self) -> str:
        with self._lock:
            lines = [
                "# HELP bartascope_uptime_seconds Seconds since process start.",
                "# TYPE bartascope_uptime_seconds gauge",
                f"bartascope_uptime_seconds {time.time() - self.started_at:.3f}",
                "# HELP bartascope_requests_total HTTP requests by endpoint "
                "and status.",
                "# TYPE bartascope_requests_total counter",
            ]
            for (endpoint, status), n in sorted(self.requests_total.items()):
                lines.append(
                    f'bartascope_requests_total{{endpoint="{endpoint}",'
                    f'status="{status}"}} {n}')

            lines += [
                "# HELP bartascope_predictions_total Predictions by label and "
                "detected language.",
                "# TYPE bartascope_predictions_total counter",
            ]
            for (label, lang), n in sorted(self.predictions_total.items()):
                lines.append(
                    f'bartascope_predictions_total{{label="{label}",'
                    f'language="{lang}"}} {n}')

            lines += [
                "# HELP bartascope_errors_total Errors by endpoint and kind.",
                "# TYPE bartascope_errors_total counter",
            ]
            for (endpoint, kind), n in sorted(self.errors_total.items()):
                lines.append(
                    f'bartascope_errors_total{{endpoint="{endpoint}",'
                    f'kind="{kind}"}} {n}')

            lines += [
                "# HELP bartascope_shed_total Requests refused under load, "
                "by endpoint and reason.",
                "# TYPE bartascope_shed_total counter",
            ]
            for (endpoint, reason), n in sorted(self.shed_total.items()):
                lines.append(
                    f'bartascope_shed_total{{endpoint="{endpoint}",'
                    f'reason="{reason}"}} {n}')

            lines += [
                "# HELP bartascope_inference_seconds Tokenisation plus ONNX "
                "Runtime execution.",
                "# TYPE bartascope_inference_seconds histogram",
            ]
            cumulative = 0
            for bound in LATENCY_BUCKETS:
                cumulative = self.latency_buckets[bound]
                lines.append(
                    f'bartascope_inference_seconds_bucket{{le="{bound}"}} '
                    f'{cumulative}')
            lines.append(
                f'bartascope_inference_seconds_bucket{{le="+Inf"}} '
                f'{self.latency_count}')
            lines.append(
                f"bartascope_inference_seconds_sum {self.latency_sum:.6f}")
            lines.append(
                f"bartascope_inference_seconds_count {self.latency_count}")
            return "\n".join(lines) + "\n"


metrics = Metrics()
