"""
Metrics collection for La Corrigeuse.

Tracks request metrics (latency, errors) and business metrics
(grading operations, token usage) via structured logging.
"""

import time
import threading
from collections import defaultdict
from typing import Dict, List
from loguru import logger


class MetricsCollector:
    """
    Collect and aggregate metrics for observability.

    Thread-safe in-memory metrics storage. Metrics are logged
    periodically and reset on restart.
    """

    def __init__(self):
        # Request metrics
        self.request_latencies: List[float] = []
        self.request_counts: Dict[str, int] = defaultdict(int)  # status_code -> count
        self.request_errors: int = 0

        # Business metrics
        self.grading_operations: int = 0
        self.token_usage: Dict[str, int] = defaultdict(int)  # session_id -> tokens

        # Active sessions
        self.active_sessions: set = set()

        # Thread lock for thread safety
        self._lock = threading.Lock()

    def record_request(self, method: str, path: str, status_code: int, latency_ms: float):
        """Record a request with its latency and status."""
        with self._lock:
            self.request_latencies.append(latency_ms)
            self.request_counts[str(status_code)] += 1
            if status_code >= 400:
                self.request_errors += 1

    def record_grading_operation(self, session_id: str, tokens_used: int):
        """Record a grading operation with token usage."""
        with self._lock:
            self.grading_operations += 1
            self.token_usage[session_id] += tokens_used

    def record_active_session(self, session_id: str):
        """Record an active session."""
        with self._lock:
            self.active_sessions.add(session_id)

    def remove_active_session(self, session_id: str):
        """Remove a session from active tracking."""
        with self._lock:
            self.active_sessions.discard(session_id)

    def get_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles (p50, p95, p99)."""
        with self._lock:
            if not self.request_latencies:
                return {"p50": 0, "p95": 0, "p99": 0}

            sorted_latencies = sorted(self.request_latencies)
            count = len(sorted_latencies)

            return {
                "p50": sorted_latencies[int(count * 0.5)],
                "p95": sorted_latencies[int(count * 0.95)],
                "p99": sorted_latencies[int(count * 0.99)]
            }

    def get_error_rate(self) -> float:
        """Calculate error rate (errors / total requests)."""
        with self._lock:
            total = sum(self.request_counts.values())
            if total == 0:
                return 0.0
            return (self.request_errors / total) * 100

    def get_requests_per_minute(self) -> int:
        """Get requests per minute (simplified - returns total since start)."""
        with self._lock:
            return sum(self.request_counts.values())

    def get_business_metrics(self) -> Dict:
        """Get business metrics summary."""
        with self._lock:
            return {
                "grading_operations": self.grading_operations,
                "total_token_usage": sum(self.token_usage.values()),
                "active_sessions": len(self.active_sessions)
            }

    def log_metrics(self):
        """Log all metrics as structured JSON."""
        metrics = {
            "request_metrics": {
                "latency_percentiles": self.get_percentiles(),
                "error_rate": self.get_error_rate(),
                "total_requests": self.get_requests_per_minute(),
                "status_codes": dict(self.request_counts)
            },
            "business_metrics": self.get_business_metrics()
        }

        logger.info("Metrics snapshot", extra={"metrics": metrics})

        return metrics


# Global singleton
_collector: MetricsCollector = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector singleton."""
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = MetricsCollector()
    return _collector
