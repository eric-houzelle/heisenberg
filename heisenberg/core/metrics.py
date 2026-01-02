import time
from dataclasses import dataclass, field
from typing import Dict, List
import logging

logger = logging.getLogger("metrics")

@dataclass
class MetricsRegistry:
    counters: Dict[str, int] = field(default_factory=dict)
    latencies: Dict[str, List[float]] = field(default_factory=dict)

    def increment(self, name: str, tags: Dict[str, str] = None):
        key = self._format_key(name, tags)
        self.counters[key] = self.counters.get(key, 0) + 1
        # In a real system, this might push to Prometheus/StatsD
        logger.debug(f"Metric inc: {key} = {self.counters[key]}")

    def record_latency(self, name: str, value_ms: float, tags: Dict[str, str] = None):
        key = self._format_key(name, tags)
        if key not in self.latencies:
            self.latencies[key] = []
        self.latencies[key].append(value_ms)
        logger.info(f"Metric latency: {key} = {value_ms}ms", extra={"latency_ms": value_ms, "metric": name})

    def _format_key(self, name: str, tags: Dict[str, str] = None) -> str:
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"

# Global instance
metrics = MetricsRegistry()
