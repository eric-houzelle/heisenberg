import pytest
from heisenberg.core.metrics import metrics

def test_metrics_recording():
    metrics.record_latency("test_op", 100.0)
    assert "test_op" in metrics.latencies
    assert metrics.latencies["test_op"][-1] == 100.0
