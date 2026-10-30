"""Tests for BenchmarkConfig."""
import pytest
from pydantic import ValidationError
from npu_kernelbench.cli.config import BenchmarkConfig


class TestBenchmarkConfig:
    def test_default_values(self):
        cfg = BenchmarkConfig()
        assert cfg.warmup_runs == 5
        assert cfg.iterations == 10
        assert cfg.timeout == 600
        assert cfg.compile_timeout == 120
        assert cfg.benchmark_reference is True
        assert cfg.npu_id == 0

    def test_custom_values(self):
        cfg = BenchmarkConfig(
            warmup_runs=5, iterations=100, timeout=300, npu_id=1
        )
        assert cfg.warmup_runs == 5
        assert cfg.iterations == 100

    def test_perf_reference_source_default(self):
        cfg = BenchmarkConfig()
        assert cfg.perf_reference_source == "measured"

    @pytest.mark.parametrize("source", ["measured", "baseline"])
    def test_perf_reference_source_valid(self, source):
        cfg = BenchmarkConfig(perf_reference_source=source)
        assert cfg.perf_reference_source == source

    def test_perf_reference_source_invalid(self):
        with pytest.raises(ValidationError):
            BenchmarkConfig(perf_reference_source="invalid")

    @pytest.mark.parametrize("field,value", [
        ("warmup_runs", -1),
        ("iterations", 0),
        ("iterations", -1),
        ("timeout", 0),
        ("compile_timeout", 0),
    ])
    def test_invalid_values_raise_error(self, field, value):
        with pytest.raises(ValidationError):
            BenchmarkConfig(**{field: value})
