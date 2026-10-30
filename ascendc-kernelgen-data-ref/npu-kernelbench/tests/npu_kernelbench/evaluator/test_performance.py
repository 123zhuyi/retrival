"""Tests for NPU timing via torch_npu profiler."""

import os

import pytest

def _npu_available() -> bool:
    if os.environ.get("NPU_KERNELBENCH_DISABLE_NPU_TESTS"):
        return False
    try:
        import torch
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except Exception:
        return False


HAS_NPU = _npu_available()

from npu_kernelbench.evaluator.performance import TimingResult, parse_kernel_durations


class TestTimingResult:
    def test_creation(self):
        tr = TimingResult(
            mean_ms=1.5, durations=[1.0, 1.2, 1.5, 1.8, 2.0]
        )
        assert tr.mean_ms == 1.5
        assert len(tr.durations) == 5


class TestParseKernelDurations:
    def test_step_trace_time_csv(self, tmp_path):
        """Reads Computing from step_trace_time.csv, divides by iterations."""
        csv = tmp_path / "step_trace_time.csv"
        csv.write_text(
            "Device_id,Step,Computing,Communication(Not Overlapped),Overlapped,"
            "Communication,Free,Stage,Bubble,"
            "Communication(Not Overlapped and Exclude Receive),Preparing\n"
            "-1,,20600,0,0,0,1716.0,1736.75,0,0,322.5\n"
        )
        mean_ms = parse_kernel_durations(str(tmp_path), iterations=10)
        # 20600 µs / 10 iterations / 1000 = 2.06 ms
        assert mean_ms == pytest.approx(2.06)

    def test_kernel_details_fallback(self, tmp_path):
        """Falls back to kernel_details.csv: sum / iterations."""
        csv = tmp_path / "kernel_details.csv"
        csv.write_text(
            "Name,Start,Duration\n"
            "kernel_A,100,1500\n"
            "kernel_B,200,2000\n"
            "kernel_C,300,500\n"
        )
        # default iterations=1: total 4000µs / 1 / 1000 = 4.0 ms
        assert parse_kernel_durations(str(tmp_path)) == pytest.approx(4.0)
        # with iterations=4: 4000µs / 4 / 1000 = 1.0 ms
        assert parse_kernel_durations(str(tmp_path), iterations=4) == pytest.approx(1.0)

    def test_empty_csv(self, tmp_path):
        csv = tmp_path / "kernel_details.csv"
        csv.write_text("")
        assert parse_kernel_durations(str(tmp_path)) is None

    def test_missing_file(self, tmp_path):
        assert parse_kernel_durations(str(tmp_path / "nonexistent")) is None


@pytest.mark.skipif(not HAS_NPU, reason="NPU hardware required")
class TestProfileKernel:
    def test_profiles_simple_kernel(self):
        import torch

        from npu_kernelbench.evaluator.performance import profile_kernel

        def dummy_kernel(x):
            return x + 1

        x = torch.randn(100, 100, device="npu:0")
        result = profile_kernel(
            dummy_kernel, (x,), npu_id=0, warmup_runs=3, iterations=5
        )
        assert result.mean_ms > 0
        assert len(result.durations) >= 1


def test_event_based_timing_fallback():
    """_event_based_timing should return expected number of durations."""
    from npu_kernelbench.evaluator.performance import _event_based_timing

    class FakeKernel:
        def __call__(self, *args):
            pass

    if not HAS_NPU:
        pytest.skip("NPU hardware required")

    import torch

    device = torch.device("npu:0")
    durations = _event_based_timing(FakeKernel(), (), device, iterations=5)
    assert len(durations) == 5
