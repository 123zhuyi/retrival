"""Tests for reward hack detection."""

import hashlib
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

from npu_kernelbench.evaluator.anti_hack import (
    RewardHackResult,
    detect_output_caching,
    detect_monkey_patch,
    detect_clock_anomaly,
    detect_kernel_signature_mismatch,
    run_all_checks,
)
from npu_kernelbench.cli.config import BenchmarkConfig
from npu_kernelbench.data.definition import Definition, TensorSpec
from npu_kernelbench.data.solution import Solution, BuildSpec, SupportedLanguages, SupportedHardware, SourceFile
from npu_kernelbench.data.trace import EvaluationStatus
from npu_kernelbench.data.workload import RandomInput, ToleranceSpec, Workload
from npu_kernelbench.evaluator.evaluator import BaselineHandle, Evaluator, PerfDiagnostics, Performance


class TestRewardHackResult:
    def test_creation(self):
        r = RewardHackResult(detected=True, hack_type="test", severity="high")
        assert r.detected is True
        assert r.hack_type == "test"
        assert r.severity == "high"
        assert r.evidence == {}

    def test_evidence(self):
        r = RewardHackResult(
            detected=True, hack_type="test",
            evidence={"key": "val"}, severity="low",
        )
        assert r.evidence["key"] == "val"


class TestDetectMonkeyPatch:
    def test_hash_match(self):
        code = "def run(x): return x"
        h = hashlib.sha256(code.encode()).hexdigest()
        result = detect_monkey_patch(code, h)
        assert result.detected is False

    def test_hash_mismatch(self):
        code = "def run(x): return x"
        result = detect_monkey_patch(code, "badhash")
        assert result.detected is True
        assert result.hack_type == "monkey_patch"

    def test_empty_hash_skips(self):
        result = detect_monkey_patch("def run(x): return x", "")
        assert result.detected is False
        assert result.evidence.get("reason", "").startswith("reference_hash empty")


class TestDetectClockAnomaly:
    def test_stable_clock(self):
        durations = [1.0, 1.01, 0.99, 1.02, 1.0] * 3  # 15 samples
        result = detect_clock_anomaly(durations, max_cv=0.15)
        assert result.detected is False

    def test_unstable_clock(self):
        durations = [1.0, 0.5, 2.0, 0.3, 3.0] * 3  # 15 samples
        result = detect_clock_anomaly(durations, max_cv=0.5)
        assert result.detected is True
        assert result.hack_type == "clock_anomaly"

    def test_insufficient_samples(self):
        result = detect_clock_anomaly([1.0], max_cv=0.15)
        assert result.detected is False


class TestDetectKernelSignatureMismatch:
    def test_no_csv(self, tmp_path):
        result = detect_kernel_signature_mismatch(
            "my_kernel", str(tmp_path)
        )
        assert result.detected is False

    def test_csv_without_kernel(self, tmp_path):
        csv = tmp_path / "kernel_details.csv"
        csv.write_text("Name,Start,Duration\n")
        result = detect_kernel_signature_mismatch(
            "my_kernel", str(tmp_path)
        )
        assert result.detected is False

    def test_matching_kernel(self, tmp_path):
        csv = tmp_path / "kernel_details.csv"
        csv.write_text("Name,Start,Duration\nmy_kernel,100,1500\n")
        result = detect_kernel_signature_mismatch(
            "my_kernel", str(tmp_path)
        )
        assert result.detected is False

    def test_mismatched_kernel(self, tmp_path):
        csv = tmp_path / "kernel_details.csv"
        csv.write_text("Name,Start,Duration\nother_kernel,100,1500\n")
        result = detect_kernel_signature_mismatch(
            "my_kernel", str(tmp_path)
        )
        assert result.detected is True

    def test_empty_declared_kernel_skips(self, tmp_path):
        csv = tmp_path / "kernel_details.csv"
        csv.write_text("Name,Start,Duration\nother_kernel,100,1500\n")
        result = detect_kernel_signature_mismatch(
            "", str(tmp_path)
        )
        assert result.detected is False

    def test_substring_match(self, tmp_path):
        csv = tmp_path / "kernel_details.csv"
        csv.write_text("Name,Start,Duration\n__aicore__my_kernel_123,100,1500\n")
        result = detect_kernel_signature_mismatch(
            "my_kernel", str(tmp_path)
        )
        assert result.detected is False


class TestRunAllChecks:
    def test_runs_all_detections(self, tmp_path_factory):
        code = "def run(x): return x"
        h = hashlib.sha256(code.encode()).hexdigest()
        results = run_all_checks(
            kernel_fn=lambda x: x,
            input_generator=lambda seed: (seed,),
            reference_code=code,
            reference_hash=h,
            durations=[1.0] * 15,
            trace_dir=str(tmp_path_factory.mktemp("trace")),
            declared_kernel="my_kernel",
        )
        hack_types = {r.hack_type for r in results}
        assert "output_caching" in hack_types
        assert "monkey_patch" in hack_types
        assert "clock_anomaly" in hack_types
        assert "kernel_signature" in hack_types

    def test_runs_only_enabled_detections(self, tmp_path_factory):
        code = "def run(x): return x"
        h = hashlib.sha256(code.encode()).hexdigest()
        results = run_all_checks(
            kernel_fn=lambda x: x,
            input_generator=lambda seed: (seed,),
            reference_code=code,
            reference_hash=h,
            durations=[1.0] * 15,
            trace_dir=str(tmp_path_factory.mktemp("trace")),
            declared_kernel="my_kernel",
            enabled=["monkey_patch", "clock_anomaly"],
        )
        hack_types = {r.hack_type for r in results}
        assert hack_types == {"monkey_patch", "clock_anomaly"}

    def test_detects_clock_anomaly_when_unstable(self, tmp_path_factory):
        code = "def run(x): return x"
        h = hashlib.sha256(code.encode()).hexdigest()
        results = run_all_checks(
            kernel_fn=lambda x: x,
            input_generator=lambda seed: (seed,),
            reference_code=code,
            reference_hash=h,
            durations=[1.0, 0.1, 5.0, 0.1, 5.0] * 3,
            trace_dir=str(tmp_path_factory.mktemp("trace")),
        )
        assert any(
            r.hack_type == "clock_anomaly" and r.detected for r in results
        )


@pytest.mark.skipif(not HAS_NPU, reason="NPU hardware required")
class TestDetectOutputCaching:
    def test_normal_kernel(self):
        import torch

        def kernel(x):
            return x + 1

        def gen_inputs(seed):
            torch.manual_seed(seed)
            x = torch.randn(10, 10, device="npu:0")
            return (x,)

        result = detect_output_caching(kernel, gen_inputs, num_runs=3)
        # Normal kernel with different inputs produces different outputs
        assert result.detected is False
        assert result.hack_type == "output_caching"

class TestAntiHackEvaluatorFlow:
    """Verify Evaluator.evaluate() surfaces REWARD_HACK via clock anomaly."""

    def _make_definition(self) -> Definition:
        return Definition(
            name="dummy",
            axes={},
            inputs={"x": TensorSpec(shape=["4", "4"], dtype="float32")},
            outputs={"y": TensorSpec(shape=["4", "4"], dtype="float32")},
            reference="def run(x): return x",
        )

    def _make_workload(self) -> Workload:
        return Workload(
            uuid="test-workload-001",
            inputs={
                "x": RandomInput(dtype="float32", shape=[4, 4]),
            },
            tolerance=ToleranceSpec(max_atol=1e-2, max_rtol=1e-2),
        )

    def _make_baseline(self, definition: Definition, workload: Workload) -> BaselineHandle:
        import torch
        x = torch.randn(4, 4, dtype=torch.float32)
        return BaselineHandle(
            definition_name=definition.name,
            device="cpu",
            workload_uuid=workload.uuid,
            inputs=[x],
            outputs={"y": x.clone()},
        )

    def test_evaluator_returns_reward_hack_on_clock_anomaly(self):
        import torch

        definition = self._make_definition()
        workload = self._make_workload()
        baseline = self._make_baseline(definition, workload)

        class ClockAnomalyEvaluator(Evaluator):
            def can_evaluate(self, definition: Definition) -> bool:
                return True

            def build_baseline(
                self,
                definition: Definition,
                workload: Workload,
                device: str,
                config: BenchmarkConfig,
            ) -> BaselineHandle:
                return baseline

            def check_correctness(
                self,
                definition: Definition,
                workload: Workload,
                baseline: BaselineHandle,
                solution_outputs: dict[str, torch.Tensor],
            ) -> tuple:
                from npu_kernelbench.data.trace import Correctness
                return Correctness(max_absolute_error=0.0, max_relative_error=0.0), True

            def eval_performance(
                self,
                solution_fn,
                inputs: list,
                definition: Definition,
                config: BenchmarkConfig,
            ) -> tuple[Performance, PerfDiagnostics]:
                # Provide highly variable durations to trigger clock anomaly (CV > 1.0).
                durations = [0.001] * 10 + [100.0] * 10
                return Performance(latency_ms=50.0), PerfDiagnostics(
                    durations=durations,
                    trace_dir="",
                )

        def solution_fn(x):
            return x.clone()

        solution = Solution(
            name="dummy_solution",
            definition=definition.name,
            author="test",
            spec=BuildSpec(
                languages=[SupportedLanguages.PYTHON],
                target_hardware=[SupportedHardware.LOCAL],
                entry_point="main.py::run",
            ),
            sources=[SourceFile(path="main.py", content="def run(x): return x")],
        )

        evaluator = ClockAnomalyEvaluator()
        config = BenchmarkConfig(enable_anti_hack=["clock_anomaly"])
        trace = evaluator.evaluate(
            definition=definition,
            workload=workload,
            solution_fn=solution_fn,
            baseline=baseline,
            config=config,
            solution=solution,
        )

        assert trace.evaluation.status == EvaluationStatus.REWARD_HACK
        assert "clock_anomaly" in trace.evaluation.log.lower()

    def test_evaluator_passes_with_stable_clock(self):
        import torch

        definition = self._make_definition()
        workload = self._make_workload()
        baseline = self._make_baseline(definition, workload)

        class StableClockEvaluator(Evaluator):
            def can_evaluate(self, definition: Definition) -> bool:
                return True

            def build_baseline(
                self,
                definition: Definition,
                workload: Workload,
                device: str,
                config: BenchmarkConfig,
            ) -> BaselineHandle:
                return baseline

            def check_correctness(
                self,
                definition: Definition,
                workload: Workload,
                baseline: BaselineHandle,
                solution_outputs: dict[str, torch.Tensor],
            ) -> tuple:
                from npu_kernelbench.data.trace import Correctness
                return Correctness(max_absolute_error=0.0, max_relative_error=0.0), True

            def eval_performance(
                self,
                solution_fn,
                inputs: list,
                definition: Definition,
                config: BenchmarkConfig,
            ) -> tuple[Performance, PerfDiagnostics]:
                durations = [1.0] * 15
                return Performance(latency_ms=1.0), PerfDiagnostics(
                    durations=durations,
                    trace_dir="",
                )

        def solution_fn(x):
            return x.clone()

        solution = Solution(
            name="dummy_solution",
            definition=definition.name,
            author="test",
            spec=BuildSpec(
                languages=[SupportedLanguages.PYTHON],
                target_hardware=[SupportedHardware.LOCAL],
                entry_point="main.py::run",
            ),
            sources=[SourceFile(path="main.py", content="def run(x): return x")],
        )

        evaluator = StableClockEvaluator()
        config = BenchmarkConfig()
        trace = evaluator.evaluate(
            definition=definition,
            workload=workload,
            solution_fn=solution_fn,
            baseline=baseline,
            config=config,
            solution=solution,
        )

        assert trace.evaluation.status == EvaluationStatus.PASSED
        assert trace.evaluation.performance is not None
        assert trace.evaluation.correctness is not None
