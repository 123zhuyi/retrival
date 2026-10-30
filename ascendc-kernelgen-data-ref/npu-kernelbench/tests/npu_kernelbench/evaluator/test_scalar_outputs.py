"""Evaluator coverage for Python scalar outputs and output-count errors."""

from __future__ import annotations

from npu_kernelbench.cli.config import BenchmarkConfig
from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.trace import EvaluationStatus
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.default import DefaultEvaluator
from npu_kernelbench.evaluator.evaluator import BaselineHandle


def _definition() -> Definition:
    return Definition(
        name="scalar_outputs",
        axes={},
        inputs={},
        outputs={"seed": {}, "offset": {}},
        reference="def run():\n    return 7, 8\n",
    )


def _evaluate(solution_fn):
    definition = _definition()
    workload = Workload(uuid="scalar-001", inputs={})
    baseline = BaselineHandle(
        definition_name=definition.name,
        device="cpu",
        workload_uuid=workload.uuid,
        inputs=[],
        outputs={"seed": 7, "offset": 8},
    )
    return DefaultEvaluator().evaluate(
        definition=definition,
        workload=workload,
        solution_fn=solution_fn,
        baseline=baseline,
        config=BenchmarkConfig(mode="correctness_only"),
    )


def test_matching_integer_scalars_pass():
    trace = _evaluate(lambda: (7, 8))

    assert trace.evaluation.status == EvaluationStatus.PASSED


def test_mismatched_integer_scalar_fails_numerically():
    trace = _evaluate(lambda: (7, 9))

    assert trace.evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL
    assert trace.evaluation.correctness.max_absolute_error == 1.0
    assert trace.evaluation.correctness.extra["output_name"] == "offset"


def test_scalar_kind_mismatch_fails_dtype_check():
    trace = _evaluate(lambda: (True, 8))

    assert trace.evaluation.status == EvaluationStatus.INCORRECT_DTYPE
    assert "bool" in trace.evaluation.log
    assert "int" in trace.evaluation.log


def test_output_count_mismatch_has_structured_shape_error():
    trace = _evaluate(lambda: (7,))

    assert trace.evaluation.status == EvaluationStatus.INCORRECT_SHAPE
    assert "Expected 2 outputs" in trace.evaluation.log
