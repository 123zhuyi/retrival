"""Tests for EvaluatorRegistry."""

import pytest
from npu_kernelbench.evaluator.registry import EvaluatorRegistry
from npu_kernelbench.evaluator.evaluator import Evaluator
from npu_kernelbench.evaluator.default import DefaultEvaluator
from npu_kernelbench.data.definition import Definition, TensorSpec, AxisConst


@pytest.fixture
def simple_definition():
    return Definition(
        name="test_gemm",
        op_type="gemm",
        axes={
            "m": AxisConst(type="const", value=16),
            "n": AxisConst(type="const", value=16),
            "k": AxisConst(type="const", value=16),
        },
        inputs={
            "a": TensorSpec(shape=["m", "k"], dtype="float32"),
            "b": TensorSpec(shape=["k", "n"], dtype="float32"),
        },
        outputs={
            "c": TensorSpec(shape=["m", "n"], dtype="float32"),
        },
        reference="def run(a, b):\n    return a @ b\n",
    )


class TestEvaluatorRegistry:
    def test_resolve_returns_default_for_gemm(self, simple_definition):
        registry = EvaluatorRegistry()
        ev = registry.resolve(simple_definition)
        assert isinstance(ev, DefaultEvaluator)

    def test_custom_evaluator_takes_priority(self, simple_definition):
        registry = EvaluatorRegistry()

        class CustomEval(Evaluator):
            def can_evaluate(self, d): return d.op_type == "gemm"
            def build_baseline(self, d, w, dev, cfg): pass
            def check_correctness(self, d, w, b, s): return (None, True)

        registry.register(CustomEval())
        ev = registry.resolve(simple_definition)
        assert isinstance(ev, CustomEval)

    def test_missing_evaluator_raises(self):
        registry = EvaluatorRegistry()
        registry._evaluators = []
        definition = Definition(
            name="test",
            op_type="unknown_type",
            axes={},
            inputs={},
            outputs={},
            reference="def run():\n    return None\n",
        )
        with pytest.raises(RuntimeError, match="No evaluator registered"):
            registry.resolve(definition)

    def test_get_registry_returns_singleton(self):
        from npu_kernelbench.evaluator.registry import get_registry
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
