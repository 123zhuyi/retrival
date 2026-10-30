"""Tests for the custom input generator registry."""

import pytest

from npu_kernelbench.data.definition import Definition, TensorSpec
from npu_kernelbench.data.workload import RandomInput, ScalarInput, Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.base import BaseInputGenerator
from npu_kernelbench.evaluator.generator import gen_inputs
from npu_kernelbench.evaluator.custominputs.default import DefaultInputGenerator
from npu_kernelbench.evaluator.custominputs.level1.iou import IOUInputGenerator
from npu_kernelbench.evaluator.custominputs.level2.top_k_top_p import (
    TopKTopPInputGenerator,
)
from npu_kernelbench.evaluator.custominputs.level3.grouped_matmul import (
    GroupedMatmulInputGenerator,
)
from npu_kernelbench.evaluator.custominputs.level4.ffn import FFNInputGenerator
from npu_kernelbench.evaluator.custominputs.level4.grouped_matmul_swiglu_quant import (
    GroupedMatmulSwigluQuantInputGenerator,
)


def _simple_definition(name: str) -> Definition:
    return Definition(
        name=name,
        axes={},
        inputs={
            "x": TensorSpec(description="input"),
            "y": TensorSpec(description="input"),
        },
        outputs={"z": TensorSpec(description="output")},
        reference="def run(x, y): return x + y",
    )


class TestInputGeneratorRegistry:
    def test_default_generator_is_last_resort(self):
        registry = get_input_generator_registry()
        definition = _simple_definition("unknown_op")
        generator = registry.resolve(definition)
        assert isinstance(generator, DefaultInputGenerator)

    def test_custom_generators_take_priority(self):
        registry = get_input_generator_registry()

        assert isinstance(registry.resolve(_simple_definition("IOU")), IOUInputGenerator)
        assert isinstance(
            registry.resolve(_simple_definition("TopKTopP")),
            TopKTopPInputGenerator,
        )
        assert isinstance(
            registry.resolve(_simple_definition("GroupedMatmul")),
            GroupedMatmulInputGenerator,
        )
        assert isinstance(
            registry.resolve(_simple_definition("GroupedMatmulSwigluQuant")),
            GroupedMatmulSwigluQuantInputGenerator,
        )
        assert isinstance(registry.resolve(_simple_definition("FFN")), FFNInputGenerator)

    def test_priority_order(self):
        from npu_kernelbench.evaluator.custominputs.registry import InputGeneratorRegistry

        registry = InputGeneratorRegistry()
        # Builtin default is registered; add low-priority custom generator first.
        registry.register(TopKTopPInputGenerator())

        class HighPriorityGenerator(BaseInputGenerator):
            names = ("TopKTopP",)
            priority = 100

            def generate(self, ctx):
                return []

        registry.register(HighPriorityGenerator())
        resolved = registry.resolve(_simple_definition("TopKTopP"))
        assert isinstance(resolved, HighPriorityGenerator)

    def test_unknown_definition_falls_back_to_default(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_simple_definition("totally_unknown"))
        assert isinstance(generator, DefaultInputGenerator)

    def test_generators_produce_ordered_outputs(self):
        definition = Definition(
            name="TopKTopP",
            axes={},
            inputs={
                "logits": TensorSpec(description="logits"),
                "p": TensorSpec(description="p"),
                "k": TensorSpec(description="k"),
            },
            outputs={"y": TensorSpec(description="output")},
            reference="def run(logits, p, k): return logits",
        )
        workload = Workload(
            uuid="topk-test",
            inputs={
                "logits": RandomInput(shape=[4, 16], dtype="float16"),
                "p": RandomInput(shape=[4], dtype="float16"),
                "k": RandomInput(shape=[4], dtype="int32"),
            },
        )
        result = gen_inputs(definition, workload, "cpu", 42)
        assert len(result) == 3
        assert result[0].shape == (4, 16)
        assert result[1].shape == (4,)
        assert result[2].shape == (4,)

    def test_default_generator_handles_scalar_and_random(self):
        definition = Definition(
            name="mixed",
            axes={},
            inputs={
                "x": TensorSpec(description="x"),
                "alpha": TensorSpec(description="alpha"),
            },
            outputs={"y": TensorSpec(description="output")},
            reference="def run(x, alpha): return x + alpha",
        )
        workload = Workload(
            uuid="mixed-test",
            inputs={
                "x": RandomInput(shape=[2, 3], dtype="float32"),
                "alpha": ScalarInput(value=2.0),
            },
        )
        from npu_kernelbench.evaluator.generator import gen_inputs

        result = gen_inputs(definition, workload, "cpu", 42)
        assert result[0].shape == (2, 3)
        assert result[1] == 2.0
