"""Migration parity checks for level3 benchmark data."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload


REPO_ROOT = Path(__file__).resolve().parents[5]
OLD_DIR = REPO_ROOT / "npu_benchmark" / "level3"
NEW_DIR = REPO_ROOT / "npu-kernelbench" / "data" / "kernel_generator" / "level3"
CONV_OLD_FILES = {
    "ConvStandard1d": OLD_DIR / "6_ConvStandard1d.py",
    "ConvStandard2d": OLD_DIR / "7_ConvStandard2d.py",
    "ConvStandard3d": OLD_DIR / "8_ConvStandard3d.py",
    "ConvDepthwise2d": OLD_DIR / "9_ConvDepthwise2d.py",
    "ConvTranspose2d": OLD_DIR / "10_ConvTranspose2d.py",
}
MODEL_DOCSTRING_OLD_FILES = {
    "GroupedMatmul": OLD_DIR / "2_GroupedMatmul.py",
    **CONV_OLD_FILES,
}


def _old_workloads():
    out = {}
    for path in OLD_DIR.glob("*.json"):
        name = path.stem.split("_", 1)[1]
        out[name] = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    return out


def _new_workloads(name: str):
    path = NEW_DIR / name / "workload.jsonl"
    return [
        Workload(**json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _definition(name: str):
    return Definition(
        **json.loads((NEW_DIR / name / "definition.json").read_text())
    )


def _old_model_docstring(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            return ast.get_docstring(node)
    raise AssertionError(f"Model class not found in {path}")


def _expected_new_type(old_type: str) -> str:
    return {
        "tensor": "random",
        "attr": "scalar",
    }[old_type]


def test_level3_operator_set_matches_old_benchmark():
    old_names = set(_old_workloads())
    new_names = {p.name for p in NEW_DIR.iterdir() if p.is_dir()}
    assert new_names == old_names


def test_level3_workloads_match_old_json_shapes_dtypes_and_attrs():
    for name, old_cases in _old_workloads().items():
        new_cases = _new_workloads(name)
        assert len(new_cases) == len(old_cases), name

        for case_index, (old_case, new_case) in enumerate(
            zip(old_cases, new_cases)
        ):
            old_inputs = old_case["inputs"]
            new_inputs = new_case.inputs
            assert list(new_inputs) == [old["name"] for old in old_inputs]

            for old_input in old_inputs:
                new_input = new_inputs[old_input["name"]]
                assert new_input.type == _expected_new_type(old_input["type"])

                if old_input["type"] == "tensor":
                    assert new_input.shape == old_input["shape"]
                    assert new_input.dtype == old_input["dtype"]
                    if old_input["dtype"].startswith(("float", "bfloat")):
                        expected_distribution = (
                            "normal" if case_index % 2 == 0 else "uniform"
                        )
                        assert new_input.distribution.value == expected_distribution
                        if expected_distribution == "normal":
                            assert new_input.mu_low == -100
                            assert new_input.mu_high == 100
                            assert new_input.sigma_low == 1
                            assert new_input.sigma_high == 25
                        else:
                            assert new_input.low == -5
                            assert new_input.high == 5
                else:
                    assert new_input.value == old_input.get("value")


def test_level3_descriptions_match_old_model_docstrings():
    for name, old_path in MODEL_DOCSTRING_OLD_FILES.items():
        assert _definition(name).description == _old_model_docstring(old_path)


def test_level3_conv_references_preserve_old_hf32_seed_and_device_setup():
    for name in CONV_OLD_FILES:
        reference = _definition(name).reference
        assert "import torch_npu" in reference
        assert "torch_npu.npu.conv.allow_hf32 = False" in reference
        assert "torch.manual_seed(hash(key) & 0xFFFFFFFF)" in reference
        assert "conv = conv.to(x.device)" in reference


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_level3_gen_inputs_matches_sample_workload_specs():
    from npu_kernelbench.evaluator.generator import gen_inputs

    for name in sorted(_old_workloads()):
        definition = _definition(name)
        workload = _new_workloads(name)[0]
        result = gen_inputs(definition, workload, "cpu", 42)

        assert len(result) == len(definition.inputs), name
        for value, input_name in zip(result, definition.inputs):
            spec = workload.inputs[input_name]
            if spec.type == "random":
                assert isinstance(value, torch.Tensor), (name, input_name)
                assert list(value.shape) == spec.shape, (name, input_name)
                assert str(value.dtype).replace("torch.", "") == spec.dtype
            elif spec.type == "scalar":
                assert value == spec.value, (name, input_name)
