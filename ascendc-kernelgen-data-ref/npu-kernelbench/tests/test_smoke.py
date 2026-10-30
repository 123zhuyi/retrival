"""Smoke test — verifies the full data pipeline loads correctly."""

from pathlib import Path

from npu_kernelbench.data import (
    Definition,
    Workload,
    load_json_file,
    load_jsonl_file,
)


def test_load_sample_definition():
    path = Path(__file__).parent / "samples" / "gelu" / "definition.json"
    d = load_json_file(Definition, path)
    assert d.name == "gelu"
    assert len(d.inputs) == 1
    assert len(d.outputs) == 1
    assert "run" in d.reference


def test_load_sample_workloads():
    path = Path(__file__).parent / "samples" / "gelu" / "workload.jsonl"
    wls = load_jsonl_file(Workload, path)
    assert len(wls) == 2
    assert wls[0].uuid == "gelu-001"
    assert wls[1].uuid == "gelu-002"
    assert wls[0].axes == {"N": 32, "D": 128}


def test_load_matmul_pytorch_definition():
    path = Path(__file__).parent / "samples" / "matmul_pytorch" / "definition.json"
    d = load_json_file(Definition, path)
    assert d.name == "matmul"
    assert len(d.inputs) == 2
    assert len(d.outputs) == 1


def test_load_matmul_pytorch_workloads():
    path = Path(__file__).parent / "samples" / "matmul_pytorch" / "workload.jsonl"
    wls = load_jsonl_file(Workload, path)
    assert len(wls) == 3


def test_load_vector_add_ascendc_definition():
    path = Path(__file__).parent / "samples" / "vector_add_ascendc" / "definition.json"
    d = load_json_file(Definition, path)
    assert d.name == "vector_add"


def test_load_matmul_triton_definition():
    path = Path(__file__).parent / "samples" / "matmul_triton" / "definition.json"
    d = load_json_file(Definition, path)
    assert d.name == "matmul_triton"


def test_load_matmul_triton_solution():
    from npu_kernelbench.data import Solution
    path = Path(__file__).parent / "samples" / "matmul_triton" / "solution.json"
    sol = load_json_file(Solution, path)
    assert sol.name == "matmul_triton"
    assert "triton" in sol.spec.languages
