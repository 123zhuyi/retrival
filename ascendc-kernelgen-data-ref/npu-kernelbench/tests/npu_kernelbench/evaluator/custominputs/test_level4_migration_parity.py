"""Migration parity checks for the Level4 benchmark data."""

from __future__ import annotations

import ast
import builtins
import importlib.util
import io
import json
import math
from pathlib import Path

import pytest

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload


REPO_ROOT = Path(__file__).resolve().parents[5]
OLD_DIR = REPO_ROOT / "npu_benchmark" / "level4"
NEW_DIR = REPO_ROOT / "npu-kernelbench" / "data" / "kernel_generator" / "level4"
INPUT_ORDERS = {
    "FFN": [
        "x", "weight1", "weight2", "activation", "expert_tokens",
        "expert_tokens_index", "bias1", "bias2", "scale", "offset",
        "deq_scale1", "deq_scale2", "antiquant_scale1",
        "antiquant_scale2", "antiquant_offset1", "antiquant_offset2",
        "inner_precise", "output_dtype",
    ],
    "FluxTimestepGuidanceProjectionEmbedding": [
        "timestep", "pooled_projections", "freqs", "timestep_linear1_weight",
        "timestep_linear1_bias", "timestep_linear2_weight",
        "timestep_linear2_bias", "text_embedder_weight", "text_embedder_bias",
    ],
    "FusedInferAttentionScore": [
        "query", "key", "value", "pse_shift", "atten_mask",
        "actual_seq_lengths", "actual_seq_lengths_kv", "dequant_scale1",
        "quant_scale1", "dequant_scale2", "quant_scale2", "quant_offset2",
        "antiquant_scale", "antiquant_offset", "block_table",
        "query_padding_size", "kv_padding_size", "key_antiquant_scale",
        "key_antiquant_offset", "value_antiquant_scale",
        "value_antiquant_offset", "key_shared_prefix", "value_shared_prefix",
        "actual_shared_prefix_len", "query_rope", "key_rope",
        "key_rope_antiquant_scale", "num_heads", "scale", "pre_tokens",
        "next_tokens", "input_layout", "num_key_value_heads", "sparse_mode",
        "inner_precise", "block_size", "antiquant_mode", "softmax_lse_flag",
        "key_antiquant_mode", "value_antiquant_mode",
    ],
    "FusionAttention": [
        "query", "key", "value", "head_num", "input_layout", "pse",
        "padding_mask", "atten_mask", "scale", "keep_prob", "pre_tockens",
        "next_tockens", "inner_precise", "prefix", "actual_seq_qlen",
        "actual_seq_kvlen", "sparse_mode", "gen_mask_parallel", "sync",
        "softmax_layout", "sink",
    ],
    "GroupedMatmulSwigluQuant": [
        "x", "weight", "weight_scale", "x_scale", "group_list",
    ],
    "GroupedMatmulSwigluQuantV2": [
        "x", "weight", "weight_scale", "x_scale", "group_list",
        "smooth_scale", "weight_assist_matrix", "bias", "dequant_mode",
        "dequant_dtype", "quant_mode", "quant_dtype", "group_list_type",
        "tuning_config",
    ],
    "LightningIndexer": [
        "query", "key", "weights", "actual_seq_lengths_query",
        "actual_seq_lengths_key", "block_table", "layout_query", "layout_key",
        "sparse_count", "sparse_mode", "pre_tokens", "next_tokens",
        "return_value",
    ],
    "LSTM": [
        "x", "weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0",
        "h_0", "c_0", "batch_first", "dropout",
    ],
    "QuantMatmul": [
        "x1", "x2", "scale", "offset", "pertoken_scale", "bias",
        "output_dtype", "group_sizes",
    ],
    "QuantMatmulReduceSum": ["x1", "x2", "x1_scale", "x2_scale"],
    "SparseFlashAttention": [
        "query", "key", "value", "sparse_indices", "scale_value",
        "block_table", "actual_seq_lengths_query", "actual_seq_lengths_kv",
        "query_rope", "key_rope", "sparse_block_size", "layout_query",
        "layout_kv", "sparse_mode", "pre_tokens", "next_tokens",
        "attention_mode", "return_softmax_lse",
    ],
    "WeightQuantBatchmatmul": [
        "x", "weight", "antiquant_scale", "antiquant_offset", "quant_scale",
        "quant_offset", "bias", "antiquant_group_size", "inner_precise",
    ],
}


def _old_paths() -> dict[str, Path]:
    return {
        path.stem.split("_", 1)[1]: path
        for path in OLD_DIR.glob("*.json")
    }


def _json_lines(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _expected_spec(operator: str, old_input: dict) -> dict:
    name = old_input["name"]
    if old_input["type"] == "attr":
        return {"type": "scalar", "value": old_input.get("value")}
    if old_input.get("shape") is None:
        return {"type": "scalar", "value": None}
    if operator == "GroupedMatmulSwigluQuantV2" and name == "group_list":
        return {"type": "scalar", "value": old_input.get("value")}
    if operator == "GroupedMatmulSwigluQuant" and name == "group_list":
        return {"type": "scalar", "value": None}
    if (
        operator == "FusedInferAttentionScore"
        and name == "block_table"
        and "value" in old_input
    ):
        return {"type": "scalar", "value": old_input["value"]}

    dtype = old_input["dtype"]
    spec = {
        "type": "random",
        "shape": list(old_input["shape"]),
        "dtype": dtype,
    }
    if name == "atten_mask":
        spec["distribution"] = "ones"
    elif dtype.startswith(("float", "bfloat")):
        if operator == "LSTM":
            spec.update(
                distribution="normal", mu_low=-100, mu_high=100,
                sigma_low=1, sigma_high=25,
            )
        elif operator == "FFN":
            low, high = (
                (-0.15, 0.25)
                if dtype in ("float16", "bfloat16")
                else (-1.0, 1.0)
            )
            spec.update(distribution="uniform", low=low, high=high)
        elif operator == "FluxTimestepGuidanceProjectionEmbedding":
            spec.update(distribution="uniform", low=-0.5, high=0.5)
        elif operator == "WeightQuantBatchmatmul":
            spec.update(distribution="uniform", low=-5.0, high=5.0)
        elif operator == "GroupedMatmulSwigluQuant" and name in (
            "weight_scale", "x_scale",
        ):
            spec.update(distribution="uniform", low=0.001, high=0.05)
        elif operator == "QuantMatmul" and name in ("scale", "pertoken_scale"):
            spec.update(distribution="uniform", low=0.0, high=1.0)
        elif operator == "QuantMatmul" and name in ("offset", "bias"):
            spec.update(distribution="uniform", low=-5.0, high=5.0)
        else:
            spec["distribution"] = "mixed"

    ranges = {
        ("FFN", "x"): [-5, 5],
        ("FFN", "weight1"): [-5, 5],
        ("FFN", "weight2"): [-5, 5],
        ("FFN", "bias1"): [0, 5],
        ("FFN", "bias2"): [0, 5],
        ("GroupedMatmulSwigluQuant", "x"): [-5, 5],
        ("GroupedMatmulSwigluQuant", "weight"): [-5, 5],
        ("GroupedMatmulSwigluQuantV2", "x"): [-128, 127],
        ("GroupedMatmulSwigluQuantV2", "weight"): [-128, 127],
        ("QuantMatmul", "x1"): [-5, 5],
        ("QuantMatmul", "x2"): [-5, 5],
        ("QuantMatmul", "bias"): [-128, 127],
        ("QuantMatmulReduceSum", "x1"): [-5, 5],
        ("QuantMatmulReduceSum", "x2"): [-5, 5],
        ("WeightQuantBatchmatmul", "weight"): [-8, 8],
    }
    if (operator, name) in ranges and dtype.startswith("int"):
        spec["range"] = ranges[(operator, name)]
    return spec


def _expected_case(operator: str, old_case: dict) -> dict[str, dict]:
    old_by_name = {item["name"]: item for item in old_case["inputs"]}
    result = {
        name: _expected_spec(operator, item)
        for name, item in old_by_name.items()
    }
    scalar = lambda name, default: old_by_name.get(name, {}).get("value", default)

    if operator == "FusedInferAttentionScore":
        layout = scalar("input_layout", "BSH")
        num_heads = scalar("num_heads", 1)
        num_kv_heads = scalar("num_key_value_heads", 0)
        if layout == "BSH" and 0 < num_kv_heads < num_heads:
            query_shape = result["query"]["shape"]
            if query_shape[-1] % num_heads == 0:
                expected = num_kv_heads * (query_shape[-1] // num_heads)
                result["key"]["shape"][-1] = expected
                result["value"]["shape"][-1] = expected

    if operator == "SparseFlashAttention":
        layout = scalar("layout_kv", "BSND")
        block_table = old_by_name.get("block_table")
        sparse_block_size = scalar("sparse_block_size", 1)
        if layout == "PA_BSND" and block_table is not None:
            max_index = math.prod(block_table["shape"])
        elif sparse_block_size > 1:
            max_index = max(
                1, old_by_name["key"]["shape"][1] // sparse_block_size
            )
        else:
            max_index = max(1, old_by_name["key"]["shape"][1])
        result["sparse_indices"]["range"] = [0, max_index]
    return result


def _definition(operator: str) -> Definition:
    return Definition(
        **json.loads(
            (NEW_DIR / operator / "definition.json").read_text(encoding="utf-8")
        )
    )


def test_level4_operator_sets_and_workload_counts_match_old_benchmark():
    old_paths = _old_paths()
    assert set(old_paths) == {path.name for path in NEW_DIR.iterdir() if path.is_dir()}
    for operator, old_path in old_paths.items():
        assert len(_json_lines(old_path)) == len(
            _json_lines(NEW_DIR / operator / "workload.jsonl")
        )


def test_level4_workload_shapes_dtypes_attrs_and_distributions_match():
    for operator, old_path in _old_paths().items():
        old_cases = _json_lines(old_path)
        new_cases = _json_lines(NEW_DIR / operator / "workload.jsonl")
        for index, (old_case, new_case) in enumerate(zip(old_cases, new_cases), 1):
            assert new_case["uuid"] == f"{operator}-{index:03d}"
            assert new_case["inputs"] == _expected_case(operator, old_case)
            Workload(**new_case)


def test_level4_descriptions_are_the_old_model_docstrings():
    for operator, old_json_path in _old_paths().items():
        tree = ast.parse(old_json_path.with_suffix(".py").read_text(encoding="utf-8"))
        model = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Model"
        )
        assert _definition(operator).description == (ast.get_docstring(model) or "")


def test_level4_reference_signatures_follow_legacy_input_order():
    for operator, expected_order in INPUT_ORDERS.items():
        definition = _definition(operator)
        tree = ast.parse(definition.reference)
        run = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        assert list(definition.inputs) == expected_order
        assert [arg.arg for arg in run.args.args] == expected_order


def test_level4_references_preserve_latest_legacy_fixes():
    markers = {
        "FluxTimestepGuidanceProjectionEmbedding": [
            "timestep.to(torch.float32) * 1000.0", "t_emb.to(torch.float32) +",
        ],
        "FusedInferAttentionScore": ["effective_nkv", "num_key_value_heads=effective_nkv"],
        "FusionAttention": ["if keep_prob < 1.0:", "keep_prob = 1.0"],
        "GroupedMatmulSwigluQuant": ["w_list = weight", "torch.cat(matmul_out, dim=0)"],
        "GroupedMatmulSwigluQuantV2": ["_cast_list_to_nz", "npu_format_cast"],
        "LightningIndexer": ["torch.argsort", "torch.gather"],
        "LSTM": ["_LSTMS", "torch.manual_seed(hash(key) & 4294967295)", "lstm.eval()"],
        "QuantMatmulReduceSum": ["for b in range(mid.shape[0])", "acc = acc + mid[b]"],
    }
    for operator, expected_markers in markers.items():
        reference = _definition(operator).reference
        for marker in expected_markers:
            assert marker in reference, (operator, marker)


try:
    import torch
    import torch_npu  # noqa: F401

    HAS_TORCH_NPU = True
except ImportError:
    HAS_TORCH_NPU = False


def _case_cost(case: dict) -> int:
    return sum(
        math.prod(item.get("shape") or [])
        for item in case["inputs"]
        if item["type"] == "tensor" and item.get("shape") is not None
    )


def _load_old_module(operator: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"old_level4_{operator}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_same_structure(old_value, new_value, label: str) -> None:
    if isinstance(old_value, torch.Tensor):
        assert isinstance(new_value, torch.Tensor), label
        assert new_value.shape == old_value.shape, label
        assert new_value.dtype == old_value.dtype, label
    elif isinstance(old_value, list):
        assert isinstance(new_value, list), label
        assert len(new_value) == len(old_value), label
        for index, (old_item, new_item) in enumerate(zip(old_value, new_value)):
            _assert_same_structure(old_item, new_item, f"{label}[{index}]")
    else:
        assert new_value == old_value, label


@pytest.mark.skipif(not HAS_TORCH_NPU, reason="torch and torch_npu are required")
@pytest.mark.parametrize("operator", sorted(INPUT_ORDERS))
def test_level4_sampled_gen_inputs_match_legacy_shape_dtype_and_values(
    operator: str,
    monkeypatch,
):
    from npu_kernelbench.evaluator.generator import gen_inputs

    old_json_path = _old_paths()[operator]
    old_cases = _json_lines(old_json_path)
    case_index = min(range(len(old_cases)), key=lambda i: _case_cost(old_cases[i]))
    old_case = old_cases[case_index]
    old_module = _load_old_module(operator, old_json_path.with_suffix(".py"))
    real_open = builtins.open

    def open_one_case(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file).name == old_json_path.name:
            return io.StringIO(json.dumps(old_case) + "\n")
        return real_open(file, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", open_one_case)
        torch.manual_seed(42)
        old_values = old_module.get_input_groups()[0]

    definition = _definition(operator)
    workload = Workload(
        **_json_lines(NEW_DIR / operator / "workload.jsonl")[case_index]
    )
    new_values = gen_inputs(
        definition, workload, "cpu", 42, case_index=case_index
    )
    assert len(old_values) == len(new_values) == len(INPUT_ORDERS[operator])
    for name, old_value, new_value in zip(
        INPUT_ORDERS[operator], old_values, new_values
    ):
        _assert_same_structure(old_value, new_value, f"{operator}.{name}")


@pytest.mark.skipif(not HAS_TORCH_NPU, reason="torch and torch_npu are required")
@pytest.mark.parametrize("operator", sorted(INPUT_ORDERS))
def test_level4_sampled_gen_inputs_are_deterministic(operator: str):
    from npu_kernelbench.evaluator.generator import gen_inputs

    old_cases = _json_lines(_old_paths()[operator])
    case_index = min(range(len(old_cases)), key=lambda i: _case_cost(old_cases[i]))
    definition = _definition(operator)
    workload = Workload(
        **_json_lines(NEW_DIR / operator / "workload.jsonl")[case_index]
    )
    first = gen_inputs(definition, workload, "cpu", 123, case_index=case_index)
    second = gen_inputs(definition, workload, "cpu", 123, case_index=case_index)
    for name, first_value, second_value in zip(
        INPUT_ORDERS[operator], first, second
    ):
        if isinstance(first_value, torch.Tensor):
            assert torch.equal(first_value, second_value), (operator, name)
        elif isinstance(first_value, list) and first_value and isinstance(first_value[0], torch.Tensor):
            assert all(torch.equal(a, b) for a, b in zip(first_value, second_value))
        else:
            assert first_value == second_value, (operator, name)


@pytest.mark.skipif(not HAS_TORCH_NPU, reason="torch and torch_npu are required")
@pytest.mark.parametrize("operator", sorted(INPUT_ORDERS))
def test_level4_sampled_reference_runs_without_generated_source(operator: str):
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")

    from npu_kernelbench.evaluator.default import _run_reference
    from npu_kernelbench.evaluator.generator import gen_inputs

    old_cases = _json_lines(_old_paths()[operator])
    case_index = min(range(len(old_cases)), key=lambda i: _case_cost(old_cases[i]))
    definition = _definition(operator)
    workload = Workload(
        **_json_lines(NEW_DIR / operator / "workload.jsonl")[case_index]
    )
    device = torch.device("npu:0")
    inputs = gen_inputs(
        definition, workload, str(device), 42, case_index=case_index
    )
    outputs = _run_reference(
        definition.reference,
        inputs,
        list(definition.outputs),
        device=device,
    )
    assert list(outputs) == list(definition.outputs), operator
