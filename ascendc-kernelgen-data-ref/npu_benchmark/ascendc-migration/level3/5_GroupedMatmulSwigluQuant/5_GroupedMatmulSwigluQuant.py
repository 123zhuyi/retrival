import torch
import torch_npu
import json
import os


class Model(torch.nn.Module):
    r"""
    torch_npu.npu_grouped_matmul + npu_swiglu + npu_dynamic_quant

    Fused grouped matmul + SwiGLU activation + dynamic quantization.

    Performs:
        1. npu_grouped_matmul(x_list=[x], weight_list=[weight[i] for i in range(e)],
           scale=weight_scale, group_list=group_list, group_type=0, group_list_type=0,
           split_item=2, output_dtype=torch.float16)
        2. npu_swiglu(matmul_out, dim=-1)
        3. npu_dynamic_quant(swiglu_out, dst_type=torch.int8)

    x: int8 [M, K]
    weight: int8 [e, K, N]
    weight_scale: float32 [e, N]
    x_scale: float32 [M]
    group_list: cumsum int list (length = number of groups)

    Returns: (out int8 [M, N//2], q_scale float [M])

    Example::
        >>> x = torch.randint(-128, 127, (1024, 7168), dtype=torch.int8).npu()
        >>> weight = torch.randint(-128, 127, (2, 7168, 4096), dtype=torch.int8).npu()
        >>> weight_scale = torch.randn(2, 4096, dtype=torch.float32).npu()
        >>> x_scale = torch.randn(1024, dtype=torch.float32).npu()
        >>> group_list = [1024]
        >>> out, q_scale = torch_npu.npu_grouped_matmul(
        ...     [x], [weight[0], weight[1]], scale=[weight_scale[0], weight_scale[1]],
        ...     group_list=group_list, group_type=0, group_list_type=0,
        ...     split_item=2, output_dtype=torch.float16)
        >>> swiglu_out = torch_npu.npu_swiglu(out, dim=-1)
        >>> q_out, qs = torch_npu.npu_dynamic_quant(swiglu_out, dst_type=torch.int8)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                weight_scale: torch.Tensor, x_scale: torch.Tensor,
                group_list: list) -> tuple:
        e = weight.shape[0]
        x_list = [x.to(torch.float16)]
        w_list = [weight[i].to(torch.float16) for i in range(e)]
        ws_list = [weight_scale[i] for i in range(e)]
        matmul_out = torch_npu.npu_grouped_matmul(
            x_list, w_list, scale=ws_list, group_list=group_list,
            group_type=0, group_list_type=0, split_item=2,
            output_dtype=torch.float16,
        )
        swiglu_out = torch_npu.npu_swiglu(matmul_out, dim=-1)
        q_out, q_scale = torch_npu.npu_dynamic_quant(swiglu_out, dst_type=torch.int8)
        return q_out, q_scale


def get_input_groups(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "5_GroupedMatmulSwigluQuant.json")

    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int32": torch.int32,
    }

    input_groups = []

    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        input_by_name = {inp["name"]: inp for inp in inputs}

        # x tensor
        x_info = input_by_name["x"]
        x_dtype_str = x_info["dtype"]
        x_dtype = dtype_map.get(x_dtype_str, torch.int8)
        x_shape = x_info["shape"]
        if idx % 2 == 0:
            raw = torch.empty(x_shape, dtype=torch.float32).uniform_(-5, 5)
            x = raw.to(x_dtype)
        else:
            raw = torch.empty(x_shape, dtype=torch.float32).uniform_(-128, 127)
            x = raw.to(x_dtype)

        # weight tensor
        w_info = input_by_name["weight"]
        w_dtype_str = w_info["dtype"]
        w_dtype = dtype_map.get(w_dtype_str, torch.int8)
        w_shape = w_info["shape"]
        if idx % 2 == 0:
            raw = torch.empty(w_shape, dtype=torch.float32).uniform_(-5, 5)
            weight = raw.to(w_dtype)
        else:
            raw = torch.empty(w_shape, dtype=torch.float32).uniform_(-128, 127)
            weight = raw.to(w_dtype)

        # weight_scale tensor
        ws_info = input_by_name["weight_scale"]
        ws_dtype = dtype_map.get(ws_info["dtype"], torch.float32)
        ws_shape = ws_info["shape"]
        if idx % 2 == 0:
            raw = torch.empty(ws_shape, dtype=torch.float32).uniform_(0.001, 1.0)
            weight_scale = raw.to(ws_dtype)
        else:
            raw = torch.empty(ws_shape, dtype=torch.float32).uniform_(-1.0, 1.0)
            weight_scale = raw.to(ws_dtype)

        # x_scale tensor
        xs_info = input_by_name["x_scale"]
        xs_dtype = dtype_map.get(xs_info["dtype"], torch.float32)
        xs_shape = xs_info["shape"]
        if idx % 2 == 0:
            raw = torch.empty(xs_shape, dtype=torch.float32).uniform_(0.001, 1.0)
            x_scale = raw.to(xs_dtype)
        else:
            raw = torch.empty(xs_shape, dtype=torch.float32).uniform_(-1.0, 1.0)
            x_scale = raw.to(xs_dtype)

        # group_list: derived from weight's first dimension (number of experts)
        # For npu_grouped_matmul with group_list_type=0 and split_item=2,
        # group_list = [M] where M is number of rows = x_shape[0]
        M = x_shape[0]
        group_list = [M]

        input_groups.append([x, weight, weight_scale, x_scale, group_list])

    return input_groups


def get_init_inputs():
    return []
