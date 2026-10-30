import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs Grouped Matmul + SwiGLU + Quant computation using NPU ops.
    Uses npu_grouped_matmul for grouped matmul, npu_swiglu for activation,
    and npu_dynamic_quant for quantization.

    PyTorch native implementation (aligned with atk reference & aclnnSwigluGatedMlp):
    def _gmm_swiglu_quant(x, weight, weight_scale, x_scale, m):
        # Step 1: int32 matmul to avoid int8 overflow
        c_temp1 = torch.matmul(x.to(torch.int32), weight.to(torch.int32))
        c_temp1 = c_temp1.to(torch.float32)
        # Step 2: apply per-channel (weight) and per-token (x) scales
        c_temp2 = c_temp1 * weight_scale                    # [t, n]
        c_temp3 = c_temp2 * x_scale.reshape(m, 1)           # [t, n]
        # Step 3: SwiGLU activation — silu(gate) * up (Ascend standard)
        gate, up = c_temp3.chunk(2, dim=-1)
        swiglu_out = gate * torch.sigmoid(gate) * up        # = silu(gate) * up
        # Step 4: dynamic quantize to int8 (per-token)
        max_abs = torch.max(torch.abs(swiglu_out), dim=-1, keepdim=True).values
        quant_scale = 127.0 / max_abs
        q_out = torch.round(swiglu_out * quant_scale).clamp(-128, 127).to(torch.int8)
        dequant_scale = (1.0 / quant_scale).squeeze(-1)
        return q_out, dequant_scale

    def process_groups(x, weight_list, weight_scale_list, x_scale, group_list):
        m = x.shape[0]
        outputs = []
        start = 0
        prev = 0
        for i, curr in enumerate(group_list.tolist()):
            count = curr - prev
            prev = curr
            if count > 0:
                q_out, q_scale = _gmm_swiglu_quant(
                    x[start:start + count],
                    weight_list[i],
                    weight_scale_list[i],
                    x_scale[start:start + count],
                    count
                )
                outputs.append((q_out, q_scale))
            start += count
        if outputs:
            output = torch.cat([o[0] for o in outputs], dim=0)
            output_scale = torch.cat([o[1] for o in outputs], dim=0)
        else:
            n = weight_list[0].shape[1]
            output = torch.empty(m, n // 2, dtype=torch.int8)
            output_scale = torch.empty(m, dtype=torch.float32)
        return output, output_scale
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, weight: list, weight_scale: list,
                x_scale: torch.Tensor, group_list: torch.Tensor):
        """
        Performs grouped matmul + SwiGLU + quant using NPU ops.
        ...
        """
        import torch_npu
        def _to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, list):
                return [_to_device(item, device) for item in obj]
            return obj

        device = x.device
        weight = _to_device(weight, device)
        weight_scale = _to_device(weight_scale, device)
        x_scale = _to_device(x_scale, device)
        group_list = _to_device(group_list, device)

        x_list = [x]
        w_list = weight
        ws_list = weight_scale

        matmul_out = torch_npu.npu_grouped_matmul(
            x_list, w_list, scale=ws_list, group_list=group_list,
            group_type=0, group_list_type=0, split_item=2, output_dtype=torch.float16)

        # FIX: npu_grouped_matmul with split_item=2 returns list[Tensor], must concat before npu_swiglu
        if isinstance(matmul_out, (list, tuple)):
            matmul_out = torch.cat(matmul_out, dim=0)

        # FIX: apply per-token x_scale dequant (npu_grouped_matmul only applied weight_scale)
        matmul_out = (matmul_out * x_scale.unsqueeze(-1)).to(matmul_out.dtype)

        swiglu_out = torch_npu.npu_swiglu(matmul_out, dim=-1)

        q_out, q_scale = torch_npu.npu_dynamic_quant(swiglu_out, dst_type=torch.int8)
        return q_out, q_scale


def generate_non_decreasing_sequence(length, upper_limit, seed: int):
    gen = torch.Generator()
    gen.manual_seed(seed)
    if length == 1:
        return torch.tensor([upper_limit], dtype=torch.int64)
    random_increments = torch.randint(1, 128, (length,), generator=gen)
    sequence = torch.cumsum(random_increments, dim=0)
    scale_factor = upper_limit / sequence[-1]
    sequence = (sequence * scale_factor).to(torch.int64)
    sequence[-1] = upper_limit
    for i in range(length - 2, -1, -1):
        if sequence[i] > sequence[i + 1]:
            sequence[i] = sequence[i + 1]
    return sequence


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "57_GroupedMatmulSwigluQuant.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    # 使用真实 int8 量化 scale 量级的小正数，避免 fp16 融合算子链路溢出产生 inf/NaN
    def small_positive_scale(shape, dtype, gen):
        return torch.rand(shape, dtype=dtype, generator=gen) * 0.049 + 0.001

    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]

        name_map = {inp["name"]: inp for inp in inputs}

        if "group_list" not in name_map:
            continue

        x_info = name_map["x"]
        weight_info = name_map["weight"]
        weight_scale_info = name_map["weight_scale"]
        x_scale_info = name_map["x_scale"]

        m = x_info["shape"][0]
        e = weight_info["shape"][0]

        gen = torch.Generator()
        gen.manual_seed(idx + 42)

        x = torch.randint(-5, 5, x_info["shape"], dtype=torch.int8, generator=gen)
        weight = [
            torch.randint(-5, 5, weight_info["shape"][1:], dtype=torch.int8, generator=gen)
            for _ in range(e)
        ]
        weight_scale = [
            small_positive_scale(weight_scale_info["shape"][1:], torch.float32, gen)
            for _ in range(e)
        ]

        x_scale = small_positive_scale(x_scale_info["shape"], torch.float32, gen)

        group_list = generate_non_decreasing_sequence(e, m, seed=idx + 42)

        input_groups.append([x, weight, weight_scale, x_scale, group_list])
    return input_groups


def get_init_inputs():
    return []