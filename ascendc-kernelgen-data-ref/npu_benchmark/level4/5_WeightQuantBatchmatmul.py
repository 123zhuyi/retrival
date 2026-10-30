import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs weight-quantized batch matrix multiplication using NPU accelerated npu_weight_quant_batchmatmul.
    Supports pertensor, perchannel, and pergroup quantization for weight matrices.
    torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, antiquant_group_size, inner_precise) -> Tensor

    CPU 标杆实现（注释内）—— 来源: analysis/正确性_V2_CPU_修复/v2_golden_cpu.py 的 normal() 路径
    ----------------------------------------------------------------------------------------
    设计要点（与朴素 PyTorch 实现的关键差异，是对齐 NPU cube 行为的修复点）：

    1. dequant 域按 x.dtype 分流（V3 修复）：
       - fp16 输入：在 fp16 域做 (weight+offset)*scale，最后再 fp16→fp32（贴合 V2 仓上原行为）
       - bf16 输入：在 fp32 域做 (weight+offset)*scale，再 cast 到 bf16（模拟 bf16 表示精度）→ 回 fp32
         （对应 NPU cube L0C 行为，避免 bf16 加法尾数损失）

    2. matmul 用 fp32 split-K 累加，BLOCK_K=16（模拟 NPU cube 单次累加深度），不是一次性 torch.matmul。
       循环内每个 partial 都是 fp32，最后输出仍是 fp32。

    3. bias / quant_scale / quant_offset 可选；quant 路径有 clamp 到 [-256,256] / [-128,127] 的边界处理。

    4. 标杆假设输入 tensor 可在任意 device，内部统一 .cpu() 计算（CPU 标杆本质）。

    PyTorch native implementation of forward function
    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                antiquant_scale: torch.Tensor, antiquant_offset=None,
                quant_scale=None, quant_offset=None, bias=None,
                antiquant_group_size=0, inner_precise=0):
        low_dtype = x.dtype
        # 输入统一搬到 CPU（标杆本质是 CPU 计算）
        weight_t = weight.cpu()
        x_t = x.cpu()
        scale_t = antiquant_scale.cpu()
        off_t = antiquant_offset.cpu() if (antiquant_offset is not None and antiquant_offset.numel() > 0) else None

        # V3 修复：按 x.dtype 决定 dequant 域
        if low_dtype == torch.bfloat16:
            # bf16 路径：在 fp32 域 dequant（避免 bf16 加法尾数损失），最后截断到 bf16 再回 fp32
            weight_f32 = weight_t.to(torch.float32)
            scale_f32 = scale_t.to(torch.float32)
            off_f32 = off_t.to(torch.float32) if off_t is not None else None
            if antiquant_group_size:
                num_groups = scale_f32.shape[0]
                gs = antiquant_group_size
                kSize = x_t.shape[-1]
                scale_exp = scale_f32.view(num_groups, 1, -1).expand(num_groups, gs, -1).reshape(-1, scale_f32.shape[-1])[:kSize, :]
                if off_f32 is not None:
                    off_exp = off_f32.view(num_groups, 1, -1).expand(num_groups, gs, -1).reshape(-1, off_f32.shape[-1])[:kSize, :]
                    weight_t = (weight_f32 + off_exp) * scale_exp
                else:
                    weight_t = weight_f32 * scale_exp
            else:
                if off_f32 is not None:
                    weight_t = (weight_f32 + off_f32) * scale_f32
                else:
                    weight_t = weight_f32 * scale_f32
            # 截断到 bf16 模拟 bf16 表示精度，再回 fp32
            weight_t = weight_t.to(torch.bfloat16).to(torch.float32)
        else:
            # fp16 路径：在 low_dtype(fp16) 域 dequant（V2 仓上原行为）
            weight_t = weight_t.to(low_dtype)
            scale_t = scale_t.to(low_dtype)
            if off_t is not None:
                off_t = off_t.to(low_dtype)
            if antiquant_group_size:
                num_groups = scale_t.shape[0]
                gs = antiquant_group_size
                kSize = x_t.shape[-1]
                scale_exp = scale_t.view(num_groups, 1, -1).expand(num_groups, gs, -1).reshape(-1, scale_t.shape[-1])[:kSize, :]
                if off_t is not None:
                    off_exp = off_t.view(num_groups, 1, -1).expand(num_groups, gs, -1).reshape(-1, off_t.shape[-1])[:kSize, :]
                    weight_t = (weight_t + off_exp) * scale_exp
                else:
                    weight_t = weight_t * scale_exp
            else:
                if off_t is not None:
                    weight_t = (weight_t + off_t) * scale_t
                else:
                    weight_t = weight_t * scale_t
            # fp16 截断再回 fp32
            weight_t = weight_t.to(torch.float16).to(torch.float32)

        # matmul：fp32 split-K 累加，BLOCK_K=16 模拟 NPU cube 累加深度
        x_f32 = x_t.to(torch.float32)
        K = x_f32.shape[-1]
        BLOCK_K = 16
        output = None
        for k0 in range(0, K, BLOCK_K):
            k1 = min(k0 + BLOCK_K, K)
            partial = torch.matmul(x_f32[:, k0:k1], weight_t[k0:k1, :])
            output = partial if output is None else output + partial

        # bias
        if bias is not None and bias.numel() != 0:
            output = output + bias.cpu().to(torch.float32)
        # quant_scale：clamp 到 [-256, 256]，转 int16
        if quant_scale is not None and quant_scale.numel() != 0:
            output = torch.clamp(torch.round(output * quant_scale.cpu()), -256, 256).to(torch.int16)
        # quant_offset：再 clamp 到 [-128, 127]
        if quant_offset is not None and quant_offset.numel() != 0:
            output = torch.clamp(output + torch.clamp(torch.round(quant_offset.cpu()), -256, 256), -128, 127)

        # 标杆最终返回 low_dtype（与 NPU op 输出 dtype 对齐）
        return output.to(low_dtype)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                antiquant_scale: torch.Tensor, antiquant_offset=None,
                quant_scale=None, quant_offset=None, bias=None,
                antiquant_group_size=0, inner_precise=0):
        """
        Performs weight-quantized batch matmul on NPU.
        Args:
            x (Tensor): Left matrix, shape [M, K], dtype float16/bfloat16.
            weight (Tensor): Right matrix (weight), shape [K, N], dtype int8.
            antiquant_scale (Tensor): Dequantization scale for weight, dtype float16/bfloat16.
            antiquant_offset (Tensor, optional): Dequantization offset for weight, dtype float16/bfloat16.
            quant_scale (Tensor, optional): Output quantization scale, dtype float32.
            quant_offset (Tensor, optional): Output quantization offset, dtype float32.
            bias (Tensor, optional): Bias term, shape [1, N] or [N], dtype float16/float32.
            antiquant_group_size (int): Group size for pergroup quantization, default 0.
            inner_precise (int): 0=high precision, 1=high performance, default 0.
        Returns:
            Tensor: Output tensor, same dtype as x.
        """
        import torch_npu
        return torch_npu.npu_weight_quant_batchmatmul(
            x, weight, antiquant_scale, antiquant_offset,
            quant_scale, quant_offset, bias, antiquant_group_size, inner_precise)


def get_input_groups():
    import json, os
    json_path = os.path.join(os.path.dirname(__file__), "5_WeightQuantBatchmatmul.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        # uniform(-5, 5)
        return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        x_info = inputs[0]
        weight_info = inputs[1]
        dtype = dtype_map[x_info["dtype"]]
        weight_dtype = {"int8": torch.int8}[weight_info["dtype"]]

        x = random_tensor(x_info["shape"], dtype)
        weight = torch.randint(-8, 8, weight_info["shape"], dtype=weight_dtype)

        antiquant_scale = None
        antiquant_offset = None
        quant_scale = None
        quant_offset = None
        bias = None
        antiquant_group_size = 0
        inner_precise = 0

        for inp in inputs[2:]:
            name = inp.get("name", "")
            if name == "antiquant_scale":
                aq_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(inp.get("dtype", "float16"), dtype)
                antiquant_scale = random_tensor(inp["shape"], aq_dtype)
            elif name == "antiquant_offset":
                aq_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(inp.get("dtype", "float16"), dtype)
                antiquant_offset = random_tensor(inp["shape"], aq_dtype)
            elif name == "quant_scale":
                quant_scale = random_tensor(inp["shape"], torch.float32)
            elif name == "quant_offset":
                quant_offset = random_tensor(inp["shape"], torch.float32)
            elif name == "bias":
                b_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(inp.get("dtype", "float16"), dtype)
                bias = random_tensor(inp["shape"], b_dtype)
            elif name == "antiquant_group_size":
                antiquant_group_size = inp["value"]
            elif name == "inner_precise":
                inner_precise = inp["value"]

        if antiquant_scale is None:
            antiquant_scale = torch.ones([1], dtype=dtype)

        input_groups.append([x, weight, antiquant_scale, antiquant_offset,
                             quant_scale, quant_offset, bias,
                             antiquant_group_size, inner_precise])
    return input_groups


def get_init_inputs():
    return []
