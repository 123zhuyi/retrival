import torch
import torch.nn as nn
import json
import os


def _crop_float32_to_19bit(scale):
    """模拟 npu_trans_quant_param / aclnnTransQuantParamV2：
    将 fp32 scale 截断到 19 位有效位（与 NPU 硬件预处理一致）。"""
    if scale.dtype != torch.float32:
        return scale
    scale_int32 = scale.view(torch.int32)
    scale_int32 = scale_int32 & 0xFFFFE000
    return scale_int32.view(torch.float32)


class Model(nn.Module):
    """
    Model that performs quantized matrix multiplication using NPU accelerated npu_quant_matmul.
    Supports int8 quantized matmul with various output data types.

    torch_npu.npu_quant_matmul(x1, x2, scale, *, offset=None, pertoken_scale=None,
                               bias=None, output_dtype=None, group_sizes=None) -> Tensor

    PyTorch native reference implementation (for correctness analysis only):
    ------------------------------------------------------------------------
    def forward_native(x1, x2, scale, offset=None, pertoken_scale=None,
                       bias=None, output_dtype=None, group_sizes=None):
        x1_fp = x1.to(torch.float32)
        x2_fp = x2.to(torch.float32)
        matmul_result = torch.matmul(x1_fp, x2_fp)

        # scale 预处理：模拟 npu_trans_quant_param
        if (scale.dtype == torch.float32 and pertoken_scale is None and
                output_dtype not in (torch.bfloat16, torch.int32)):
            scale_fp = _crop_float32_to_19bit(scale)
        else:
            scale_fp = scale.to(torch.float32)

        # int32 bias：始终在 scale 之前（与官方标杆一致）
        if bias is not None and bias.dtype == torch.int32:
            matmul_result = matmul_result + bias.to(torch.float32)

        if pertoken_scale is not None:
            pertoken_scale_fp = pertoken_scale.to(torch.float32)
            if pertoken_scale_fp.dim() == 1:
                pertoken_scale_fp = pertoken_scale_fp.view(-1, 1)
            matmul_result_fp = matmul_result * pertoken_scale_fp * scale_fp
        else:
            matmul_result_fp = matmul_result * scale_fp

        if bias is not None and bias.dtype != torch.int32:
            matmul_result_fp = matmul_result_fp + bias.to(torch.float32)

        if offset is not None:
            matmul_result_fp = matmul_result_fp + offset.to(torch.float32)

        if output_dtype is not None:
            matmul_result_fp = matmul_result_fp.to(output_dtype)

        if output_dtype is not None and torch.is_floating_point(matmul_result_fp):
            info = torch.finfo(output_dtype)
            matmul_result_fp = torch.clamp(matmul_result_fp, info.min, info.max)

        return matmul_result_fp

    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1, x2, scale, offset=None, pertoken_scale=None,
                bias=None, output_dtype=None, group_sizes=None):
        """

        Args:
            x1 (Tensor): Left matrix, shape 2-6D, dtype int8.
            x2 (Tensor): Right matrix, shape 2-6D, dtype int8.
            scale (Tensor): Quantization scale factor, dtype float32/bfloat16.
            offset (Tensor, optional): Quantization offset, dtype float32.
            pertoken_scale (Tensor, optional): Per-token scale, shape [m], dtype float32.
            bias (Tensor, optional): Bias term, shape [n] or [batch, 1, n], dtype bfloat16/float32/int32.
            output_dtype (str or torch.dtype, optional): Output data type, supports float16/bfloat16/int32/int8.
            group_sizes (list[int], optional): Group quantization granularity.

        Returns:
            Tensor: Quantized matmul result.
        """
        import torch_npu
        return torch_npu.npu_quant_matmul(
            x1, x2, scale,
            offset=offset,
            pertoken_scale=pertoken_scale,
            bias=bias,
            output_dtype=output_dtype,
            group_sizes=group_sizes
        )


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "4_QuantMatmul.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]

        x1_info = inputs[0]
        x2_info = inputs[1]
        scale_info = inputs[2]

        x_dtype_map = {
            "int8": torch.int8,
        }
        x1_dtype = x_dtype_map[x1_info["dtype"]]
        x2_dtype = x_dtype_map[x2_info["dtype"]]
        x1 = torch.randint(-5, 5, x1_info["shape"], dtype=x1_dtype)
        x2 = torch.randint(-5, 5, x2_info["shape"], dtype=x2_dtype)

        scale_dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        scale_dtype = scale_dtype_map.get(scale_info["dtype"], torch.float32)
        scale = torch.rand(scale_info["shape"], dtype=scale_dtype)

        offset = None
        pertoken_scale = None
        bias = None
        output_dtype = None
        group_sizes = None

        for inp in inputs[3:]:
            name = inp.get("name", "")
            if name == "offset":
                offset_dtype_map = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                }
                offset_dtype = offset_dtype_map.get(inp.get("dtype", "float32"), torch.float32)
                offset = torch.rand(inp["shape"], dtype=offset_dtype) * 10 - 5
            elif name == "pertoken_scale":
                pertoken_scale = torch.rand(inp["shape"], dtype=torch.float32)
            elif name == "bias":
                bias_dtype_map = {
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "int32": torch.int32,
                }
                bias_dtype = bias_dtype_map.get(inp.get("dtype", "float32"), torch.float32)
                if bias_dtype == torch.int32:
                    bias = torch.randint(-128, 127, inp["shape"], dtype=torch.int32)
                else:
                    bias = torch.rand(inp["shape"], dtype=bias_dtype) * 10 - 5
            elif name == "output_dtype":
                dtype_str_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "int32": torch.int32,
                    "int8": torch.int8,
                }
                output_dtype = dtype_str_map.get(inp["value"], inp["value"])
            elif name == "group_sizes":
                group_sizes = inp.get("value")
        if (scale.dtype == torch.float32 and
                pertoken_scale is None and
                output_dtype not in (torch.bfloat16, torch.int32)):
            scale = _crop_float32_to_19bit(scale)

        input_groups.append([x1, x2, scale, offset, pertoken_scale, bias, output_dtype, group_sizes])
    return input_groups


def get_init_inputs():
    return []