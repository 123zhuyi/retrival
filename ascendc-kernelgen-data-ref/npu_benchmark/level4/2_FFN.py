import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs FFN (Feed-Forward Network) computation using NPU accelerated npu_ffn.
    out = activation(x * W1 + b1) * W2 + b2
    torch_npu.npu_ffn(x, weight1, weight2, activation, *, expert_tokens=None, expert_tokens_index=None, bias1=None, bias2=None, scale=None, offset=None, deq_scale1=None, deq_scale2=None, antiquant_scale1=None, antiquant_scale2=None, antiquant_offset1=None, antiquant_offset2=None, inner_precise=None, output_dtype=None) -> Tensor

    PyTorch native implementation of forward function (aligned with aclnnFFNV2 & torch_npu.npu_ffn behavior):
    def _apply_activation(x, activation):
        if activation == "fastgelu":
            return x * torch.sigmoid(1.7 * x)
        elif activation == "gelu":
            return torch.nn.functional.gelu(x, approximate="tanh")
        elif activation == "relu":
            return torch.nn.functional.relu(x)
        elif activation == "silu":
            return torch.nn.functional.silu(x)
        elif activation == "geglu":
            gate, up = x.chunk(2, dim=-1)
            return torch.nn.functional.gelu(gate, approximate="tanh") * up
        elif activation == "swiglu":
            gate, up = x.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate) * up
        elif activation == "reglu":
            gate, up = x.chunk(2, dim=-1)
            return torch.nn.functional.relu(gate) * up
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def _dequantize(x, deq_scale):
        return x.to(deq_scale.dtype) * deq_scale

    def _quantize_activation(x_fp32, scale, offset):
        scale_fp32 = scale.to(torch.float32)
        offset_fp32 = offset.to(torch.float32)
        q = torch.round(x_fp32 * scale_fp32 + offset_fp32)
        q = torch.clamp(q, -128, 127)
        return q.to(torch.int8)

    def _ffn_standard(x, weight1, weight2, activation, bias1, bias2, inner_precise, output_dtype):
        compute_dtype = torch.float32 if inner_precise == 0 else x.dtype
        x_orig_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        x_c = x_2d.to(compute_dtype)
        w1_c = weight1.to(compute_dtype)
        w2_c = weight2.to(compute_dtype)
        out1 = torch.matmul(x_c, w1_c)
        if bias1 is not None:
            out1 = out1 + bias1.to(compute_dtype)
        out1_activated = _apply_activation(out1, activation)
        out2 = torch.matmul(out1_activated, w2_c)
        if bias2 is not None:
            out2 = out2 + bias2.to(compute_dtype)
        if output_dtype is not None:
            out2 = out2.to(output_dtype)
        else:
            out2 = out2.to(x.dtype)
        out_shape = list(x_orig_shape[:-1]) + [out2.shape[-1]]
        out2 = out2.view(out_shape)
        return out2

    def _ffn_quant(x, weight1, weight2, activation, bias1, bias2, scale, offset, deq_scale1, deq_scale2, output_dtype):
        x_orig_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        x_q = x_2d.to(torch.int64)
        w1_c = weight1.to(torch.int64)
        out1_int32 = torch.matmul(x_q, w1_c)
        if bias1 is not None:
            out1_int32 = out1_int32 + bias1.to(torch.int64)
        out1_fp = _dequantize(out1_int32, deq_scale1)
        out1_activated = _apply_activation(out1_fp, activation)
        out1_q = _quantize_activation(out1_activated, scale, offset)
        w2_c = weight2.to(torch.int64)
        out2_int32 = torch.matmul(out1_q.to(torch.int64), w2_c)
        if bias2 is not None:
            out2_int32 = out2_int32 + bias2.to(torch.int64)
        out2 = _dequantize(out2_int32, deq_scale2)
        if output_dtype is not None:
            out2 = out2.to(output_dtype)
        else:
            out2 = out2.to(torch.float16)
        out_shape = list(x_orig_shape[:-1]) + [out2.shape[-1]]
        out2 = out2.view(out_shape)
        return out2

    def _ffn_pseudo_quant(x, weight1, weight2, activation, bias1, bias2, antiquant_scale1, antiquant_scale2, antiquant_offset1, antiquant_offset2, output_dtype):
        x_orig_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        compute_dtype = x.dtype
        # Aligned with aclnnFFNV2: y = activation(x * ((W1 + antiquantOffset1) * antiquantScale1) + b1) * ((W2 + antiquantOffset2) * antiquantScale2) + b2
        w1_fp = (weight1.to(compute_dtype) + antiquant_offset1.to(compute_dtype)) * antiquant_scale1.to(compute_dtype)
        w2_fp = (weight2.to(compute_dtype) + antiquant_offset2.to(compute_dtype)) * antiquant_scale2.to(compute_dtype)
        x_c = x_2d.to(compute_dtype)
        w1_c = w1_fp.to(compute_dtype)
        w2_c = w2_fp.to(compute_dtype)
        out1 = torch.matmul(x_c, w1_c)
        if bias1 is not None:
            out1 = out1 + bias1.to(compute_dtype)
        out1_activated = _apply_activation(out1, activation)
        out2 = torch.matmul(out1_activated, w2_c)
        if bias2 is not None:
            out2 = out2 + bias2.to(compute_dtype)
        if output_dtype is not None:
            out2 = out2.to(output_dtype)
        else:
            out2 = out2.to(x.dtype)
        out_shape = list(x_orig_shape[:-1]) + [out2.shape[-1]]
        out2 = out2.view(out_shape)
        return out2

    def _ffn_standard_moe_expert(x, weight1, weight2, activation, bias1, bias2, inner_precise, output_dtype):
        compute_dtype = torch.float32 if inner_precise == 0 else x.dtype
        x_c = x.to(compute_dtype)
        w1_c = weight1.to(compute_dtype)
        w2_c = weight2.to(compute_dtype)
        out1 = torch.matmul(x_c, w1_c)
        if bias1 is not None:
            out1 = out1 + bias1.to(compute_dtype)
        out1_activated = _apply_activation(out1, activation)
        out2 = torch.matmul(out1_activated, w2_c)
        if bias2 is not None:
            out2 = out2 + bias2.to(compute_dtype)
        if output_dtype is not None:
            out2 = out2.to(output_dtype)
        else:
            out2 = out2.to(x.dtype)
        return out2

    def _ffn_quant_moe_expert(x, weight1, weight2, activation, bias1, bias2, scale, offset, deq_scale1, deq_scale2, output_dtype):
        x_q = x.to(torch.int64)
        w1_c = weight1.to(torch.int64)
        out1_int32 = torch.matmul(x_q, w1_c)
        if bias1 is not None:
            out1_int32 = out1_int32 + bias1.to(torch.int64)
        out1_fp = _dequantize(out1_int32, deq_scale1)
        out1_activated = _apply_activation(out1_fp, activation)
        out1_q = _quantize_activation(out1_activated, scale, offset)
        w2_c = weight2.to(torch.int64)
        out2_int32 = torch.matmul(out1_q.to(torch.int64), w2_c)
        if bias2 is not None:
            out2_int32 = out2_int32 + bias2.to(torch.int64)
        out2 = _dequantize(out2_int32, deq_scale2)
        if output_dtype is not None:
            out2 = out2.to(output_dtype)
        else:
            out2 = out2.to(torch.float16)
        return out2

    def _ffn_pseudo_quant_moe_expert(x, weight1, weight2, activation, bias1, bias2, antiquant_scale1, antiquant_scale2, antiquant_offset1, antiquant_offset2, output_dtype):
        compute_dtype = x.dtype
        # Aligned with aclnnFFNV2: (W + antiquantOffset) * antiquantScale
        w1_fp = (weight1.to(compute_dtype) + antiquant_offset1.to(compute_dtype)) * antiquant_scale1.to(compute_dtype)
        w2_fp = (weight2.to(compute_dtype) + antiquant_offset2.to(compute_dtype)) * antiquant_scale2.to(compute_dtype)
        x_c = x.to(compute_dtype)
        w1_c = w1_fp.to(compute_dtype)
        w2_c = w2_fp.to(compute_dtype)
        out1 = torch.matmul(x_c, w1_c)
        if bias1 is not None:
            out1 = out1 + bias1.to(compute_dtype)
        out1_activated = _apply_activation(out1, activation)
        out2 = torch.matmul(out1_activated, w2_c)
        if bias2 is not None:
            out2 = out2 + bias2.to(compute_dtype)
        if output_dtype is not None:
            out2 = out2.to(output_dtype)
        else:
            out2 = out2.to(x.dtype)
        return out2

    def _ffn_moe(x, weight1, weight2, activation, expert_tokens, bias1, bias2, scale, offset, deq_scale1, deq_scale2, antiquant_scale1, antiquant_scale2, antiquant_offset1, antiquant_offset2, inner_precise, output_dtype, is_quant, is_pseudo_quant):
        x_orig_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])
        total_tokens = x_2d.shape[0]
        expert_tokens_list = expert_tokens if isinstance(expert_tokens, list) else expert_tokens.tolist()
        num_experts = len(expert_tokens_list)
        out_chunks = [None] * total_tokens
        current_token = 0
        for expert_id in range(num_experts):
            num_tokens = expert_tokens_list[expert_id]
            if num_tokens == 0:
                continue
            x_expert = x_2d[current_token:current_token + num_tokens]
            if is_pseudo_quant:
                out_expert = _ffn_pseudo_quant_moe_expert(
                    x_expert, weight1[expert_id], weight2[expert_id], activation,
                    bias1[expert_id] if bias1 is not None else None,
                    bias2[expert_id] if bias2 is not None else None,
                    antiquant_scale1[expert_id] if antiquant_scale1 is not None else None,
                    antiquant_scale2[expert_id] if antiquant_scale2 is not None else None,
                    antiquant_offset1[expert_id] if antiquant_offset1 is not None else None,
                    antiquant_offset2[expert_id] if antiquant_offset2 is not None else None,
                    output_dtype)
            elif is_quant:
                out_expert = _ffn_quant_moe_expert(
                    x_expert, weight1[expert_id], weight2[expert_id], activation,
                    bias1[expert_id] if bias1 is not None else None,
                    bias2[expert_id] if bias2 is not None else None,
                    scale, offset, deq_scale1, deq_scale2, output_dtype)
            else:
                out_expert = _ffn_standard_moe_expert(
                    x_expert, weight1[expert_id], weight2[expert_id], activation,
                    bias1[expert_id] if bias1 is not None else None,
                    bias2[expert_id] if bias2 is not None else None,
                    inner_precise, output_dtype)
            for i in range(num_tokens):
                out_chunks[current_token + i] = out_expert[i:i + 1]
            current_token += num_tokens
        out = torch.cat(out_chunks, dim=0)
        out_shape = list(x_orig_shape[:-1]) + [out.shape[-1]]
        out = out.view(out_shape)
        return out

    def forward(self, x: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor,
                activation: str, expert_tokens=None, expert_tokens_index=None,
                bias1=None, bias2=None, scale=None, offset=None,
                deq_scale1=None, deq_scale2=None,
                antiquant_scale1=None, antiquant_scale2=None,
                antiquant_offset1=None, antiquant_offset2=None,
                inner_precise=None, output_dtype=None):
        is_quant = scale is not None or deq_scale1 is not None
        is_pseudo_quant = antiquant_scale1 is not None or antiquant_scale2 is not None
        is_moe = expert_tokens is not None
        if is_moe:
            return _ffn_moe(x, weight1, weight2, activation, expert_tokens,
                            bias1, bias2, scale, offset, deq_scale1, deq_scale2,
                            antiquant_scale1, antiquant_scale2, antiquant_offset1, antiquant_offset2,
                            inner_precise, output_dtype, is_quant, is_pseudo_quant)
        if is_pseudo_quant:
            return _ffn_pseudo_quant(x, weight1, weight2, activation, bias1, bias2,
                                     antiquant_scale1, antiquant_scale2,
                                     antiquant_offset1, antiquant_offset2, output_dtype)
        if is_quant:
            return _ffn_quant(x, weight1, weight2, activation, bias1, bias2,
                              scale, offset, deq_scale1, deq_scale2, output_dtype)
        return _ffn_standard(x, weight1, weight2, activation, bias1, bias2, inner_precise, output_dtype)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor,
                activation: str, expert_tokens=None, expert_tokens_index=None,
                bias1=None, bias2=None, scale=None, offset=None,
                deq_scale1=None, deq_scale2=None,
                antiquant_scale1=None, antiquant_scale2=None,
                antiquant_offset1=None, antiquant_offset2=None,
                inner_precise=None, output_dtype=None):
        """
        Performs FFN computation on NPU.

        Args:
            x (Tensor): Input tensor of shape [M, K1] (or up to 8D).
            weight1 (Tensor): Weight for first matmul, shape [K1, N1] or [E, K1, N1].
            weight2 (Tensor): Weight for second matmul, shape [K2, N2] or [E, K2, N2].
            activation (str): Activation function, supports fastgelu/gelu/relu/silu/geglu/swiglu/reglu.
            expert_tokens (list, optional): Token count per expert.
            expert_tokens_index (list, optional): Token index per expert.
            bias1 (Tensor, optional): Bias for first matmul.
            bias2 (Tensor, optional): Bias for second matmul.
            scale (Tensor, optional): Quantization scale.
            offset (Tensor, optional): Quantization offset.
            deq_scale1 (Tensor, optional): Dequantization scale for first matmul.
            deq_scale2 (Tensor, optional): Dequantization scale for second matmul.
            antiquant_scale1 (Tensor, optional): Anti-quantization scale for first matmul.
            antiquant_scale2 (Tensor, optional): Anti-quantization scale for second matmul.
            antiquant_offset1 (Tensor, optional): Anti-quantization offset for first matmul.
            antiquant_offset2 (Tensor, optional): Anti-quantization offset for second matmul.
            inner_precise (int, optional): 0 for high precision, 1 for high performance.
            output_dtype (ScalarType, optional): Output data type for quantization scenario.

        Returns:
            Tensor: Output tensor with same dimensions as x.
        """
        import torch_npu
        kwargs = {}
        if expert_tokens is not None:
            kwargs['expert_tokens'] = expert_tokens
        if expert_tokens_index is not None:
            kwargs['expert_tokens_index'] = expert_tokens_index
        if bias1 is not None:
            kwargs['bias1'] = bias1
        if bias2 is not None:
            kwargs['bias2'] = bias2
        if scale is not None:
            kwargs['scale'] = scale
        if offset is not None:
            kwargs['offset'] = offset
        if deq_scale1 is not None:
            kwargs['deq_scale1'] = deq_scale1
        if deq_scale2 is not None:
            kwargs['deq_scale2'] = deq_scale2
        if antiquant_scale1 is not None:
            kwargs['antiquant_scale1'] = antiquant_scale1
        if antiquant_scale2 is not None:
            kwargs['antiquant_scale2'] = antiquant_scale2
        if antiquant_offset1 is not None:
            kwargs['antiquant_offset1'] = antiquant_offset1
        if antiquant_offset2 is not None:
            kwargs['antiquant_offset2'] = antiquant_offset2
        if inner_precise is not None:
            kwargs['inner_precise'] = inner_precise
        if output_dtype is not None:
            kwargs['output_dtype'] = output_dtype
        return torch_npu.npu_ffn(x, weight1, weight2, activation, **kwargs)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "2_FFN.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        """限制值域，避免 fp16/bf16 matmul 累加溢出与近零相对误差"""
        if dtype in (torch.float16, torch.bfloat16):
            # 非对称值域：避开零附近，降低 activation 相对误差
            return torch.empty(shape, dtype=dtype).uniform_(-0.15, 0.25)
        else:
            return torch.empty(shape, dtype=dtype).uniform_(-1.0, 1.0)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
            "int32": torch.int32,
        }

        x_info = inputs[0]
        weight1_info = inputs[1]
        weight2_info = inputs[2]
        activation_info = inputs[3]

        x_dtype = dtype_map[x_info["dtype"]]
        w_dtype = dtype_map[weight1_info["dtype"]]

        if x_dtype == torch.int8:
            x = torch.randint(-5, 5, x_info["shape"], dtype=x_dtype)
        else:
            x = random_tensor(x_info["shape"], x_dtype)
        if w_dtype == torch.int8:
            weight1 = torch.randint(-5, 5, weight1_info["shape"], dtype=w_dtype)
            weight2 = torch.randint(-5, 5, weight2_info["shape"], dtype=w_dtype)
        else:
            weight1 = random_tensor(weight1_info["shape"], w_dtype)
            weight2 = random_tensor(weight2_info["shape"], w_dtype)
        activation = activation_info["value"]

        expert_tokens = None
        expert_tokens_index = None
        bias1 = None
        bias2 = None
        scale = None
        offset = None
        deq_scale1 = None
        deq_scale2 = None
        antiquant_scale1 = None
        antiquant_scale2 = None
        antiquant_offset1 = None
        antiquant_offset2 = None
        inner_precise = None
        output_dtype = None

        for inp in inputs[4:]:
            name = inp.get("name", "")
            if name == "expert_tokens":
                expert_tokens = inp.get("value")
            elif name == "expert_tokens_index":
                expert_tokens_index = inp.get("value")
            elif name == "bias1":
                b1_dtype = dtype_map.get(inp["dtype"], x_dtype)
                if b1_dtype in (torch.int32, torch.int8):
                    bias1 = torch.randint(0, 5, inp["shape"], dtype=b1_dtype)
                else:
                    bias1 = random_tensor(inp["shape"], b1_dtype)
            elif name == "bias2":
                b2_dtype = dtype_map.get(inp["dtype"], x_dtype)
                if b2_dtype in (torch.int32, torch.int8):
                    bias2 = torch.randint(0, 5, inp["shape"], dtype=b2_dtype)
                else:
                    bias2 = random_tensor(inp["shape"], b2_dtype)
            elif name == "scale":
                scale = random_tensor(inp["shape"], dtype_map.get(inp["dtype"], torch.float32))
            elif name == "offset":
                offset = random_tensor(inp["shape"], dtype_map.get(inp["dtype"], torch.float32))
            elif name == "deq_scale1":
                ds1_dtype = dtype_map.get(inp["dtype"], torch.float32)
                deq_scale1 = random_tensor(inp["shape"], ds1_dtype)
            elif name == "deq_scale2":
                ds2_dtype = dtype_map.get(inp["dtype"], torch.float32)
                deq_scale2 = random_tensor(inp["shape"], ds2_dtype)
            elif name == "antiquant_scale1":
                antiquant_scale1 = random_tensor(inp["shape"], dtype_map.get(inp["dtype"], x_dtype))
            elif name == "antiquant_scale2":
                antiquant_scale2 = random_tensor(inp["shape"], dtype_map.get(inp["dtype"], x_dtype))
            elif name == "antiquant_offset1":
                antiquant_offset1 = random_tensor(inp["shape"], dtype_map.get(inp["dtype"], x_dtype))
            elif name == "antiquant_offset2":
                antiquant_offset2 = random_tensor(inp["shape"], dtype_map.get(inp["dtype"], x_dtype))
            elif name == "inner_precise":
                inner_precise = inp["value"]
            elif name == "output_dtype":
                output_dtype_map = {0: torch.float16, 1: torch.bfloat16}
                output_dtype = output_dtype_map.get(inp["value"], torch.float16)

        input_groups.append([x, weight1, weight2, activation,
                             expert_tokens, expert_tokens_index,
                             bias1, bias2, scale, offset,
                             deq_scale1, deq_scale2,
                             antiquant_scale1, antiquant_scale2,
                             antiquant_offset1, antiquant_offset2,
                             inner_precise, output_dtype])
    return input_groups


def get_init_inputs():
    return []