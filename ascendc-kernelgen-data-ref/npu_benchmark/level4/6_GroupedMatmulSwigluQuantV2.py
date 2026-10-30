import torch
import torch.nn as nn
import json
import os
import torch_npu

# 必须开启：否则 npu_format_cast 到 FRACTAL_NZ 会被强制回落为 base format，
# 导致 aclnnGroupedMatmulSwigluQuantWeightNzV2 报 "storageShape dimnum must be 5"。
torch.npu.config.allow_internal_format = True

class Model(nn.Module):
    """
    Model that performs Grouped Matmul + SwiGLU + Quant computation using NPU accelerated npu_grouped_matmul_swiglu_quant_v2.
    Fuses grouped matrix multiplication, SwiGLU activation, and quantization.

    Pure PyTorch reference implementation (decomposing the fused operator into basic ops):

        def grouped_matmul_swiglu_quant_torch(x, weight, weight_scale, x_scale, group_list, **kwargs):
            m, k = x.shape
            e = len(weight)

            # Step 1: Resolve group_list (cumsum or count)
            if group_list_type == 0:
                # group_list is cumsum: [0, count_0, count_0+count_1, ..., m]
                cumsum = group_list
                token_counts = cumsum[1:] - cumsum[:-1]
            else:
                # group_list is raw counts
                token_counts = group_list
                cumsum = torch.zeros(e + 1, dtype=torch.int64, device=x.device)
                cumsum[1:] = torch.cumsum(token_counts, dim=0)

            # Step 2: Dequantize x and weight for each expert group
            outputs = []
            for g in range(e):
                start = cumsum[g].item()
                end = cumsum[g + 1].item()

                # Slice tokens belonging to this expert
                x_g = x[start:end, :]        # [t, k] int8
                x_s_g = x_scale[start:end]    # [t] float32

                # Dequantize x (per-token): x_deq[t, :] = x[t, :] * x_scale[t]
                x_deq = x_g.to(torch.float32) * x_s_g.unsqueeze(1)  # [t, k]

                # Dequantize weight (per-channel): W_deq[c, :] = W[c, :] * W_scale[c]
                W_g = weight[g]                 # [k, n] int8
                W_s_g = weight_scale[g]          # [1] float32 or [n] float32

                if dequant_mode == 0:
                    # Left pertoken, right perchannel
                    W_deq = W_g.to(torch.float32) * W_s_g  # [k, n]
                else:
                    # Left pertoken, right pergroup (not implemented in pure torch)
                    W_deq = W_g.to(torch.float32) * W_s_g.unsqueeze(0)

                # Step 3: Matmul (grouped GEMM)
                hidden = x_deq @ W_deq  # [t, n] float32

                # Apply bias if provided
                if bias is not None:
                    hidden = hidden + bias[g].to(torch.float32)

                # Apply weight_assist_matrix if provided
                if weight_assist_matrix is not None and weight_assist_matrix[g] is not None:
                    hidden = hidden @ weight_assist_matrix[g].to(torch.float32)

                # Step 4: SwiGLU activation
                # SwiGLU splits hidden dim in half: g = sigmoid(x[:n/2]) * x[n/2:]
                n = hidden.shape[1]
                half = n // 2
                left = hidden[:, :half]
                right = hidden[:, half:]
                gate = torch.sigmoid(left)
                swiglu_out = gate * right  # [t, half]

                # Step 5: Apply smooth_scale if provided
                if smooth_scale is not None:
                    smooth_g = smooth_scale[start:end, :half]
                    swiglu_out = swiglu_out * smooth_g

                # Step 6: Quantize output
                # Per-token quantization
                if quant_mode == 0:
                    # Find scale per token
                    abs_max = swiglu_out.abs().max(dim=1, keepdim=True).values  # [t, 1]
                    out_scale_g = abs_max / 127.0  # [t, 1]
                    q_out = (swiglu_out / out_scale_g).round().clamp(-128, 127).to(torch.int8)
                    out_scale_g = out_scale_g.squeeze(1)
                else:
                    # Per-channel quantization (fallback to per-token)
                    abs_max = swiglu_out.abs().max()
                    out_scale_g = torch.full((end - start,), abs_max.item() / 127.0, device=x.device)
                    q_out = (swiglu_out / out_scale_g.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

                outputs.append((q_out, out_scale_g))

            # Step 7: Concatenate all expert outputs
            output = torch.cat([o[0] for o in outputs], dim=0)    # [m, half] int8
            output_scale = torch.cat([o[1] for o in outputs], dim=0)  # [m] float32

            return output, output_scale
    """

    def __init__(self):
        super(Model, self).__init__()


    def _cast_list_to_nz(self, tensor_list):
        """将 List[Tensor] 中的每个 NPU Tensor 转为 NZ 格式（ACL_FORMAT_NZ = 29）。
        aclnnGroupedMatmulSwigluQuantWeightNzV2 要求 weight 的 storageShape.dimnum == 5。"""
        if tensor_list is None:
            return None
        result = []
        for t in tensor_list:
            if not isinstance(t, torch.Tensor):
                result.append(t)
                continue
            if t.device.type != 'npu':
                t = t.npu()
            fmt = torch_npu.get_npu_format(t)
            if fmt != 29:
                t = torch_npu.npu_format_cast(t, 29)
            result.append(t)
        return result

    def forward(self, x: torch.Tensor, weight: list, weight_scale: list,
                x_scale: torch.Tensor, group_list: torch.Tensor,
                smooth_scale=None, weight_assist_matrix=None, bias=None,
                dequant_mode=0, dequant_dtype=0, quant_mode=0, quant_dtype=0,
                group_list_type=0, tuning_config=None):
        """
        Performs grouped matmul + SwiGLU + quant on NPU.

        Args:
            x (Tensor): Left matrix for matmul, shape [m, k], dtype int8.
            weight (TensorList): Weight matrices, shape [e, k, n], dtype int8/int32.
            weight_scale (TensorList): Weight quantization scales, dtype float32.
            x_scale (Tensor): Left matrix quantization scale, shape [m], dtype float32.
            group_list (Tensor): Token count per group, shape [e], dtype int64.
            smooth_scale (Tensor, optional): Quantization smooth scales, dtype float32.
            weight_assist_matrix (TensorList, optional): Weight assist matrix, dtype float32.
            bias (Tensor, optional): Matmul bias, shape 2D, dtype int32.
            dequant_mode (int): Dequantization mode, 0=left pertoken right perchannel, 1=left pertoken right pergroup.
            dequant_dtype (int): Dequantization dtype, reserved, default 0.
            quant_mode (int): Quantization mode after SwiGLU, 0=pertoken, 1=perchannel.
            quant_dtype (int): Quantized low-bit dtype, 0=int8, 1=float8_e8m0, 2=float8_e5m2, 3=float8_e4m3.
            group_list_type (int): Group list input type, 0=cumsum, 1=count.
            tuning_config (List[int], optional): Tuning configuration.

        Note:
            This implementation includes an internal device-alignment helper that
            recursively moves Tensors inside list arguments (weight, weight_scale,
            weight_assist_matrix) to the same device as x, because the validation
            script does not handle list-of-tensor device migration.

        Returns:
            tuple: (output, output_scale) where output is int8 [m, n] and output_scale is float [m].
        """
        # ---- Device alignment helper: recursively move Tensors inside lists to x's device ----
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
        if smooth_scale is not None:
            smooth_scale = _to_device(smooth_scale, device)
        if weight_assist_matrix is not None:
            weight_assist_matrix = _to_device(weight_assist_matrix, device)
        if bias is not None:
            bias = _to_device(bias, device)
        weight = self._cast_list_to_nz(weight)
        if weight_assist_matrix is not None:
            weight_assist_matrix = self._cast_list_to_nz(weight_assist_matrix)

        return torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x, weight, weight_scale, x_scale, group_list,
            smooth_scale=smooth_scale, weight_assist_matrix=weight_assist_matrix,
            bias=bias, dequant_mode=dequant_mode, dequant_dtype=dequant_dtype,
            quant_mode=quant_mode, quant_dtype=quant_dtype,
            group_list_type=group_list_type, tuning_config=tuning_config)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "6_GroupedMatmulSwigluQuantV2.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        """独立随机选择正态/均匀分布"""
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)
        else:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]

        x_info = inputs[0]
        weight_info = inputs[1]
        weight_scale_info = inputs[2]
        x_scale_info = inputs[3]
        group_list_info = inputs[4]
        x = torch.randint(-128, 127, x_info["shape"], dtype=torch.int8)
        weight = [torch.randint(-128, 127, weight_info["shape"], dtype=torch.int8)]
        weight_scale = [random_tensor(weight_scale_info["shape"], torch.float32)]
        x_scale = random_tensor(x_scale_info["shape"], torch.float32)

        group_list = torch.tensor(group_list_info["value"], dtype=torch.int64)

        smooth_scale = None
        weight_assist_matrix = None
        bias = None
        dequant_mode = 0
        dequant_dtype = 0
        quant_mode = 0
        quant_dtype = 0
        group_list_type = 0
        tuning_config = None

        for inp in inputs[5:]:
            name = inp.get("name", "")
            if name == "smooth_scale":
                smooth_scale = None
            elif name == "weight_assist_matrix":
                weight_assist_matrix = None
            elif name == "bias":
                bias = None
            elif name == "dequant_mode":
                dequant_mode = inp["value"]
            elif name == "dequant_dtype":
                dequant_dtype = inp["value"]
            elif name == "quant_mode":
                quant_mode = inp["value"]
            elif name == "quant_dtype":
                quant_dtype = inp["value"]
            elif name == "group_list_type":
                group_list_type = inp["value"]
            elif name == "tuning_config":
                tuning_config = inp.get("value")

        input_groups.append([x, weight, weight_scale, x_scale, group_list,
                             smooth_scale, weight_assist_matrix, bias,
                             dequant_mode, dequant_dtype, quant_mode, quant_dtype,
                             group_list_type, tuning_config])
    return input_groups


def get_init_inputs():
    return []