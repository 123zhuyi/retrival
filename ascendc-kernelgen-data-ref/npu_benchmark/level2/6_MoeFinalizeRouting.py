import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Pure PyTorch high-precision (fp32) implementation of MoE finalize routing.

    Replaces torch_npu.npu_moe_finalize_routing with pure torch ops using fp32
    arithmetic to avoid precision loss from fp16 accumulation.

    Operator semantics:
        expanded_permuted_rows: (NUM_ROWS*K, H) for non-drop, or (E, C, H) for drop mode
        scales: (NUM_ROWS, K) expert weights
        expanded_src_to_dst_row: (NUM_ROWS*K) index mapping
        export_for_source_row: (NUM_ROWS, K) expert assignments per row
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, expanded_permuted_rows: torch.Tensor, skip1: torch.Tensor, skip2: torch.Tensor,
                bias: torch.Tensor, scales: torch.Tensor, expanded_src_to_dst_row: torch.Tensor,
                export_for_source_row: torch.Tensor, drop_pad_mode: int = 0) -> torch.Tensor:
        orig_dtype = expanded_permuted_rows.dtype

        # Cast to fp32 for high-precision computation
        rows = expanded_permuted_rows.float()
        s = scales.float()

        if drop_pad_mode in (0, 2):
            num_rows_k, H = rows.shape
            NUM_ROWS = s.shape[0]
            K = s.shape[1]
            idx = expanded_src_to_dst_row.long()

            gathered = rows[idx]
            if drop_pad_mode == 0:
                gathered = gathered.view(K, NUM_ROWS, H).transpose(0, 1)
            else:
                gathered = gathered.view(NUM_ROWS, K, H)

            output = (gathered * s.unsqueeze(-1)).sum(dim=1)

        else:
            E, C, H = rows.shape
            NUM_ROWS = s.shape[0]
            K = s.shape[1]
            idx = expanded_src_to_dst_row.long()

            output = torch.zeros(NUM_ROWS, H, device=rows.device, dtype=torch.float32)
            for r in range(NUM_ROWS):
                for k in range(K):
                    if drop_pad_mode == 1:
                        flat_idx = idx[r + k * NUM_ROWS].item()
                    else:
                        flat_idx = idx[r * K + k].item()
                    expert = flat_idx // C
                    slot = flat_idx % C
                    if 0 <= expert < E and 0 <= slot < C:
                        output[r] += rows[expert, slot] * s[r, k]

        # Bias (computed in fp32)
        if bias is not None and export_for_source_row is not None:
            bias_f32 = bias.float()
            row_bias = torch.zeros(NUM_ROWS, H, device=output.device, dtype=torch.float32)
            for k in range(K):
                expert_idx = export_for_source_row[:, k].long()
                row_bias += bias_f32[expert_idx] * s[:, k:k + 1]
            output = output + row_bias

        # Skip connections (cast to fp32, add, cast back)
        if skip1 is not None:
            output = output + skip1.float()
        if skip2 is not None:
            output = output + skip2.float()

        return output.to(orig_dtype)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "6_MoeFinalizeRouting.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        expanded_permuted_rows_info = inputs[0]
        skip1_info = inputs[1]
        skip2_info = inputs[2]
        bias_info = inputs[3]
        scales_info = inputs[4]
        expanded_src_to_dst_row_info = inputs[5]
        export_for_source_row_info = inputs[6]
        drop_pad_mode_info = inputs[7]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[expanded_permuted_rows_info["dtype"]]

        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            expanded_permuted_rows = torch.normal(mu, sigma, expanded_permuted_rows_info["shape"], dtype=dtype)
        else:
            mu = 0.0
            sigma = 1.0
            expanded_permuted_rows = torch.empty(expanded_permuted_rows_info["shape"], dtype=dtype).uniform_(-5, 5)

        if skip1_info.get("shape") is None:
            skip1 = None
            skip2 = None
        else:
            if idx % 2 == 0:
                skip1 = torch.normal(mu, sigma, skip1_info["shape"], dtype=dtype)
                skip2 = torch.normal(mu, sigma, skip2_info["shape"], dtype=dtype)
            else:
                skip1 = torch.empty(skip1_info["shape"], dtype=dtype).uniform_(-5, 5)
                skip2 = torch.empty(skip2_info["shape"], dtype=dtype).uniform_(-5, 5)

        if bias_info.get("shape") is None:
            bias = None
        else:
            if idx % 2 == 0:
                bias = torch.normal(mu, sigma, bias_info["shape"], dtype=dtype)
            else:
                bias = torch.empty(bias_info["shape"], dtype=dtype).uniform_(-5, 5)

        if idx % 2 == 0:
            scales = torch.normal(mu, sigma, scales_info["shape"], dtype=dtype)
        else:
            scales = torch.empty(scales_info["shape"], dtype=dtype).uniform_(-5, 5)
        expanded_src_to_dst_row = torch.randint(0, max(1, expanded_permuted_rows_info["shape"][0]),
                                                expanded_src_to_dst_row_info["shape"], dtype=torch.int32)
        export_for_source_row = torch.randint(0, 8, export_for_source_row_info["shape"], dtype=torch.int32)
        drop_pad_mode = drop_pad_mode_info["value"]
        input_groups.append(
            [expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, export_for_source_row,
             drop_pad_mode])
    return input_groups


def get_init_inputs():
    return []