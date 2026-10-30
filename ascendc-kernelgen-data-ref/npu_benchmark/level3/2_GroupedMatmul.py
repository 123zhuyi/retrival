import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    Model that performs grouped matrix multiplication using torch.nn.functional.grouped_mm.
    Supports both 3D inputs (direct grouping) and 2D inputs with offset-based grouping.

    Native reference implementation (end offsets semantics, aligned with F.grouped_mm):
        def forward_native(A, B, offs):
            # A: (total_rows, K), B: (num_groups, K, N)
            # offs: (num_groups,) end offsets, offs[i] = exclusive end of group i
            outputs = []
            start = 0
            for i in range(B.shape[0]):
                end = offs[i]
                outputs.append(A[start:end] @ B[i])
                start = end
            return torch.cat(outputs, dim=0)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, offsets: torch.Tensor = None) -> torch.Tensor:
        """
        Applies grouped matrix multiplication between A and B.

        Args:
            A (torch.Tensor): Left operand tensor.
                - 3D shape: (num_groups, m, k) - groups are directly enumerated
                - 2D shape: (total_rows, k) - rows are grouped by end offsets
            B (torch.Tensor): Right operand tensor.
                - Shape: (num_groups, k, n) for common forward pass
            offsets (torch.Tensor): 1D int32 tensor defining group **end** indices in A.
                - Required when A is 2D (total_rows, k).
                - Length: exactly num_groups (matches first dimension of B).
                - Format: [end_0, end_1, ..., end_{G-1}] where end_i = exclusive end of group i.
                - Group 0: rows A[0 : offsets[0]]
                - Group i: rows A[offsets[i-1] : offsets[i]] for i > 0.
                - Must be strictly increasing; offsets[-1] <= total_rows.
                  Elements beyond offsets[-1] are ignored.

        Returns:
            torch.Tensor: Concatenated results of each per-group GEMM operation.
        """
        if hasattr(F, 'grouped_mm'):
            return F.grouped_mm(A, B, offs=offsets)
        else:
            return torch._grouped_mm(A, B, offs=offsets)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "2_GroupedMatmul.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        A_info = inputs[0]
        B_info = inputs[1]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[A_info["dtype"]]

        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            A = torch.normal(mu, sigma, A_info["shape"], dtype=dtype)
        else:
            A = torch.empty(A_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)

        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            B = torch.normal(mu, sigma, B_info["shape"], dtype=dtype)
        else:
            B = torch.empty(B_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)

        if len(inputs) > 2:
            offsets_info = inputs[2]
            if offsets_info.get("type") == "tensor":
                total_rows = A_info["shape"][0]
                num_groups = B_info["shape"][0]
                group_size = total_rows // num_groups
                # end offsets: offs[i] marks the exclusive end of group i
                # aligns with F.grouped_mm docstring: "offs[i] marks the end of group i"
                offsets = torch.tensor(
                    [(i + 1) * group_size for i in range(num_groups)],
                    dtype=torch.int32
                )
                input_groups.append([A, B, offsets])
            else:
                input_groups.append([A, B, None])
        else:
            input_groups.append([A, B, None])

    return input_groups


def get_init_inputs():
    return []