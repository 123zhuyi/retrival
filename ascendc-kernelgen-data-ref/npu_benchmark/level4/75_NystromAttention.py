import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, num_landmarks):
        torch.manual_seed(42)
        batch, n_heads, sequence, head_dim = q.shape
        q = q.float()
        k = k.float()
        v_float = v.float()
        landmarks = min(sequence, num_landmarks)
        scale = 1.0 / math.sqrt(head_dim)
        if landmarks >= sequence:
            output = torch.matmul(
                torch.softmax(
                    torch.matmul(q, k.transpose(-2, -1)) * scale,
                    dim=-1,
                ),
                v_float,
            )
        else:
            q_landmarks = F.adaptive_avg_pool1d(
                q.permute(0, 1, 3, 2).reshape(
                    batch * n_heads, head_dim, sequence
                ),
                landmarks,
            ).reshape(
                batch, n_heads, head_dim, landmarks
            ).transpose(-2, -1)
            k_landmarks = F.adaptive_avg_pool1d(
                k.permute(0, 1, 3, 2).reshape(
                    batch * n_heads, head_dim, sequence
                ),
                landmarks,
            ).reshape(
                batch, n_heads, head_dim, landmarks
            ).transpose(-2, -1)
            kernel_1 = torch.softmax(
                torch.matmul(q, k_landmarks.transpose(-2, -1)) * scale,
                dim=-1,
            )
            kernel_2 = torch.softmax(
                torch.matmul(
                    q_landmarks, k_landmarks.transpose(-2, -1)
                ) * scale,
                dim=-1,
            )
            kernel_3 = torch.softmax(
                torch.matmul(q_landmarks, k.transpose(-2, -1)) * scale,
                dim=-1,
            )

            absolute = kernel_2.abs()
            max_row = absolute.sum(dim=-1).amax(dim=-1)
            max_column = absolute.sum(dim=-2).amax(dim=-1)
            denominator = (max_row * max_column).clamp_min(1e-6)
            inverse = kernel_2.transpose(-2, -1)
            inverse = inverse / denominator.unsqueeze(-1).unsqueeze(-1)
            identity = torch.eye(
                landmarks, device=q.device, dtype=q.dtype
            ).view(1, 1, landmarks, landmarks)
            for _ in range(6):
                product = torch.matmul(kernel_2, inverse)
                inverse = 0.25 * torch.matmul(
                    inverse,
                    13.0 * identity
                    - torch.matmul(
                        product,
                        15.0 * identity
                        - torch.matmul(
                            product, 7.0 * identity - product
                        ),
                    ),
                )
            output = torch.matmul(
                kernel_1,
                torch.matmul(inverse, torch.matmul(kernel_3, v_float)),
            )

        output = output.to(dtype=v.dtype).transpose(1, 2).contiguous()
        return output.view(batch, sequence, n_heads * head_dim)


def get_input_groups():
    torch.manual_seed(42)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    json_path = os.path.splitext(__file__)[0] + ".json"
    with open(json_path, "r", encoding="utf-8-sig") as file:
        cases = [json.loads(line) for line in file if line.strip()]

    def random_tensor(spec, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.rand(
            tuple(spec["shape"]),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype=dtype_map[spec["dtype"]]).npu()

    input_groups = []
    for case_index, case in enumerate(cases):
        specs = {item["name"]: item for item in case["inputs"]}
        input_groups.append([
            random_tensor(specs["q"], 42 + case_index * 3),
            random_tensor(specs["k"], 43 + case_index * 3),
            random_tensor(specs["v"], 44 + case_index * 3),
            specs["num_landmarks"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
