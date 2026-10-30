import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    MoE Router: expert selection via Softmax + TopK.

    Computes softmax over gate logits then selects the top-k experts
    per token with their corresponding routing weights.

    Input:  (num_tokens, num_experts), bfloat16  — gate logits
    Output: topk_ids  (num_tokens, topk), int64
            topk_vals (num_tokens, topk), bfloat16 — routing weights
    """

    def __init__(self):
        super().__init__()
        # 无参数，保留空缓存以备统一
        self._cache = {}

    def forward(self, gate_logits: torch.Tensor, topk: int):
        """
        Args:
            gate_logits: (num_tokens, num_experts)
            topk: number of experts to select per token
        Returns:
            topk_ids: (num_tokens, topk) int64
            topk_vals: (num_tokens, topk) same dtype as gate_logits
        """
        torch.manual_seed(42)
        # Softmax in float32 for numerical stability
        probs = torch.softmax(gate_logits.float(), dim=-1)
        topk_vals, topk_ids = torch.topk(probs, k=topk, dim=-1)
        return topk_ids, topk_vals.to(gate_logits.dtype)


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "93_Router.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)
        else:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        gate_info = next(inp for inp in inputs if inp["name"] == "gate_logits")
        dtype = dtype_map[gate_info["dtype"]]
        gate_logits = random_tensor(gate_info["shape"], dtype)

        topk_info = next(inp for inp in inputs if inp["name"] == "topk")
        topk = topk_info["value"]

        input_groups.append([gate_logits, topk])
    return input_groups


def get_init_inputs():
    return []