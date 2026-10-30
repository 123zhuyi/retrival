import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.inner_dim = 3072
        self.time_embed_dim = 768

    def forward(self, timestep: torch.Tensor, pooled_projections: torch.Tensor, freqs: torch.Tensor,
                timestep_linear1_weight: torch.Tensor, timestep_linear1_bias: torch.Tensor,
                timestep_linear2_weight: torch.Tensor, timestep_linear2_bias: torch.Tensor,
                text_embedder_weight: torch.Tensor, text_embedder_bias: torch.Tensor):
        batch_size = timestep.shape[0]
        # embedding 在 fp32 下完成，不提前 cast 到输入 dtype
        t_emb = timestep.to(torch.float32) * 1000.0
        t_emb = t_emb.unsqueeze(-1) * freqs.to(torch.float32).unsqueeze(0)
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
        # linear 在 fp32 下完成；SiLU 在 fp32 下完成，linear2 之前不做低精度 cast
        t_emb = F.linear(t_emb, timestep_linear1_weight.to(torch.float32),
                         timestep_linear1_bias.to(torch.float32))
        t_emb = F.silu(t_emb)
        t_emb = F.linear(t_emb, timestep_linear2_weight.to(torch.float32),
                         timestep_linear2_bias.to(torch.float32)).to(timestep_linear2_bias.dtype)
        text_emb = F.linear(pooled_projections.to(torch.float32), text_embedder_weight.to(torch.float32),
                            text_embedder_bias.to(torch.float32)).to(text_embedder_bias.dtype)
        # 加法在 fp32 下完成，再 cast 到 result_type
        out_dtype = torch.result_type(t_emb, text_emb)
        conditioning = (t_emb.to(torch.float32) + text_emb.to(torch.float32)).to(out_dtype)
        return conditioning


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "60_FluxTimestepGuidanceProjectionEmbedding.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        """使用有界均匀分布，避免 fp16/bf16 累加溢出产生 NaN/Inf"""
        return torch.empty(shape, dtype=dtype).uniform_(-0.5, 0.5)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        name_map = {inp["name"]: inp for inp in inputs}

        dtype = name_map["timestep"]["dtype"]
        bias_dtype = name_map["timestep_linear1_bias"]["dtype"]

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        dt = dtype_map[dtype]
        bdt = dtype_map[bias_dtype]

        # 所有浮点 tensor 改用独立随机分布，保留 .npu()
        timestep = random_tensor(name_map["timestep"]["shape"], dt).npu()
        pooled_projections = random_tensor(name_map["pooled_projections"]["shape"], dt).npu()
        freqs = random_tensor(name_map["freqs"]["shape"], dt).npu()
        timestep_linear1_weight = random_tensor(name_map["timestep_linear1_weight"]["shape"], dt).npu()
        timestep_linear1_bias = random_tensor(name_map["timestep_linear1_bias"]["shape"], bdt).npu()
        timestep_linear2_weight = random_tensor(name_map["timestep_linear2_weight"]["shape"], dt).npu()
        timestep_linear2_bias = random_tensor(name_map["timestep_linear2_bias"]["shape"], bdt).npu()
        text_embedder_weight = random_tensor(name_map["text_embedder_weight"]["shape"], dt).npu()
        text_embedder_bias = random_tensor(name_map["text_embedder_bias"]["shape"], bdt).npu()

        input_groups.append([timestep, pooled_projections, freqs,
                             timestep_linear1_weight, timestep_linear1_bias,
                             timestep_linear2_weight, timestep_linear2_bias,
                             text_embedder_weight, text_embedder_bias])
    return input_groups


def get_init_inputs():
    return []