import json
import math
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q,
        k,
        v,
        dy,
        softmax_max,
        softmax_sum,
        attention_in,
        causal,
        window_left,
        window_right,
        softcap,
    ):
        torch.manual_seed(0)
        in_dtype = q.dtype
        qf = q.float().transpose(1, 2)
        kf = k.float().transpose(1, 2)
        vf = v.float().transpose(1, 2)
        dyf = dy.float().transpose(1, 2)
        attention_in_f = attention_in.float().transpose(1, 2)
        batch_size, n_heads_q, seq_len_q, d_qk = qf.shape
        n_heads_kv = kf.shape[1]
        seq_len_kv = kf.shape[2]
        if n_heads_q % n_heads_kv != 0:
            raise ValueError("query heads must be divisible by KV heads")

        scale = 1.0 / math.sqrt(d_qk)
        group = n_heads_q // n_heads_kv
        if group > 1:
            kf = kf.repeat_interleave(group, dim=1)
            vf = vf.repeat_interleave(group, dim=1)

        scores_before_softcap = torch.matmul(qf, kf.transpose(-1, -2)) * scale
        if softcap > 0.0:
            tanh_scores = torch.tanh(scores_before_softcap / softcap)
            scores = softcap * tanh_scores
        else:
            tanh_scores = torch.zeros_like(scores_before_softcap)
            scores = scores_before_softcap

        row = torch.arange(seq_len_q, device=q.device).unsqueeze(1)
        column = torch.arange(seq_len_kv, device=q.device).unsqueeze(0)
        relative = column - (row + seq_len_kv - seq_len_q)
        atten_mask = torch.zeros(
            (seq_len_q, seq_len_kv), dtype=torch.bool, device=q.device
        )
        if causal:
            atten_mask = atten_mask | (relative > 0)
        if window_left >= 0:
            atten_mask = atten_mask | (relative < -window_left)
        if window_right >= 0:
            atten_mask = atten_mask | (relative > window_right)

        has_mask = causal or window_left >= 0 or window_right >= 0
        if has_mask:
            scores = scores.masked_fill(
                atten_mask.unsqueeze(0).unsqueeze(0), -40000.0
            )

        x_max = softmax_max[..., :1].float()
        x_sum = softmax_sum[..., :1].float()
        probabilities = torch.exp(scores - x_max) / x_sum
        if has_mask:
            probabilities = probabilities.masked_fill(
                atten_mask.unsqueeze(0).unsqueeze(0), 0.0
            )

        d_probabilities = torch.matmul(dyf, vf.transpose(-1, -2))
        d_rowsum = (dyf * attention_in_f).sum(dim=-1, keepdim=True)
        d_scores = probabilities * (d_probabilities - d_rowsum)
        if softcap > 0.0:
            d_scores = d_scores * (1.0 - tanh_scores * tanh_scores)
        d_scores = d_scores * scale

        dq = torch.matmul(d_scores, kf)
        dk = torch.matmul(d_scores.transpose(-1, -2), qf)
        dv = torch.matmul(probabilities.transpose(-1, -2), dyf)

        if group > 1:
            dk = dk.view(
                batch_size, n_heads_kv, group, seq_len_kv, d_qk
            ).sum(dim=2)
            dv = dv.view(
                batch_size, n_heads_kv, group, seq_len_kv, vf.shape[-1]
            ).sum(dim=2)

        return (
            dq.transpose(1, 2).to(in_dtype),
            dk.transpose(1, 2).to(k.dtype),
            dv.transpose(1, 2).to(v.dtype),
        )


def get_input_groups():
    torch.manual_seed(0)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    json_path = os.path.splitext(__file__)[0] + ".json"
    with open(json_path, "r", encoding="utf-8-sig") as file:
        cases = [json.loads(line) for line in file if line.strip()]

    input_groups = []
    for case_index, case in enumerate(cases):
        specs = {item["name"]: item for item in case["inputs"]}
        q_generator = torch.Generator()
        k_generator = torch.Generator()
        v_generator = torch.Generator()
        dy_generator = torch.Generator()
        q_generator.manual_seed(case_index * 4 + 1)
        k_generator.manual_seed(case_index * 4 + 2)
        v_generator.manual_seed(case_index * 4 + 3)
        dy_generator.manual_seed(case_index * 4 + 4)
        q = torch.rand(
            tuple(specs["q"]["shape"]),
            generator=q_generator,
            dtype=torch.float32,
        ).to(dtype=dtype_map[specs["q"]["dtype"]]).npu()
        k = torch.rand(
            tuple(specs["k"]["shape"]),
            generator=k_generator,
            dtype=torch.float32,
        ).to(dtype=dtype_map[specs["k"]["dtype"]]).npu()
        v = torch.rand(
            tuple(specs["v"]["shape"]),
            generator=v_generator,
            dtype=torch.float32,
        ).to(dtype=dtype_map[specs["v"]["dtype"]]).npu()
        dy = torch.rand(
            tuple(specs["dy"]["shape"]),
            generator=dy_generator,
            dtype=torch.float32,
        ).to(dtype=dtype_map[specs["dy"]["dtype"]]).npu()
        causal = specs["causal"]["value"]
        window_left = specs["window_left"]["value"]
        window_right = specs["window_right"]["value"]
        softcap = specs["softcap"]["value"]

        with torch.no_grad():
            qf = q.float().transpose(1, 2)
            kf = k.float().transpose(1, 2)
            vf = v.float().transpose(1, 2)
            batch_size, n_heads_q, seq_len_q, d_qk = qf.shape
            n_heads_kv = kf.shape[1]
            seq_len_kv = kf.shape[2]
            if n_heads_q % n_heads_kv != 0:
                raise ValueError("query heads must be divisible by KV heads")

            group = n_heads_q // n_heads_kv
            if group > 1:
                kf = kf.repeat_interleave(group, dim=1)
                vf = vf.repeat_interleave(group, dim=1)

            scale = 1.0 / math.sqrt(d_qk)
            scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
            if softcap > 0.0:
                scores = softcap * torch.tanh(scores / softcap)

            row = torch.arange(seq_len_q, device=q.device).unsqueeze(1)
            column = torch.arange(seq_len_kv, device=q.device).unsqueeze(0)
            relative = column - (row + seq_len_kv - seq_len_q)
            atten_mask = torch.zeros(
                (seq_len_q, seq_len_kv), dtype=torch.bool, device=q.device
            )
            if causal:
                atten_mask = atten_mask | (relative > 0)
            if window_left >= 0:
                atten_mask = atten_mask | (relative < -window_left)
            if window_right >= 0:
                atten_mask = atten_mask | (relative > window_right)

            has_mask = causal or window_left >= 0 or window_right >= 0
            if has_mask:
                scores = scores.masked_fill(
                    atten_mask.unsqueeze(0).unsqueeze(0), -40000.0
                )
            x_max = scores.amax(dim=-1, keepdim=True)
            exp_scores = torch.exp(scores - x_max)
            x_sum = exp_scores.sum(dim=-1, keepdim=True)
            probabilities = exp_scores / x_sum
            if has_mask:
                probabilities = probabilities.masked_fill(
                    atten_mask.unsqueeze(0).unsqueeze(0), 0.0
                )
            attention_in = torch.matmul(probabilities, vf)

            stats_shape = list(x_max.shape[:-1]) + [8]
            softmax_max = x_max.expand(stats_shape).contiguous()
            softmax_sum = x_sum.expand(stats_shape).contiguous()

        input_groups.append([
            q,
            k,
            v,
            dy,
            softmax_max.detach(),
            softmax_sum.detach(),
            attention_in.transpose(1, 2).to(q.dtype).detach(),
            causal,
            window_left,
            window_right,
            softcap,
        ])
    return input_groups


def get_init_inputs():
    return []
