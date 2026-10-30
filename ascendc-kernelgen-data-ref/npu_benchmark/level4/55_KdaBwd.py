import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    chunk_kda 反向 —— KDA 门控 delta 规则线性注意力的梯度 golden reference。
    对标 fla-org/flash-linear-attention 的 chunk_kda 反向
    (fla/ops/kda/chunk_bwd.py 的 chunk_kda_bwd, dk 双路径见
    chunk_kda_bwd_kernel_wy_dqkg_fused; dtype/None 契约见
    fla/ops/kda/chunk.py 的 ChunkKDAFunction; 前向数学定义见
    fla/ops/kda/naive.py, 官方测试以 autograd 跑 naive 当反向标杆;
    共享部件 bwd_dhu 在 fla/ops/common/chunk_delta_h.py)。
    采用 **显式反向公式**，前向中间量由 get_input_groups 提供。
    vLLM 是推理框架无训练反向 kernel, 该算子只有上游 fla 官方实现。

    数学语义：
      前向（fp32）：S ← S·exp(g_t)；v_new = v_t − S_decᵀk_t（擦除门）；
                   S ← S_dec + (β_t k_t)⊗v_new（写入门）；o_t = (scale·q_t)ᵀS
      损失：L = <o, do> + <final_state, dht>
    反向递推（从 T-1 到 0）：
      dS 初始化为 dht（若存在）或零
      对每个 t：
        S_t = S_dec_t + β_t * k_t ⊗ v_new_t
        dq_t = scale * (S_t @ do_t)          # q 梯度
        dS += (scale·q_t) ⊗ do_t             # 来自输出路径（scale 只乘一次）
        # 写入门路径
        dv_new = β_t * (dS^T @ k_t)
        dβ_t = (dS * (k_t ⊗ v_new_t)).sum()  # β 是标量（每个头）
        # k 同时出现在写入门与擦除门, dk 为两路径之和：
        dk_t = β_t * (dS @ v_new_t)          # 写入门贡献
               − S_dec_t @ dv_new            # 擦除门贡献（v_new 对 k 的 VJP）
        # 擦除路径
        dv_t = dv_new
        dS_dec = dS − k_t ⊗ dv_new            # 因为 v_new = v_t − S_dec^T k_t
        # 衰减路径
        dg_t = (dS_dec * S_dec_t).sum(dim=-1) # [B,HV,K]
        dS = dS_dec * exp(g_t)                # 传给上一步
      dh0 = dS（若 h0 存在）
    上述公式已经 PyTorch autograd 决定性数值验证（全部梯度 ~1e-15）。

    布局约定：同原 kernel，但 q/k 在 forward 中会进行 GVA 扩展
    （repeat_interleave）。q/k 契约是已预归一化（l2norm 在 kernel 外层,
    官方 use_qk_l2norm_in_kernel=True 时反向外层套 l2norm_bwd, 不在
    本算子覆盖内）。无 varlen 分支（官方 chunk_kda 反向支持 cu_seqlens,
    本 golden 声明域不含, JSON 不出现 cu_seqlens）。
    返回梯度：dq,dk,dv,dg,dbeta,dh0（均回铸输入 dtype，dh0 为 fp32;
    h0 为空张量 [0,...] 时 dh0 返回 [0,HV,K,V] 空张量）。
    数值路径：官方 chunked kernel 为 bf16 中间量 + fp32 累加 + log2 域
    （exp2/RCP_LN2）换算; 本 golden 全程 fp32 逐步伴随, 比 kernel 更精确,
    对拍按 bf16 管线常规容差判定。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, v, g, beta, do, dht, scale, h0,
                S_dec_stack, v_new_stack):
        """
        参数：
          q,k,v,g,beta: 原始输入叶子张量（fp32，不要求梯度）
          do: 输出梯度系数 [B,T,HV,V]
          dht: 末状态梯度 [B,HV,K,V] 或 None
          scale: 缩放因子
          h0: 初始状态 [B,HV,K,V] 或 None
          S_dec_stack: [T,B,HV,K,V] 每步衰减后的状态 fp32
          v_new_stack: [T,B,HV,V]   每步擦除后的向量 fp32
        返回：
          dq, dk, dv, dg, dbeta, dh0
        """
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        G = HV // H

        qf, kf, vf = q.float(), k.float(), v.float()
        gf, bf = g.float(), beta.float()
        dof = do.float()

        q_exp = qf.repeat_interleave(G, dim=2) * scale
        k_exp = kf.repeat_interleave(G, dim=2)

        dq_exp = torch.zeros_like(q_exp)
        dk_exp = torch.zeros_like(k_exp)
        dv = torch.zeros_like(vf)
        dg = torch.zeros_like(gf)
        dbeta = torch.zeros_like(bf)

        if dht is not None:
            dS = dht.float().clone()
        else:
            dS = torch.zeros(B, HV, K, V, device=q.device)
        for t in range(T - 1, -1, -1):
            S_dec = S_dec_stack[t]
            v_new = v_new_stack[t]

            beta_t = bf[:, t]
            k_t = k_exp[:, t]
            # 外积以 matmul 实现（数学恒等；NPU eager 广播外积
            # unsqueeze(-1)*unsqueeze(-2) 在该平台极度劣化, 单算子可达秒级）
            S_t = S_dec + beta_t.unsqueeze(-1).unsqueeze(-1) * (
                k_t.unsqueeze(-1) @ v_new.unsqueeze(-2))
            do_t = dof[:, t]
            dq_exp[:, t] = scale * (S_t @ do_t.unsqueeze(-1)).squeeze(-1)
            # 输出路径: q_exp 已含 scale, 此处不再乘（否则 scale² 污染 dS 递推）
            dS = dS + q_exp[:, t].unsqueeze(-1) @ do_t.unsqueeze(-2)
            dv_new = beta_t.unsqueeze(-1) * (dS.transpose(-2, -1) @ k_t.unsqueeze(-1)).squeeze(-1)
            # dk = 写入门 β(dS@v_new) + 擦除门 -(S_dec@dv_new)
            dk_exp[:, t] = beta_t.unsqueeze(-1) * (dS @ v_new.unsqueeze(-1)).squeeze(-1) \
                - (S_dec @ dv_new.unsqueeze(-1)).squeeze(-1)
            # dβ = Σ_kv dS⊙(k⊗v_new) = kᵀ(dS@v_new), 数学恒等
            dbeta[:, t] = (k_t * (dS @ v_new.unsqueeze(-1)).squeeze(-1)).sum(-1)
            dv[:, t] = dv_new
            dS_dec = dS - k_t.unsqueeze(-1) @ dv_new.unsqueeze(-2)
            g_t = gf[:, t]
            dg[:, t] = (dS_dec * S_dec).sum(dim=-1)
            dS = dS_dec * g_t.unsqueeze(-1).exp()

        dq = dq_exp.reshape(B, T, H, G, K).sum(dim=3)
        dk = dk_exp.reshape(B, T, H, G, K).sum(dim=3)

        if h0 is not None:
            dh0 = dS
        else:
            dh0 = torch.empty(0, HV, K, V, dtype=torch.float32, device=q.device)
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                dg.to(g.dtype), dbeta.to(beta.dtype), dh0)


_GATE_LOGIT_NORMALIZER = [1.0, 1.0, 1.0, 1.0, 10.0, 0.1]


def get_input_groups():
    def json_path():
        # 验证/性能脚本会把本文件拷贝成 {op_name}_torch.py，
        # 因此除同名 .json 外还需回退到去掉 _torch 后缀的名字。
        here = os.path.dirname(os.path.abspath(__file__))
        stem = os.path.splitext(os.path.basename(__file__))[0]
        names = [stem]
        if stem.endswith("_torch"):
            names.append(stem[: -len("_torch")])
        for name in names:
            path = os.path.join(here, name + ".json")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"case 描述文件未找到: {names} @ {here}")

    with open(json_path(), "r", encoding="utf-8-sig") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    input_groups = []
    for case_idx, case in enumerate(cases):
        # 每 case 独立播种（种子 3407+case_idx），Generator 穿线保证
        # 输入可复现且与 case 顺序/子集无关
        gen = torch.Generator().manual_seed(3407 + case_idx)

        def random_tensor(shape, dtype):
            # 标准输入分布: 50% 均匀 [-5, 5] + 50% 正态 (mu/sigma 随机)
            if torch.rand(1, generator=gen).item() < 0.5:
                return torch.empty(shape, dtype=torch.float32).uniform_(
                    -5.0, 5.0, generator=gen).to(dtype)
            mu = float(torch.empty(1).uniform_(-5.0, 5.0,
                                               generator=gen).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0,
                                                  generator=gen).item())
            return torch.normal(mu, sigma, shape, generator=gen,
                                dtype=torch.float32).to(dtype)

        inputs = case["inputs"]

        def info(name):
            return next(i for i in inputs if i["name"] == name)

        q_dtype = dtype_map[info("q")["dtype"]]
        v_dtype = dtype_map[info("v")["dtype"]]
        scale = info("scale")["value"]
        B, T, H, K = info("q")["shape"]
        HV, V = info("v")["shape"][2], info("v")["shape"][3]
        G = HV // H
        # 契约: q/k 已预归一化（l2norm 在 kernel 外层）
        q = F.normalize(random_tensor((B, T, H, K), torch.float32),
                        p=2, dim=-1).to(q_dtype)
        k = F.normalize(random_tensor((B, T, H, K), torch.float32),
                        p=2, dim=-1).to(q_dtype)
        v = random_tensor(info("v")["shape"], v_dtype)
        # 物理约束: g 是 K 维逐通道 log 域衰减, 恒 <= 0（logsigmoid）;
        # normalizer 循环提供门控强弱多样性（沿用 level4 机制）
        g = (F.logsigmoid(random_tensor(info("g")["shape"], torch.float32))
             / _GATE_LOGIT_NORMALIZER[case_idx % len(_GATE_LOGIT_NORMALIZER)])
        g = g.to(dtype_map[info("g")["dtype"]])
        # 物理约束: beta 是 sigmoid 类门控, 恒在 (0, 1)
        beta = torch.rand(info("beta")["shape"], generator=gen,
                          dtype=torch.float32).to(
                              dtype_map[info("beta")["dtype"]])
        h0_shape = info("initial_state")["shape"]
        if h0_shape[0] == 0:
            h0 = None
        else:
            h0 = random_tensor(h0_shape, torch.float32)
        do = random_tensor(info("do")["shape"], v_dtype)
        dht_shape = info("dht")["shape"]
        if dht_shape[0] == 0:
            dht = None
        else:
            dht = random_tensor(dht_shape, torch.float32)

        # 前向递推预计算中间量栈（与 level4 逐行同构）
        qf = q.float().repeat_interleave(G, dim=2) * scale
        kf = k.float().repeat_interleave(G, dim=2)
        vf, gf, bf = v.float(), g.float(), beta.float()

        S = qf.new_zeros(B, HV, K, V)
        if h0 is not None:
            S = S + h0

        S_dec_list = []
        v_new_list = []
        for t in range(T):
            k_t = kf[:, t]       # [B,HV,K]
            v_t = vf[:, t]       # [B,HV,V]
            g_t = gf[:, t]       # [B,HV,K]
            b_t = bf[:, t]       # [B,HV]

            S_dec = S * g_t.unsqueeze(-1).exp()          # 衰减
            # 擦除
            v_new = v_t - (S_dec.transpose(-2, -1) @ k_t.unsqueeze(-1)).squeeze(-1)  # [B,HV,V]
            # 更新
            S = S_dec + b_t.unsqueeze(-1).unsqueeze(-1) * k_t.unsqueeze(-1) * v_new.unsqueeze(-2)

            S_dec_list.append(S_dec)
            v_new_list.append(v_new)

        # 返回 (输入叶子, do, dht, scale, h0, 中间量栈)
        S_dec_stack = torch.stack(S_dec_list, dim=0)
        v_new_stack = torch.stack(v_new_list, dim=0)
        input_groups.append([q, k, v, g, beta, do, dht, scale, h0,
                             S_dec_stack, v_new_stack])
    return input_groups


def get_init_inputs():
    return []
