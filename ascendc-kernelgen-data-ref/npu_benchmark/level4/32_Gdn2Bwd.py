import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gdn2Bwd —— GDN-2（Gated DeltaNet 2）反向（显式对偶递推）
#
# 对标 ops-ref `chunk_gated_delta_rule` / fla chunk_gdn2_bwd 的逐 token 精确伴随
# （数据依赖使反向天然逐 token 串行）。前向（逐 token，S ∈ R^{K×V}）：
#   S_dec = Diag(exp(g_t))·S_{t-1}          # 衰减
#   erase = (b_t·k_t)^T·S_dec                # 擦除门读取
#   v_new = w_t·v_t − erase                  # 新写入向量
#   S_t   = S_dec + k_t ⊗ v_new              # 更新
#   o_t   = (scale·q_t)^T·S_t                # 输出
# 反向（dS 从 dht 初始化，逆序传播）：
#   dqs_t = S_t @ do_t ; dq_t = scale·dqs_t
#   dS   += (scale·q_t) ⊗ do_t
#   dk_t  = dS @ v_new_t                     # 秩一更新路径
#   dv_new = dS^T @ k_t
#   dw_t  = dv_new·v_t ; dv_t = dv_new·w_t
#   derase = −dv_new
#   dbk   = S_dec_t @ derase                 # 擦除门读取路径
#   dS_dec = dS + (b_t·k_t) ⊗ derase
#   db_t  = dbk·k_t ; dk_t += dbk·b_t
#   dg_t  = Σ_V(dS_dec·S_dec_t)              # 衰减路径
#   dS   <- dS_dec·exp(g_t)                  # 传给上一步
#   dh0   = 递推结束时的 dS
#
# 前向中间量（S_dec_stack/v_new_stack）由 get_input_groups 预计算并传入
# （对应 kernel 前向导出的中间量），forward 内零前向计算、仅做反向递推。
# 全程 fp32 累加；输出回铸输入 dtype（dh0 恒 fp32）。q/k 的 L2 归一化（kernel
# use_qk_l2norm）在进入递推前先施加，并在输出时经链式法则传回 q/k 梯度。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """GDN-2 chunkwise 反向（融合显式递推版）。"""

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, g, b, w, initial_state, do, dht,
                scale, use_qk_l2norm_in_kernel,
                S_dec_stack, v_new_stack, S_final, o,
                cu_seqlens=None):
        """
        q,k,v,g,b,w : 输入叶子张量
        initial_state : [B,H,K,V] 或 [N,H,K,V] fp32 或 None（shape[0]==0 表示无）
        do : [B,T,H,V] 输出梯度；dht : 末状态梯度（shape[0]==0 表示无）
        S_dec_stack : [T,B,H,K,V] 每步衰减状态 fp32；v_new_stack : [T,B,H,V]
        S_final, o : 前向最终状态/输出（未使用，仅保持接口）
        cu_seqlens : varlen 前缀和，None 表示等长 [B,T]
        返回 (dq, dk, dv, dg, db, dw, dh0)
        """
        B, T, H, K = q.shape
        V = v.shape[-1]

        qf, kf, vf = q.float(), k.float(), v.float()
        gf, bf, wf = g.float(), b.float(), w.float()
        dof = do.float()

        if use_qk_l2norm_in_kernel:
            q_rstd = torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
            k_rstd = torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
            qn = qf * q_rstd
            kn = kf * k_rstd
        else:
            qn, kn = qf, kf
            q_rstd = k_rstd = None

        dq = torch.zeros_like(qf)
        dk = torch.zeros_like(kf)
        dv = torch.zeros_like(vf)
        dg = torch.zeros_like(gf)
        db = torch.zeros_like(bf)
        dw = torch.zeros_like(wf)

        has_init = initial_state is not None and initial_state.shape[0] > 0
        has_dht = dht is not None and dht.shape[0] > 0

        if cu_seqlens is not None and len(cu_seqlens) > 0:
            # varlen：B 必须为 1，按 segment 独立反向
            N = len(cu_seqlens) - 1
            dh0 = torch.zeros(N, H, K, V, dtype=torch.float32, device=q.device)
            for n in range(N):
                s, e = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
                dS0 = dht[n:n + 1].float() if has_dht else \
                    torch.zeros(1, H, K, V, device=q.device)
                r = self._segment_bwd(
                    qn[:, s:e], kn[:, s:e], vf[:, s:e],
                    gf[:, s:e], bf[:, s:e], wf[:, s:e],
                    dof[:, s:e], S_dec_stack, v_new_stack,
                    s, e, dS0, scale)
                dq[:, s:e], dk[:, s:e], dv[:, s:e] = r[0], r[1], r[2]
                dg[:, s:e], db[:, s:e], dw[:, s:e] = r[3], r[4], r[5]
                dh0[n] = r[6][0]
        else:
            dS0 = dht.float() if has_dht else \
                torch.zeros(B, H, K, V, device=q.device)
            dq, dk, dv, dg, db, dw, dh0 = self._segment_bwd(
                qn, kn, vf, gf, bf, wf, dof,
                S_dec_stack, v_new_stack, 0, T, dS0, scale)

        if use_qk_l2norm_in_kernel:
            dq = q_rstd * (dq - qn * (dq * qn).sum(-1, keepdim=True))
            dk = k_rstd * (dk - kn * (dk * kn).sum(-1, keepdim=True))

        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
                dg.to(g.dtype), db.to(b.dtype), dw.to(w.dtype), dh0)

    @staticmethod
    def _segment_bwd(qn, kn, vf, gf, bf, wf, dof,
                     S_dec_stack, v_new_stack, s, e, dS0, scale):
        """对 token 区间 [s, e) 执行反向递推（dS 从 dS0 初始化）。"""
        dq = torch.zeros_like(qn)
        dk = torch.zeros_like(kn)
        dv = torch.zeros_like(vf)
        dg = torch.zeros_like(gf)
        db = torch.zeros_like(bf)
        dw = torch.zeros_like(wf)
        dS = dS0
        for t in range(e - 1, s - 1, -1):
            S_dec = S_dec_stack[t]
            v_new = v_new_stack[t]
            rel = t - s

            # S_t = S_dec + k_t ⊗ v_new
            S_t = S_dec + kn[:, rel].unsqueeze(-1) * v_new.unsqueeze(-2)
            # 输出路径
            dq[:, rel] = scale * (S_t * dof[:, rel].unsqueeze(-2)).sum(-1)
            dS = dS + (scale * qn[:, rel]).unsqueeze(-1) * dof[:, rel].unsqueeze(-2)
            # 更新路径
            dk_t = (dS * v_new.unsqueeze(-2)).sum(-1)
            dv_new = (dS * kn[:, rel].unsqueeze(-1)).sum(-2)
            dw[:, rel] = dv_new * vf[:, rel]
            dv[:, rel] = dv_new * wf[:, rel]
            # 擦除门读取路径
            derase = -dv_new
            dbk = (S_dec * derase.unsqueeze(-2)).sum(-1)
            dS_dec = dS + (bf[:, rel] * kn[:, rel]).unsqueeze(-1) * derase.unsqueeze(-2)
            db[:, rel] = dbk * kn[:, rel]
            dk_t = dk_t + dbk * bf[:, rel]
            # 衰减路径
            dg[:, rel] = (dS_dec * S_dec).sum(-1)
            dS = dS_dec * gf[:, rel].unsqueeze(-1).exp()
            dk[:, rel] = dk_t
        return dq, dk, dv, dg, db, dw, dS


def get_input_groups():
    """
    生成所有测试用例输入，执行完整前向递推，保存每步 S_dec/v_new，连同 S_final/o。
    返回元组 (q, k, v, g, b, w, initial_state, do, dht, scale, use_l2norm,
              S_dec_all, v_new_all, S_final, o, cu_seqlens)
    """
    json_path = os.path.join(os.path.dirname(__file__), "32_Gdn2Bwd.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        if torch.rand(1).item() < 0.5:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)
        else:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,
    }

    input_groups = []
    for case_idx, case in enumerate(cases):
        torch.manual_seed(3407 + case_idx)

        shapes, dtypes = {}, {}
        cu_seqlens, scale, use_l2norm = None, None, 1
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "cu_seqlens":
                cu_seqlens = torch.tensor(inp["value"], dtype=torch.int64)
            elif name == "scale":
                scale = inp["value"]
            elif name == "use_qk_l2norm_in_kernel":
                use_l2norm = inp["value"]

        B, T, H, K = shapes["q"]
        V = shapes["v"][-1]

        q = F.normalize(random_tensor(shapes["q"], torch.float32), p=2, dim=-1).to(dtypes["q"])
        k = F.normalize(random_tensor(shapes["k"], torch.float32), p=2, dim=-1).to(dtypes["k"])
        v = random_tensor(shapes["v"], dtypes["v"])
        g = torch.empty(shapes["g"], dtype=torch.float32).uniform_(-1.0, -0.02).to(dtypes["g"])
        b = torch.rand(shapes["b"], dtype=dtypes["b"])
        w = torch.rand(shapes["w"], dtype=dtypes["w"])

        if shapes["initial_state"][0] == 0:
            initial_state = None
        else:
            initial_state = random_tensor(shapes["initial_state"], torch.float32)
        if shapes["dht"][0] == 0:
            dht = None
        else:
            dht = random_tensor(shapes["dht"], torch.float32)
        do = random_tensor(shapes["do"], dtypes["do"])

        # ---- 前向递推（fp32），保存 S_dec/v_new ----
        qf = q.float()
        kf = k.float()
        if use_l2norm:
            q_rstd = torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
            k_rstd = torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
            qn = qf * q_rstd
            kn = kf * k_rstd
        else:
            qn, kn = qf, kf
        vf, gf, bf, wf = v.float(), g.float(), b.float(), w.float()

        S_dec_all = []
        v_new_all = []
        o = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)
        has_init = initial_state is not None and initial_state.shape[0] > 0

        if cu_seqlens is not None and len(cu_seqlens) > 0:
            N = len(cu_seqlens) - 1
            for n in range(N):
                s, e = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
                S = qf.new_zeros(1, H, K, V)
                if has_init:
                    S = S + initial_state[n:n + 1].float()
                for t in range(s, e):
                    S_dec = S * gf[:, t].unsqueeze(-1).exp()
                    erase = ((bf[:, t].unsqueeze(-1) * kn[:, t].unsqueeze(-1)) * S_dec).sum(-2)
                    v_new = wf[:, t] * vf[:, t] - erase
                    S = S_dec + kn[:, t].unsqueeze(-1) * v_new.unsqueeze(-2)
                    o[:, t] = (qn[:, t].unsqueeze(-1) * S * scale).sum(-2)
                    S_dec_all.append(S_dec)
                    v_new_all.append(v_new)
        else:
            S = qf.new_zeros(B, H, K, V)
            if has_init:
                S = S + initial_state.float()
            for t in range(T):
                S_dec = S * gf[:, t].unsqueeze(-1).exp()
                erase = ((bf[:, t].unsqueeze(-1) * kn[:, t].unsqueeze(-1)) * S_dec).sum(-2)
                v_new = wf[:, t] * vf[:, t] - erase
                S = S_dec + kn[:, t].unsqueeze(-1) * v_new.unsqueeze(-2)
                o[:, t] = (qn[:, t].unsqueeze(-1) * S * scale).sum(-2)
                S_dec_all.append(S_dec)
                v_new_all.append(v_new)

        S_final = S
        S_dec_stack = torch.stack(S_dec_all, dim=0)
        v_new_stack = torch.stack(v_new_all, dim=0)

        input_groups.append((q, k, v, g, b, w, initial_state, do, dht,
                             scale, use_l2norm,
                             S_dec_stack, v_new_stack, S_final, o, cu_seqlens))
    return input_groups


def get_init_inputs():
    return []
