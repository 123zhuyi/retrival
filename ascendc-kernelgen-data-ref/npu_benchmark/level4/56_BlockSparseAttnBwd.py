import torch
import torch.nn as nn
import json
import os

# kernel 固定的块粒度（block streaming / blocksparse 均为 128×128 块）
BLOCK = 128


def _ceil_div(x, y):
    return (x + y - 1) // y


def _streaming_keep_blocks(sq, sk, causal, device):
    """块级 streaming 掩码的公共部分：返回 (jj, mr, valid_row)。
    jj        [1, 1, ncol]  列块号
    mr        [1, nrow, 1]  每行允许的最右块号（不含）
    valid_row [1, nrow, 1]  causal 时前 start 行整行不保留
    调用方再按各头的 sink/local 组合出 keep。"""
    nrow, ncol = _ceil_div(sq, BLOCK), _ceil_div(sk, BLOCK)
    ii = torch.arange(nrow, device=device).view(1, nrow, 1)
    jj = torch.arange(ncol, device=device).view(1, 1, ncol)
    if causal:
        start = max((sq - sk) // BLOCK, 0)
        mr = _ceil_div(max(sk - sq, 0), BLOCK) + 1 + (ii - start)
        valid_row = ii >= start
    else:
        mr = torch.full_like(ii, ncol)
        valid_row = torch.ones(1, nrow, 1, dtype=torch.bool, device=device)
    return jj, mr, valid_row

def _build_blocked_all(bs_head_idx, bs_ranks, blk_mask_b,
                       st_head_idx, st_sink, st_local,
                       sq, sk, is_causal, exact_streaming, H, device):
    """返回 [H, sq, sk] 的 blocked 掩码；无需任何掩码时返回 None。
    所有索引/参数张量均在计时循环外预计算（修改点 ④），此处零 GPU 同步。"""
    has_bs = bs_head_idx is not None
    has_st = st_head_idx is not None
    if not is_causal and not has_bs and not has_st:
        return None                                    # 全 dense 且非因果：原实现同样跳过
    blocked = torch.zeros(H, sq, sk, dtype=torch.bool, device=device)

    if has_bs:
        bm = blk_mask_b[bs_ranks]                      # [nb, nrow, ncol]，gather 一次
        nb_, nr_, nc_ = bm.shape
        # 128×128 块展开：expand+reshape（与 repeat_interleave 逐元素一致，
        # 后者在本平台走 AiCPU 慢路径）
        bm = (bm[:, :, None, :, None]
              .expand(nb_, nr_, BLOCK, nc_, BLOCK)
              .reshape(nb_, nr_ * BLOCK, nc_ * BLOCK))
        blocked[bs_head_idx] = ~bm[:, :sq, :sk]

    if has_st:
        if exact_streaming:
            row = torch.arange(sq, device=device).view(1, sq, 1)
            col = torch.arange(sk, device=device).view(1, 1, sk)
            ns = st_head_idx.numel()
            sink = st_sink.view(ns, 1, 1)
            local = st_local.view(ns, 1, 1)
            # 与官方 construct_exact_streaming_mask 逐元素一致（已实测）
            blocked_st = torch.logical_or(
                col > torch.minimum(row + sk - sq, torch.full_like(col, sk)),
                torch.logical_and(col < row + sk - sq - (local - 1),
                                  col >= sink))
            blocked[st_head_idx] = blocked_st
        else:
            jj, mr, valid_row = _streaming_keep_blocks(sq, sk, is_causal, device)
            ns = st_head_idx.numel()
            sink = st_sink.view(ns, 1, 1)
            local = st_local.view(ns, 1, 1)
            # keep = valid_row & (窗口 | sink)，blocked 取其反
            ncol = _ceil_div(sk, BLOCK)
            win = (jj >= (mr - local).clamp(min=0)) & (jj < mr.clamp(max=ncol))
            keep_blk = valid_row & (win | (jj < sink))   # [ns, nrow, ncol]
            ns_, nr_, nc_ = keep_blk.shape
            keep_blk = (keep_blk[:, :, None, :, None]
                        .expand(ns_, nr_, BLOCK, nc_, BLOCK)
                        .reshape(ns_, nr_ * BLOCK, nc_ * BLOCK))
            blocked[st_head_idx] = ~keep_blk[:, :sq, :sk]

    if is_causal:
        row = torch.arange(sq, device=device).view(1, sq, 1)
        col = torch.arange(sk, device=device).view(1, 1, sk)
        blocked |= col > row + (sk - sq)               # 原实现 keep & causal_keep 的对偶
    return blocked


class Model(nn.Module):
    """
    block_sparse_attn_func 反向 —— MIT Han Lab Block-Sparse-Attention
    的梯度 golden reference（方案 B：前向中间量预给，纯反向）。

    算子出处：
      mit-han-lab/Block-Sparse-Attention（基于 FlashAttention-2.4.2 改造）
      - 公开接口:  block_sparse_attn/block_sparse_attn_interface.py
                  BlockSparseAttnFunc.backward
    输入约束：
      q/k/v/dout dtype 一致（fp16/bf16）；q [total_q,H,D]，
      k/v [total_k,HK,D]，H % HK == 0；dout 与 q 同形；
      head_mask_type [H]（0=dense / 1=blocksparse / -1=streaming）；
      streaming_info [2H]；base_blockmask [B, nblk, nrow, ncol]；
      exact_streaming=True 时仅支持因果（官方限制）。

    输出（对 varlen 打包输入）：
      dq: [total_q, H, D]   回铸 q.dtype
      dk: [total_k, HK, D]  回铸 k.dtype
      dv: [total_k, HK, D]  回铸 v.dtype

    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type,
                streaming_info, base_blockmask, dout,
                softmax_max, softmax_sum, attention_in,
                softmax_scale, is_causal, exact_streaming):
        total_q, H, D = q.shape
        total_k, HK, _ = k.shape
        B = cu_seqlens_q.shape[0] - 1
        dtype = q.dtype
        G = H // HK
        scale = softmax_scale if softmax_scale is not None else D ** -0.5

        dq = torch.zeros(total_q, H, D, dtype=torch.float32, device=q.device)
        dk = torch.zeros(total_k, HK, D, dtype=torch.float32, device=q.device)
        dv = torch.zeros(total_k, HK, D, dtype=torch.float32, device=q.device)

        # 元数据一次同步取回（int32 小向量拼接后单次 tolist；
        # 避免逐 .item()/nonzero 的反复 host-device 往返）
        meta = torch.cat([cu_seqlens_q, cu_seqlens_k,
                          head_mask_type, streaming_info]).tolist()
        cuq = meta[:B + 1]
        cuk = meta[B + 1:2 * B + 2]
        hmt = meta[2 * B + 2:2 * B + 2 + H]
        si = meta[2 * B + 2 + H:]

        # 头下标（主机侧构造后一次上送，与设备 nonzero 逐元素一致）
        bs_head_l = [i for i, t in enumerate(hmt) if t == 1]
        st_head_l = [i for i, t in enumerate(hmt) if t == -1]
        bs_head_idx = torch.tensor(bs_head_l, device=q.device) if bs_head_l else None
        st_head_idx = torch.tensor(st_head_l, device=q.device) if st_head_l else None
        bs_ranks = torch.arange(len(bs_head_l), device=q.device) if bs_head_l else None
        st_sink = torch.tensor([si[2 * i] for i in st_head_l],
                               device=q.device) if st_head_l else None
        st_local = torch.tensor([si[2 * i + 1] for i in st_head_l],
                                device=q.device) if st_head_l else None

        for b in range(B):
            qs, qe = cuq[b], cuq[b + 1]
            ks, ke = cuk[b], cuk[b + 1]
            sq, sk = qe - qs, ke - ks
            q_b = q[qs:qe].float()                               # [sq, H, D]
            # GQA 头复用：expand+reshape（与 repeat_interleave(G, dim=1)
            # 逐元素一致，后者在本平台走 AiCPU 慢路径）
            k_b = (k[ks:ke].float()[:, :, None, :]
                   .expand(sk, HK, G, D).reshape(sk, H, D))      # [sk, H, D]
            v_b = (v[ks:ke].float()[:, :, None, :]
                   .expand(sk, HK, G, D).reshape(sk, H, D))      # [sk, H, D]
            dy_b = dout[qs:qe].float()                           # [sq, H, D]
            o_b = attention_in[qs:qe].float()                    # 前向输出，预给

            S = torch.einsum('t h d, s h d -> h t s', q_b * scale, k_b)
            blocked = _build_blocked_all(
                bs_head_idx, bs_ranks,
                base_blockmask[b] if bs_head_idx is not None else None,
                st_head_idx, st_sink, st_local,
                sq, sk, is_causal, exact_streaming, H, q.device)
            if blocked is not None:
                S.masked_fill_(blocked, float('-inf'))
            m = softmax_max[b, :, :sq].unsqueeze(-1)               # [H, sq, 1] 预给
            l = softmax_sum[b, :, :sq].unsqueeze(-1)               # [H, sq, 1] 预给
            P = torch.exp(S - m) / l
            P = torch.nan_to_num(P, nan=0.0)                       # 全掩码行 P=0
            delta = (dy_b * o_b).sum(dim=-1).transpose(0, 1).unsqueeze(-1)  # rowsum(o⊙dY) [H,sq,1]
            dP = torch.einsum('t h d, s h d -> h t s', dy_b, v_b)
            dS = P * (dP - delta)
            dv_full = torch.einsum('h t s, t h d -> s h d', P, dy_b)          # dV = Pᵀ·dY
            dq_b = torch.einsum('h t s, s h d -> t h d', dS, k_b) * scale     # dQ = dS·K·scale
            dk_full = torch.einsum('h t s, t h d -> s h d', dS, q_b) * scale  # dK = dSᵀ·Q·scale

            dq[qs:qe] = dq_b
            dk[ks:ke] = dk_full.view(sk, HK, G, D).sum(dim=2)
            dv[ks:ke] = dv_full.view(sk, HK, G, D).sum(dim=2)

        return dq.to(dtype), dk.to(dtype), dv.to(dtype)


def _json_path():
    # 验证/性能脚本会把本文件拷贝/改名（如 {op}_torch.py），
    # 除同名 .json 外回退到去掉 _torch 后缀与原始 case 文件名。
    here = os.path.dirname(os.path.abspath(__file__))
    stem = os.path.splitext(os.path.basename(__file__))[0]
    names = [stem]
    if stem.endswith("_torch"):
        names.append(stem[: -len("_torch")])
    names.append("100_BlockSparseAttnBwd")
    for name in names:
        path = os.path.join(here, name + ".json")
        if os.path.isfile(path):
            return path
    return os.path.join(here, names[-1] + ".json")


def get_input_groups():
    with open(_json_path(), "r", encoding="utf-8-sig") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}

    def random_tensor(shape, dtype):
        # 标准分布：50% 均匀 U[-5,5] + 50% 正态 N(mu, sigma)
        if torch.rand(1).item() < 0.5:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)
        else:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)

    def gen_base_blockmask(B, nblk, nrow, ncol, sparsity, causal):
        # 输入生成侧（不计时），保持原实现不动，与官方
        # generate_base_sparsity_mask 逐行同构
        bm = torch.zeros(B, nblk, nrow, ncol, dtype=torch.bool)
        for b in range(B):
            for h in range(nblk):
                s = sparsity
                if s != 0.0 and s != 1.0:
                    for i in range(nrow):
                        idx = nrow - i - 1
                        avail = max(0, ncol - i) if causal else ncol
                        num_one = max(1, int(s * avail))
                        bm[b, h, idx, torch.randperm(avail)[:num_one]] = True
                elif s == 1.0:
                    bm[b, h] = True
        return bm

    input_groups = []
    for case_idx, case in enumerate(cases):
        # 每 case 独立播种，输入可复现且与 case 顺序/子集无关
        torch.manual_seed(3407 + case_idx)

        inputs = case["inputs"]

        def info(name):
            return next(i for i in inputs if i["name"] == name)

        dt = dtype_map[info("q")["dtype"]]
        # varlen 序列配置 / GQA / 稀疏度 / streaming 参数改由 JSON attrs 提供，
        # 使单个 .py 可驱动任意数量（如 50）个 case。
        q_lens = list(info("q_lens")["value"])
        k_lens = list(info("k_lens")["value"])
        B = len(q_lens)
        H, D = info("q")["shape"][1], info("q")["shape"][2]
        HK = int(info("hk")["value"])
        scale = info("softmax_scale")["value"]
        is_causal = int(info("is_causal")["value"])
        exact_streaming = int(info("exact_streaming")["value"])
        sparsity = float(info("sparsity")["value"])
        sink = int(info("sink")["value"])
        local = int(info("local")["value"])

        q = random_tensor(info("q")["shape"], dt)
        k = random_tensor(info("k")["shape"], dt)
        v = random_tensor(info("v")["shape"], dt)
        dout = random_tensor(info("dout")["shape"], dt)

        cu_q = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32)
        cu_k = torch.tensor([0] + list(torch.tensor(k_lens).cumsum(0)), dtype=torch.int32)

        ns, nb = H // 3, H // 3
        head_mask_type = torch.tensor([0] * (H - ns - nb) + [1] * nb + [-1] * ns,
                                      dtype=torch.int32)
        streaming_info = torch.tensor([sink, local] * H, dtype=torch.int32)

        max_q, max_k = max(q_lens), max(k_lens)
        nrow = (max_q + BLOCK - 1) // BLOCK
        ncol = (max_k + BLOCK - 1) // BLOCK
        base_blockmask = gen_base_blockmask(B, nb, nrow, ncol,
                                            sparsity, is_causal)

        G = H // HK
        sc = scale if scale is not None else D ** -0.5
        softmax_max = torch.zeros(B, H, max_q, dtype=torch.float32)
        softmax_sum = torch.ones(B, H, max_q, dtype=torch.float32)
        attention_in = torch.zeros(sum(q_lens), H, D, dtype=torch.float32)

        # ════════════════════════════════════════════════════════════════
        # 修改点 ⑤：统计量预计算与 forward 共用同一个向量化掩码构造函数
        # （原实现此处复制了与 forward 相同的 for b / for h 双重循环 +
        #  _head_keep_mask；虽不计时，但两份代码有漂移风险，且更慢）。
        # 索引预计算方式与 forward 的修改点 ④ 完全一致。
        # ════════════════════════════════════════════════════════════════
        cuq, cuk = cu_q.tolist(), cu_k.tolist()
        hmt_l = head_mask_type.tolist()
        is_bs = torch.tensor([t == 1 for t in hmt_l])
        is_st = torch.tensor([t == -1 for t in hmt_l])
        bs_head_idx = is_bs.nonzero(as_tuple=True)[0]
        st_head_idx = is_st.nonzero(as_tuple=True)[0]
        bs_ranks = (is_bs.cumsum(0) - 1)[bs_head_idx] if bs_head_idx.numel() else None
        st_sink = streaming_info[2 * st_head_idx] if st_head_idx.numel() else None
        st_local = streaming_info[2 * st_head_idx + 1] if st_head_idx.numel() else None
        bs_head_idx = bs_head_idx if bs_head_idx.numel() else None
        st_head_idx = st_head_idx if st_head_idx.numel() else None

        for b in range(B):
            qs, qe = cuq[b], cuq[b + 1]
            ks, ke = cuk[b], cuk[b + 1]
            sq, sk = qe - qs, ke - ks
            q_b = q[qs:qe].float()
            k_b = k[ks:ke].float().repeat_interleave(G, dim=1)
            v_b = v[ks:ke].float().repeat_interleave(G, dim=1)
            S = torch.einsum('t h d, s h d -> h t s', q_b * sc, k_b)
            # 原: for h in range(H): keep = _head_keep_mask(...); S[h].masked_fill_
            blocked = _build_blocked_all(
                bs_head_idx, bs_ranks,
                base_blockmask[b] if bs_head_idx is not None else None,
                st_head_idx, st_sink, st_local,
                sq, sk, is_causal, exact_streaming, H, q.device)
            if blocked is not None:
                S.masked_fill_(blocked, float('-inf'))
            m = S.amax(dim=-1)                          # [H, sq]
            p = torch.nan_to_num(torch.exp(S - m.unsqueeze(-1)), nan=0.0)
            l = p.sum(dim=-1)                           # [H, sq]，全掩码行为 0
            P = torch.nan_to_num(p / l.unsqueeze(-1), nan=0.0)
            # 全掩码行存 max=0/sum=1：golden 重算的 S 该行全 -inf，
            # exp(-inf-0)=0 → P 行恒 0，统计量取值不影响语义
            row_empty = (l == 0)
            m = torch.where(row_empty, torch.zeros_like(m), m)
            l = torch.where(row_empty, torch.ones_like(l), l)
            softmax_max[b, :, :sq] = m
            softmax_sum[b, :, :sq] = l
            o_b = torch.einsum('h t s, s h d -> t h d', P, v_b)
            attention_in[qs:qe] = o_b                   # fp32 存储，保 delta 精度

        input_groups.append([q, k, v, cu_q, cu_k, head_mask_type, streaming_info,
                             base_blockmask, dout,
                             softmax_max, softmax_sum, attention_in,
                             scale, is_causal, exact_streaming])
    return input_groups


def get_init_inputs():
    return []