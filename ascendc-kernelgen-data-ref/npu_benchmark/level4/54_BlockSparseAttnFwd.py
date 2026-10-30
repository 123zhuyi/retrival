import torch
import torch.nn as nn
import json
import math
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
    """返回 [H, sq, sk] 的 blocked 掩码（True = 屏蔽）；无需任何掩码时返回 None。
    全部头一次向量化构造，与逐头实现逐元素一致。"""
    has_bs = bs_head_idx is not None
    has_st = st_head_idx is not None
    if not is_causal and not has_bs and not has_st:
        return None
    blocked = torch.zeros(H, sq, sk, dtype=torch.bool, device=device)

    if has_bs:
        bm = blk_mask_b[bs_ranks]                      # [nb, nrow, ncol]，gather 一次
        nb_, nr_, nc_ = bm.shape
        # 128×128 块展开：expand+reshape（纯数据搬运，与 repeat_interleave 逐元素一致，
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
            # 与官方 construct_exact_streaming_mask 逐元素一致
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
        blocked |= col > row + (sk - sq)               # 与因果掩码取交的对偶
    return blocked


class Model(nn.Module):
    """
    block_sparse_attn_func 前向 —— MIT Han Lab Block-Sparse-Attention
    （varlen 打包布局的块稀疏注意力）golden reference。

    算子出处：
      mit-han-lab/Block-Sparse-Attention（基于 FlashAttention 2.4.2 修改）
      - 公开接口:  block_sparse_attn/block_sparse_attn_interface.py
                  block_sparse_attn_func（BlockSparseAttnFunc，fwd+bwd 均支持）
      - 语义依据:  block_sparse_tests/fwd_bwd/test_correctness/utils.py
                  attention_blocksparse_ref（官方单测的 PyTorch reference）
                  及 mask 生成器 generate_base_sparsity_mask /
                  generate_streaming_mask / construct_exact_streaming_mask
      数学语义逐行对齐上述 reference（fp32 计算、mask 约定、全掩码行处置）。

    数学语义（varlen 打包，逐序列、逐头）：
      输入  q: [total_q, H, D]，k/v: [total_k, HK, D]（MHA/GQA/MQA：
            G = H // HK，k/v 沿头维 repeat_interleave(G) 复用），
            cu_seqlens_q/k: [B+1] int32（变长序列边界），
            head_mask_type: [H] int32（0=dense，1=blocksparse，-1=streaming），
            streaming_info: [2H] int32（逐头 [sink_blocks, local_blocks]；
            exact_streaming=True 时为 token 粒度的 [sink_tokens, local_tokens]），
            base_blockmask: [B, nblk, nrow, ncol] bool（True=保留；
            nrow=ceil(round128(max_seqlen_q)/128)，ncol 同；仅 blocksparse 头使用，
            第 r 个 blocksparse 头（head_mask_type 中第 r 个 1）取 base_blockmask[:, r]）
      对序列 b、头 h 的 token 级保留掩码 keep[b,h] ∈ {0,1}^{sq×sk}：
        dense:        全 1
        blocksparse:  base_blockmask[b, r] 按 128×128 展开后裁到 [sq, sk]
        streaming（块粒度，exact_streaming=False）: 行块 i 保留
                      前 sink 个列块 与 对角前 local 个列块，即列块区间
                      [max(mr-local,0), mr)，mr = ceil(max(sk-sq,0)/128)+1+(i-i0)，
                      i0 = max((sq-sk)//128, 0)（causal），非因果 mr=ncol
        streaming（token 粒度，exact_streaming=True，仅因果）:
                      keep = (col <= min(row+sk-sq, sk)) &
                             ~( (col < row+sk-sq-(local-1)) & (col >= sink) )
        causal 时以上所有掩码再与因果掩码取交：keep &= (col <= row + sk - sq)
        （底右对齐，sk-sq 为序列长度差；掩码位置 softmax 前填 -inf）
      注意力（fp32）：
        scores = (q·softmax_scale) @ k^T     （scale 默认 D^-0.5，乘在 q 上）
        scores[masked] = -inf
        attn = softmax(scores, dim=-1)；attn[masked] = 0
        （全掩码行 softmax 为 NaN，经此步置 0 —— 与官方 reference 一致，
          即全掩码 query 行输出定义为 0）
        o = attn @ v，回铸 q.dtype → [total_q, H, D]

    dtype 约定：内部全程 fp32（官方 reference upcast=True 路径）；输出回铸
      q.dtype。kernel 仅支持 fp16/bf16，故用例不含 fp32。

    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type,
                streaming_info, base_blockmask, softmax_scale,
                is_causal, exact_streaming):
        # ---- 形状自洽校验（对齐 block_sparse_attn_func 入口）----
        total_q, H, D = q.shape
        total_k, HK, _ = k.shape
        assert k.shape == v.shape and k.shape[2] == D
        assert H % HK == 0, f"H({H}) 必须整除 HK({HK})"
        assert q.dtype == k.dtype == v.dtype
        B = cu_seqlens_q.shape[0] - 1
        assert cu_seqlens_k.shape[0] == B + 1
        assert head_mask_type.shape == (H,) and streaming_info.shape == (2 * H,)

        # 元数据一次同步取回（int32 小向量拼接后单次 tolist；
        # 避免逐 .item()/nonzero 的反复 host-device 往返）
        meta = torch.cat([cu_seqlens_q, cu_seqlens_k,
                          head_mask_type, streaming_info]).tolist()
        cuq = meta[:B + 1]
        cuk = meta[B + 1:2 * B + 2]
        hmt_l = meta[2 * B + 2:2 * B + 2 + H]
        si_l = meta[2 * B + 2 + H:]
        assert cuq[-1] == total_q and cuk[-1] == total_k
        nblk = sum(1 for t in hmt_l if t == 1)
        nrow = (max(cuq[i + 1] - cuq[i] for i in range(B)) + BLOCK - 1) // BLOCK
        ncol = (max(cuk[i + 1] - cuk[i] for i in range(B)) + BLOCK - 1) // BLOCK
        assert base_blockmask.shape == (B, nblk, nrow, ncol), \
            f"base_blockmask 形状错误: {base_blockmask.shape} vs {(B, nblk, nrow, ncol)}"
        if exact_streaming:
            assert is_causal, "exact_streaming 仅支持因果（官方断言）"

        dtype = q.dtype
        G = H // HK
        scale = softmax_scale if softmax_scale is not None else D ** -0.5
        out = torch.zeros_like(q, dtype=torch.float32)

        # 头下标预计算（主机侧构造后一次上送，与设备 nonzero 逐元素一致）：
        # blocksparse 头及其在 base_blockmask 第 1 维的序号、
        # streaming 头及其 sink/local
        bs_head_l = [i for i, t in enumerate(hmt_l) if t == 1]
        st_head_l = [i for i, t in enumerate(hmt_l) if t == -1]
        bs_head_idx = torch.tensor(bs_head_l, device=q.device) if bs_head_l else None
        st_head_idx = torch.tensor(st_head_l, device=q.device) if st_head_l else None
        bs_ranks = torch.arange(len(bs_head_l), device=q.device) if bs_head_l else None
        st_sink = torch.tensor([si_l[2 * i] for i in st_head_l],
                               device=q.device) if st_head_l else None
        st_local = torch.tensor([si_l[2 * i + 1] for i in st_head_l],
                                device=q.device) if st_head_l else None

        for b in range(B):
            qs, qe = cuq[b], cuq[b + 1]
            ks, ke = cuk[b], cuk[b + 1]
            sq, sk = qe - qs, ke - ks
            q_b = q[qs:qe].float()                       # [sq, H, D]
            # GQA 头复用：expand+reshape（与 repeat_interleave(G, dim=1) 逐元素一致，
            # 后者在本平台走 AiCPU 慢路径）
            k_b = (k[ks:ke].float()[:, :, None, :]
                   .expand(sk, HK, G, D).reshape(sk, H, D))
            v_b = (v[ks:ke].float()[:, :, None, :]
                   .expand(sk, HK, G, D).reshape(sk, H, D))

            scores = torch.einsum('t h d, s h d -> h t s',
                                  q_b * scale, k_b)    # [H, sq, sk] fp32
            blocked = _build_blocked_all(
                bs_head_idx, bs_ranks,
                base_blockmask[b] if bs_head_idx is not None else None,
                st_head_idx, st_sink, st_local,
                sq, sk, is_causal, exact_streaming, H, q.device)
            if blocked is not None:
                scores.masked_fill_(blocked, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            if blocked is not None:
                attn.masked_fill_(blocked, 0.0)        # 全掩码行 NaN → 0
            out[qs:qe] = torch.einsum('h t s, s h d -> t h d', attn, v_b)
        return out.to(dtype)


def _json_path():
    # 验证/性能脚本会把本文件拷贝/改名（如 {op}_torch.py），
    # 除同名 .json 外回退到去掉 _torch 后缀的名字。
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
        assert len(k_lens) == B
        H, D = info("q")["shape"][1], info("q")["shape"][2]
        HK = int(info("hk")["value"])
        assert H % HK == 0
        scale = info("softmax_scale")["value"]
        is_causal = int(info("is_causal")["value"])
        exact_streaming = int(info("exact_streaming")["value"])
        sparsity = float(info("sparsity")["value"])
        sink = int(info("sink")["value"])
        local = int(info("local")["value"])

        q = random_tensor(info("q")["shape"], dt)
        k = random_tensor(info("k")["shape"], dt)
        v = random_tensor(info("v")["shape"], dt)

        cu_q = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32)
        cu_k = torch.tensor([0] + list(torch.tensor(k_lens).cumsum(0)), dtype=torch.int32)

        # 官方头配比：1/3 streaming + 1/3 blocksparse + 其余 dense
        ns, nb = H // 3, H // 3
        head_mask_type = torch.tensor([0] * (H - ns - nb) + [1] * nb + [-1] * ns,
                                      dtype=torch.int32)
        streaming_info = torch.tensor([sink, local] * H, dtype=torch.int32)

        max_q, max_k = max(q_lens), max(k_lens)
        nrow = (max_q + BLOCK - 1) // BLOCK
        ncol = (max_k + BLOCK - 1) // BLOCK
        base_blockmask = gen_base_blockmask(B, nb, nrow, ncol,
                                            sparsity, is_causal)

        # JSON 形状自洽校验
        assert info("q")["shape"] == [sum(q_lens), H, D]
        assert info("k")["shape"] == [sum(k_lens), HK, D]
        assert info("v")["shape"] == [sum(k_lens), HK, D]
        assert info("cu_seqlens_q")["shape"] == [B + 1]
        assert info("cu_seqlens_k")["shape"] == [B + 1]
        assert info("head_mask_type")["shape"] == [H]
        assert info("streaming_info")["shape"] == [2 * H]
        assert info("base_blockmask")["shape"] == [B, nb, nrow, ncol]

        input_groups.append([q, k, v, cu_q, cu_k, head_mask_type, streaming_info,
                             base_blockmask, scale, is_causal, exact_streaming])
    return input_groups


def get_init_inputs():
    return []