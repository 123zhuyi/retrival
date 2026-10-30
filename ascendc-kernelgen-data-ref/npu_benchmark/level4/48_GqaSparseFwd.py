import json
import os

import torch
import torch.nn as nn

SPARSE_BLOCK_SIZE = 128  # 一个稀疏块 == 一个 KV 页


class Model(nn.Module):
    """
    MiniMax-M3 GQA block-sparse attention (prefill) 的性能基线（golden），
    对齐 vllm-project/vllm @ ee5a89f4d7b8
    `vllm/models/minimax_m3/common/ops/sparse_attn.py` 的
    `minimax_m3_sparse_attn`（block_size_q == 1 路径，kernel
    `_gqa_sparse_fwd_kernel`）。

    逐点语义（与 kernel 的对应关系）：
    - 可见块数：q_abs = prefix_len + j（第 b 个请求第 j 个 query），
      valid_blocks = (q_abs + 128) // 128，real_topk = min(topk,
      valid_blocks)；kernel 顺序读前 real_topk 个槽位、尾部不读。本实现
      用等价掩码：slot j 有效 ⟺ j*BS <= q_abs（j < valid_blocks 的
      整数等价式，j < topk 由形状保证），尾部槽位 -inf 屏蔽，与"不读"
      数学等价。
    - KV 布局 [num_blocks, KH, 128, 2D]，K=[..., :D]、V=[..., D:]；
      逻辑块经 block_table[b, blk] 翻页。gather 用两级 index_select
      平铺索引（先 block_table 后 kv_cache，int32 索引无 IndexCheck
      伴随 kernel），一次性取回 [TQ, KH, topk*128, 2D]。
    - 掩码：pos <= q_abs（因果）。kernel 另有 pos < seq_len 一项——
      合法 prefill 输入域内 q_abs = prefix + j <= seq_len - 1 恒成立，
      被因果项蕴含，故 lim = q_abs（域内逐点等价；输入生成器保证
      选中块号 < valid_blocks，与 kernel 读取范围一致）。
    - GQA：q 头连续分组 [kh*G, (kh+1)*G)（kernel pid_h = kh*G 同序）；
      query reshape [TQ, KH, G, D] 直接按 kv head 分组计算，无 KV 头拷贝。
    - QK/softmax/PV 精度路径逐点镜像 kernel 的 **online base-2 softmax**
      （非单遍自然底 softmax）：按 topk 槽位顺序逐块迭代（kernel 顺序读
      前 real_topk 个槽位），qk = dot(q,k) * sm_scale_log2e 在 log2 域
      （sm_scale_log2e = sm_scale * 1.4426950409，kernel host 侧常量），
      running max m_i 逐块重缩放 acc（exp2 因子），lse 以
      m_ij + log2(exp2(lse - m_ij) + l_ij) 递推，末步 acc * exp2(m - lse)
      归一。无效槽位（j >= real_topk）以全 -inf 行喂入，更新为精确
      空操作（p=0、rescale=1、lse 不变），与 kernel 不读该槽位等价。
      16-bit 输入时 QK 为精确乘积 + fp32 累加，PV 前 P（未归一、相对
      当前块 max）舍回 v.dtype（kernel `tl.dot(p.to(v.dtype), v)` 同）；
      fp32 输入时 kernel 两处 `tl.dot` 走 TF32（10 位尾数，
      orig_vllm_sparse_attn/sparse_attn.py:172 QK、:200 PV 两处，
      PV 前 `p.to(v.dtype)` 对 fp32 为恒等），本实现对 fp32 case 的
      QK 两侧与 PV 两侧操作数做 TF32 RZ 截断（位模式清低 13 位）镜像。
    - 全掩码行：合法输入下不可达（每个 query 至少看见自己 pos=q_abs），
      nan_to_num 为防御性置 0，不影响域内语义。
    - 输出 fp32 累加后 cast 回 q.dtype（kernel 同）。
    - 单请求（B==1）走省略 req_id 的等价快路径（req_id 恒 0）；
      列索引 arange 按 (device, size) 缓存，warmup 后零开销。

    与 level4 初版的差异：5-D 高级索引 gather（NPU 上落到
    IndexAiCpu，单次 ~377us）改两级 index_select；einsum 改批量
    matmul（k 切片 strides 直接进 GEMM，免布局拷贝）；req_id 的
    repeat_interleave（~75us）改比较+求和；arange 全部缓存；
    .long() 转换消除；掩码构造改 4-D 广播（免 repeat_interleave
    展开）。语义不变，小 shape 固定开销 ~630us -> ~87us。
    """

    def __init__(self):
        super().__init__()
        self._col_cache = {}

    @staticmethod
    def _tf32_round(x):
        # tl.dot fp32 操作数走 TF32（保留 10 位尾数）。该 GPU 上 fp32→tf32
        # 转换为 RZ（直接截断低 13 位）——41 号算子用 RZ 在 CPU 精确复现
        # 了 GPU 实测误差（6.5465e-3 vs 6.55e-3），RNA 不符
        xi = x.contiguous().view(torch.int32)
        return (xi & ~0x1FFF).view(torch.float32)

    def _col(self, device, size):
        key = (str(device), size)
        if key not in self._col_cache:
            self._col_cache[key] = torch.arange(size, device=device,
                                                dtype=torch.int32)
        return self._col_cache[key]

    def forward(self, q, kv_cache, topk_idx, block_table, cu_seqlens_q,
                seq_lens, prefix_lens, sm_scale):
        total_q, H, D = q.shape
        KH = kv_cache.shape[1]
        BS = SPARSE_BLOCK_SIZE
        G = H // KH
        topk = topk_idx.shape[-1]
        B = cu_seqlens_q.numel() - 1
        MB = block_table.shape[1]
        device = q.device
        is_fp32 = q.dtype == torch.float32
        # kernel host 侧 sm_scale_log2e = sm_scale * 1.4426950409（fp64 乘积
        # 以 fp32 标量入 kernel），QK 全程 log2 域
        scale2 = sm_scale * 1.4426950409

        pid = self._col(device, total_q)
        if B == 1:
            # 单请求：req_id 恒 0，q_abs = prefix + j（[1] 广播免索引）
            q_abs = pid + prefix_lens.to(device)                    # [TQ]
            rows = torch.zeros(total_q, dtype=torch.int64, device=device)
        else:
            cu = cu_seqlens_q.to(device)
            rows = (pid[:, None] >= cu[1:][None, :]).sum(1)         # [TQ]
            delta = prefix_lens.to(device) - cu[:-1]                # [B]
            q_abs = pid + torch.index_select(delta, 0, rows)        # [TQ]

        # kernel: valid_blocks = (q_abs + 128) // 128, real_topk = min(topk, ·)
        real_topk = torch.clamp((q_abs + BS) // BS, max=topk)       # [TQ]

        q_g = q.float().view(total_q * KH, G, D)
        if is_fp32:
            q_g = self._tf32_round(q_g)

        # online base-2 softmax 状态（kernel: m_i/lse_i fp32, acc fp32）
        m_i = torch.full((total_q * KH, G), float("-inf"), device=device)
        lse_i = torch.full((total_q * KH, G), float("-inf"), device=device)
        acc = torch.zeros(total_q * KH, G, D, device=device)

        tk = topk_idx.permute(1, 0, 2)                              # [TQ,KH,tk]
        bt_flat = block_table.to(device).reshape(-1)
        kv_flat = kv_cache.view(-1, BS * 2 * D)
        col_kh = self._col(device, KH)
        col_bs = self._col(device, BS)
        q_abs_r = q_abs.repeat_interleave(KH)                       # [TQ*KH]

        # 逐槽位迭代（kernel 顺序读前 real_topk 槽；无效槽喂全 -inf 行，
        # 更新为精确空操作）
        for j in range(topk):
            slot_ok = j < real_topk                                 # [TQ]
            blk = tk[:, :, j].clamp(min=0).long()                   # [TQ,KH]
            pages = torch.index_select(
                bt_flat, 0, (rows[:, None] * MB + blk).reshape(-1)
            ).view(total_q, KH)
            kv_idx = (pages.long() * KH + col_kh[None, :]).reshape(-1)
            kv = torch.index_select(kv_flat, 0, kv_idx).view(
                total_q * KH, BS, 2 * D).float()
            k_b = kv[..., :D]
            v_b = kv[..., D:]
            if is_fp32:
                k_b = self._tf32_round(k_b)

            qk = torch.bmm(q_g, k_b.transpose(1, 2)) * scale2
            pos = blk.reshape(-1)[:, None] * BS + col_bs[None, :]   # [TQ*KH,BS]
            invalid = (pos > q_abs_r[:, None]) \
                | (~slot_ok).repeat_interleave(KH)[:, None]
            qk = qk.masked_fill(invalid[:, None, :], float("-inf"))

            m_ij = torch.maximum(m_i, qk.max(dim=-1).values)
            p = torch.nan_to_num(torch.exp2(qk - m_ij.unsqueeze(-1)),
                                 nan=0.0)
            l_ij = p.sum(dim=-1)
            alpha = torch.nan_to_num(torch.exp2(m_i - m_ij), nan=0.0)
            if is_fp32:
                p_r = self._tf32_round(p)
                v_b = self._tf32_round(v_b)
            else:
                # kernel: tl.dot(p.to(v.dtype), v) —— 未归一 P 先舍回 KV dtype
                p_r = p.to(kv_cache.dtype).float()
            acc = acc * alpha.unsqueeze(-1) + torch.bmm(p_r, v_b)
            # kernel: lse_i = m_ij + log2(exp2(lse_i - m_ij) + l_ij)
            lse_new = m_ij + torch.log2(torch.exp2(lse_i - m_ij) + l_ij)
            upd = slot_ok.repeat_interleave(KH)[:, None]
            lse_i = torch.where(
                upd, torch.nan_to_num(lse_new, nan=float("-inf")), lse_i)
            m_i = torch.where(upd, m_ij, m_i)

        o = acc * torch.exp2(m_i - lse_i).unsqueeze(-1)
        return o.view(total_q, H, D).to(q.dtype)


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
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,  # topk_idx/block_table 仅声明形状, 实际由逻辑生成
    }
    BS = SPARSE_BLOCK_SIZE

    def random_tensor(shape, dtype, gen):
        # 标准输入分布: 50% 均匀 [-5, 5] + 50% 正态 (mu ∈ [-5, 5], sigma ∈ [0.1, 2])
        if torch.rand(1, generator=gen).item() < 0.5:
            return torch.empty(shape, dtype=torch.float32).uniform_(
                -5.0, 5.0, generator=gen).to(dtype)
        mu = float(torch.empty(1).uniform_(-5.0, 5.0, generator=gen).item())
        sigma = float(torch.empty(1).uniform_(0.1, 2.0, generator=gen).item())
        return torch.normal(mu, sigma, shape, generator=gen,
                            dtype=torch.float32).to(dtype)

    input_groups = []
    for case_idx, case in enumerate(cases):
        # 每 case 独立播种，保证输入可复现且与 case 顺序/子集无关
        gen = torch.Generator().manual_seed(3407 + case_idx)

        shapes, dtypes = {}, {}
        cu_seqlens_q = seq_lens = prefix_lens = None
        sm_scale = None
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "cu_seqlens_q":
                cu_seqlens_q = inp["value"]
            elif name == "seq_lens":
                seq_lens = inp["value"]
            elif name == "prefix_lens":
                prefix_lens = inp["value"]
            elif name == "sm_scale":
                sm_scale = inp["value"]

        total_q, H, D = shapes["q"]
        KH = shapes["kv_cache"][1]
        topk = shapes["topk_idx"][-1]
        B = len(seq_lens)
        assert cu_seqlens_q[0] == 0 and cu_seqlens_q[-1] == total_q
        assert shapes["topk_idx"] == [KH, total_q, topk]
        assert H % KH == 0
        q_lens = [cu_seqlens_q[b + 1] - cu_seqlens_q[b] for b in range(B)]
        # prefill 语义: kv 总长 = prefix + 本次 query 长度
        assert all(s == p + ql for s, p, ql in zip(seq_lens, prefix_lens, q_lens))

        q = random_tensor(shapes["q"], dtypes["q"], gen)
        kv_cache = random_tensor(shapes["kv_cache"], dtypes["kv_cache"], gen)
        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
        prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32)

        # 分页: 每请求逻辑块数 = ceil(seq_len/128), 页 id 随机打乱 + 3 冗余页
        nblocks = [(s + BS - 1) // BS for s in seq_lens]
        total_pages = sum(nblocks) + 3
        assert shapes["kv_cache"][0] == total_pages
        max_blocks = max(nblocks)
        perm = torch.randperm(total_pages, generator=gen)
        block_table = torch.zeros(B, max_blocks, dtype=torch.int32)
        off = 0
        for b, n in enumerate(nblocks):
            block_table[b, :n] = perm[off:off + n].int()
            off += n

        # topk_idx [KH, total_q, topk]: 每个 (kh, query token) 从因果可见块中
        # 无放回采样 real_topk 个逻辑块号并升序排列 (模拟 indexer 输出, 无哨兵);
        # 尾部 (kernel 不读取) 填范围内的随机块号
        topk_idx = torch.zeros(KH, total_q, topk, dtype=torch.int32)
        for b in range(B):
            qs = cu_seqlens_q[b]
            for j in range(q_lens[b]):
                q_abs = prefix_lens[b] + j
                valid_blocks = (q_abs + BS) // BS
                n_sel = min(topk, valid_blocks)
                for kh in range(KH):
                    sel = torch.randperm(valid_blocks, generator=gen)[:n_sel].sort().values
                    topk_idx[kh, qs + j, :n_sel] = sel.int()
                    if n_sel < topk:
                        topk_idx[kh, qs + j, n_sel:] = torch.randint(
                            0, valid_blocks, (topk - n_sel,), generator=gen,
                            dtype=torch.int32)

        input_groups.append([q, kv_cache, topk_idx, block_table, cu_seqlens_q_t,
                             seq_lens_t, prefix_lens_t, sm_scale])
    return input_groups


def get_init_inputs():
    return []
