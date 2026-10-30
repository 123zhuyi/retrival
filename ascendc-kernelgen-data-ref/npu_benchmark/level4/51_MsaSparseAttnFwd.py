import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    SM100 MSA block-sparse attention forward (prefill) 标杆。
    对标 fmha_sm100 的 sparse_atten_func
    (vllm-project/MSA @ 9175bebf, python/fmha_sm100/cute/interface.py),
    vLLM 调用点: MiniMaxM3SparseMSAImpl.forward 的 prefill 路径
    (vllm/models/minimax_m3/nvidia/sparse_attention_msa.py),
    即 MiniMax-M3 lightning indexer 选出 top-k KV 块后的主稀疏注意力。

    数学定义 (对每个 batch b、请求内查询序号 qi、q-head h, 所属 kv-head
    g = h // (Hq/Hkv); GQA 组内所有 q-head 共享同一块选择):
        可见块集合 S(g, b, qi) = { q2k 选择的块 }   (由 k2q CSR 逆推还原)
        可见 token: t ∈ S 展开的 blk_kv 个位置, 且
            seqused 掩码: t < seqused_k[b]
            因果掩码 (causal=1, bottom-right 对齐):
                t ≤ qi + (seqused_k[b] - qlen_b)
        scores[t] = softmax_scale * q[b, qi, h] · k[b, t, g]
        out[b, qi, h] = softmax(scores | 可见) · v[b, :, g]
        无可见 token 的行输出 0 (kernel 同语义: row_has_value 处理)

    布局约定:
        q              [total_q, Hq, 128]          bf16 (varlen 拼接)
        k, v           [num_pages, Hkv, blk_kv, 128] bf16 paged KV 缓存;
                       kv 位置 t -> 页 page_table[b, t // blk_kv],
                       页内偏移 t % blk_kv
        k2q_row_ptr    [Hkv, total_rows + 1]       int32 CSR 行指针;
                       全局行 r <-> (batch b, batch 内块号 j), 按 "块号优先、
                       批次次之" 列优先线性化 (build_k2q_csr 的打包顺序)
        k2q_q_indices  [Hkv, total_q * topK]       int32, 每行内存 batch 局部
                       q 序号 (按 (行, q) 稳定排序), 尾部 -1 填充
        cu_seqlens_q   [B+1] int32  q 前缀和
        cu_seqlens_k   [B+1] int32  kv 前缀和 (与 seqused_k 一致)
        seqused_k      [B]   int32  每请求有效 kv 长度
        page_table     [B, max_pages] int32
        topK           int, 每查询选中的 KV 块数 (kernel 支持 4/8/16/32)
        blk_kv         int, 稀疏块大小 (= 页大小, MiniMax-M3 为 128)
        causal         int (0/1), MSA prefill 路径恒为 1
        softmax_scale  float
        输出           [total_q, Hq, 128] bf16
    实现说明 (向量化版本):
        - CSR 行逆映射由 meshgrid 一次性构造, 与 _build_packed_row_map 的
          "块号优先、批次次之" 顺序逐位一致 (纯整数索引, 无数值差异)。
        - 注意力计算按 (batch, kv-head) 语义分组 (varlen 请求与 GQA 头组
          是语义边界; 全量物化选中 token 需 TB 级内存, 不可全批量):
          每组将选中块展开为 token 索引 [qlen, topK*blk_kv], 一次 gather
          取 k/v, token 级掩码同时覆盖 seqused 截断与因果 (bottom-right),
          masked softmax 后加权求和。与逐行实现数学等价, 仅存在 fp32 归约
          顺序差异 (实测 bf16 输出逐位一致或极少数 1-ULP)。
    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, k2q_row_ptr, k2q_q_indices,
                cu_seqlens_q, cu_seqlens_k, seqused_k, page_table,
                topK, max_seqlen_q, max_seqlen_k, blk_kv, causal,
                softmax_scale):
        device = q.device
        total_q, Hq, D = q.shape
        Hkv = k.shape[1]
        G = Hq // Hkv
        B = cu_seqlens_q.numel() - 1
        cu_q = cu_seqlens_q.long()
        topK, blk_kv = int(topK), int(blk_kv)

        # 主机侧元数据合并为一次同步取回（整数索引无数值差异；
        # 单次 cat+tolist 避免多次 host-device 往返）
        meta = torch.cat([cu_seqlens_q, cu_seqlens_k, seqused_k]).tolist()
        cuq = meta[:B + 1]
        cuk = meta[B + 1:2 * B + 2]
        seqused = meta[2 * B + 2:]

        # ---- 1. CSR 全局行 -> (batch, 块号) 逆映射 (与 build_k2q_csr 一致) ----
        # "块号优先、批次次之" 列优先线性化；行数规模小（total_rows），
        # 在 CPU 构造后一次上送，无数值差异
        kv_lens = [cuk[b + 1] - cuk[b] for b in range(B)]
        rows_pb = [(L + blk_kv - 1) // blk_kv for L in kv_lens]
        total_rows = k2q_row_ptr.shape[1] - 1
        max_rows = max(rows_pb) if B > 0 else 0
        row_batch_l, row_block_l = [], []
        for j in range(max_rows):
            for b in range(B):
                if j < rows_pb[b]:
                    row_batch_l.append(b)
                    row_block_l.append(j)
        assert len(row_batch_l) == total_rows, "k2q_row_ptr 行数与 cu_seqlens_k 不一致"
        row_batch = torch.tensor(row_batch_l, dtype=torch.long, device=device)
        row_block = torch.tensor(row_block_l, dtype=torch.long, device=device)

        # ---- 2. k2q CSR -> q2k [Hkv, total_q, topK] (集合语义还原) ----
        # 全设备向量化（零额外同步）：searchsorted 展开 CSR 行，
        # 与逐 head repeat_interleave 的 (行, q) 扫描序逐位一致；
        # 稳定排序键只取 qg：同一 q 的条目在 CSR 行扫描序下本就按 batch 内
        # 块号升序（行线性化是块号外层、批次内层，单 q 只属于一个 batch），
        # 稳定排序保持该次序；fp32 键在 total_q < 2^24 下精确且留在 AiCore。
        counts = (k2q_row_ptr[:, 1:] - k2q_row_ptr[:, :-1]).long()  # [Hkv, R]
        cum = counts.cumsum(dim=1)                       # 非降序，可 searchsorted
        nnz_total = cum[:, -1:]                          # [Hkv, 1]
        T = k2q_q_indices.shape[1]                       # total_q * topK（槽位上界）
        pos = torch.arange(T, device=device, dtype=torch.long)
        pos_h = pos.unsqueeze(0).expand(Hkv, T).contiguous()  # [Hkv, T]
        rows_h = torch.searchsorted(cum, pos_h, right=True)
        valid = pos_h < nnz_total                        # 尾部 -1 填充槽位无效
        rows_c = rows_h.clamp(max=total_rows - 1)
        qg = cu_q[row_batch[rows_c]] + k2q_q_indices.long()   # [Hkv, T] 全局 q 序号
        jb = row_block[rows_c]                           # [Hkv, T] batch 内块号
        key = torch.where(valid, qg.float(),
                          torch.full_like(qg, float("inf"), dtype=torch.float32))
        order = torch.argsort(key, dim=1, stable=True)   # [Hkv, T]
        qg_s = qg.gather(1, order)
        jb_s = jb.gather(1, order)
        valid_s = valid.gather(1, order)
        is_start = torch.ones(Hkv, T, dtype=torch.bool, device=device)
        is_start[:, 1:] = qg_s[:, 1:] != qg_s[:, :-1]
        first_pos = torch.where(is_start, pos_h, pos_h - pos_h - 1) \
            .cummax(dim=1).values                        # 最近一个组起点位置
        rank = pos_h - first_pos                         # 同一 q 内按块号的名次
        keep = valid_s & (rank < topK)
        h_idx = torch.arange(Hkv, device=device, dtype=torch.long) \
            .unsqueeze(1).expand(Hkv, T)
        q2k = torch.full((Hkv, total_q, topK), -1, dtype=torch.long, device=device)
        flat = (h_idx * (total_q * topK) + qg_s * topK + rank)
        q2k.view(-1)[flat[keep]] = jb_s[keep]

        # ---- 3. 块稀疏注意力 (fp32 内部计算) ----
        # 按 batch 语义分组（varlen 请求是语义边界），batch 内全部 kv-head
        # 一次批量 einsum 完成；qlen 按显存预算分块，k_g 物化不超过 ~1GB。
        # 与逐 (batch, kv-head) 实现数学等价（同一点积序列，仅批量维度合并）。
        out = torch.zeros(total_q, Hq, D, dtype=torch.float32, device=device)
        arange_blk = torch.arange(blk_kv, device=device)
        S_all = topK * blk_kv
        chunk = max(1, (1 << 30) // max(1, Hkv * S_all * D * 4))
        for b in range(B):
            q0, q1 = cuq[b], cuq[b + 1]
            qlen = q1 - q0
            if qlen == 0:
                continue
            L = int(seqused[b])
            if L == 0:
                continue
            nblocks = (L + blk_kv - 1) // blk_kv
            pages = page_table[b, :nblocks].long()
            k_bT = k[pages].reshape(nblocks * blk_kv, Hkv, D)[:L] \
                .float().permute(1, 0, 2)                # [Hkv, L, D]
            v_bT = v[pages].reshape(nblocks * blk_kv, Hkv, D)[:L] \
                .float().permute(1, 0, 2)
            q_pos = torch.arange(qlen, device=device) + (L - qlen)  # bottom-right
            sel = q2k[:, q0:q1]                          # [Hkv, qlen, topK]
            q_b = q[q0:q1].float().view(qlen, Hkv, G, D) * softmax_scale

            for c0 in range(0, qlen, chunk):
                c1 = min(c0 + chunk, qlen)
                sel_c = sel[:, c0:c1]                    # [Hkv, cq, topK]
                cq = c1 - c0
                # 选中块展开的 token 位置 [Hkv, cq, S];
                # token 级有效性同时覆盖: -1 填充 / 越界块 / seqused 截断 / 因果
                tok4 = sel_c[..., None] * blk_kv + arange_blk   # [Hkv, cq, topK, blk]
                tok_idx = tok4.reshape(Hkv, cq, S_all)
                tok_valid = ((sel_c[..., None] >= 0) & (sel_c[..., None] < nblocks)
                             & (tok4 < L))
                if causal:
                    tok_valid &= tok4 <= q_pos[c0:c1].view(1, cq, 1, 1)
                tok_valid = tok_valid.reshape(Hkv, cq, S_all)

                idx = tok_idx.clamp(0, L - 1)
                gidx = idx.reshape(Hkv, -1, 1).expand(-1, -1, D)
                k_g = k_bT.gather(1, gidx).reshape(Hkv, cq, S_all, D)
                v_g = v_bT.gather(1, gidx).reshape(Hkv, cq, S_all, D)
                scores = torch.einsum("qhgd,hqsd->hqgs", q_b[c0:c1], k_g)
                scores = scores.masked_fill(~tok_valid.unsqueeze(2), float("-inf"))
                attn = torch.softmax(scores, dim=-1)
                attn = torch.nan_to_num(attn, nan=0.0)   # 无可见 token 的行输出 0
                out[q0 + c0:q0 + c1] = torch.einsum(
                    "hqgs,hqsd->qhgd", attn, v_g).reshape(cq, Hq, D)
        return out.to(torch.bfloat16)


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

    def random_tensor(shape, dtype):
        # 标准输入分布: 50% 均匀 [-5, 5] + 50% 正态 (mu ∈ [-5, 5], sigma ∈ [0.1, 2])
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
    }

    input_groups = []
    for case_idx, case in enumerate(cases):
        torch.manual_seed(3407 + case_idx)

        shapes, dtypes, attrs = {}, {}, {}
        for inp in case["inputs"]:
            if inp.get("type") == "tensor":
                shapes[inp["name"]] = inp["shape"]
                if inp["dtype"] in dtype_map:  # int32 索引张量在下方直接构造
                    dtypes[inp["name"]] = dtype_map[inp["dtype"]]
            else:
                attrs[inp["name"]] = inp["value"]

        total_q, Hq, D = shapes["q"]
        num_pages, Hkv, blk, _ = shapes["k"]
        cu_q = torch.tensor(attrs["cu_seqlens_q"], dtype=torch.int32)
        cu_k = torch.tensor(attrs["cu_seqlens_k"], dtype=torch.int32)
        seqused = torch.tensor(attrs["seqused_k"], dtype=torch.int32)
        topK = int(attrs["topK"])
        blk_kv = int(attrs["blk_kv"])
        causal = int(attrs["causal"])
        scale = float(attrs["softmax_scale"])
        max_q = int(attrs["max_seqlen_q"])
        max_k = int(attrs["max_seqlen_k"])
        B = cu_q.numel() - 1
        assert blk == blk_kv and int(cu_q[-1]) == total_q
        kv_lens = [int(seqused[b]) for b in range(B)]
        rows_pb = [(L + blk_kv - 1) // blk_kv for L in kv_lens]
        total_rows = sum(rows_pb)
        assert shapes["k2q_row_ptr"] == [Hkv, total_rows + 1]
        assert shapes["k2q_q_indices"] == [Hkv, total_q * topK]

        q = random_tensor(shapes["q"], dtypes["q"])
        k = random_tensor(shapes["k"], dtypes["k"])
        v = random_tensor(shapes["v"], dtypes["v"])

        # ---- 物理合法的 q2k 选择 (物理约束: indexer 只会从因果可见块中选) ----
        # 每个 (b, h, qi): 可见块 = [0, (qi + L - qlen)//blk], 从中选 <= topK 个
        q2k = torch.full((Hkv, total_q, topK), -1, dtype=torch.int32)
        for b in range(B):
            q0, q1 = int(cu_q[b]), int(cu_q[b + 1])
            qlen = q1 - q0
            L = kv_lens[b]
            nb = rows_pb[b]
            for h in range(Hkv):
                for qi in range(qlen):
                    visible = min((qi + L - qlen) // blk_kv + 1, nb)
                    if visible <= 0:
                        continue
                    if visible <= topK:
                        chosen = torch.arange(visible, dtype=torch.int32)
                    else:
                        chosen = torch.randperm(visible)[:topK].sort().values.int()
                    q2k[h, q0 + qi, :chosen.numel()] = chosen
        row_of = {}
        r = 0
        for j in range(max(rows_pb)):
            for b in range(B):
                if j < rows_pb[b]:
                    row_of[(b, j)] = r
                    r += 1
        row_ptr = torch.zeros(Hkv, total_rows + 1, dtype=torch.int32)
        qidx = torch.full((Hkv, total_q * topK), -1, dtype=torch.int32)
        for h in range(Hkv):
            rows_flat, q_flat = [], []
            for b in range(B):
                q0, q1 = int(cu_q[b]), int(cu_q[b + 1])
                sel = q2k[h, q0:q1].long()            # [qlen, topK]
                qlen = q1 - q0
                rm = torch.tensor([row_of[(b, j)] for j in range(rows_pb[b])])
                valid = sel >= 0
                rows_flat.append(rm[sel.clamp(min=0)][valid])
                q_flat.append(torch.arange(qlen).unsqueeze(1)
                              .expand(qlen, topK)[valid])
            rows_cat = torch.cat(rows_flat)
            q_cat = torch.cat(q_flat)
            counts = torch.zeros(total_rows, dtype=torch.long)
            counts.scatter_add_(0, rows_cat, torch.ones_like(rows_cat))
            row_ptr[h, 1:] = counts.cumsum(0).int()
            order = torch.argsort(rows_cat * (total_q + 1) + q_cat, stable=True)
            qidx[h, :rows_cat.numel()] = q_cat[order].int()
        max_pages = shapes["page_table"][1]
        assert total_rows <= num_pages
        perm = torch.randperm(num_pages)
        page_table = torch.zeros(B, max_pages, dtype=torch.int32)
        cursor = 0
        for b in range(B):
            nb = rows_pb[b]
            page_table[b, :nb] = perm[cursor:cursor + nb].int()
            cursor += nb

        input_groups.append([q, k, v, row_ptr, qidx, cu_q, cu_k, seqused,
                             page_table, topK, max_q, max_k, blk_kv, causal,
                             scale])
    return input_groups


def get_init_inputs():
    return []
