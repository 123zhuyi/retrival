import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Batch decode MLA attention with paged KV cache —— golden reference，
    按 vLLM 官方 Triton MLA decode kernel 语义对齐（本版接口与掩码约定
    均跟随 Triton kernel，而非 FlashInfer CUDA 版）。

    上游：vllm-project/vllm @ main
      - kernel: vllm/v1/attention/ops/triton_decode_attention.py
        · `_fwd_grouped_kernel_stage1`（H > 1 路径, IS_MLA=True）：
          qk = dot(q[..., :Dc], kv[..., :Dc]) + dot(q[..., Dc:], kv[..., Dc:])
          （Dc=kv_lora_rank=512 与 Dr=64 两个 tl.dot 分开累加, fp32 累加器;
           Lk==576 时 BLOCK_DMODEL=512/BLOCK_DPE=64; H=1 时走
           `_fwd_kernel_stage1` 全维点积, 语义相同）
          qk *= sm_scale（自然对数域, 不乘 log2e —— 与 FlashInfer 不同）
          logit_cap>0: qk = cap * tanh(qk / cap)（先 cap 后 mask）
          V = kv[..., :Dc]（IS_MLA: v = tl.trans(k), 矩阵吸收, ckv 兼作 K/V）
          在线 softmax（自然 exp）, 部分 LSE = m + ln(l)（自然对数域）
        · `_fwd_kernel_stage2`：跨 split 归并, o = Σ exp(lse_i - m) o_i / l,
          最终 lse = m + ln(l)（自然对数, 随 vLLM impl 以 q.dtype 返回）
      - 调用现场: vllm/v1/attention/backends/mla/triton_mla.py
        （TritonMLAImpl._forward_decode: q = cat([q_nope, q_pe], -1),
         kv cache 单 buffer [num_pages, page_size, 1, Dc+Dr],
         req_to_token [B, max_pages] + B_Seqlen）
      仓库：https://github.com/vllm-project/vllm

    torch 原生小算子拼接参考实现（内部 fp32，输出回铸 q.dtype）：
        每个请求 b 经 req_to_token 页表 gather 出 kv_len 个 token：
            k = kv[..., :Dc], pe = kv[..., Dc:]
            s = sm_scale * (q[..., :Dc]·k + q[..., Dc:]·pe)
            logit_cap: s = cap * tanh(s/cap)（在 -inf 掩码之前, 与 kernel 同序）
            o = softmax(s) @ k
            lse = logsumexp(s)（自然对数域）

    与 Triton kernel 的语义边界说明：
      1. split-KV（num_kv_splits）仅为调度参数（vLLM 启发式
         next_pow2(max(1, max_len//512)), 上限 sm_count*2），不改变数学语义，
         故不作为算子入参；golden 以稠密单遍 softmax 表达。
      2. RoPE 不在 kernel 内（q_pe/kpe 视为已加 rope），golden 同样不重算。
      3. kernel 无 sliding window（无 window_left 参数），本 golden 不含该语义；
         原 FlashInfer 对齐版 golden 中 window/cap 为超集语义，本版只保留 cap
         （Triton kernel 有 logit_cap, 顺序为 cap→mask）。
      4. LSE 为自然对数域（Triton kernel），与 FlashInfer CUDA 版的 log2 域
         相差 ln2 倍；仅在 return_lse=True 时可见，o 不受影响。
      5. FP8（k_scale/v_scale）路径未纳入；kv_lens=0 上游行为未定义, 不覆盖。
      6. 无因果掩码：decode 单 query token 注意全部 kv token。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, kv_cache, req_to_token, kv_lens, sm_scale,
                kv_lora_rank, logit_cap=0.0, return_lse=False):
        B, H, Dqk = q.shape
        page_size = kv_cache.shape[1]
        Dc = int(kv_lora_rank)                       # V = ckv = kv[..., :Dc]
        device = q.device

        kv_lens = kv_lens.long()                     # [B]，即 kernel 的 B_Seqlen
        L = int(kv_lens.max().item())
        P = (kv_lens + page_size - 1) // page_size   # 每请求页数
        Pmax = int(P.max().item())

        # 页表 gather：req_to_token [B, max_pages] -> 页号 -> 拼稠密 KV
        # （kernel: page = req_to_token[b, n // page_size], 页内 n % page_size；
        #   仅前 kv_len 个 token 有效, 与 kernel 的 offs_n < split_kv_end 一致）
        j = torch.arange(Pmax, device=device)
        pages = req_to_token.long()[:, :Pmax]        # [B, Pmax]
        kv = kv_cache[pages].float()                 # [B, Pmax, page_size, 1, Dqk]
        kv = kv.reshape(B, Pmax * page_size, Dqk)[:, :L]

        k = kv[..., :Dc]                             # [B, L, Dc]（兼作 V）
        pe = kv[..., Dc:]                            # [B, L, Dr]

        # 分体点积求和后单次乘 sm_scale（与 grouped kernel 两个 tl.dot 同构）
        qf = q.float()
        s = (qf[..., :Dc] @ k.transpose(1, 2)
             + qf[..., Dc:] @ pe.transpose(1, 2))    # [B, H, L]
        s = s * sm_scale
        if logit_cap and logit_cap > 0:
            # kernel 顺序：先 tanh cap，再 -inf 掩码
            s = logit_cap * torch.tanh(s / logit_cap)
        col = torch.arange(L, device=device)
        pad_mask = col[None, :] >= kv_lens[:, None]  # [B, L]
        s = s.masked_fill(pad_mask[:, None, :], float("-inf"))

        m = s.max(dim=-1, keepdim=True).values
        e = torch.exp(s - m)
        denom = e.sum(dim=-1, keepdim=True)
        o = (e / denom) @ k                          # [B, H, Dc]
        lse = (m.squeeze(-1) + torch.log(denom.squeeze(-1))).to(q.dtype)

        o = o.to(q.dtype)
        if return_lse:
            return o, lse
        return o


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _random_tensor(shape, dtype, gen):
    # 标准输入分布: 50% 均匀 [-5, 5] + 50% 正态 (mu ∈ [-5, 5], sigma ∈ [0.1, 2])
    if torch.rand(1, generator=gen).item() < 0.5:
        return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0, generator=gen)
    mu = float(torch.empty(1).uniform_(-5.0, 5.0, generator=gen).item())
    sigma = float(torch.empty(1).uniform_(0.1, 2.0, generator=gen).item())
    return torch.normal(mu, sigma, shape, generator=gen).to(dtype)


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

    input_groups = []
    for case_index, case in enumerate(cases):
        specs = {item["name"]: item for item in case["inputs"]}

        def attr(name, default):
            item = specs.get(name)
            return item["value"] if item is not None else default

        kv_lens = attr("kv_lens", None)
        page_size = attr("page_size", None)
        sm_scale = attr("sm_scale", None)
        kv_lora_rank = attr("kv_lora_rank", None)
        logit_cap = attr("logit_cap", 0.0)
        return_lse = attr("return_lse", False)

        # 每 case 独立播种（与 FlashInfer 布局旧版同种子流, 保证底层数据一致）:
        # q_nope/q_pe 与 ckv/kpe 分别生成后拼接为 kernel 的合并布局。
        B, H, Dqk = specs["q"]["shape"]
        Dc = kv_lora_rank
        gen_qn = torch.Generator().manual_seed(42 + case_index * 4)
        gen_qp = torch.Generator().manual_seed(43 + case_index * 4)
        gen_ckv = torch.Generator().manual_seed(44 + case_index * 4)
        gen_kpe = torch.Generator().manual_seed(45 + case_index * 4)
        gen_meta = torch.Generator().manual_seed(1000 + case_index)

        q_nope = _random_tensor([B, H, Dc], _DTYPE_MAP[specs["q"]["dtype"]], gen_qn)
        q_pe = _random_tensor([B, H, Dqk - Dc],
                              _DTYPE_MAP[specs["q"]["dtype"]], gen_qp)
        q = torch.cat([q_nope, q_pe], dim=-1).contiguous()  # [B, H, Dqk]

        # 分页 KV cache: 页 id 随机打乱 + 3 个不被引用的冗余页
        npages = [(l + page_size - 1) // page_size for l in kv_lens]
        total_pages = sum(npages) + 3
        num_pages, ps, Hc, _ = specs["kv_cache"]["shape"]
        assert num_pages == total_pages and ps == page_size and Hc == 1
        perm = torch.randperm(total_pages, generator=gen_meta)

        ckv = _random_tensor([num_pages, page_size, 1, Dc],
                             _DTYPE_MAP[specs["kv_cache"]["dtype"]], gen_ckv)
        kpe = _random_tensor([num_pages, page_size, 1, Dqk - Dc],
                             _DTYPE_MAP[specs["kv_cache"]["dtype"]], gen_kpe)
        kv_cache = torch.cat([ckv, kpe], dim=-1).contiguous()

        # req_to_token [B, max_npages]（vLLM 版页表; 空位补 0, kernel 不读）
        max_npages = max(npages)
        req_to_token = torch.zeros(B, max_npages, dtype=torch.int32)
        off = 0
        for b, n in enumerate(npages):
            req_to_token[b, :n] = perm[off:off + n].int()
            off += n

        input_groups.append([q, kv_cache, req_to_token,
                             torch.tensor(kv_lens, dtype=torch.int32),
                             sm_scale, kv_lora_rank, logit_cap, return_lse])
    return input_groups


def get_init_inputs():
    return []
