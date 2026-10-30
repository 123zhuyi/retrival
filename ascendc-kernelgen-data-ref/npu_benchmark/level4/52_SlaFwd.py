import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import os


class Model(nn.Module):
    """
        SparseLinearAttention 前向的性能基线（thu-ml SLA；语义经核对与官方一致，
    无需修改；level5 版相对 level4 仅做格式迁移）。
    稀疏注意力 + 线性注意力可训练融合 golden reference。

    算子出处：
      thu-ml/SLA（main 分支）
      - 公开接口:  sparse_linear_attention/core.py  SparseLinearAttention
      - 语义依据:  sparse_linear_attention/utils.py  get_block_map（块选择）
                  sparse_linear_attention/kernel.py  _attn_fwd（块稀疏注意力）
                  core.py  forward（feature_map / calc_linear / proj_l 融合）
      数学语义逐行对齐上述三处官方实现。

    数学语义（q/k/v: [B, H, L, D]，非因果）：
      1) 块选择（get_block_map）：
         arg_k = k - mean(k, dim=-2)              （SageAttention smooth-k）
         qm = 块均值(q, BLKQ)，km = 块均值(arg_k, BLKK)
         （尾块按实际 token 数取均值，与官方 compress_kernel 一致）
         score = qm @ km^T  → [B,H,MQ,NK]
         real_topk = min(NK, int(topk_ratio * NK))
         每个 query 块选分数最高的 real_topk 个 key 块（torch.topk）
         ※ 选择路径遵循官方精度：块均值与块分数均回铸 bf16/fp16 后 topk
           （离散选择对舍入敏感，fp32 理想化会与 kernel 选块不一致；
           官方 compress_kernel fp32 求和、bf16 存储、bf16 matmul 得分数）
         ※ 官方未处理 real_topk=0（int(ratio*NK)<1 时 l_i=0 → NaN），
           golden 同样未定义该行为，用例均保证 real_topk ≥ 1
      2) 稀疏分支 o_s（_attn_fwd）：对每个 query 块，仅在选中的 key 块上做
         标准 softmax 注意力，计算精度路径镜像 kernel：
         scores = tl.dot(q, k)（fp16/bf16 精确乘积 + fp32 累加）× scale；
         online softmax 的 p 在 fp32 下求 exp2 与分母 l_i（未舍入 p 之和），
         但 PV 前 p 舍回 v.dtype（kernel: tl.dot(p.to(v.dtype), v)），
         o_s = (p_dtype @ v) / l_i 落盘时再舍回 v.dtype（OS = empty_like(v)）。
         golden 用批量 softmax 等值 online（仅 fp32 累加顺序差异 ~1e-7）。
      3) 线性分支 o_l（calc_linear + proj_l，core.py 官方 torch 路径，
         全程在模型 dtype（fp16/bf16）下执行）：
         c_q = φ(q)，c_k = φ(k)（逐 token 特征图，默认 φ=softmax(dim=-1)，
         可选 elu+1 / relu；fp16/bf16 tensor 上的 torch 算子按 fp32 逐元素
         计算后舍回，golden 以 fp32 计算 + .to(dtype) 镜像）；
         matmul 均为 fp32 累加 + dtype 输出（torch tensor core 口径）：
         kvsum = (c_k^T @ v).to(dtype)；ksum = c_k.sum(-2).to(dtype)；
         分子 num = (c_q @ kvsum).to(dtype)；
         分母 den = (1e-5 + (c_q*ksum).to(dtype).sum(-1).to(dtype)).to(dtype)；
         o_l = (num / den).to(dtype)（fp16 除法按 fp32 逐元素算后舍回）；
         o_l = (o_l @ W_l^T).to(dtype)（autocast 下 Linear 权重落 dtype）
      4) 融合：o = (o_s + o_l) 回铸 q.dtype（两侧均已为 dtype 值，
         fp32 相加后一次舍入 = dtype 加法语义）。

    """

    def __init__(self):
        super(Model, self).__init__()

    # query 块维分块大小: 控制 gather/scores 内存为
    # B*H*chunk × real_topk*BLKK × D 个 fp32
    _MQ_CHUNK = 16

    @staticmethod
    def _block_mean(x, BLK):
        device = x.device
        B, H, L, D = x.shape
        LB = (L + BLK - 1) // BLK
        pad = LB * BLK - L
        xp = F.pad(x.float(), (0, 0, 0, pad)).view(B, H, LB, BLK, D)
        cnt = torch.full((LB,), BLK, dtype=torch.float32, device=device)
        if pad:
            cnt[-1] = BLK - pad
        return (xp.sum(3) / cnt.view(1, 1, LB, 1)).to(x.dtype)

    def forward(self, q, k, v, topk, feature_map, BLKQ, BLKK, W_l):
        device = q.device
        B, H, L, D = q.shape
        assert k.shape == v.shape == q.shape, "SLA 要求 q/k/v 同形状（MHA）"
        assert q.dtype == k.dtype == v.dtype
        assert isinstance(feature_map, str) and feature_map in ("elu", "relu", "softmax")
        assert 0.0 < topk <= 1.0

        dtype = q.dtype

        # ---- 1. 块选择
        arg_k = k - k.float().mean(dim=-2, keepdim=True).to(dtype)
        qm = self._block_mean(q, BLKQ)
        km = self._block_mean(arg_k, BLKK)
        pooled = (qm.float() @ km.float().transpose(-1, -2)).to(dtype)
        NK = pooled.shape[-1]
        real_topk = min(NK, int(topk * NK))
        assert real_topk >= 1, "real_topk=0 时官方 kernel 输出 NaN，未定义"
        lut = torch.topk(pooled.float(), real_topk, dim=-1, sorted=False).indices
        # 按块号升序, 与原实现 blk_mask.nonzero() 的 cat 顺序一致
        lut = lut.sort(dim=-1).values                            # [B,H,MQ,real_topk]

        scale = D ** -0.5
        if feature_map == "softmax":
            fmap = lambda x: F.softmax(x, dim=-1)
        elif feature_map == "elu":
            fmap = lambda x: F.elu(x) + 1
        else:
            fmap = lambda x: F.relu(x)

        MQ = pooled.shape[2]
        kf, vf, qf = k.float(), v.float(), q.float()

        # ---- 2. 稀疏分支 o_s (_attn_fwd 精度路径镜像, 向量化) ----
        # 选中块展开的 token 位置 [B,H,MQ,real_topk*BLKK];
        # 尾 key 块越界列由 tok_idx < L 掩码覆盖
        tok_idx = lut[..., None] * BLKK + torch.arange(BLKK, device=device)
        S = real_topk * BLKK
        tok_valid = (tok_idx < L).reshape(B, H, MQ, S)
        tok_idx = tok_idx.reshape(B, H, MQ, S)

        k2 = kf.reshape(B * H, L, D)
        v2 = vf.reshape(B * H, L, D)
        idx2 = tok_idx.reshape(B * H, MQ, S).clamp(0, L - 1)   # 非法位置先钳位再屏蔽
        tv2 = tok_valid.reshape(B * H, MQ, 1, S)
        # q 零填充到 MQ*BLKQ 并预乘 scale, 尾部垃圾行最后切掉
        qpad = F.pad(qf.reshape(B * H, L, D),
                     (0, 0, 0, MQ * BLKQ - L)).reshape(B * H, MQ, BLKQ, D) * scale

        o_s = torch.empty(B * H, MQ, BLKQ, D, dtype=torch.float32, device=device)
        for s0 in range(0, MQ, self._MQ_CHUNK):
            e0 = min(MQ, s0 + self._MQ_CHUNK)
            gi = idx2[:, s0:e0].unsqueeze(-1).expand(B * H, e0 - s0, S, D)
            k_g = torch.gather(k2.unsqueeze(1).expand(B * H, e0 - s0, L, D), 2, gi)
            v_g = torch.gather(v2.unsqueeze(1).expand(B * H, e0 - s0, L, D), 2, gi)
            sc = torch.einsum("bmqd,bmsd->bmqs", qpad[:, s0:e0], k_g)
            sc = sc.masked_fill(~tv2[:, s0:e0], float("-inf"))
            # kernel: p 以 fp32 求 exp2 与分母 l_i，PV 前舍回 v.dtype，
            # o_s = (p_dtype @ v) / l_i，落盘舍回 v.dtype
            e = torch.exp(sc - sc.amax(dim=-1, keepdim=True))
            l_i = e.sum(-1, keepdim=True)
            acc = torch.einsum("bmqs,bmsd->bmqd",
                               e.to(dtype).float(), v_g)
            o_s[:, s0:e0] = (acc / l_i).to(dtype).float()
        o_s = o_s.reshape(B, H, MQ * BLKQ, D)[:, :, :L]

        # ---- 3. 线性分支 o_l (core.py 官方路径, 全程模型 dtype) ----
        # fp16/bf16 上的 torch 算子 = fp32 逐元素/累加计算后舍回 dtype，
        # golden 以 fp32 计算 + .to(dtype) 逐步镜像存储舍入
        c_q = fmap(qf).to(dtype).float()
        c_k = fmap(kf).to(dtype).float()
        kvsum = (c_k.transpose(-1, -2) @ vf).to(dtype).float()
        ksum = c_k.sum(dim=-2, keepdim=True).to(dtype).float()
        num = (c_q @ kvsum).to(dtype).float()
        den = (c_q * ksum).to(dtype).float().sum(-1, keepdim=True).to(dtype).float()
        den = (1e-5 + den).to(dtype).float()
        o_l = (num / den).to(dtype).float()
        o_l = (o_l @ W_l.float().transpose(-1, -2)).to(dtype).float()

        # ---- 4. 融合 ----
        return (o_s + o_l).to(dtype)


def get_input_groups():
    def json_path():
        here = os.path.dirname(os.path.abspath(__file__))
        stem = os.path.splitext(os.path.basename(__file__))[0]
        names = [stem]
        if stem.endswith("_torch"):
            names.append(stem[: -len("_torch")])
        names.append("101_SlaFwd")
        for name in names:
            path = os.path.join(here, name + ".json")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"case 描述文件未找到: {names} @ {here}")

    with open(json_path(), "r", encoding="utf-8-sig") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}

    def random_tensor(shape, dtype, gen):
        if torch.rand(1, generator=gen).item() < 0.5:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)
        else:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0, generator=gen).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0, generator=gen).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)

    input_groups = []
    for case_idx, case in enumerate(cases):
        gen = torch.Generator().manual_seed(42 + case_idx)
        inputs = case["inputs"]

        def info(name):
            return next(i for i in inputs if i["name"] == name)

        dt = dtype_map[info("q")["dtype"]]
        q = random_tensor(info("q")["shape"], dt, gen)
        k = random_tensor(info("k")["shape"], dt, gen)
        v = random_tensor(info("v")["shape"], dt, gen)
        topk = info("topk")["value"]
        feature_map = info("feature_map")["value"]
        BLKQ = info("BLKQ")["value"]
        BLKK = info("BLKK")["value"]

        B, H, L, D = info("q")["shape"]
        assert info("k")["shape"] == [B, H, L, D]
        assert info("v")["shape"] == [B, H, L, D]

        W_l = torch.randn(D, D, generator=gen,
                          dtype=torch.float32) / math.sqrt(D)

        input_groups.append([
            q, k, v, topk, feature_map, BLKQ, BLKK, W_l.to(dt)
        ])
    return input_groups


def get_init_inputs():
    return []