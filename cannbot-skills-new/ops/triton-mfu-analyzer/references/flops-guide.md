# 怎么写 flops.py

MFU 的分子完全由这个文件决定，写错了后面全错，而且**不会报错**——只会得到一个
看起来很合理的数字。这是整条链路上唯一没有校验的环节。

## 接口

放在算子工作目录下，文件名必须是 `flops.py`：

```python
def flops(case) -> float:
    """ops-profiling `--mfu` 要求的接口。case 是用例文件里的一行。"""

def flops_parts(case) -> dict:      # 可选，mfu_decompose.py 用
    """把理论 FLOPs 拆成若干段，各段之和必须等于 flops(case)。"""

KERNEL_PARTS = {"kernel_a": "段名1", "kernel_b": "段名2"}   # 可选，同上
```

`case` 的结构就是用例 jsonl 的一行：

```json
{"inputs": [
  {"name": "x", "type": "tensor", "dtype": "float16", "shape": [3, 317, 1408]},
  {"name": "n_heads", "type": "attr", "dtype": "int", "value": 22},
  {"name": "causal", "type": "attr", "dtype": "bool", "value": true}
]}
```

tensor 项有 `shape`/`dtype`，标量项有 `value`。建议先转成 `{name: item}` 的字典再取。

## 口径（对齐官方 operator_mfu）

只统计**语义必需的核心 GEMM**，后处理一律不计：

| 算子 | 公式 |
|---|---|
| GEMM / MatMul | `2 × M × K × N`（batch 维连乘进去） |
| Linear | `2 × prod(input.shape[:-1]) × in_features × out_features` |
| Attention | `2 × score_elems × q_dim + 2 × score_elems × value_dim` |
| Conv | `2 × Cout × Hout × Wout × Cin × KH × KW × batch`（im2col 等价 GEMM） |

`score_elems` 是**语义上必需**的 (query, key) 对数，按 batch × head 累加：

| 场景 | score_elems |
|---|---|
| 稠密非 causal | `B × H × S_q × S_kv` |
| causal（标准下三角） | `B × H × S(S+1)/2` —— 上三角被 mask 成 -inf，语义上不需要算 |
| causal（右下对齐，`S_q ≠ S_kv`） | `B × H × Σᵢ min(S_kv − S_q + i + 1, S_kv)` |
| 局部窗口 | `B × H × S_q × window` |
| 块稀疏 | `B × H × Σ(选中块的元素数)` |
| 稀疏 topk | `Σ_token min(topk × block_size, S_kv)` |
| 变长 | 按 `cu_seqlens` / `actual_seq_lengths` 逐 batch 累加，**不能用 padding 后的 shape** |

### 怎么判断算子带不带 causal mask

**只看 attr 名字会漏。** 实测扫描 103 个算子，causal 语义有 6 种不同编码：

| attr 名 | 代表算子 | 备注 |
|---|---|---|
| `causal` | 20_FlashAttentionV2、30_PagedAttention、31_MLA | `True/False` 或 `1/0` |
| `is_causal` | 99/100_BlockSparseAttn、104/105_FlexAttention | 同上 |
| **`sparse_mode`** | **7_SparseFlashAttention**、5_LightningIndexer | `3` = right-down causal，`0` = 无 |
| `window_left`/`window_right` | 28/29_FlashAttention | `window_right=0` 等价 causal |
| `window_size` | 14_WindowAttention、23_LocalAttention | 局部窗口，同样缩减 |
| `head_mask_type`（tensor） | 99/100_BlockSparseAttn | 值运行时才知道 |

**决定性证据是 torch 参考实现的 forward。** （源目录里是 `<Op>_torch.py` 或 `<Op>.py`，
含 `class Model`；工作目录里落地后叫 `model.py`。**不要去读 triton 实现**——
它只提供 kernel 耗时作分母，算子语义由 torch 实现定义。）grep 这几个写法：

```
masked_fill(...) + float("-inf") / -2e35 / -40000.0
torch.triu / torch.tril
column > row  /  col > row  /  relative > 0
```

四类典型实现：

```python
# A 坐标比较（20_FlashAttentionV2）
scores = scores.masked_fill((column > row).unsqueeze(0).unsqueeze(0), float("-inf"))

# B relative 偏移（28_FlashAttentionFwd，支持 S_q != S_kv）
relative = column - (row + seq_len_kv - seq_len_q)
mask = mask | (relative > 0)

# C torch.triu（104_FlexAttentionFwd）
causal_mask = torch.triu(torch.ones(...), diagonal=1).bool()

# D 阈值累加（7_SparseFlashAttention，right-down causal）
threshold = (cur_s2 - cur_s1) + s1_idx + 1
```

**注意右下对齐变体**：当 `S_q ≠ S_kv`（decode / prefix-cache），causal 不是标准下三角，
而是右下角对齐的梯形——query token `i` 能看到 KV 的前 `(S_kv − S_q) + i + 1` 行。

**不计**：softmax、mask、scale、transpose、gather、norm、激活。

Elementwise / Norm 这类 memory-bound 算子分子太小，MFU 没有意义，不要采；
它们该看 MemoryUtilization（ops-profiling 的 `--compare` 模式）。

## 三条必须遵守的规则

### 1. 先读 torch 参考实现，不要看算子名字

算子名字经常只描述最显眼的那一段。实测踩过的例子：`20_FlashAttentionV2` 的
`Model.forward` 里除 attention 外还有 **4 个 `nn.Linear`**（q/k/v/out 投影），
投影占理论 FLOPs 的 **94.7%**。只算 attention 会让分子小 18 倍，MFU 直接失去意义。

判断标准是**语义必需**：参考实现里为了产出这个算子的输出而必须做的 GEMM，
全都要计入，不管它在名字里有没有体现。

### 2. 分子与实现无关

只算语义必需量，实现里的冗余一律**不计**：

- 多趟 GEMM（如 online softmax 为了拿 m/l 把 QK dot 做两遍）
- 块粒度 mask 导致对角块内算了又丢掉的元素
- head_dim / K 维 pad 到 BLOCK 的补零
- split-K / split-P 的重复累加

冗余会拉长耗时、把 MFU 拉低——**这正是我们想看到的**。把冗余算进分子等于把
实现的浪费伪装成有效算力，MFU 就退化成了 PMU 占空比。

### 3. 维度要泛化

用例集里同一个算子的 shape 维数可能变。用 `prod(shape[:-2])` 之类的写法，
不要写死 `shape[0] * shape[1]`：

```python
batch = 1
for d in a["shape"][:-2]:
    batch *= d
```

## 模板

**MatMul**

```python
def flops(case) -> float:
    a, b = case["inputs"][0], case["inputs"][1]
    m, k = a["shape"][-2], a["shape"][-1]
    n = b["shape"][-1]
    batch = 1
    for d in a["shape"][:-2]:
        batch *= d
    return float(2 * batch * m * k * n)
```

**FlashAttention（q/k/v 形状直接可得）**

```python
def flops(case) -> float:
    q, v = case["inputs"][0], case["inputs"][2]
    *lead, s, d = q["shape"]            # 末两维 seq_len × head_dim
    rows = 1
    for x in lead:
        rows *= x                       # batch × heads
    score_elems = rows * s * s          # causal 改成 rows * s * (s + 1) / 2
    return float(2 * score_elems * d + 2 * score_elems * v["shape"][-1])
```

**稀疏 / 变长（逐 token 累加）**

```python
def _cumsum(lengths, total):
    if not lengths:
        return [total]
    if lengths[-1] >= total * 0.9:      # 已经是 cumsum 形式
        return list(lengths)
    return [sum(lengths[:i + 1]) for i in range(len(lengths))]

def _batch_of(cumsum_list, t_idx):
    b = 0
    while b < len(cumsum_list) and t_idx >= cumsum_list[b]:
        b += 1
    return min(b, len(cumsum_list) - 1)

# 每个 query token 实际选中的 KV 行数
total_sel = 0
for i in range(t1):
    b = _batch_of(asq_cumsum, i)
    cur_s2 = ask_cumsum[b] - (ask_cumsum[b - 1] if b > 0 else 0)
    total_sel += min(topk * block_size, cur_s2)
score_elems = total_sel * n2 * g
```

完整文件见 `examples/flops_sparse_flash_attention.py`（含 rope 拼接、三种 layout）。

**多段算子（带拆分钩子）** —— 见 `examples/flops_mha_block.py`，是实测用过的完整文件。

### 无 FLOPs 的段怎么处理

gather / combine / layout transpose 这类段不产生语义 FLOPs，但耗时必须计入总时间。
做法是在 `flops_parts` 里给它们一个 0 值条目，并在 `KERNEL_PARTS` 里映射过去：

```python
return {"attn": flops, "gather": 0.0, "combine": 0.0, "layout": 0.0}

KERNEL_PARTS = {
    "_sfa_spec_attn_kernel": "attn",
    "_sfa_gather_kv_kernel": "gather",        # 0 FLOPs，只看耗时
    "_sfa_split_combine_kernel": "combine",
}
```

这样 `mfu_decompose.py` 会单独列出这些段的耗时。实测靠这个发现某 case 的 gather
吃掉了 50.6% 的时间——不这么做只会看到一个笼统的低 MFU，定位不到是哪一段。

## 写完怎么验

1. **手算一个 case 对拍。** 挑最大的那个 case 手工算一遍，和 `flops(case)` 比。
2. **量级自检。** `mfu_decompose.py` 会打出 actual TFLOPS。如果某个 case 的
   actual TFLOPS **超过了芯片峰值**，分子一定写大了（或时间取错了）。
3. **拆分自检。** 提供了 `flops_parts` 时，`mfu_decompose.py` 会校验各段之和
   与 `flops()` 的偏差，超过 1e-6 会打 `[warn]`。
4. **KERNEL_PARTS 覆盖自检。** 未映射到任何段的 kernel 会以 `[warn]` 列出来，
   它们的耗时不计入任何段但仍计入总时间——看到 warn 就补进映射表。
5. **tracer 独立对拍。** `check_flops.py` 会 monkey-patch `torch.matmul` /
   `F.linear` / `torch.bmm` 跑一遍参考实现，统计实际 GEMM FLOPs：

   ```bash
   python3 scripts/check_flops.py --op-dir <工作目录>
   python3 scripts/check_flops.py --op-dir . --cases 0,9,50   # 只查几个
   python3 scripts/check_flops.py --op-dir . --no-trace       # 跳过 tracer
   ```

   | tracer/flops.py 比值 | 判断 |
   |---|---|
   | ≈ 1.0 | 分子写对了 |
   | ≈ 2.0 | 大概率漏了 causal 缩减 |
   | > 2 | 可能把实现冗余算进了分子，或漏了某段 GEMM 使基数偏小 |
   | < 0.5 | 大概率漏算了某些 GEMM（如投影） |

   注意 tracer 统计的是**参考实现实际执行**的 GEMM，含 causal 算完再 mask 这类冗余，
   所以带 causal 的算子比值略大于 1 是正常的（实测 20_FlashAttentionV2 为 1.05）。
