# mhc_post 算子正确性证明

## 1. 论文公式

mHC论文 (arXiv:2512.24880) Equation 3:

```text
x_{l+1} = H_l^{res} · x_l + H_l^{post}^T · F(H_l^{pre} · x_l, W_l)
```

其中：

- `x_l`: 输入特征 [batch, n×C] (n个streams，每个C维)
- `H_l^{pre}`: 输入映射 [1, n]，用于聚合n个streams到1个
- `H_l^{post}`: 输出映射 [1, n]，用于分发1个到n个streams
- `F(...)`: 分支模块输出 (如Attention, FFN)
- `H_l^{res}`: 残差映射 [n, n]

**mhc_post 对应的就是 `H_l^{post}^T · F(...)`** 这部分。

## 2. 具体计算过程

假设：

- branch_output = F(...) 的输出，shape = [batch, seq_len, dim]
- h_post = H_l^{post}，shape = [num_streams] (经softmax归一化)

mhc_post 计算：

```python
output = H_post^T ⊗ branch_output
```

展开为：

```text
output[b, s, seq, d] = branch_output[b, seq, d] × h_post[s]
```

最终reshape为：

```text
output[(b × num_streams + s), seq, d] = branch_output[b, seq, d] × h_post[s]
```

## 3. PyTorch参考实现 (tokenbender/mHC)

来自 hyper_connections_mhc.py 的 depth_connection 函数：

```python
def depth_connection(self, branch_output, residuals, *, beta):
    # beta 就是 h_post
    
    # einsum: "b ... d, s -> b ... s d"
    # 含义: 对每个位置，branch_output乘以每个stream的权重
    output = einsum(branch_output, beta, "b ... d, s -> b ... s d")
    
    # rearrange: "b ... s d -> (b s) ... d"
    # 含义: 将batch和stream维度合并
    output = rearrange(output, "b ... s d -> (b s) ... d")
    
    return output
```

## 4. 数学等价性验证

einsum "b ... d, s -> b ... s d" 的含义：

- 输入1: branch_output [b, seq, d]
- 输入2: beta [s]
- 输出: [b, seq, s, d]
- 计算: output[b,seq,s,d] = branch_output[b,seq,d] × beta[s]

rearrange "b ... s d -> (b s) ... d" 的含义：

- 输入: [b, seq, s, d]
- 输出: [b×s, seq, d]
- 计算: output[b×s+i, seq, d] = input[b, seq, i, d]

组合起来：

```text
final_output[b×num_streams + s, seq, d] = branch_output[b, seq, d] × h_post[s]
```

这正是我们要实现的公式！
