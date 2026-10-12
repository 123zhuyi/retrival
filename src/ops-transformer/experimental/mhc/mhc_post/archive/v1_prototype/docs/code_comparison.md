# mhc_post 代码级别正确性证明

## 1. 论文公式

```text
mhc_post: output[b*s+i, seq, d] = branch_output[b, seq, d] × h_post[i]
```

## 2. PyTorch参考实现 (tokenbender/mHC)

```python
# 来源: hyper_connections_mhc.py depth_connection()

def depth_connection(self, branch_output, residuals, *, beta):
    # beta 就是 h_post, shape [num_streams]
    
    # Step 1: einsum广播乘法
    # "b ... d, s -> b ... s d" 的含义:
    #   - 输入1 branch_output: [batch, seq, dim]
    #   - 输入2 beta: [streams]
    #   - 输出: [batch, seq, streams, dim]
    #   - 计算: out[b,seq,s,d] = branch_output[b,seq,d] * beta[s]
    output = einsum(branch_output, beta, "b ... d, s -> b ... s d")
    
    # Step 2: reshape合并batch和stream维度
    # "b ... s d -> (b s) ... d" 的含义:
    #   - 输入: [batch, seq, streams, dim]
    #   - 输出: [batch*streams, seq, dim]
    #   - 计算: out[b*s+i, seq, d] = in[b, seq, i, d]
    output = rearrange(output, "b ... s d -> (b s) ... d")
    
    return output
```

## 3. CPU参考实现 (我的实现)

```cpp
// 来源: test_mhc_post.cpp mhc_post_cpu()

void mhc_post_cpu(
    const float* branch_output,  // [batch, seq_len, dim]
    const float* h_post,         // [num_streams]
    float* output,               // [batch * num_streams, seq_len, dim]
    int64_t batch,
    int64_t seq_len,
    int64_t dim,
    int64_t num_streams
) {
    // 遍历每个batch
    for (int64_t b = 0; b < batch; ++b) {
        // 遍历每个stream
        for (int64_t s = 0; s < num_streams; ++s) {
            // 获取当前stream的权重
            float weight = h_post[s];
            
            // 计算输出的batch索引: b*num_streams + s
            // 这对应 rearrange "b s -> (b s)" 的语义
            int64_t out_batch_idx = b * num_streams + s;
            
            // 遍历每个序列位置和维度
            for (int64_t seq = 0; seq < seq_len; ++seq) {
                for (int64_t d = 0; d < dim; ++d) {
                    // 输入索引: branch_output[b, seq, d]
                    int64_t in_idx = b * seq_len * dim + seq * dim + d;
                    
                    // 输出索引: output[b*s+i, seq, d]
                    int64_t out_idx = out_batch_idx * seq_len * dim + seq * dim + d;
                    
                    // 核心计算: 乘法
                    // 对应论文公式: output[b*s+i] = branch_output[b] * h_post[i]
                    output[out_idx] = branch_output[in_idx] * weight;
                }
            }
        }
    }
}
```

**为什么CPU实现正确？**

| 论文公式 | CPU代码 | 说明 |
|---------|---------|------|
| `output[b*s+i, ...]` | `out_batch_idx = b * num_streams + s` | batch和stream索引合并方式一致 |
| `branch_output[b, ...]` | `in_idx = b * seq_len * dim + ...` | 输入只依赖batch索引 |
| `h_post[i]` | `weight = h_post[s]` | 权重只依赖stream索引 |
| `*` | `output[...] = branch_output[...] * weight` | 简单乘法 |

## 4. NPU Kernel实现 (我的实现)

```cpp
// 来源: mhc_post_kernel.cpp

class MhcPostKernel {
public:
    __aicore__ inline void Init(...) {
        // 每个block处理一个(batch, stream)组合
        // block_idx范围: [0, batch * num_streams)
        int64_t block_idx = GetBlockIdx();
        
        // 解析batch索引和stream索引
        // 这对应 "(b s)" -> "b, s" 的反向操作
        this->batch_idx = block_idx / num_streams;
        this->stream_idx = block_idx % num_streams;
        
        // 输入偏移: 只依赖batch_idx
        // 对应论文公式中 branch_output[b, ...]
        int64_t input_offset = this->batch_idx * this->batch_elements;
        
        // 输出偏移: 依赖batch_idx和stream_idx
        // 对应论文公式中 output[b*s+i, ...]
        int64_t output_offset = (this->batch_idx * num_streams + this->stream_idx) * this->batch_elements;
        
        // 设置GM指针...
    }
    
    __aicore__ inline void LoadWeight() {
        // 加载当前stream的权重
        // 对应论文公式中 h_post[i]
        this->weight_value = this->gm_h_post.GetValue(this->stream_idx);
    }
    
    __aicore__ inline void Compute(int64_t length) {
        LocalTensor<float> inLocal = inQueue.DeQue<float>();
        LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        
        // 核心计算: 向量乘标量
        // 对应论文公式: output = branch_output * h_post[i]
        // Muls是AscendC的向量乘标量指令
        Muls(outLocal, inLocal, this->weight_value, length);
        
        outQueue.EnQue(outLocal);
        inQueue.FreeTensor(inLocal);
    }
};
```

**为什么NPU实现正确？**

| 论文公式 | NPU代码 | 说明 |
|---------|---------|------|
| 遍历所有(b,s)组合 | `block_idx ∈ [0, batch*num_streams)` | 并行处理每个组合 |
| `b = idx / s` | `batch_idx = block_idx / num_streams` | 解析batch索引 |
| `i = idx % s` | `stream_idx = block_idx % num_streams` | 解析stream索引 |
| `branch_output[b]` | `input_offset = batch_idx * elements` | 输入偏移计算 |
| `output[b*s+i]` | `output_offset = (b*s+i) * elements` | 输出偏移计算 |
| `h_post[i]` | `weight = gm_h_post.GetValue(stream_idx)` | 加载权重 |
| `output = input * weight` | `Muls(out, in, weight, len)` | 向量乘标量 |

## 5. 数学等价性证明

**给定:**

- `branch_output`: shape [B, S, D]
- `h_post`: shape [N]
- `output`: shape [B*N, S, D]

**论文公式:**

```text
output[b*N + n, s, d] = branch_output[b, s, d] × h_post[n]
其中 b ∈ [0,B), n ∈ [0,N), s ∈ [0,S), d ∈ [0,D)
```

**CPU实现等价性:**

```cpp
for b in [0, B):
  for n in [0, N):
    out_idx = b * N + n          // ← 对应 output[b*N + n, ...]
    weight = h_post[n]            // ← 对应 h_post[n]
    for s in [0, S):
      for d in [0, D):
        output[out_idx, s, d] = branch_output[b, s, d] * weight
        //                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        //                      对应 branch_output[b, s, d] × h_post[n]
```

**NPU实现等价性:**

```cpp
// 并行化: 每个block处理一个(b, n)组合
block_idx ∈ [0, B*N):
  b = block_idx / N               // ← 解析batch索引
  n = block_idx % N               // ← 解析stream索引
  weight = h_post[n]              // ← 对应 h_post[n]
  
  // 向量化: 处理整个[S, D]平面
  output[block_idx, :, :] = branch_output[b, :, :] * weight
  //                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  //                        对应 branch_output[b, s, d] × h_post[n]
```

由于 `block_idx = b * N + n`，所以 `output[block_idx]` 等价于 `output[b*N + n]`

因此，NPU实现与论文公式完全等价。
