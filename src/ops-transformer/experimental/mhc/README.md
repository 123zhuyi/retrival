# mHC 昇腾 AscendC 算子

面向昇腾 NPU 的 mHC（Manifold-Constrained Hyper-Connections，流形约束超连接）算子 AscendC 实现，此算子由智子芯元 KernelCAT 智能体生成。

论文：[arXiv:2512.24880](https://arxiv.org/abs/2512.24880)

## 论文对应关系

mHC 层更新公式:

```text
x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l, W_l)
```

| 公式项 | 算子 | 功能说明 |
|------|----------|----------|
| `H_pre · x` | mhc_pre | N 路流 → 1 路（加权求和） |
| `H_post^T · y` | mhc_post | 1 路 → N 路流（广播缩放） |
| `H_res · x` | mhc_res | N 路 → N 路（流间混合） |

以上算子仅处理张量收缩运算。训练逻辑（Sinkhorn 投影、权重约束等）由上层框架负责。

## 算子公式

| 算子 | 公式 |
|----------|--------|
| **mhc_pre** | `out[b,s,d] = Σ_n h[n] × x[b*N+n,s,d]` |
| **mhc_post** | `out[b*N+n,s,d] = x[b,s,d] × h[n]` |
| **mhc_res** | `out[b*N+t,s,d] = Σ_r h[r,t] × x[b*N+r,s,d]` |

## 数据类型与精度

| 数据类型 | 精度（对比 einsum） |
|-------|----------------------|
| fp32 | bit-exact |
| fp16 | atol=1e-4, rtol=1e-3 |
| bf16 | atol=1e-3, rtol=4e-3 |

## 实现范围

权重张量（`h_pre`、`h_post`、`h_res`）为**逐层静态参数**:

- 形状: `[num_streams]` 或 `[num_streams, num_streams]`
- 在 batch 与序列维度上共享
- 与 [tokenbender/mHC](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections) 开源实现保持一致

逐样本或逐 token 的动态权重需要扩展接口。

## 硬件要求

| 项目 | 规格 |
|------|------|
| **运行设备** | Atlas A2 系列 |
| **CANN 版本** | 8.3.RC2 / 8.5 (已验证) |
| **驱动版本** | 25.0.rc1.2 |

## 编译构建

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

cd mhc_pre  # 或 mhc_post, mhc_res
mkdir -p build && cd build
cmake .. -DSOC_VERSION=ascend910b2 && make -j
cd .. && python setup.py build_ext --inplace
```

## 测试

```bash
# C++
cd build && LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH ./test_multi_dtype

# Python
LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH python mhc_pre_ops.py
```

## 使用方式

```python
import mhc_pre_ext

x = torch.randn(B * N, S, D, device='npu')  # [batch*streams, seq, dim]
h = torch.randn(N, device='npu')            # [streams]
out = mhc_pre_ext.forward(x, h)             # [batch, seq, dim]
```

## 性能（对比 torch.einsum, Ascend 910B2）

| 算子 | 加速比 |
|----------|--------|
| mhc_pre | 24x ~ 52x |
| mhc_post | 2x ~ 5x |
| mhc_res | 24x ~ 50x |

## KernelCAT内测申请

KernelCAT限时免费内测中，欢迎体验：https://kernelcat.autokernel.cn

## 参考文献

本算子实现参考了 mHC 论文：

```bibtex
@article{xie2025mhc,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={Xie, Zhenda and Wei, Yixuan and Cao, Huanqi and Zhao, Chenggang and Deng, Chengqi and Li, Jiashi and Dai, Damai and Gao, Huazuo and Chang, Jiang and Yu, Kuai and Zhao, Liang and Zhou, Shangyan and Xu, Zhean and Zhang, Zhengyan and Zeng, Wangding and Hu, Shengding and Wang, Yuqing and Yuan, Jingyang and Wang, Lean and Liang, Wenfeng and others},
  journal={arXiv preprint arXiv:2512.24880},
  year={2025}
}
```

## 许可证

CANN Open Software License Agreement Version 2.0 - 详见 [LICENSE](../../LICENSE) 文件。
