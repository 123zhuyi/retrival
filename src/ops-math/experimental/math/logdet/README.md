# Logdet

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | √ |

## 功能说明

- 算子功能：计算输入方阵或方阵 batch `self` 的行列式自然对数 `log(det(self))`。
- 输出语义：
  - `det(self) > 0` 时，输出 `log(det(self))`
  - `det(self) = 0` 时，输出 `-inf`
  - `det(self) < 0` 时，输出 `NaN`

- 计算公式：

$$
\mathrm{out} = \log(\det(\mathrm{self}))
$$

其中 `self` 的最后两维必须构成方阵，整体 shape 为 `(*, n, n)`。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| `self` | 输入 | 输入方阵或方阵 batch | `FLOAT` | `ND` | `(*, n, n)`，rank >= 2 |
| `out` | 输出 | 行列式自然对数结果 | `FLOAT` | `ND` | `self.shape[:-2]` |

## 约束说明

- 仅支持 `float32`
- 仅支持 `ND`
- `self` 最后两维必须相等
- 支持非连续 Tensor
- 支持高维输入（大于 8 维时内部会做 reshape 归一化）

## 目录说明

| 目录 / 文件 | 说明 |
| ---- | ---- |
| [`docs/`](./docs) | 需求、设计、测试、规格与开发日志 |
| [`op_api/`](./op_api) | ACLNN 两段式接口实现 |
| [`op_host/`](./op_host) | Host 侧注册、InferShape、Tiling |
| [`op_kernel/`](./op_kernel) | Ascend C Kernel 实现 |
| [`tests/st/`](./tests/st) | ST 测试工程 |
| [`tests/ut/`](./tests/ut) | UT 测试工程 |

## 调用与文档入口

| 内容 | 路径 |
| ---- | ---- |
| ACLNN 接口说明 | [docs/aclnnLogdet.md](./docs/aclnnLogdet.md) |
