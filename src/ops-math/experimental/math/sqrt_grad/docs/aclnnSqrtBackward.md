# aclnnSqrtBackward

## 支持产品

| 产品 | 是否支持 |
| :--- | :---: |
| `ascend910b` | 是 |

## 功能

`aclnnSqrtBackward` 计算 `sqrt` 前向输入的梯度，输入为前向输出 `y` 和上游梯度 `dy`，输出为 `z`。

## 数学定义

逐元素计算：

```text
tmp = 0.5 * dy
z = tmp != 0 ? tmp / y : 0
```

其中：

- `float32` 路径直接在原 dtype 上执行 `Muls + Compare + Div + Select`。
- `float16` / `bfloat16` 路径先升到 `float32` 计算，再 cast 回输出 dtype。

## 接口

```cpp
aclnnStatus aclnnSqrtBackwardGetWorkspaceSize(
    const aclTensor  *y, 
    const aclTensor  *dy, 
    aclTensor        *z, 
    uint64_t         *workspaceSize, 
    aclOpExecutor   **executor);

aclnnStatus aclnnSqrtBackward(
    void              *workspace, 
    uint64_t           workspaceSize, 
    aclOpExecutor     *executor, 
    const aclrtStream  stream);
```

## 参数

| Name | Description | Dtype | Format | Shape | Required |
| --- | --- | --- | --- | --- | --- |
| `y` | 前向 `sqrt` 的输出 | `float32` / `float16` / `bfloat16` | `ND` | 0 到 8 维 | Yes |
| `dy` | 上游反向梯度 | `float32` / `float16` / `bfloat16` | `ND` | 与 `y` 完全一致 | Yes |
| `z` | 反向输出梯度 | `float32` / `float16` / `bfloat16` | `ND` | 与 `y` 完全一致 | Yes |
| `workspaceSize` | 返回 workspace 大小 | `uint64_t *` | - | - | Yes |
| `executor` | 返回执行器 | `aclOpExecutor **` | - | - | Yes |

## 约束

- 不支持 broadcast。
- `y`、`dy`、`z` 的 dtype 必须完全一致。
- `y`、`dy`、`z` 的 shape 必须完全一致。
- 当前仅支持 `ND`，维度数不超过 8。

## 返回值

- `ACLNN_SUCCESS`：成功。
- `ACLNN_ERR_PARAM_INVALID`：shape、dtype、rank 或格式不满足约束。
- `ACLNN_ERR_PARAM_NULLPTR`：输入或输出指针为空。
- `ACLNN_ERR_INNER_CREATE_EXECUTOR`：内部执行器创建失败。
