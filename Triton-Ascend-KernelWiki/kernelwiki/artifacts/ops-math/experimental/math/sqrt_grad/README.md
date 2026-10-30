# sqrt_grad

`sqrt_grad` 为前向 `sqrt` 的反向梯度算子 ACLNN 绑定工程，对外暴露接口 `aclnnSqrtBackward`。

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| `ascend910b` | 是 |

## 数学语义

输入前向输出 `y` 和上游梯度 `dy`，输出 `z`。逐元素语义严格对齐已验证 AscendC kernel：

```text
tmp = 0.5 * dy
z = tmp != 0 ? tmp / y : 0
```

实现保持 `CopyIn -> Compute -> CopyOut` 流水顺序不变。`float32` 直接计算；`float16` 与 `bfloat16` 先升到 `float32` 计算，再回写原 dtype。

## 参数与约束

| Name | Role | Dtype | Format | Shape |
| --- | --- | --- | --- | --- |
| `y` | 输入，前向 `sqrt` 的输出 | `float32` / `float16` / `bfloat16` | `ND` | 0 到 8 维 |
| `dy` | 输入，上游梯度 | `float32` / `float16` / `bfloat16` | `ND` | 与 `y` 完全一致 |
| `z` | 输出，前向输入的梯度 | `float32` / `float16` / `bfloat16` | `ND` | 与 `y` 完全一致 |

约束：

- 不支持 broadcast，`y`、`dy`、`z` 的 shape 和 dtype 必须完全一致。
- 支持 0 维标量和动态 shape / 动态 rank。
- 当前 host/kernel 只支持 `ND`。
- 当 `0.5 * dy == 0` 时直接输出 `0`，避免进入除法结果路径。

## ACLNN 接口

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

## 构建与验证

```bash
source ${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/set_env.sh
cd <ops-repo>

bash build.sh --experimental --ops=sqrt_grad -j8 -O2
bash build.sh --experimental --ops=sqrt_grad -u --opapi -j8 -O2
bash build.sh --pkg --experimental --soc=ascend910b --ops=sqrt_grad --vendor_name=custom -j8 -O2
./build_out/cann-ops-math-custom_linux-${HOST_ARCH}.run
```

Example 编译与运行：

```bash
cd <ops-repo>/experimental/math/sqrt_grad/examples
./run.sh
```
