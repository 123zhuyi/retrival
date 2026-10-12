# aclnnFmodScalar & aclnnInplaceFmodScalar

## 功能说明

对 `self` 和 host scalar `other` 执行 Mod 取余计算：`out = self - other * trunc(self / other)`。`out` shape 需要与 `self` 一致。

## 接口原型

```cpp
aclnnStatus aclnnFmodScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnFmodScalar(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnInplaceFmodScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnInplaceFmodScalar(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
```

## 约束

- `self`、`out` 支持 ND，维度不超过 8。
- aclnn 层支持 DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8、INT16 类型推导；AICore kernel 覆盖 BFLOAT16、FLOAT16、FLOAT32、INT32，其余类型走 AICPU fallback。BFLOAT16 仅在支持该数据类型的 NPU 平台上走 AICore。
- **INT16 同数据类型计算、以及 `self`/`other` 分别为 INT16 与 BFLOAT16/FLOAT16/FLOAT32（数据类型不同）的混合数据类型计算，仅在 Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品上由 AICore 支持**；其余产品不在该增强范围内。已有的 BFLOAT16/FLOAT16/FLOAT32/INT32 同数据类型计算，以及 DOUBLE/INT64/INT8/UINT8 的 AICPU 回退，在原支持产品上的行为保持不变。
- 精度：针对 `self`/`other` 商值较大（大 \|self/other\|）的场景，Atlas A2/A3 上的 AICore 计算路径引入了数值稳定性增强算法，相比朴素截断取余（trunc-mod）实现降低了大商场景下的精度损失风险。
- `out` shape 必须等于 `self` shape。

## 样例

见 `../examples/test_aclnn_fmod_scalar.cpp`。
