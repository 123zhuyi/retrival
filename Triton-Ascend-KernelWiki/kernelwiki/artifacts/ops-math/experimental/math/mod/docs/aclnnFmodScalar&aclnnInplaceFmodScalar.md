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
- aclnn 层支持 DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8 类型推导；AICore kernel 覆盖 BFLOAT16、FLOAT16、FLOAT32、INT32，其余类型走 AICPU fallback。BFLOAT16 仅在支持该数据类型的 NPU 平台上走 AICore。
- `out` shape 必须等于 `self` shape。

## 样例

见 `examples/test_aclnn_fmod_scalar.cpp`。
