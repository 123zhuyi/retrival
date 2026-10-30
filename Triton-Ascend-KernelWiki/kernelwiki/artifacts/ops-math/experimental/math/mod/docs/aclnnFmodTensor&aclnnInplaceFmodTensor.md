# aclnnFmodTensor & aclnnInplaceFmodTensor

## 功能说明

对 `self` 和 tensor `other` 执行 Mod 取余计算：`out = self - other * trunc(self / other)`。`other` 需要能广播到 `self`，`out` shape 需要与 `self` 一致。

## 接口原型

```cpp
aclnnStatus aclnnFmodTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnFmodTensor(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnInplaceFmodTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnInplaceFmodTensor(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
```

## 约束

- `self`、`other`、`out` 支持 ND，维度不超过 8。
- aclnn 层支持 DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8 类型推导；AICore kernel 覆盖 BFLOAT16、FLOAT16、FLOAT32、INT32，其余类型走 AICPU fallback。
- `out` shape 必须等于 `self` shape。

## 样例

见 `examples/test_aclnn_fmod_tensor.cpp`。
