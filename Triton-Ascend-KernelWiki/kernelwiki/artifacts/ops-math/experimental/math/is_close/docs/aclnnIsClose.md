# aclnnIsClose

## Description

`aclnnIsClose` compares two tensors element by element and writes a bool output
tensor. The comparison rule is:

```text
abs(x1 - x2) <= atol + rtol * abs(x2)
```

This experimental AscendC implementation supports ND broadcasting between
`x1` and `x2`.

## API

Each aclnn operator uses a two-stage API. Call
`aclnnIsCloseGetWorkspaceSize` first to get the workspace size and executor,
then call `aclnnIsClose` to run the computation.

```cpp
aclnnStatus aclnnIsCloseGetWorkspaceSize(
    const aclTensor* x1,
    const aclTensor* x2,
    double rtol,
    double atol,
    bool equalNan,
    aclTensor* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnIsClose(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## Parameters

| Name | I/O | Description |
| :-- | :-- | :-- |
| x1 | input | First input tensor. Supports `ACL_FLOAT`, `ACL_FLOAT16`, `ACL_BF16`, and `ACL_INT32`. Format is `ACL_FORMAT_ND`. |
| x2 | input | Second input tensor. Dtype must match `x1`; shape must be broadcastable with `x1`. Format is `ACL_FORMAT_ND`. |
| rtol | input | Relative tolerance. |
| atol | input | Absolute tolerance. |
| equalNan | input | Whether two `NaN` values at the same position are treated as close. |
| y | output | Bool output tensor. Shape must equal the broadcast result of `x1` and `x2`; dtype is `ACL_BOOL`. |
| workspaceSize | output | Device workspace size required by the executor. |
| executor | output | Operator executor returned by the first-stage API. |
| workspace | input | Device workspace pointer. It can be `nullptr` when `workspaceSize` is `0`. |
| stream | input | ACL stream used to launch the operator. |

Broadcast rank is limited to 8 dimensions.

## Example

See [test_aclnn_is_close.cpp](../examples/test_aclnn_is_close.cpp).

Build and run a custom experimental package example:

```bash
bash build.sh --run_example is_close eager cust --vendor_name=custom --experimental
```
