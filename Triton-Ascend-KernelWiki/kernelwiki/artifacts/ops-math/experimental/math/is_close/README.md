# IsClose

## Product Support

| Product | Supported |
| :-- | :--: |
| Atlas A2 training/inference series | Yes |

## Function

`IsClose` compares two tensors element by element and returns a bool tensor.
For each element, the result is true when:

```text
abs(x1 - x2) <= atol + rtol * abs(x2)
```

For floating-point inputs, exact equality is also treated as close. When
`equal_nan` is true, `NaN` values at the same position are treated as close.

## Parameters

| Name | I/O | Description | Type | Format |
| :-- | :-- | :-- | :-- | :-- |
| x1 | input | First input tensor. | FLOAT, FLOAT16, BFLOAT16, INT32 | ND |
| x2 | input | Second input tensor. Must have the same dtype as `x1` and be broadcastable with `x1`. | FLOAT, FLOAT16, BFLOAT16, INT32 | ND |
| rtol | attr | Relative tolerance. Default is `1e-5`. | FLOAT | - |
| atol | attr | Absolute tolerance. Default is `1e-8`. | FLOAT | - |
| equal_nan | attr | Whether two `NaN` values are treated as close. Default is `false`. | BOOL | - |
| y | output | Bool result tensor. Shape is the broadcast result of `x1` and `x2`. | BOOL | ND |

## Constraints

- `x1` and `x2` must satisfy ND broadcast rules. The output shape must exactly match the broadcast result.
- Broadcast rank is limited to 8 dimensions.
- `x1` and `x2` must have the same dtype.
- Supported input dtypes are `FLOAT`, `FLOAT16`, `BFLOAT16`, and `INT32`.

## Usage

| Mode | Example | Description |
| :-- | :-- | :-- |
| aclnn | [test_aclnn_is_close](./examples/test_aclnn_is_close.cpp) | Calls `IsClose` through the two-stage `aclnnIsClose` API. |
