# ArgMaxWithValue

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |

## 功能说明

- 算子功能：在指定维度 `dimension` 上，返回输入张量 `x` 的**最大值** `values` 及其
  **首次出现的下标** `indice`。等价于内置 `aclnnMaxDim`（底层 IR 为 ArgMaxWithValue）。
- 相等最大值取首个下标；支持 NaN 传播语义（与 torch.max(dim) 一致）。

## 参数说明

| 参数 | 输入/输出/属性 | 数据类型 | 数据格式 | 说明 |
| :--- | :------------- | :------- | :------- | :--- |
| x | 输入 | float16 / float / bfloat16 / int16 | ND | 1–8 维，支持非连续 |
| dimension | 属性 | int | - | 归约轴，[-x.dim(), x.dim()) |
| keep_dims | 属性 | bool | - | 是否保留归约轴，默认 false |
| indice | 输出 | int32 | ND | 最大值下标，与 values 同 shape |
| values | 输出 | 同 x | ND | 最大值，keep_dims=false 时降一维 |

## 约束说明

- `dimension` 取值需在合法范围内。
- 规约轴长度必须大于 0；空规约轴属于未定义输入。

## 调用说明

| 调用方式 | 样例 | 说明 |
| :------- | :--- | :--- |
| aclnn 接口 | [examples/test_aclnn_arg_max_with_value.cpp](examples/test_aclnn_arg_max_with_value.cpp) | 5 组自检用例，覆盖 4 dtype、负轴、keepdim、ties、非连续输入 |
