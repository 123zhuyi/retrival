# aclnnDivV3

## 功能描述

根据指定的舍入模式（mode）对两个张量执行除法运算：

- **mode=0 (RealDiv)**：直接浮点除法，不做舍入
  $$ out_i = self_i / other_i $$

- **mode=1 (TruncDiv)**：除法后向零取整
  $$ out_i = \text{trunc}(self_i / other_i) $$

- **mode=2 (FloorDiv)**：除法后向下取整
  $$ out_i = \lfloor self_i / other_i \rfloor $$

## 接口原型

```cpp
aclnnStatus aclnnDivV3GetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* other,
    int64_t          mode,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor);

aclnnStatus aclnnDivV3(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream);
```

## 参数说明

| 参数名        | 输入/输出 | 说明                                       |
| :------------ | :-------- | :----------------------------------------- |
| self          | 输入      | 被除数，device 侧 aclTensor               |
| other         | 输入      | 除数，device 侧 aclTensor                 |
| mode          | 输入      | 舍入模式：0/1/2                            |
| out           | 输出      | 结果张量，device 侧 aclTensor             |
| workspaceSize | 输出      | 所需 workspace 大小                        |
| executor      | 输出      | op 执行器                                  |
| workspace     | 输入      | workspace 内存起始地址                     |
| stream        | 输入      | acl stream 流                              |

## 支持的数据类型

| self     | other    | out      |
| :------- | :------- | :------- |
| FLOAT32  | FLOAT32  | FLOAT32  |
| FLOAT16  | FLOAT16  | FLOAT16  |
| BFLOAT16 | BFLOAT16 | BFLOAT16 |
| INT32    | INT32    | INT32    |
| INT16    | INT16    | INT16    |

## 广播规则

- self 和 other 的 shape 需满足 NumPy 广播规则。
- out 的 shape 必须等于广播后的 shape。
- 广播在 aclnn 接口层通过 `l0op::BroadcastTo` 完成，kernel 层仅执行逐元素计算。

## 约束说明

- self 和 other 的数据类型必须一致。
- out 的数据类型必须与 self 一致。
- 数据格式仅支持 ND。
- 维度数不超过 8 维。
- mode 值只能为 0、1 或 2。

## 调用示例

参见 [test_aclnn_div_v3.cpp](../examples/test_aclnn_div_v3.cpp)。
