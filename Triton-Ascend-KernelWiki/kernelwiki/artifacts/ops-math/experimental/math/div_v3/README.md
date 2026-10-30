# DivV3（带模式的除法 / DivMod）

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------- |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √        |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √        |

## 功能说明

- 算子功能：根据 `mode` 参数完成不同舍入模式的除法计算，对应官方 `aclnnDivMod` 语义。
- 计算公式：
  - **mode=0 (RealDiv)**：$$ y = x1 / x2 $$
  - **mode=1 (TruncDiv)**：$$ y = \text{trunc}(x1 / x2) $$
  - **mode=2 (FloorDiv)**：$$ y = \lfloor x1 / x2 \rfloor $$
- 本算子使用原生 Ascend C LocalTensor 手写流水线实现（CopyIn / Compute / CopyOut 三级流水），
  而非高级 DAG/BroadcastSch 模板，用于学习和对比低级实现方式。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述                               | 数据类型 | 数据格式 |
| :----- | :------------- | :--------------------------------- | :------- | :------- |
| x1     | 输入           | 被除数张量                         | 见下方   | ND       |
| x2     | 输入           | 除数张量                           | 见下方   | ND       |
| mode   | 属性           | 舍入模式：0=RealDiv, 1=Trunc, 2=Floor | int32  | -        |
| y      | 输出           | 除法计算的结果                     | 见下方   | ND       |

### 数据类型支持

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - FLOAT32、FLOAT16、BFLOAT16、INT32、INT16

### 数据格式

- 仅支持 ND 格式。

## 广播策略

- 广播操作在 aclnn 接口层（op_api）通过调用 `l0op::BroadcastTo` 完成。
- Kernel 层假设输入 x1、x2 已经是广播后的同形状张量，执行逐元素计算。
- 输入 shape 需满足 NumPy 广播规则，输出 shape 为广播后的 shape。

## 实现架构

```text
aclnnDivV3 (op_api 层)
  ├── 参数检查 (dtype / shape / format)
  ├── Contiguous 转换
  ├── BroadcastTo 广播对齐
  ├── l0op::DivV3 → kernel 调度
  │     └── DivV3 kernel (手写流水线)
  │           ├── CopyIn:  GM → UB (DataCopyPad)
  │           ├── Compute: 根据 mode 分支计算
  │           │     ├── mode=0: Div
  │           │     ├── mode=1: Div → Trunc (或 Cast CAST_TRUNC)
  │           │     └── mode=2: Div → Floor
  │           └── CopyOut: UB → GM (DataCopyPad)
  └── ViewCopy 输出
```

## Kernel 内部类型转换策略

| 输入类型  | 计算路径                                    |
| :-------- | :------------------------------------------ |
| float32   | 直接 Div，按 mode 执行 Floor/Trunc          |
| float16   | Cast → float32 → Div+mode → Cast 回 float16 |
| bfloat16  | Cast → float32 → Div+mode → Cast 回 bfloat16|
| int32     | Cast → float32 → Div+mode → Cast 回 int32   |
| int16     | Cast → float32 → Div+mode → Cast 回 int16   |

## 相比 DivV2 的改进

1. **新增 mode 属性**：支持三种除法模式，覆盖 DivMod 完整语义
2. **DataCopyPad**：尾块搬运使用 DataCopyPad 避免非对齐数据丢失
3. **成员变量初始化**：所有基础类型成员变量声明时初始化为 0
4. **op_api 层广播**：在 aclnn 层完成广播，kernel 只做逐元素计算
5. **完善边界处理**：增加 Floor 模式的 tmpBuf 支持（Floor 指令需要额外缓冲区）

## 调用说明

| 调用方式  | 样例代码                                              | 说明                                  |
| :-------- | :---------------------------------------------------- | :------------------------------------ |
| aclnn接口 | [test_div_v3](./examples/test_aclnn_div_v3.cpp)       | 通过 aclnnDivV3 接口调用 DivV3 算子   |
