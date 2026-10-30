# MaskedScale

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |     √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- **算子功能**：完成elementwise计算
- **计算公式**：

  $$
  out = self \times mask \times scale
  $$

## 调用说明

当前算子作为自定义算子通过图模式下发执行，`aclnnMaskedScale` 两段式接口仅供内部使用，不对外暴露。

## 算子输入输出

- **参数说明：**

  - self(计算输入)：公式中的输入`self`，Device侧Tensor。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、BFLOAT16、FLOAT，[数据格式](../../../docs/zh/context/data_format.md)支持ND。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、BFLOAT16、FLOAT，[数据格式](../../../docs/zh/context/data_format.md)支持ND。

  - mask(计算输入)：公式中的`mask`，Device侧Tensor，shape需要与self一致。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持UINT8、INT8、FLOAT16、FLOAT，[数据格式](../../../docs/zh/context/data_format.md)支持ND。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT8、INT8、FLOAT16、FLOAT，[数据格式](../../../docs/zh/context/data_format.md)支持ND。

  - scale(float, 计算输入)：标量缩放系数，数据类型支持FLOAT（非Tensor）。

  - y(计算输出)：公式中的`out`，Device侧Tensor，数据类型和shape需要与self一致。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、BFLOAT16、FLOAT，[数据格式](../../../docs/zh/context/data_format.md)支持ND。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、BFLOAT16、FLOAT，[数据格式](../../../docs/zh/context/data_format.md)支持ND。

## 约束说明

- 确定性计算：
  - MaskedScale默认确定性实现。
- shape约束：
  - self、mask和y的shape需要一致。
  - 当前实现的元素总数不能超过uint32_t可表示范围。
  - 输入/属性命名采用 `self`、`mask`、`scale`、`y`，与内部 aclnn 接口保持一致。
  - FLOAT16 路径中 `scale` 按 half 标量参与向量乘，存在标量精度截断；FLOAT/BFLOAT16 路径按 float 标量计算。
