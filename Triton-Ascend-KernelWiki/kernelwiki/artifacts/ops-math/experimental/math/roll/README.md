# Roll

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| --- | --- | --- | --- | --- |
| boxw987 | 个人开发者 | Roll | 2026/06 | Roll 算子适配开源仓 |

## 支持的产品型号

- Atlas A2 训练系列产品

## 算子描述

- 功能描述

  `Roll` 沿给定维度对输入张量执行循环位移；当 `dims` 为空时，先按逻辑视图展平，再执行一维循环位移，最后按原始形状输出。

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Roll</th></tr>
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td rowspan="1" align="center">算子输入</td><td align="center">x</td><td align="center">tensor</td><td align="center">uint8, int8, bfloat16, float16, float32, int32, uint32</td><td align="center">ND</td></tr>
    <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">tensor</td><td align="center">与 x 相同</td><td align="center">ND</td></tr>
    <tr><td rowspan="2" align="center">属性</td><td align="center">shifts</td><td align="center">listInt</td><td align="center">整型列表</td><td align="center">-</td></tr>
    <tr><td align="center">dims</td><td align="center">listInt</td><td align="center">整型列表</td><td align="center">-</td></tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">roll</td></tr>
  </table>

## 约束与限制

- 仅支持 `ND` 格式。
- 支持 0 维到 8 维输入。
- `dims` 为空时，`shifts` 长度必须为 1。
- `dims` 非空时，`shifts` 与 `dims` 长度必须一致。
- `dims` 取值范围为 `[-rank, rank)`。

## 算子使用

使用该算子前，请参考[社区版 CANN 开发套件包安装文档](../../../docs/zh/invocation/quick_op_invocation.md)完成开发运行环境部署。

### 编译部署

```bash
cd ${git_clone_path}/ops-math
bash build.sh --pkg --experimental --soc=ascend910b --ops=roll
./build_out/cann-ops-<vendor_name>-linux.<arch>.run
```

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| aclnn 调用 | [test_aclnn_roll.cpp](./examples/test_aclnn_roll.cpp) | 通过 [aclnnRoll](./docs/aclnnRoll.md) 接口方式调用 Roll 算子 |
