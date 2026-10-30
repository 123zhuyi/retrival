# Sqrt

## 贡献说明

| 贡献者      | 贡献方              | 贡献算子 | 贡献时间       | 贡献内容     |
|----------|------------------|------|------------|----------|
| Nice_try | 西北工业大学-智能感知交互实验室 | Sqrt | 2024/12/24 | 新增Sqrt算子 |
| liuxiqiang | 个人开发者 | Sqrt | 2025/11/12 | Sqrt算子适配开源仓 |

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 200I/500 A2 推理产品

## 算子描述

- 功能描述

  `Sqrt`算子返回输入数据经过开方运算的结果。

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Sqrt</th></tr>
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td rowspan="2" align="center">算子输入</td></tr>

    <tr><td align="center">x</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>

    <tr><td rowspan="1" align="center">算子输出</td>
    <td align="center">y</td><td align="center">tensor</td><td align="center">float32,float16,bfloat16</td><td align="center">ND</td></tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">sqrt</td></tr>
  </table>

## 约束与限制

- x，y，out的数据类型仅支持float32,float16,bfloat16，数据格式仅支持ND

## 算子使用

使用该算子前，请参考[社区版CANN开发套件包安装文档](../../../docs/zh/invocation/quick_op_invocation.md)完成开发运行环境的部署。

### 编译部署

  - 进入到仓库目录

    ```bash
    cd ${git_clone_path}/ops-math
    ```

  - 执行编译

    ```bash
    bash build.sh --pkg --experimental --soc=ascend910b --ops=sqrt
    ```

  - 部署算子包

    ```bash
    ./build_out/cann-ops-<vendor_name>-linux.<arch>.run
    ```

### 算子调用

  - 执行调用

    ```bash
    bash build.sh --run_example sqrt eager cust --vendor_name=custom
    ```

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sqrt](./examples/test_aclnn_sqrt.cpp) | 通过[aclnnSqrt](./docs/aclnnSqrt.md)接口方式调用Sqrt算子。 |
