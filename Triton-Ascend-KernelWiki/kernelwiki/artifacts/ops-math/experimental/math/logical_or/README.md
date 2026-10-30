# aclnnLogicalOr

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|----|----|----|------|------|
| hky | 浙江工业大学-智能计算实验室 | LogicalOr | 2025/06/24 | 新增LogicalOr算子，实现了对输入数据计算逻辑或，获取输出数据的功能。|

## 支持的产品型号

-Atlas A2 训练系列产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 算子描述

- 功能描述

  LogicalOr算子的主要功能是对输入的两个数值（或张量）进行逐元素逻辑或运算。在数学和工程领域中，比较是一个基础且常见的操作，被广泛应用于图像处理、信号处理、逻辑运算等多个领域。LogicalOr算子能够高效地处理批量数据的比较，支持布尔类型的输入。

- 原型信息

  <table>
  <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">LogicalOr</th></tr>
  <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td rowspan="2" align="center">算子输入</td></tr>
  <tr><td align="center">x1</td><td align="center">tensor</td><td align="center">bool</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">算子输入</td></tr>
  <tr><td align="center">x2</td><td align="center">tensor</td><td align="center">bool</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td>
  <td align="center">y</td><td align="center">tensor</td><td align="center">bool</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">logical_or</td></tr>
  </table>

## 约束与限制

  - x1，x2，out的数据类型只支持BOOL，数据格式只支持ND

## 算子使用

使用该算子前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

### 编译部署

  - 进入到仓库目录

    ```bash
    cd ${git_clone_path}/ops-math
    ```

  - 执行编译

    ```bash
    bash build.sh --pkg --experimental --soc=ascend910b --ops=logical_or
    ```

  - 部署算子包

    ```bash
    ./build_out/cann-ops-<vendor_name>-linux.<arch>.run
    ```

### 算子调用

  - 执行调用

    ```bash
    bash build.sh --run_example --experimental logical_or eager cust --vendor_name=custom
    ```

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_logical_or](./examples/test_aclnn_logical_or.cpp) | 通过[aclnnLogical](./docs/aclnnLogicalOr.md)接口方式调用LogicalOr算子。 |
