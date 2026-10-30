# Sort

## 贡献说明

| 贡献者      | 贡献方              | 贡献算子 | 贡献时间       | 贡献内容     |
|----------|------------------|------|------------|----------|
| Zhoujianhua | 个人开发者 | Sort | 2025/12/18 | 新增Sort算子 |

## 支持的产品型号

- Atlas A2 训练系列产品（Ascend 910B）
- Atlas 200I/500 A2 推理产品（Ascend 310B）
- Atlas 推理系列产品 AI Core（Ascend 310P）
- Atlas 训练系列产品（Ascend 910）

## 算子描述

- 功能描述

  `Sort`算子沿指定轴对输入张量进行排序，返回排序后的数值张量和对应的索引张量。

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">Sort</th></tr> 
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
    <tr><td rowspan="2" align="center">算子输入</td>
    <td align="center">x</td><td align="center">tensor</td><td align="center">float16、float</td><td align="center">ND</td></tr>  
    <tr><td align="center">index</td><td align="center">tensor</td><td align="center">uint32</td><td align="center">ND</td></tr> 
    
    <tr><td rowspan="2" align="center">算子输出</td>
    <td align="center">y1</td><td align="center">tensor</td><td align="center">float16、float</td><td align="center">ND</td></tr>  
    <tr><td align="center">dstIndex</td><td align="center">tensor</td><td align="center">uint32</td><td align="center">ND</td></tr>  
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">sort</td></tr>  
  </table>

## 约束与限制

- 支持 float16, float32 数据类型。
- 输入 shape 暂时只支持二维，可以指定其中任一维度排序，输出排序结果以及排序后的索引顺序（可选）。
- 支持升序和降序排序，排序的稳定性取决于sort接口。
- 不支持广播机制，仅支持对给定维度的独立排序。

## 算子使用

使用该算子前，请参考[社区版CANN开发套件包安装文档](../../../docs/zh/invocation/quick_op_invocation.md)完成开发运行环境的部署。

### 编译部署

  - 进入到仓库目录

    ```bash
    cd ${git_clone_path}/ops-math
    ```

  - 执行编译

    ```bash
    bash build.sh --experimental --ops=sort_v2 --soc=ascend910b --pkg
    ```

  - 部署算子包

    ```bash
    ./build_out/cann-ops-math-custom_linux-aarch64.run
    ```

### 算子调用

  - 执行调用

    ```bash
    bash build.sh --run_example sort_v2 eager cust --vendor_name=custom
    ```    

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sort_v2](./examples/test_aclnn_sort_v2.cpp) | 通过aclnnSortV2接口方式调用Sort算子。 |
