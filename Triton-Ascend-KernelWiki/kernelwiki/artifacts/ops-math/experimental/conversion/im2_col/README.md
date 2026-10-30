# Im2Col

## 贡献说明

| 贡献者      | 贡献方              | 贡献算子 | 贡献时间       | 贡献内容     |
|----------|------------------|------|------------|----------|
| Qiu Zhuang | 哈尔滨工业大学-苏统华团队 | Im2Col | 2025/12/12 | 新增Im2Col算子 |

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 200I/500 A2 推理产品

## 算子描述

- 算子功能：图像到列，滑动局部窗口数据转为列向量，拼接为大张量。从批处理输入张量中提取滑动窗口。考虑一个形状为（N, C, H, W）或 (C, H, W) 的批处理input张量，其中N是批处理维度， C是通道维度， 而 H, W 表示图像大小，此操作将input的空间维度内的每个滑动kernel_size大小的块展平为（N, C $\times \prod$（kernel_szie）, L）的3-D 或 （C $\times \prod$（kernel_szie）, L）的2-D 的 output张量的列（即最后一维），而L是这些块的总数。

- 计算公式：
  $L = \prod_{d} \lfloor \frac{spatial_size[d] + 2 \times padding[d] - dilation[d] \times （kernel_size[d] -1） -1}{stride[d]} + 1$ \rfloor, 其中spatial_size由上述input张量的H,W构成。
  $dilation[d]$ 是膨胀系数（默认为1）

- 原型信息

<table style="undefined;table-layout: fixed; width: 966px"> <colgroup> <col style="width: 144px"> <col style="width: 166px"> <col style="width: 290px"> <col style="width: 264px"> <col style="width: 102px"> </colgroup> <thead> <tr> <th>参数名</th> <th>输入/输出/属性</th> <th>描述</th> <th>数据类型</th> <th>数据格式</th> </tr> </thead> <tbody> <tr> <td>x</td> <td>输入张量</td> <td>必选输入张量，shape为3维或4维</td> <td>FLOAT、FLOAT16</td> <td>ND</td> </tr> <tr> <td>kernel_h</td> <td>属性</td> <td>卷积核的高度，可选属性，默认值为2</td> <td>INT</td> <td>-</td> </tr> <tr> <td>kernel_w</td> <td>属性</td> <td>卷积核的宽度，可选属性，默认值为2</td> <td>INT</td> <td>-</td> </tr> <tr> <td>stride_val</td> <td>属性</td> <td>步长，可选属性，默认值为1</td> <td>INT</td> <td>-</td> </tr> <tr> <td>padding_val</td> <td>属性</td> <td>填充大小，可选属性，默认值为0</td> <td>INT</td> <td>-</td> </tr> <tr> <td>z</td> <td>输出张量</td> <td>必选输出张量，shape根据输入参数推导得出</td> <td>FLOAT、FLOAT16</td> <td>ND</td> </tr> </tbody> </table>

## 约束与限制

### 1. 输入约束

- **输入张量x的形状**：必须为3维或4维
  - 4维格式：[N, C, H, W]（批处理模式）
  - 3维格式：[C, H, W]（单样本模式）
  - 其中 N ≥ 1，C ≥ 1，H ≥ 1，W ≥ 1

### 2. 参数约束

- **kernel_h**：卷积核高度，必须为正整数，满足 1 ≤ kernel_h ≤ H + 2 × padding_val
- **kernel_w**：卷积核宽度，必须为正整数，满足 1 ≤ kernel_w ≤ W + 2 × padding_val
- **stride_val**：步长，必须为正整数，stride_val ≥ 1
- **padding_val**：填充大小，必须为非负整数，padding_val ≥ 0
- **dilation**：膨胀系数，必须为正整数，dilation ≥ 1(当前实现下只支持为1的情况)

### 3. 计算约束

- **输出尺寸必须为正数**：
  $out\_H = \lfloor \frac{H + 2 \times padding\_val - dilation \times (kernel\_h - 1) - 1}{stride\_val} + 1 \rfloor ≥ 1$

  $out\_W = \lfloor \frac{W + 2 \times padding\_val - dilation \times (kernel\_w - 1) - 1}{stride\_val} + 1 \rfloor ≥ 1$

- **有效卷积核约束**：
  $kernel\_h ≤ H + 2 \times padding\_val$
  $kernel\_w ≤ W + 2 \times padding\_val$

### 4. 内存约束

- **输出张量内存大小**：
  $output\_size = N \times C \times out\_H \times out\_W \times element\_size$
  必须在设备内存限制范围内

### 5. 数据类型约束

- 输入张量x和输出张量z的数据类型必须一致
- 支持的数据类型组合：
  - 输入：FLOAT，输出：FLOAT
  - 输入：FLOAT16，输出：FLOAT16

### 6. 格式约束

- 输入和输出张量必须为ND（NCHW）格式
- 不支持其他数据格式（如NHWC、NC1HWC0等）

### 7. 特殊情况

- **当padding_val > 0时**：会在输入张量的H和W维度两侧进行对称填充
- **当dilation > 1时**：卷积核元素之间存在间隔，实际感受野增大
- **当stride_val > 1时**：输出特征图尺寸会相应减小

### 8. 错误条件

- 如果计算出的输出尺寸out_H或out_W ≤ 0，算子应返回错误
- 如果kernel_h或kernel_w大于填充后的输入尺寸，算子应返回错误
- 如果输入张量维度不符合要求（非3维或4维），算子应返回错误

## 算子使用

使用该算子前，请参考[社区版CANN开发套件包安装文档](../../../docs/zh/invocation/quick_op_invocation.md)完成开发运行环境的部署。

### 编译部署

  - 进入到仓库目录

    ```bash
    cd ${git_clone_path}/ops-math
    ```

  - 执行编译

    ```bash
    bash build.sh --experimental --ops=im2_col --soc=ascend910b --pkg
    ```

  - 部署算子包

    ```bash
    ./build_out/cann-ops-math-custom_linux-aarch64.run
    ```

### 算子调用

  - 执行调用

    ```bash
    bash build.sh --run_example im2_col eager cust --vendor_name=custom
    ```

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_im2_col](./examples/test_aclnn_im2_col.cpp) | 通过[aclnnIm2Col]接口方式调用Im2Col算子。 |
