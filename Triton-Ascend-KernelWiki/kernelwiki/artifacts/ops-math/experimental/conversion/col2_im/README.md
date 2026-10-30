# Col2Im

## 贡献说明

| 贡献者      | 贡献方              | 贡献算子 | 贡献时间       | 贡献内容     |
|----------|------------------|------|------------|----------|
| Qiu Zhuang | 哈尔滨工业大学-苏统华团队 | Col2Im | 2025/12/12 | 新增Col2Im算子 |

## 支持的产品型号

- Atlas A2 训练系列产品
- Atlas 200I/500 A2 推理产品

## 算子描述

- 算子功能：列到图像，滑动窗口列向量重组为原始图像。从形状为（N, C $\times \prod$（kernel_szie）, L）或（C $\times \prod$（kernel_szie）, L）的列矩阵中重构出形状为（N, C, H, W）或 (C, H, W) 的图像张量，其中L是滑动窗口块的总数。

- 计算公式：
  输入：列矩阵 $col$ 形状为 $(N, C \times kH \times kW, L)$ 或 $(C \times kH \times kW, L)$
  输出：图像 $output$ 形状为 $(N, C, H, W)$ 或 $(C, H, W)$
  其中：$L = out\_H \times out\_W = \frac{H + 2 \times padding\_val - dilation \times (kernel\_h - 1) - 1}{stride\_val} + 1 \times \frac{W + 2 \times padding\_val - dilation \times (kernel\_w - 1) - 1}{stride\_val} + 1$

- 原型信息

<table style="undefined;table-layout: fixed; width: 966px"> <colgroup> <col style="width: 144px"> <col style="width: 166px"> <col style="width: 290px"> <col style="width: 264px"> <col style="width: 102px"> </colgroup> <thead> <tr> <th>参数名</th> <th>输入/输出/属性</th> <th>描述</th> <th>数据类型</th> <th>数据格式</th> </tr> </thead> <tbody> <tr> <td>col</td> <td>输入张量</td> <td>必选输入张量，形状为（N, C*kH*kW, L）或（C*kH*kW, L）</td> <td>FLOAT、FLOAT16</td> <td>ND</td> </tr> <tr> <td>kernel_h</td> <td>属性</td> <td>卷积核的高度，可选属性，默认值为2</td> <td>INT</td> <td>-</td> </tr> <tr> <td>kernel_w</td> <td>属性</td> <td>卷积核的宽度，可选属性，默认值为2</td> <td>INT</td> <td>-</td> </tr> <tr> <td>output_h</td> <td>属性</td> <td>输出图像的高度，可选属性，默认值为4</td> <td>INT</td> <td>-</td> </tr> <tr> <td>output_w</td> <td>属性</td> <td>输出图像的宽度，可选属性，默认值为4</td> <td>INT</td> <td>-</td> </tr> <tr> <td>stride_val</td> <td>属性</td> <td>步长，可选属性，默认值为1</td> <td>INT</td> <td>-</td> </tr> <tr> <td>padding_val</td> <td>属性</td> <td>填充大小，可选属性，默认值为0</td> <td>INT</td> <td>-</td> </tr> <tr> <td>dilation_val</td> <td>属性</td> <td>膨胀系数，可选属性，默认值为1</td> <td>INT</td> <td>-</td> </tr> <tr> <td>x</td> <td>输出张量</td> <td>必选输出张量，形状为（N, C, H, W）或（C, H, W）</td> <td>FLOAT、FLOAT16</td> <td>ND</td> </tr> </tbody> </table>

## 约束与限制

### 1. 输入约束

- **输入张量col的形状**：必须为2维或3维
  - 3维格式：[N, C*kH*kW, L]（批处理模式）
  - 2维格式：[C*kH*kW, L]（单样本模式）
  - 其中 N ≥ 1，C ≥ 1，kH ≥ 1，kW ≥ 1，L ≥ 1

### 2. 参数约束

- **kernel_h**：卷积核高度，必须为正整数，默认值为2，满足 1 ≤ kernel_h
- **kernel_w**：卷积核宽度，必须为正整数，默认值为2，满足 1 ≤ kernel_w
- **output_h**：输出图像高度，必须为正整数，默认值为4
- **output_w**：输出图像宽度，必须为正整数，默认值为4
- **stride_val**：步长，必须为正整数，默认值为1，stride_val ≥ 1
- **padding_val**：填充大小，必须为非负整数，默认值为0，padding_val ≥ 0
- **dilation_val**：膨胀系数，必须为正整数，默认值为1，dilation_val ≥ 1

### 3. 计算约束

- **输出尺寸计算**：
  输出图像尺寸为 (output_h, output_w)，由属性直接指定

- **输入维度一致性约束**：
  $输入第二维度 = C \times kernel\_h \times kernel\_w$，且必须能被整除

  $输入第三维度 L = out\_H \times out\_W$

  其中 $out\_H$ 和 $out\_W$ 由以下公式计算：

  $out\_H = \lfloor \frac{output\_h + 2 \times padding\_val - dilation\_val \times (kernel\_h - 1) - 1}{stride\_val} + 1 \rfloor$

  $out\_W = \lfloor \frac{output\_w + 2 \times padding\_val - dilation\_val \times (kernel\_w - 1) - 1}{stride\_val} + 1 \rfloor$

- **有效卷积核约束**：
  $kernel\_h ≤ output\_h + 2 \times padding\_val$
  $kernel\_w ≤ output\_w + 2 \times padding\_val$

### 4. 内存约束

- **输出张量内存大小**：
  $output\_size = N \times C \times output\_h \times output\_w \times element\_size$
  必须在设备内存限制范围内

### 5. 数据类型约束

- 输入张量col和输出张量x的数据类型必须一致
- 支持的数据类型组合：
  - 输入：FLOAT，输出：FLOAT
  - 输入：FLOAT16，输出：FLOAT16

### 6. 格式约束

- 输入和输出张量必须为ND（NCHW）格式
- 不支持其他数据格式（如NHWC、NC1HWC0等）

### 7. 特殊情况

- **当padding_val > 0时**：会在输出张量的H和W维度两侧进行对称填充的逆操作
- **当dilation_val > 1时**：卷积核元素之间存在间隔，实际感受野增大
- **当stride_val > 1时**：输出特征图尺寸会相应增大

### 8. 与Im2Col的关系约束

- **可逆性**：当使用相同的kernel_h、kernel_w、stride_val、padding_val、dilation_val参数，并且output_h、output_w等于原始输入图像尺寸时，Col2Im应该是Im2Col的逆操作

### 9. 错误条件

- 如果输入第二维度不能被kernel_h × kernel_w整除，算子应返回错误
- 如果kernel_h或kernel_w大于填充后的输出尺寸，算子应返回错误
- 如果计算出的out_H或out_W ≤ 0，算子应返回错误
- 如果输入第三维度L ≠ out_H × out_W，算子应返回错误
- 如果输入张量维度不符合要求（非2维或3维），算子应返回错误

## 算子使用

使用该算子前，请参考[社区版CANN开发套件包安装文档](../../../docs/zh/invocation/quick_op_invocation.md)完成开发运行环境的部署。

### 编译部署

  - 进入到仓库目录

    ```bash
    cd ${git_clone_path}/ops-math
    ```

  - 执行编译

    ```bash
    bash build.sh --experimental --ops=col2_im --soc=ascend910b --pkg
    ```

  - 部署算子包

    ```bash
    ./build_out/cann-ops-math-custom_linux-aarch64.run
    ```

### 算子调用

  - 执行调用

    ```bash
    bash build.sh --run_example col2_im eager cust --vendor_name=custom
    ```

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_col2_im](./examples/test_aclnn_col2_im.cpp) | 通过[aclnnCol2Im]接口方式调用Col2Im算子。 |
