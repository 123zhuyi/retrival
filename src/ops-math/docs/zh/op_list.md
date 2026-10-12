# 算子列表

> **使用说明**：
>
> - **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、examples、文档等，目录介绍参见[项目目录](./install/dir_structure.md)。
> - **算子执行硬件单元**：大部分算子运行在AI Core，少部分算子运行在AI CPU。默认情况下，项目中提到的算子一般指AI Core算子。关于AI Core和AI CPU详细介绍参见[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中“概念原理和术语 > 硬件架构与数据处理原理”。
> - **算子接口列表**：为方便调用算子，CANN提供一套C API执行算子，一般以aclnn为前缀，全量接口参见[aclnn列表](op_api_list.md)。
> - **V版本演进说明**：部分算子存在多个V版本，使用时选择最高V版本即可（高版本算子已兼容低版本算子的所有能力）。

项目提供的所有算子分类和算子列表如下：

<table><thead>
  <tr>
    <th rowspan="2">算子分类</th>
    <th rowspan="2">算子目录</th>
    <th colspan="2">算子实现</th>
    <th>aclnn调用</th>
    <th>图模式调用</th>
    <th rowspan="2">算子执行硬件单元</th>
    <th rowspan="2">说明</th>
  </tr>
  <tr>
    <th>op_kernel</th>
    <th>op_host</th>
    <th>op_api</th>
    <th>op_graph</th>
  </tr></thead>
<tbody>
  <tr>
    <td>math</td>
    <td><a href="../../math/abs/README.md">abs</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>为输入张量的每一个元素取绝对值。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/accumulate_nv2/README.md">accumulate_nv2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对多个相同形状、相同数据类型的输入张量进行逐元素累加，输出累加后的张量。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/acos/README.md">acos</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>该算子对输入的每个元素进行反余弦操作后输出。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/acosh/README.md">acosh</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>计算逆双曲余弦（inverse hyperbolic cosine），支持FLOAT、FLOAT16、BF16数据类型。</td>
  </tr>
    <tr>
    <td>math</td>
    <td><a href="../../math/acosh_grad/README.md">acosh_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CORE</td>
    <td>计算Acosh算子的反向梯度，公式z = dy / sqrt(y² - 1)。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/acos_grad/README.md">acos_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CORE</td>
    <td>计算Acos算子的反向梯度，公式z = -dy / sqrt(1 - y²)。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/ada_cast/README.md">ada_cast</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CORE</td>
    <td>uint16→float16 类型转换并按 pixel 归一化（HDRnet 白电平缩放），公式 y = SaturateCast_FP16(float32(x) × (1/pixel))。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/add/README.md">add</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CORE/AI CPU</td>
    <td>该算子用于完成加法计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/add_v2/README.md">add_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对两个输入张量执行逐元素加法，兼容TensorFlow AddV2语义，支持广播与空Tensor。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/add_lora/README.md">add_lora</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将输入x根据输入索引indices，分别和对应的weightA，weightB相乘，然后将结果累加到输入y上并输出。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/add_mat_mat_elements/README.md">add_mat_mat_elements</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入张量a、b进行逐元素相乘后，与输入张量c按标量alpha、beta进行加权求和，并输出结果张量cOut。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/add_n/README.md">add_n</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>对输入进行主元素相加求和操作。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/addcdiv/README.md">addcdiv</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>张量运算函数，用于执行乘除加组合操作，将张量除法（带缩放）+ 张量加法合并为单个操作。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/addcmul/README.md">addcmul</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>张量运算函数，用于执行乘除加组合操作，将张量乘法（带缩放）+ 张量加法合并为单个操作。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/addr/README.md">addr</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>求一维向量vec1和vec2的外积得到一个二维矩阵，并将外积结果矩阵乘一个系数后和自身乘系数相加后输出。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/adjacent_difference/README.md">adjacent_difference</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>比较输入张量相邻元素的差异，若相邻元素相同，返回0，否则返回1。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/affine_grid">affine_grid</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>给定一组3维的仿射参数矩阵(theta)以及输出图像的大小(size)，生成一个2D或3D的网格。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/angle_v2/README.md">angle_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>为输入张量的每一个元素取角度（单位：弧度）。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/arg_max_v2/README.md">arg_max_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回张量在指定维度（dim）上的最大值的索引，并保存到out张量中。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/arg_max_with_value/README.md">arg_max_with_value</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回Tensor指定维度的最大值及其索引位置。最大值保存到out中，最大值的索引保存到indices中。如果keepdim为false，则不保留对应的轴；如果为true，则保留指定轴的维度值为1。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/arg_min/README.md">arg_min</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回tensor中指定轴的最小值索引，并保存到out中。如果keepdim为false，则不保留对应的轴；如果为true，则保留指定轴的维度值为1。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/arg_min_with_value/README.md">arg_min_with_value</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回self中指定维度的最小值及其索引位置。最小值保存到out中，最小值的索引保存到indices中。如果keepdim为false，则不保留对应的轴；如果为true，则保留指定轴的维度值为1。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/asin/README.md">asin</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU </td>
    <td>该算子对输入的每个元素进行反正弦操作后输出。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/asinh/README.md">asinh</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>该算子对输入Tensor中的每个元素进行反双曲正弦操作后输出。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/asin_grad/README.md">asin_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Asin（反正弦）算子的反向计算，根据正向输入x和上游梯度dy，计算输入梯度dx。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/asinh_grad/README.md">asinh_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Asinh（反双曲正弦）算子的反向计算，根据正向输入y和上游梯度dy，计算输入梯度dx。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/assign_add/README.md">assign_add</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子完成在原有tensor上的加法计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/assign_sub/README.md">assign_sub</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子完成在原有tensor上的减法计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/atan/README.md">atan</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>该算子对输入的每个元素进行反正切操作后输出。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/atan2/README.md">atan2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算输入张量self中每个元素和输入标量other的反正切（四象限）。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/atanh/README.md">atanh</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>计算逆双曲正切（inverse hyperbolic tangent），支持FLOAT、FLOAT16、BF16数据类型。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/atan_grad/README.md">atan_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Atan（反正切）算子的反向计算，根据正向输入x和上游梯度dy，计算输入梯度dx，用于神经网络反向传播。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/axpy/README.md">axpy</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>源操作数中每个元素与标量求积后和目的操作数中的对应元素相加。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/axpy_v2/README.md">axpy_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>源操作数中每个元素与张量求积后和目的操作数中的对应元素相加。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bessel_i0e/README.md">bessel_i0e</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算指数缩放的第一类零阶修正贝塞尔函数，公式y = exp(-|x|) * I0(x)，仅支持图模式调用。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bessel_i1e/README.md">bessel_i1e</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算指数缩放修正贝塞尔函数（第一类，阶数1）。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bias/README.md">bias</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>按照axis、num_axes和bias_from_blob推导bias的广播形状，计算y = x + broadcast(bias)。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bias_add/README.md">bias_add</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>偏置加法。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bias_add_grad/README.md">bias_add_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算偏置的梯度。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bincount/README.md">bincount</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算非负整数数组中每个数的频率。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bitwise_and/README.md">bitwise_and</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算输入tensor中每个元素和输入标量的按位与结果。输入的tensor与other必须是整型或者BOOL型变量。当输入为BOOL时，计算逻辑与的结果。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bitwise_not/README.md">bitwise_not</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bitwise_or/README.md">bitwise_or</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算输入张量self中每个元素和输入标量other的按位或。输入self和other必须是整数或布尔类型，对于布尔类型，计算逻辑或。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bitwise_xor/README.md">bitwise_xor</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cast/README.md">cast</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>将输入tensor转换为指定的dtype类型。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cdist/README.md">cdist</a></td>
    <td> </td>
    <td> </td>
    <td>√</td>
    <td> </td>
    <td> </td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cdist_grad/README.md">cdist_grad</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/ceil/README.md">ceil</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子用于返回输入Tensor中每个元素向上取整的结果。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cholesky/README.md">cholesky</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/compare_and_bit_pack/README.md">compare_and_bit_pack</a></td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>将输入与阈值进行比较，将比较结果的比特位打包为一个uint8输出。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/complex/README.md">complex</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/complex_abs/README.md">complex_abs</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/conj/README.md">conj</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>返回复数张量的共轭，对输入每个复数元素取共轭（实部不变、虚部取反）。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cos/README.md">cos</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回输入张量元素进行余弦操作后的结果。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cosh/README.md">cosh</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>返回输入张量元素进行的双曲余弦值结果。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cosine_similarity/README.md">cosine_similarity</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>计算两个输入张量沿指定维度的余弦相似度。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cross/README.md">cross</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>该算子对输入Tensor完成linear_cross运算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cummax/README.md">cummax</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cummin/README.md">cummin</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cumprod/README.md">cumprod</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cumulative_logsumexp/README.md">cumulative_logsumexp</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入张量沿指定维度计算累积log-sum-exp。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cumsum/README.md">cumsum</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子对输入张量self的元素，按照指定维度dim依次进行累加。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cumsum_cube/README.md">cumsum_cube</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/data_compare/README.md">data_compare</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>逐元素比较两个相同shape和dtype的输入张量，统计差异超出容差范围的元素总个数。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/dawsn/README.md">dawsn</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算Dawson积分 F(x) = exp(-x^2) * integral_0^x exp(t^2) dt。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/diag_part/README.md">diag_part</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>获取输入张量对角线元素。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/digamma/README.md">digamma</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/div/README.md">div</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>张量除法计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/truncate_div/README.md">truncate_div</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>完成截断除法计算，结果向零取整。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/div_no_nan/README.md">div_no_nan</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>张量除法计算，在分母可能为0的情况下，可以帮助避免由于分母为0导致的NaN值问题。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/dot/README.md">dot</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算两个输入一维张量的点积结果。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/eltwise/README.md">eltwise</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对1~32个同shape、同dtype的输入张量执行逐元素操作，支持PRODUCT（逐元素乘积）、SUM（逐元素加权求和）、MAX（逐元素取最大值）三种计算模式。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/equal/README.md">equal</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>该算子计算两个Tensor是否有相同的大小和元素。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/erf/README.md">erf</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>返回输入张量中每个元素对应的误差函数的值。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/erfc/README.md">erfc</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回输入张量中每个元素对应的互补误差函数的值</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/exp/README.md">exp</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>返回一个新的张量，该张量的每个元素都是输入张量对应元素的指数。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/exp_segsum_grad/README.md">exp_segsum_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Segsum的反向计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/expand/README.md">expand</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>将输入tensor广播到指定的shape。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/expint/README.md">expint</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算指数积分函数。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/expm1/README.md">expm1</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>以输入张量为指数，计算自然常数e的幂，并对指数计算结果进行减1计算。对于输入张量取值较小的场景，提供比直接用公式计算结果更高的精度。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/eye/README.md">eye</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回一个对角线值为1其余位置为0的二维张量。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/floor/README.md">floor</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回输入Tensor中每个元素向下取整，并将结果回填到输入Tensor中。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/floor_div/README.md">floor_div</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>完成除法计算，对结果向下取整。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/floor_mod/README.md">floor_mod</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算两个输入张量的逐元素模运算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/fresnel_sin/README.md">fresnel_sin</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算Fresnel正弦积分。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/fresnel_cos/README.md">fresnel_cos</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算Fresnel余弦积分。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/fused_mul_add/README.md">fused_mul_add</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>三元逐元素融合算子，将Mul、Add融合为单次计算，公式y = x1 * x2 + x3。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/fused_mul_add_add/README.md">fused_mul_add_add</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>四元逐元素融合算子，将Mul、Add、Add融合为单次计算，公式y = x1 * x2 + x3 + x4。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/fused_mul_add_n/README.md">fused_mul_add_n</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将mul和addn算子进行融合，要求addn的n为2，mul的其中一个输入必须是scalar或者只包含一个数的tensor。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/fused_mul_addn_l2loss/README.md">fused_mul_addn_l2loss</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>图融合内部算子，将Mul+AddN+L2Loss子图合并，逐元素计算y1=x1*x3[0]+x2，同时对x1全量平方归约得到标量y2=Σx1²/2。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/gcd/README.md">gcd</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算两个输入张量的最大公约数。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/ger/README.md">ger</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>求一维向量vec1和vec2的外积得到一个二维矩阵。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/get_shape/README.md">get_shape</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>获取一个或多个输入 tensor 的 shape 信息，按输入顺序将各维度的尺寸值拼接为一维 int32 tensor 输出。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/greater/README.md">greater</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子用于判断输入Tensor中的每个元素是否大于other Scalar的值。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/greater_equal/README.md">greater_equal</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>返回一个BOOL类型的张量，里面每一个值代表输入张量对应位置值是否大于等于另一个张量值或者scalar值。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/approximate_equal/README.md">approximate_equal</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>逐元素近似相等校验算子，用于判断两个张量的对应元素是否在允许的误差范围内近似相等，输出布尔类型结果。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/grouped_bias_add_grad/README.md">grouped_bias_add_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>分组偏置加法（GroupedBiasAdd）的反向计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/hans_decode/README.md">hans_decode</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对压缩后的张量基于PDF进行解码，同时基于mantissa重组恢复张量。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/hans_encode/README.md">hans_encode</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入张量指数位所在字节实现PDF统计，按PDF分布统计进行无损压缩。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/histogram_fixed_width/README.md">histogram_fixed_width</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算张量直方图。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/histogram_v2/README.md">histogram_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算张量直方图。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/inplace_top_k_distance/README.md">inplace_top_k_distance</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>将一组已排序的距离数据与一组新的距离数据合并重排后取前k个数据，结果原地更新到输入中。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/inv/README.md">inv</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入张量的每一个元素计算其倒数。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/inv_grad/README.md">inv_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Inv（取倒数）算子的反向梯度，计算y = -grad * x * x。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/invert/README.md">invert</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>为输入张量的每一个元素取反。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_close/README.md">is_close</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回一个带有布尔元素的新张量，判断给定的self和other是否彼此接近，如果值接近，则返回True，否则返回False。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_finite/README.md">is_finite</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>判断输入张量哪些元素是有限数值，即不是inf、-inf或nan。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_inf/README.md">is_inf</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>判断张量中哪些元素是无限大值，即为inf、-inf。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_nan/README.md">is_nan</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>判断张量中哪些元素是nan。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_neg_inf/README.md">is_neg_inf</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>判断输入张量的元素是否为负无穷。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_pos_inf/README.md">is_pos_inf</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>判断输入张量的元素是否为正无穷。</td>
  </tr>
    <tr>
    <td>math</td>
    <td><a href="../../math/kl_div_v2/README.md">kl_div_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算KL散度。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/left_shift/README.md">left_shift</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>AI CPU</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/lerp/README.md">lerp</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>根据给定的权重，在起始和结束Tensor之间进行线性插值，返回插值后的Tensor。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/less/README.md">less</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>该算子用于判断输入self中的每个元素是否小于输入other的值。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/less_equal/README.md">less_equal</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>判断输入self中的元素值是否小于等于other的值，并将self的每个元素的值与other值的比较结果写入out中。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/lgamma/README.md">lgamma</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/lin_space/README.md">lin_space</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>生成一个等间隔数值序列。创建一个大小为steps的1维向量，其值从start起始到stop结束（包含）线性均匀分布。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/qr/README.md">linalg_qr</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/log/README.md">log</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子对输入张量x的元素，逐元素进行对数计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/log1p/README.md">log1p</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入Tensor完成log1p运算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/log_add_exp/README.md">log_add_exp</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/logdet/README.md">logdet</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/logical_and/README.md">logical_and</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对两个输入张量的对应元素执行「与逻辑」判断，输出布尔型张量（True/False）。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/logical_not/README.md">logical_not</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算给定输入Tensor的逐元素逻辑非。如果未指定输出类型，输出Tensor是bool类型。如果输入Tensor不是bool类型，则将零视为False，非零视为True。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/logical_or/README.md">logical_or</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>完成给定输入张量元素的逻辑或运算。当两个输入张量为非bool类型时，0被视为False，非0被视为True。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/logspace">logspace</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>创建一个大小为steps的一维张量，其值在base^start到base^end上对数尺度上均匀间隔。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/masked_scale/README.md">masked_scale</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/maximum/README.md">maximum</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回两个输入张量中的最大值组成的新张量。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/minimum/README.md">minimum</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回两个输入张量中的最小值组成的新张量。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/mod/README.md">mod</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>张量取余计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/mul/README.md">mul</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>返回两个张量元素之间的乘积结果组成的新张量。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/mul_addn/README.md">mul_addn</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>实现N>=2个mul和addn融合计算，减少搬运时间和内存的占用。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/mul_no_nan/README.md">mul_no_nan</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>二元逐元素安全乘法，当乘数x2为0时结果返回0，避免0*inf、0*nan产生NaN污染。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/muls/README.md">muls</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回输入张量元素与指定scalar值乘积组成的新张量。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/nan_to_num/README.md">nan_to_num</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>将输入中的NaN、正无穷大和负无穷大值分别替换为nan、posinf、neginf指定的值。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/ndtri/README.md">ndtri</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>逆标准正态累积分布函数（probit），对输入概率张量逐元素计算标准正态分位点。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/neg/README.md">neg</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>计算每个元素的相反数。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/non_finite_check/README.md">non_finite_check</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>检测输入tensor_list中是否存在非有限数值（NaN、Inf、-Inf）。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/not_equal/README.md">not_equal</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子逐元素比较两个输入张量是否不相等。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/one_hot/README.md">one_hot</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将离散标签进行二进制编码转换。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/ones_like/README.md">ones_like</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>返回形状和类型相同的张量，所有元素都设置为1。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/pdist/README.md">pdist</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/polar/README.md">polar</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>构建一个复数张量，其元素为对应于极坐标的笛卡尔坐标，其中绝对值为abs，角度为angle。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/population_count/README.md">population_count</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>逐位取出二进制每一位，累加所有为1的比特。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/pow/README.md">pow</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>根据输入指数和幂张量进行指数计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/pows/README.md">pows</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对input中的每个元素应用指数为exponent的幂运算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/precision_compare">precision_compare</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>计算两个Tensor元素是否相同，进行精度比对，返回算子执行的状态码。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/prod_force_se_a/README.md">prod_force_se_a</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>DeepPot-SE模型的反向力计算算子。将神经网络对描述符的导数与描述符对坐标的导数逐元素相乘后归约累加，得到每个原子的三维力向量。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/prod_env_mat_a_calc_rij/README.md">prod_env_mat_a_calc_rij</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>DeepMD环境矩阵的近邻计算，输出邻居索引、平方距离与中心原子指向邻居的位移向量。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/qr/README.md">q_r</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/ragged_bin_count/README.md">ragged_bin_count</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>按splits划分的ragged row，统计每个bin的出现次数或权重和。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/range/README.md">range</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>从start到end，按照step间隔长度生成等差数列组成的张量。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/rank/README.md">rank</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/real/README.md">real</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回复数张量的实部，若输入为实数张量则按原值返回。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/real_div/README.md">real_div</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子按元素逐个返回x1/x2的结果。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reciprocal/README.md">reciprocal</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回一个具有每个输入元素倒数的新张量。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reciprocal_grad/README.md">reciprocal_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算Reciprocal算子的反向梯度。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_all/README.md">reduce_all</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core/AI CPU</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_any/README.md">reduce_any</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_log_sum/README.md">reduce_log_sum</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_log_sum_exp/README.md">reduce_log_sum_exp</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_max/README.md">reduce_max</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_mean/README.md">reduce_mean</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子对一个多维向量按照指定的维度求平均值。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_min/README.md">reduce_min</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_nansum/README.md">reduce_nansum</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_prod/README.md">reduce_prod</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_std_v2/README.md">reduce_std_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算指定维度的标准差。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_std_v2_update/README.md">reduce_std_v2_update</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>接收原始输入和预计算均值，沿指定维度归约计算方差或标准差，支持贝塞尔校正与keepdim。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_std_with_mean/README.md">reduce_std_with_mean</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_sum/README.md">reduce_sum</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>返回给定维度中输入张量每行的和。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_var/README.md">reduce_var</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回指定维度的方差。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/rfft1_d/README.md">rfft1_d</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>对输入张量self进行RFFT（傅里叶变换）计算，输出是一个包含非负频率的复数张量。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/right_shift/README.md">right_shift</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>该算子按元素计算输入张量x与y的按位右移操作。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/round/README.md">round</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>该算子将输入张量的值舍入到最接近的整数，若该值与两个整数距离一样则向偶数取整。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/rsqrt/README.md">rsqrt</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>求input(Tensor)每个元素的平方根的倒数。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/rsqrt_grad/README.md">rsqrt_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>rsqrt的反向计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/scale/README.md">scale</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入x按scale缩放并可选叠加bias：y = x * scale + bias，支持axis/num_axes/scale_from_blob控制的通道广播。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/search_sorted/README.md">search_sorted</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子用于在一个已排序的张量sorted_sequence中查找给定张量values应该插入的位置。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/segsum/README.md">segsum</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>进行分段和计算。生成对角线为1、上三角为0的半可分矩阵。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/select/README.md">select</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>根据条件选取self或other中元素并返回（支持广播）。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/select_v2">select_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>根据条件选取self或other中元素并返回（支持广播）。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/get_dynamic_dims/README.md">get_dynamic_dims</a></td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>根据shape_info中标记的未知维度，从动态输入张量中提取实际维度值并输出。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/shape/README.md">shape</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/shape_n/README.md">shape_n</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sign/README.md">sign</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入的tensor逐元素进行Sign符号函数的运算并输出结果tensor。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sign_bits_pack/README.md">sign_bits_pack</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将浮点数（float16/float32）的符号位提取并打包为uint8字节，每8个符号位打包为1个字节。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sign_bits_unpack/README.md">sign_bits_unpack</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将UINT8类型的1-bit Adam符号位拆包为FLOAT32或FLOAT16类型的张量。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/signbit/README.md">signbit</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>判断输入self中的每个元素符号位是否为1，返回一个Tensor。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/silent_check/README.md">silent_check</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/silent_check_v2/README.md">silent_check_v2</a></td>
    <td> </td>
    <td> </td>
    <td>√</td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sin/README.md">sin</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入Tensor完成sin运算</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sparse_bincount/README.md">sparse_bincount</a></td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>统计稀疏张量中每个值的出现次数或加权累加和。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sinc/README.md">sinc</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入Tensor完成sinc运算</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sinh/README.md">sinh</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>该算子对输入的每个元素进行双曲正弦操作后输出。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sinkhorn/README.md">sinkhorn</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>计算Sinkhorn距离，可以用于MoE模型中的专家路由。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/slogdet/README.md">slogdet</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>计算输入self的行列式的符号和自然对数。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sort/README.md">sort</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将输入tensor中的元素根据指定维度进行升序/降序，并且返回对应的index值。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sort_with_index/README.md">sort_with_index</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将输入tensor按照元素值大小进行排序，index值跟随对应元素值进行排序。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/spence/README.md">spence</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算Spence函数（dilogarithm）：S(x) = -integral_0^x ln(1-t)/t dt。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sqrt/README.md">sqrt</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>完成非负数平方根计算，负数情况返回nan。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sqrt_grad/README.md">sqrt_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子用于完成梯度平方根计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/square/README.md">square</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子对输入Tensor逐元素计算平方值。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/square_sum_all/README.md">square_sum_all</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>分别计算两个同Shape输入Tensor的全元素平方和，输出两个独立标量。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/square_sum_v1/README.md">square_sum_v1</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>用于计算输入张量在指定轴上的平方和。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/squared_difference/README.md">squared_difference</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>为第一个输入张量减去第二个输入张量，并计算其平方值。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/stft/README.md">stft</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算输入在滑动窗口内的傅里叶变换。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sub/README.md">sub</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>完成减法计算，被减数按alpha进行缩放。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sub_mul_concat/README.md">sub_mul_concat</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>内部融合算子，由CANN既有图融合流程将Sub、Mul和Concat/ConcatV2/ConcatV2D原始图转换后生成，不提供公开ACLNN或直接构造接口。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/svd/README.md">svd</a></td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>计算一个或多个矩阵的奇异值分解。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/tan/README.md">tan</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>正切三角函数。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/tanh/README.md">tanh</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>激活函数。返回与输入tensor shape相同的tensor，对输入tensor进行elementwise的计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/tanh_grad/README.md">tanh_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>Tanh的反向计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/tensor_equal/README.md">tensor_equal</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算两个Tensor是否有相同的大小和元素，返回一个Bool类型。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/tile/README.md">tile</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>对输入tensor沿着repeats中对每个维度指定的复制次数进行复制。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/top_k_pq_distance_v2/README.md">top_k_pq_distance_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>借助分组极值剪枝，在最后一维上查找k个最大或最小的元素及其索引。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/top_k_v2/README.md">top_k_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回输入Tensor在指定维度上的k个极值及索引，支持通过输入Tensor指定k值。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/topk/README.md">topk</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>沿指定维度找出输入张量中最大或最小的k个元素及其索引。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/top_k_v2_d/README.md">top_k_v2_d</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>沿指定维度找出输入张量中最大或最小的k个元素及其索引，兼容TensorFlow TopKV2。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/top_k_pq_distance/README.md">top_k_pq_distance</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>基于分组极值剪枝的TopK距离计算，用于向量检索场景。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/trace/README.md">trace</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>计算矩阵从左上角开始的主对角线元素的和。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/transform_bias_rescale_qkv/README.md">transform_bias_rescale_qkv</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>一个用于处理多头注意力机制中查询（Query）、键（Key）、值（Value）向量的接口，用于调整这些向量的偏置（Bias）和缩放（Rescale）因子。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/triangular_solve/README.md">triangular_solve</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>求解一个具有方形上或下三角形可逆矩阵A和多个右侧b的方程组。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/trunc/README.md">trunc</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入张量逐元素截断取整，向零方向舍入。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/truncate_mod/README.md">truncate_mod</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>张量取余计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/xdivy/README.md">xdivy</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算x / y，x == 0时结果为0，支持broadcast。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/xlogy/README.md">xlogy</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算x1 * log(x2)，x1 == 0时结果为0，支持broadcast。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/xlogy/README.md">x_log_y</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>计算x1 * log(x2)，逐元素运算，支持广播</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/xlog1py/README.md">xlog1py</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算x * log(1 + y)，当x == 0时结果为0。支持broadcast。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/zeta/README.md">zeta</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>计算Hurwitz zeta函数 ζ(x, q)。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/zero_op/README.md">zero_op</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将张量填充为全零。</td>
   </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/ball_query/README.md">ball_query</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Ball Query用于查找点p1指定球半径范围内的所有点（在实现中设置了K的上限）。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/calc_buckets_limit_and_offset/README.md">calc_buckets_limit_and_offset</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>根据ivf_counts和bucket_list计算每个桶的limits和offsets，用于向量检索场景。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/chunk_cat/README.md">chunk_cat</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将tensors中所有tensor先按照维度dim切分为numChunks块，再按照dim后一维进行级联，最后转换为out的数据类型。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/circular_pad/README.md">circular_pad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>使用输入循环填充输入tensor的最后两维。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/circular_pad_grad/README.md">circular_pad_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>circular_pad的反向传播。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/coalesce_sparse/README.md">coalesce_sparse</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>实现对Coo_Tensor优化的方法coalesce()方法。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/coordinates_1d_to_2d/README.md">coordinates_1d_to_2d</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>将1D坐标转换为2D坐标，根据shape信息计算行列索引。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/depth_to_space/README.md">depth_to_space</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子通过对输入张量的深度（通道）维度进行重新排列，将其转换为空间维度（高度和宽度）。具体来说，它将输入张量的深度维度按照指定的块大小（block_size）进行划分，并将这些深度块重新排列到空间维度中，从而增加空间维度的大小，同时减少通道维度的深度。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/diag/README.md">diag</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>创建一个以小于等于4维的输入张量为对角线元素的对角矩阵。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/diag_flat/README.md">diag_flat</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>创建一个以输入数组为对角线元素的平铺对角矩阵。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/diag_v2/README.md">diag_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>根据输入的二维张量，提取由diagonal指定的对角线元素。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/feeds_repeat/README.md">feeds_repeat</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对于输入feeds，根据输入feeds_repeat_times，将对应的feeds的第0维上的数据复制对应的次数，并将输出y的第0维padding到output_feeds_size的大小。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/fill_diagonal_v2/README.md">fill_diagonal_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将指定值填充到矩阵的主对角线上。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/masked_select_v3/README.md">masked_select_v3</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>根据mask是否为True，选出input中对应位置的值，input和mask满足广播规则，结果为一维Tensor。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/mem_set_v2/README.md">mem_set_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>给输入tensor赋指定的值。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v3_grad_replicate/README.md">pad_v3_grad_replicate</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>padv3 2D的反向传播。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v3_grad_replication/README.md">pad_v3_grad_replication</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>padv3 3D的反向传播。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v4_grad/README.md">pad_v4_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>pad之后的输入的反向传播。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/reflection_pad3d_grad/README.md">reflection_pad3d_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>计算aclnnReflectionPad3d api的反向传播。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/stack_ball_query/README.md">stack_ball_query</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Stack Ball Query是KNN的替代方案，用于查找点p1指定半径范围内的所有点（在实现中设置了K的上限）。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/slice_last_dim/README.md">slice_last_dim</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入张量的最后一维进行切片，通过start、end指定切片范围，stride指定步长。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/slice_with_axes/README.md">slice_with_axes</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>沿指定axes从输入张量中提取切片，每个axis由offsets指定起始位置、size指定切片长度。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/slice_write/README.md">slice_write</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>将value tensor写入x tensor的指定位置（由begin指定偏移）。这是一个原地操作，输出与输入x共用同一块内存。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/strided_slice_assign_v2/README.md">strided_slice_assign_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>StridedSliceAssign是一种张量切片赋值操作，它可以将张量inputValue的内容，赋值给目标张量varRef中的指定位置。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/transpose_v2/README.md">transpose_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>实现张量的维度置换（Permutation）操作，按照指定的顺序重新排列输入张量的维度。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/unfold_grad/README.md">unfold_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>实现Unfold算子的反向功能，计算相应的梯度。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/concat_d/README.md">concat_d</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将tensors中所有tensor按照维度dim进行级联，除了dim对应的维度以外的维度必须一致。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/concat_dv2/README.md">concat_dv2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>将tensors中所有tensor按照维度concat_dim进行级联，除了dim对应的维度以外的维度必须一致。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/concat_v2/README.md">concat_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>该算子沿指定维度将多个输入张量进行拼接。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/clip_by_norm_no_div_sum/README.md">clip_by_norm_no_div_sum</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对4个支持广播的输入张量执行ClipByNormNoDivSum运算。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/clip_by_value/README.md">clip_by_value</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子将输入的所有元素限制在[clipValueMin,clipValueMax]范围内。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/clip_by_value_v2/README.md">clip_by_value_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>该算子将输入的所有元素限制在[clipValueMin,clipValueMax]范围内。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/invert_permutation/README.md">invert_permutation</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/conjugate_transpose/README.md">conjugate_transpose</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>按perm指定的维度顺序对输入x做转置并取共轭，实数类型等价于普通Transpose。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/unpack/README.md">unpack</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>将张量沿着某一维度拆分为多个子张量。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/as_strided/README.md">as_strided</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>允许用户通过指定新的形状和步长来创建一个与原张量共享相同数据内存的张量视图。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/assign/README.md">assign</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将一个输入tensor的值填充到另一个输入tensor中。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/broadcast_to/README.md">broadcast_to</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将输入tensor广播到指定shape。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/batch_to_space/README.md">batch_to_space</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将批次维度的数据重新排列到空间维度，并裁剪空间维度。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/batch_to_space_nd/README.md">batch_to_space_nd</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将批次维度的数据重新排列到空间维度，并裁剪空间维度。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/case_condition/README.md">case_condition</a></td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>根据三个输入值与阈值的关系，输出离散case编号。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/concat/README.md">concat</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>用于沿指定维度将多个输入Tensor进行拼接，输出包含所有输入数据按顺序拼接后的Tensor。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/confusion_transpose_d/README.md">confusion_transpose_d</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>融合reshape和transpose运算。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/contiguous/README.md">contiguous</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/dynamic_partition/README.md">dynamic_partition</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>用于根据分区索引将输入数据动态分割成多个张量。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/fill/README.md">fill</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入张量填充指定标量值。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/flatten/README.md">flatten</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/im2col/README.md">im2col</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>图像到列，滑动局部窗口数据转为列向量，拼接为大张量。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/masked_fill/README.md">masked_fill</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将输入Tensor`x`中mask位置为`True`的元素填充指定的值。`mask`必须与`x`的shape相同或可广播。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/mirror_pad/README.md">mirror_pad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>以镜像方式填充输入tensor的边界。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/npu_format_cast/README.md">npu_format_cast</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pack/README.md">pack</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>通过沿轴维度打包，将值中的张量列表打包成一个比值中的每个张量高一维度的张量。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad/README.md">pad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入Tensor做填充。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v2/README.md">pad_v2</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入tensor进行指定值的常量填充。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v3/README.md">pad_v3</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入Tensor做填充。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v3_grad/README.md">pad_v3_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>pad_v3之后的输入的反向传播。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/reshape/README.md">reshape</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>将输入张量重解释为目标形状。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/roll/README.md">roll</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>沿指定维度滚动张量元素。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/slice/README.md">slice</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>根据给定的范围和步长，从输入张量中提取子张量。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/space_to_batch/README.md">space_to_batch</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将空间维度的数据按块重新排列到批次维度，并对空间维度补零。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/space_to_batch_nd/README.md">space_to_batch_nd</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将多个空间维度的数据按块重新排列到批次维度，并对空间维度补零。支持任意数量的空间维度。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/sparse_reshape/README.md">sparse_reshape</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将稀疏张量的indices从输入shape重塑到目标shape。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/split/README.md">split</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将张量沿指定维度split_dim平均拆分为num_split份更小的张量。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/split_v/README.md">split_v</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>将输入tensor指定维度按照指定长度进行切分，最后一块长度可能不足。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/squeeze/README.md">squeeze</a></td>
    <td>×</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>移除输入张量中大小为1的维度。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/strided_slice/README.md">strided_slice</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>按照指定的起始、结束位置和步长，从输入张量中提取一个子张量。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/strided_slice_grad/README.md">strided_slice_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将子张量的梯度映射回原始张量的对应位置。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/strided_slice_v3/README.md">strided_slice_v3</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>按照指定的起始、结束位置和步长，从输入张量中提取子张量。</td>
   </tr>
   <tr>
    <td>conversion</td>
    <td><a href="../../conversion/tensor_move/README.md">tensor_move</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将输入tensor的值搬运到输出tensor中。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/tensor_redirect/README.md">tensor_redirect</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将输入张量的数据逐bit拷贝到输出张量，恒等映射。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/tile_with_axis/README.md">tile_with_axis</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>沿指定维度复制输入Tensor数据，扩展输出Tensor。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/trans_data/README.md">trans_data</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core/AI CPU</td>
    <td>实现张量的数据格式转换。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/transpose/README.md">transpose</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>完成张量转置。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/tril/README.md">tril</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将输入的self张量的最后二维（按shape从左向右数）沿对角线的右上部分置零。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/triu/README.md">triu</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将输入的self张量的最后二维（按shape从左向右数）沿对角线的左下部分置零。参数diagonal可正可负，默认为零，正数表示主对角线向右上方向移动，负数表示主对角线向左下方向移动。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/unsqueeze/README.md">unsqueeze</a></td>
    <td>×</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>在指定位置插入一个大小为1的维度。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/view_copy/README.md">view_copy</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/identity/README.md">identity</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/identity_n/README.md">identity_n</a></td>
    <td>×</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/matrix_set_diag/README.md">matrix_set_diag</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将输入tensor的对角线元素替换为对角线tensor的值。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/matrix_set_diag_v2/README.md">matrix_set_diag_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将输入tensor的对角线元素替换为对角线tensor的值。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/matrix_diag/README.md">matrix_diag</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回一个给定批量对角值的批量对角Tensor。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/matrix_diag_part/README.md">matrix_diag_part</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回批处理张量最内层矩阵的主对角线元素。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/matrix_diag_v3/README.md">matrix_diag_v3</a></td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>根据输入的单条或多条对角线值生成矩阵，对角线带之外的位置使用padding_value填充。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/matrix_diag_part_v3/README.md">matrix_diag_part_v3</a></td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>从输入矩阵或批量矩阵的最后两维中提取一条或多条对角线。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/split_d/README.md">split_d</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>将张量沿指定维度split_dim平均拆分为num_split份更小的张量。与Split算子不同，split_dim作为属性而非输入提供。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/strided_slice_assign/README.md">strided_slice_assign</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将输入张量input_value的内容赋值给目标张量var中由begin、end、strides指定的切片位置，切片以外的区域保持var原值不变（in-place语义）。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/drop_out_do_mask/README.md">drop_out_do_mask</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>根据mask中对应bit位的值，将输入中的元素置零，并按照scale放大或者置零。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/drop_out_do_mask_v3/README.md">drop_out_do_mask_v3</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/drop_out_do_mask_v3_d/README.md">drop_out_do_mask_v3_d</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/drop_out_v3/README.md">drop_out_v3</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>按照概率p随机将输入中的元素置零，并将输出按照1/(1-p)的比例缩放。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/drop_out_v3_grad/README.md">drop_out_v3_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>DropOut反向算子，根据mask和scale计算输入梯度。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/dsa_gen_bit_mask/README.md">dsa_gen_bit_mask</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/dsa_random_normal/README.md">dsa_random_normal</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/dsa_random_uniform/README.md">dsa_random_uniform</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/random_standard_normal_v2/README.md">random_standard_normal_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>生成标准正态分布的随机数列。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/random_uniform_int_v2/README.md">random_uniform_int_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>生成均匀分布的随机整数数列。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/random_uniform_v2/README.md">random_uniform_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>生成均匀分布的随机浮点数数列。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/sim_thread_exponential/README.md">sim_thread_exponential</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_bernoulli/README.md">stateless_bernoulli</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>从伯努利分布中提取二进制随机数。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_drop_out_gen_mask/README.md">stateless_drop_out_gen_mask</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>根据随机种子和随机偏移量和保活系数生成随机掩码mask。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_exponential/README.md">stateless_exponential</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>生成指数分布的随机数。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_random/README.md">stateless_random</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>返回从[from, to - 1]范围中抽取离散均匀分布的随机数。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_random_choice_with_mask/README.md">stateless_random_choice_with_mask</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>根据输入x中值为true，选择出对应索引列表，其次根据seed, offset调用philox_random生成索引列表长度个随机数，利用随机数对索引列表做洗牌算法，得到随机交换后的索引列表，返回count个。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_random_normal_v2/README.md">stateless_random_normal_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>生成正态分布随机数。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_random_normal_v3/README.md">stateless_random_normal_v3</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>生成正态分布随机数。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_normal/README.md">stateless_normal</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>生成正态分布随机数(与H20 GPU比特对齐)。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_random_uniform_v2/README.md">stateless_random_uniform_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>生成均匀分布随机数。</td>
  </tr>
  <tr>
      <td>math</td>
      <td><a href="../../math/amp_update_scale/README.md">amp_update_scale</a></td>
      <td>√</td>
      <td>√</td>
      <td>√</td>
      <td>×</td>
      <td>AI Core</td>
      <td>实现AMP训练中的动态Scale更新，根据当前scale值、growth_tracker计数器以及是否发现Inf/NaN，动态调整loss scale大小。</td>
  </tr>
  <tr>
      <td>math</td>
      <td><a href="../../math/prod_virial_se_a/README.md">prod_virial_se_a</a></td>
      <td>√</td>
      <td>√</td>
      <td>×</td>
      <td>√</td>
      <td>AI Core</td>
      <td>计算深度势能模型中基于SE原子嵌入描述符的A类型位力张量。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_uniform/README.md">stateless_uniform</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>生成均匀分布随机数。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_random_uniform_v3/README.md">stateless_random_uniform_v3</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>生成均匀分布随机数。</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_randperm/README.md">stateless_randperm</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/stateless_truncated_normal_v2/README.md">stateless_truncated_normal_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td> </td>
    <td>√</td>
    <td>AI Core</td>
    <td>生成截断正态分布的随机数</td>
  </tr>
  <tr>
    <td>random</td>
    <td><a href="../../random/truncated_normal_v2/README.md">truncated_normal_v2</a></td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
</tbody>
</table>
