# 算子接口（aclnn）

## 使用说明

为方便调用算子，提供一套基于C的API（以aclnn为前缀API），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式被称为“单算子API调用”，简称aclnn调用。

- **头文件/库文件**

  调用算子API时，需引用依赖的头文件和库文件，一般头文件默认在`${INSTALL_DIR}/include/aclnnop`，库文件默认在`${INSTALL_DIR}/lib64`，具体文件如下：

  - 头文件：①方式1 （推荐）：引用算子仓总头文件aclnn\_ops\_\$\{ops\_project\}.h。②方式2：引用单个算子API的头文件aclnn\_\*.h。
  - 库文件：引用算子仓对应的库文件libopapi_${ops_project}.so。注意，原所有算子仓总库文件libopapi.so后续会废弃，不推荐使用，也不支持与单个算子仓库文件同时使用。

  ${INSTALL_DIR}表示CANN安装后文件路径；\$\{ops\_project\}表示算子仓名（如math、nn、cv、transformer），请改为实际算子仓名。

- **V版本演进说明**

  请注意，部分API存在多个V版本，使用时选择最高V版本即可（高版本API已兼容低版本API的所有能力）。

## 接口列表

> **确定性简介**：
>
> - 配置说明：因CANN或NPU型号不同等原因，可能无法保证同一个算子多次运行结果一致。在相同条件下（平台、设备、版本号和其他随机性参数等），部分算子接口可通过`aclrtCtxSetSysParamOpt`（参见[《Runtime运行时 API》](https://hiascend.com/document/redirect/CannCommunityRuntimeApi)）开启确定性算法，使多次运行结果一致。
> - 性能说明：同一个算子采用确定性计算通常比非确定性慢，因此模型单次运行性能可能会下降。但在实验、调试调测等需要保证多次运行结果相同来定位问题的场景，确定性计算可以提升效率。
> - 线程说明：同一线程中只能设置一次确定性状态，多次设置以最后一次有效设置为准。有效设置是指设置确定性状态后，真正执行了一次算子任务下发。如果仅设置，没有算子下发，只能是确定性变量开启但未下发给算子，因此不执行算子。
>   解决方案：暂不推荐一个线程多次设置确定性。该问题在二进制开启和关闭情况下均存在，在后续版本中会解决该问题。
> - 符号说明：表中“-”符号表示该接口暂不支持当前列产品。

算子接口列表如下：

|    接口名   |   说明     | 确定性说明（A2/A3） | 确定性说明（Ascend 950） |
| ----------- | ------------------- | --------- | --------- |
| [aclnnAbs](../../math/abs/docs/aclnnAbs.md)     | 实现张量的绝对值运算。    | 默认确定性实现| 默认确定性实现|
| [aclnnAcos&aclnnInplaceAcos](../../math/acos/docs/aclnnAcos&aclnnInplaceAcos.md) | 对输入矩阵的每个元素进行反余弦操作后输出。                   | 默认确定性实现| - |
| [aclnnAcosh&aclnnInplaceAcosh](../../math/acosh/docs/aclnnAcosh&aclnnInplaceAcosh.md) | 对输入Tensor中的每个元素进行反双曲余弦操作后输出。           | 默认确定性实现| - |
| [aclnnAdd&aclnnInplaceAdd](../../math/add/docs/aclnnAdd&aclnnInplaceAdd.md) | 完成加法计算。        | 默认确定性实现| 默认确定性实现 |
| [aclnnAddcdiv&aclnnInplaceAddcdiv](../../math/addcdiv/docs/aclnnAddcdiv&aclnnInplaceAddcdiv.md) | 完成加法计算。                                               | 默认确定性实现| 默认确定性实现 |
| [aclnnAddcmul&aclnnInplaceAddcmul](../../math/addcmul/docs/aclnnAddcmul&aclnnInplaceAddcmul.md) | 执行tensor1与tensor2的逐元素乘法，将结果乘以标量值value并与输入self/selfRef做逐元素加法。 | 默认确定性实现| 默认确定性实现 |
| [aclnnAddLora](../../math/add_lora/docs/aclnnAddLora.md)        | 为神经网络添加LoRA（Low-Rank Adaptation）层功能，通过低秩分解减少参数数量。 | 默认确定性实现| - |
| [aclnnAddN](../../math/add_n/docs/aclnnAddN.md)    |  对输入进行主元素相加求和操作。  | 默认确定性实现 | - |
| [aclnnAddr&aclnnInplaceAddr](../../math/addr/docs/aclnnAddr&aclnnInplaceAddr.md) | 求一维向量vec1和vec2的外积得到一个二维矩阵，并将外积结果矩阵乘一个系数后和自身乘系数相加后输出。 | 默认确定性实现| 默认确定性实现|
| [aclnnAdds](../../math/add/docs/aclnnAdds.md) | 完成加法计算。 | 默认确定性实现|  默认确定性实现 |
| [aclnnAddV3&aclnnInplaceAddV3](../../math/add/docs/aclnnAddV3&aclnnInplaceAddV3.md) | 完成加法计算。                                               | - |  默认确定性实现 |
| [aclnnAffineGrid](../../math/affine_grid/docs/aclnnAffineGrid.md) | 给定一组3维的仿射参数矩阵(theta)以及输出图像的大小(size)，生成一个2D或3D的网格，该网格表示仿射后图像的点在原图像上的坐标。            | 默认确定性实现| - |
| [aclnnAll](../../math/reduce_all/docs/aclnnAll.md)              | 对于给定维度dim中的每一维，如果输入Tensor中该维度对应的所有元素计算为True，则返回True，否则返回False。 | 默认确定性实现| 默认确定性实现|
| [aclnnAmax](../../math/reduce_max/docs/aclnnAmax.md)            | 返回张量在指定维度(dim)上每个切片的最大值。                  | 默认确定性实现| 默认确定性实现|
| [aclnnAmin](../../math/reduce_min/docs/aclnnAmin.md)            | 返回张量在指定维度(dim)上每个切片的最小值。                  | 默认确定性实现| 默认确定性实现|
| [aclnnAminmax](../../math/reduce_min/docs/aclnnAminmax.md)      | 返回输入张量在指定维度上每行的最小值和最大值。               | 默认确定性实现| 默认确定性实现|
| [aclnnAminmaxAll](../../math/reduce_min/docs/aclnnAminmaxAll.md) | 返回输入张量在所有维度上的最小值和最大值。                   | 默认确定性实现| 默认确定性实现|
| [aclnnAminmaxDim](../../math/reduce_min/docs/aclnnAminmaxDim.md) | 返回输入张量在指定维度上的最小值和最大值。                   | 默认确定性实现| 默认确定性实现|
| [aclnnAngleV2](../../math/angle_v2/docs/aclnnAngleV2.md) | 为输入张量的每一个元素取角度（单位：弧度）。 | 默认确定性实现 | 默认确定性实现 |
| [aclnnAny](../../math/reduce_any/docs/aclnnAny.md)              | 对于给定维度dim中的每一维，如果输入Tensor中该维度对应的任意元素计算为True，则返回True，否则返回False。 | 默认确定性实现| 默认确定性实现|
| [aclnnArange](../../math/range/docs/aclnnArange.md)             | 从start起始到end结束按照step的间隔获取值，并保存到输出的1维张量，其中数据范围为[start,end)。 | 默认确定性实现|  默认确定性实现  |
| [aclnnArgMax](../../math/arg_max_v2/docs/aclnnArgMax.md)        | 返回张量在指定维度（dim）上的最大值的索引，并保存到out张量中。 | 默认确定性实现| 默认确定性实现|
| [aclnnArgMin](../../math/arg_min/docs/aclnnArgMin.md)           | 返回tensor中指定轴的最小值索引，并保存到out中。              | 默认确定性实现| 默认确定性实现|
| [aclnnArgsort](../../math/sort/docs/aclnnArgsort.md)           | 将输入tensor中的元素根据某个维度进行升序/降序排序，返回对应的index值。              | 默认确定性实现| 默认确定性实现  |
| [aclnnAsin&aclnnInplaceAsin](../../math/asin/docs/aclnnAsin&aclnnInplaceAsin.md) | 对输入矩阵的每个元素进行反正弦操作后输出。                   | 默认确定性实现| 默认确定性实现 |
| [aclnnAsinh&aclnnInplaceAsinh](../../math/asinh/docs/aclnnAsinh&aclnnInplaceAsinh.md) | 对输入Tensor中的每个元素进行反双曲正弦操作后输出。           | 默认确定性实现| - |
| [aclnnAtan&aclnnInplaceAtan](../../math/atan/docs/aclnnAtan&aclnnInplaceAtan.md) | 对输入矩阵的每个元素进行反正切操作后输出。                   | 默认确定性实现| 默认确定性实现|
| [aclnnAtan2&aclnnInplaceAtan2](../../math/atan2/docs/aclnnAtan2&aclnnInplaceAtan2.md) | 对输入张量self和other进行逐元素的反正切运算。                | 默认确定性实现| 默认确定性实现  |
| [aclnnAtanh&aclnnInplaceAtanh](../../math/atanh/docs/aclnnAtanh&aclnnInplaceAtanh.md) | 对输入Tensor中的每个元素进行反双曲正切操作后输出。           | 默认确定性实现| - |
| [aclnnBatchNormStats](../../math/reduce_std_with_mean/docs/aclnnBatchNormStats.md) | 计算单卡输入数据的均值和标准差的倒数。                       | 默认确定性实现| 默认确定性实现|
| [aclnnBernoulli&aclnnInplaceBernoulli](../../random/stateless_bernoulli/docs/aclnnBernoulli&aclnnInplaceBernoulli.md) | 从伯努利分布中提取二进制随机数（0或1），prob为生成二进制随机数的概率，输入的张量用于指定shape。  | 默认确定性实现|  默认确定性实现 |
| [aclnnBernoulliTensor&aclnnInplaceBernoulliTensor](../../random/stateless_bernoulli/docs/aclnnBernoulliTensor&aclnnInplaceBernoulliTensor.md) | 从伯努利分布中提取二进制随机数（0或1），prob为生成二进制随机数的概率，输入的张量用于指定shape。   | 默认确定性实现| 默认确定性实现 |
| [aclnnBincount](../../math/bincount/docs/aclnnBincount.md) | 计算非负整数数组中每个数的频率。                             | 默认确定性实现| 默认非确定性实现，支持配置开启 |
| [aclnnBitwiseAndScalar](../../math/bitwise_and/docs/aclnnBitwiseAndScalar.md) | 计算输入tensor中每个元素和输入标量的按位与结果。             | 默认确定性实现| 默认确定性实现 |
| [aclnnBitwiseAndTensor](../../math/bitwise_and/docs/aclnnBitwiseAndTensor.md) | 输入为BOOL型tensor时，进行逻辑与运算；输入为INT型时，进行位与运算。 | 默认确定性实现| 默认确定性实现 |
| [aclnnBitwiseAndTensorOut&aclnnInplaceBitwiseAndTensorOut](../../math/bitwise_and/docs/aclnnBitwiseAndTensorOut&aclnnInplaceBitwiseAndTensorOut.md) | 输入为BOOL型tensor时，进行逻辑与运算；输入为INT型时，进行位与运算。 | - | 默认确定性实现 |
| [aclnnBitwiseNot](../../math/bitwise_not/docs/aclnnBitwiseNot.md) | 输入为BOOL型tensor时，进行逻辑非运算；输入为INT型时进行按位非运算。 | 默认确定性实现| 默认确定性实现|
| [aclnnBitwiseOrScalar&aclnnInplaceBitwiseOrScalar](../../math/bitwise_or/docs/aclnnBitwiseOrScalar&aclnnInplaceBitwiseOrScalar.md) | 计算输入张量self中每个元素和输入标量other的按位或。          | 默认确定性实现| 默认确定性实现 |
| [aclnnBitwiseOrTensor&aclnnInplaceBitwiseOrTensor](../../math/bitwise_or/docs/aclnnBitwiseOrTensor&aclnnInplaceBitwiseOrTensor.md) | 计算张量self中每个元素与other张量中对应位置的元素的按位或。  | 默认确定性实现| 默认确定性实现 |
| [aclnnBitwiseXorScalar&aclnnInplaceBitwiseXorScalar](../../math/bitwise_xor/docs/aclnnBitwiseXorScalar&aclnnInplaceBitwiseXorScalar.md) | 计算输入张量self中每个元素和输入标量other的按位异或，输入self和other必须是整数或布尔类型，对于布尔类型，计算逻辑异或。 | 默认确定性实现| 默认确定性实现 |
| [aclnnBitwiseXorTensor&aclnnInplaceBitwiseXorTensor](../../math/bitwise_xor/docs/aclnnBitwiseXorTensor&aclnnInplaceBitwiseXorTensor.md) | 计算输入张量self中每个元素与输入张量other中对应位置元素的按位异或，输入self和other必须是整数或布尔类型，对于布尔类型，计算逻辑异或。 | 默认确定性实现| 默认确定性实现 |
| [aclnnCalculateConvolutionWeightSize](../../conversion/trans_data/docs/aclnnCalculateConvolutionWeightSize.md)   | 在Convolution算子NCHW格式输入下，计算需要申请的weight的大小，仅支持Float16数据类型，该接口仅仅用于判断对weight Tensor进行预处理需要使用多少size才可使Convolution算子执行性能最优。 | 默认确定性实现| - |
| [aclnnCalculateMatmulWeightSize](../../conversion/trans_data/docs/aclnnCalculateMatmulWeightSize.md)   | 在Matmul算子ND格式输入下，计算需要申请的weight的大小，该接口仅仅用于判断对weight Tensor进行预处理需要使用多少size才可使Matmul算子执行性能最优。 | 默认确定性实现|  默认确定性实现 |
| [aclnnCalculateMatmulWeightSizeV2](../../conversion/trans_data/docs/aclnnCalculateMatmulWeightSizeV2.md)      | 在Matmul算子ND格式输入下，计算如果要转换到NZ格式下需要占用的空间大小（单位为元素个数），该接口仅仅用于判断对weight Tensor预处理需要使用多少size才可使Matmul算子执行性能最优。 | 默认确定性实现| 默认确定性实现  |
| [aclnnCast](../../math/cast/docs/aclnnCast.md)                  | 实现张量数据类型转换。                                       | 默认确定性实现| 默认确定性实现|
| [aclnnCat](../../conversion/concat_d/docs/aclnnCat.md)            | 将tensors中所有tensor按照维度dim进行级联。    | 默认确定性实现| 默认确定性实现  |
| [aclnnCdist](../../math/cdist/docs/aclnnCdist.md)            | 计算两个向量集合中每个点之间的p范数距离。    | 默认确定性实现| - |
| [aclnnCdistBackward](../../math/cdist_grad/docs/aclnnCdistBackward.md)            | 完成aclnnCdist的反向    | 默认确定性实现| 默认确定性实现 |
| [aclnnCeil&aclnnInplaceCeil](../../math/ceil/docs/aclnnCeil&aclnnInplaceCeil.md) | 返回输入tensor中每个元素向上取整的结果。                     | 默认确定性实现| 默认确定性实现|
| [aclnnChannelShuffle](../../conversion/transpose/docs/aclnnChannelShuffle.md) | 将(\*, C, H, W)张量的channels分成g个组，然后将每个通道组中的通道进行随机重排，最后将所有通道合并输出，同时保持最终输出张量的shape和输入张量保持一致。 | 默认确定性实现| 默认确定性实现|
| [aclnnChunkCat](../../conversion/chunk_cat/docs/aclnnChunkCat.md) | 将tensors中所有tensor先按照维度dim切分为numChunks块，再按照dim后一维进行级联，最后转换为out的数据类型。 | 默认确定性实现| 默认确定性实现|
| [aclnnCircularPad2d](../../conversion/circular_pad/docs/aclnnCircularPad2d.md) | 使用输入循环填充输入tensor的最后两维。                       | 默认确定性实现| 默认确定性实现|
| [aclnnCircularPad2dBackward](../../conversion/circular_pad_grad/docs/aclnnCircularPad2dBackward.md) | circular_pad2d的反向传播，前向计算参考aclnnCircularPad2d。   | 默认确定性实现| 默认确定性实现|
| [aclnnCircularPad3d](../../conversion/circular_pad/docs/aclnnCircularPad3d.md) | 使用输入循环填充输入tensor的最后三维。                       | 默认确定性实现| 默认确定性实现|
| [aclnnCircularPad3dBackward](../../conversion/circular_pad_grad/docs/aclnnCircularPad3dBackward.md) | aclnnCircularPad3d的反向传播。                               | 默认确定性实现| 默认确定性实现|
| [aclnnClamp](../../conversion/clip_by_value_v2/docs/aclnnClamp.md) | 将输入的所有元素限制在[min,max]范围内，如果min为None，则没有下限，如果max为None，则没有上限。 | 默认确定性实现| 默认确定性实现 |
| [aclnnClampMax&aclnnInplaceClampMax](../../conversion/clip_by_value_v2/docs/aclnnClampMax&aclnnInplaceClampMax.md) | 将输入的所有元素限制在[-inf,max]范围内。 | 默认确定性实现| 默认确定性实现 |
| [aclnnClampMaxTensor&aclnnInplaceClampMaxTensor](../../conversion/clip_by_value_v2/docs/aclnnClampMaxTensor&aclnnInplaceClampMaxTensor.md) | 将输入的所有元素限制在[-inf, max]范围内。 | 默认确定性实现| 默认确定性实现 |
| [aclnnClampMin](../../conversion/clip_by_value_v2/docs/aclnnClampMin.md) | 将输入的所有元素限制在[min, inf]范围内。 | 默认确定性实现| 默认确定性实现 |
| [aclnnClampMinTensor&aclnnInplaceClampMinTensor](../../conversion/clip_by_value_v2/docs/aclnnClampMinTensor&aclnnInplaceClampMinTensor.md) | 将输入的所有元素限制在[min, inf]范围内。 | 默认确定性实现| 默认确定性实现 |
| [aclnnClampTensor](../../conversion/clip_by_value_v2/docs/aclnnClampTensor.md) | 将输入的所有元素限制在[min, max]范围内，如果min缺省，则无下限，如果max缺省，则无上限。 | 默认确定性实现| 默认确定性实现 |
| [aclnnComplex](../../math/complex/docs/aclnnComplex.md)         | 输入两个Shape满足broadcast关系、Dtype一致的Tensor。               | 默认确定性实现| 默认确定性实现 |
| [aclnnConfusionTranspose](../../conversion/confusion_transpose_d/docs/aclnnConfusionTranspose.md) | 融合reshape和transpose运算。 | 默认确定性实现| 默认确定性实现 |
| [aclnnConstantPadNd](../../conversion/pad_v3/docs/aclnnConstantPadNd.md) | 对输入的张量self，以pad参数为基准进行数据填充，填充值为value。 | 默认确定性实现| 默认确定性实现|
| [aclnnCos&aclnnInplaceCos](../../math/cos/docs/aclnnCos&aclnnInplaceCos.md) | 对输入矩阵的每个元素进行余弦操作后输出。                     | 默认确定性实现| - |
| [aclnnCosh&aclnnInplaceCosh](../../math/cosh/docs/aclnnCosh&aclnnInplaceCosh.md) | 双曲函数，根据公式返回一个新的tensor。结果的形状与输入tensor相同。 | 默认确定性实现| - |
| [aclnnCummax](../../math/cummax/docs/aclnnCummax.md)            | 计算self中的累积最大值，并返回最大值以及对应的索引。         | 默认确定性实现| - |
| [aclnnCummin](../../math/cummin/docs/aclnnCummin.md)            | 计算self中的累积最小值，并返回最小值以及对应的索引。         | 默认确定性实现| 默认确定性实现  |
| [aclnnCumprod&aclnnInplaceCumprod](../../math/cumprod/docs/aclnnCumprod&aclnnInplaceCumprod.md) | 新增aclnnCumprod接口，cumprod函数用于计算输入张量在指定维度上的累积乘积。 | 默认确定性实现| - |
| [aclnnCumsum](../../math/cumsum/docs/aclnnCumsum.md)            | 对输入张量self的元素，按照指定维度dim依次进行累加，并将结果保存到输出张量out中。 | 默认确定性实现| 默认确定性实现|
| [aclnnCumsumV2](../../math/cumsum/docs/aclnnCumsumV2.md)        | 对输入张量self的元素，按照指定维度dim依次进行累加，并将结果保存到输出张量out中。 | 默认确定性实现| 默认确定性实现|
| [aclnnDiag](../../conversion/diag_v2/docs/aclnnDiag.md) | 如果输入是向量（一维向量），则返回二维矩阵张量，其中input元素为对角线;如果输入是二维张量，则输出一维向量，取值为diagonal指定的输入矩阵的对角线元素。 | 默认确定性实现|  默认确定性实现 |
| [aclnnDiagFlat](../../conversion/diag_flat/docs/aclnnDiagFlat.md) | 生成对角线张量。如果输入self为一维张量，则返回二维张量，self里元素为对角线值；如果输入self是二维及以上张量，则先进行扁平化（化简为一维张量），再转化为第一种场景处理。 | 默认确定性实现| - |
| [aclnnDigamma](../../math/digamma/docs/aclnnDigamma.md) | 为输入张量的每一个元素进行digamma数学运算。 | 默认确定性实现| 默认确定性实现 |
| [aclnnDiv&aclnnInplaceDiv](../../math/div/docs/aclnnDiv&aclnnInplaceDiv.md) | 完成除法计算。                                               | 默认确定性实现| 默认确定性实现 |
| [aclnnDivMod&aclnnInplaceDivMod](../../math/div/docs/aclnnDivMod&aclnnInplaceDivMod.md) | 完成除法计算，并根据mode参数选择舍入操作。    | 默认确定性实现| 默认确定性实现 |
| [aclnnDivMods&aclnnInplaceDivMods](../../math/div/docs/aclnnDivMods&aclnnInplaceDivMods.md) | 完成除法计算，并根据mode参数选择舍入操作。    | 默认确定性实现| 默认确定性实现 |
| [aclnnDivs&aclnnInplaceDivs](../../math/div/docs/aclnnDivs&aclnnInplaceDivs.md) | 完成除法计算。                                               | 默认确定性实现| 默认确定性实现 |
| [aclnnDot](../../math/dot/docs/aclnnDot.md)                     | 计算两个一维张量的点积结果。                                 | 默认确定性实现| 默认确定性实现|
| [aclnnDropout](../../random/dsa_gen_bit_mask/docs/aclnnDropout.md)         | 训练过程中，按照概率p随机将输入中的元素置零，并将输出按照1/(1-p)的比例缩放。 | 默认确定性实现| - |
| [aclnnDropoutBackward](../../random/drop_out_do_mask/docs/aclnnDropoutBackward.md)         | 训练过程中，根据mask中对应bit位的值，将输入中的元素置零，并按照scale放大或者置零。若mask对应比特位为1，则gradOutput相应的元素放大。若mask中比特位为0，则gradOutput相应的元素置零。特别地，若scale为0，则将所有元素置为0；若scale为1，则不改变gradOutput的元素。建议与aclnnDropoutGenMask或者aclnnDropoutGenMaskV2组合使用。 | 默认确定性实现|  默认确定性实现 |
| [aclnnDropoutDoMask](../../random/drop_out_do_mask/docs/aclnnDropoutDoMask.md)         | 按照概率prob随机将输入中的元素置零，并将输出按照1/(1-prob)的比例放大。 | 默认确定性实现| 默认确定性实现  |
| [aclnnDropoutGenMask](../../random/dsa_gen_bit_mask/docs/aclnnDropoutGenMask.md)         | 训练过程中，按照概率prob生成mask，用于元素置零。 | 默认确定性实现| 默认确定性实现  |
| [aclnnDropoutGenMaskV2](../../random/dsa_gen_bit_mask/docs/aclnnDropoutGenMaskV2.md)         | 训练过程中，按照概率prob生成mask，用于元素置零。 | 默认确定性实现| 默认确定性实现  |
| [aclnnDropoutGenMaskV2Tensor](../../random/dsa_gen_bit_mask/docs/aclnnDropoutGenMaskV2Tensor.md)         | 训练过程中，按照概率prob生成mask，用于元素置零。 | 默认确定性实现| 默认确定性实现 |
| [aclnnDropoutV3](../../random/drop_out_v3/docs/aclnnDropoutV3.md)         | 按照概率p随机将输入中的元素置零，并将输出按照1/(1-p)的比例缩放。 | 默认确定性实现| 默认确定性实现  |
| [aclnnEqScalar&aclnnInplaceEqScalar](../../math/equal/docs/aclnnEqScalar&aclnnInplaceEqScalar.md) | 计算self中的元素的值与other的值是否相等，将self每个元素与other的值的比较结果写入out中。 | 默认确定性实现| 默认确定性实现 |
| [aclnnEqTensor&aclnnInplaceEqTensor](../../math/equal/docs/aclnnEqTensor&aclnnInplaceEqTensor.md) | 计算两个Tensor中的元素是否相等，返回一个Tensor，self=other的为True(1.)，否则为False(0.)。 | 默认确定性实现| 默认确定性实现 |
| [aclnnEqual](../../math/tensor_equal/docs/aclnnEqual.md)               | 计算两个Tensor是否有相同的大小和元素，返回一个Bool类型。     | 默认确定性实现| 默认确定性实现  |
| [aclnnErf&aclnnInplaceErf](../../math/erf/docs/aclnnErf&aclnnInplaceErf.md) | 返回输入Tensor中每个元素对应的误差函数的值。                 | 默认确定性实现| 默认确定性实现|
| [aclnnErfc&aclnnInplaceErfc](../../math/erfc/docs/aclnnErfc&aclnnInplaceErfc.md) | 返回输入Tensor中每个元素对应的误差互补函数的值。             | 默认确定性实现| 默认确定性实现 |
| [aclnnExp&aclnnInplaceExp](../../math/exp/docs/aclnnExp&aclnnInplaceExp.md) | 返回一个新的张量，该张量的每个元素都是输入张量对应元素的指数。 | 默认确定性实现| 默认确定性实现|
| [aclnnExp2&aclnnInplaceExp2](../../math/pow/docs/aclnnExp2&aclnnInplaceExp2.md) | self每个元素作为基数2的幂完成计算。                          | 默认确定性实现| 默认确定性实现 |
| [aclnnExpand](../../math/expand/docs/aclnnExpand.md)            | 将输入张量self广播成指定size的张量。                         | 默认确定性实现| 默认确定性实现|
| [aclnnExpm1&aclnnInplaceExpm1](../../math/expm1/docs/aclnnExpm1&aclnnInplaceExpm1.md) | 以输入的self为指数，计算自然常数e的幂，并对指数计算结果进行减1计算。 | 默认确定性实现| 默认确定性实现|
| [aclnnExpSegsum](../../math/segsum/docs/aclnnExpSegsum.md)      | 进行分段和计算。生成对角线为1、上三角为0的半可分矩阵。 | 默认确定性实现| - |
| [aclnnExpSegsumBackward](../../math/exp_segsum_grad/docs/aclnnExpSegsumBackward.md)      | [aclnnExpSegsum](../../math/segsum/docs/aclnnExpSegsum.md)的反向计算。  | 默认确定性实现| - |
| [aclnnEye](../../math/eye/docs/aclnnEye.md)                     | 返回一个二维张量，该张量的对角线上元素值为1，其余元素值为0。 | 默认确定性实现| 默认确定性实现  |
| [aclnnFlatten](../../conversion/flatten/docs/aclnnFlatten.md)   | 将输入Tensor，基于给定的axis，扁平化为一个2D的Tensor。       | 默认确定性实现| 默认确定性实现|
| [aclnnFloor&aclnnInplaceFloor](../../math/floor/docs/aclnnFloor&aclnnInplaceFloor.md) | 返回输入Tensor中每个元素向下取整，并将结果回填到输入Tensor中。 | 默认确定性实现| 默认确定性实现|
| [aclnnFloorDivide&aclnnInplaceFloorDivide](../../math/floor_div/docs/aclnnFloorDivide&aclnnInplaceFloorDivide.md) |  完成除法计算，对余数向下取整。                                                            | 默认确定性实现| 默认确定性实现 |
| [aclnnFloorDivides&aclnnInplaceFloorDivides](../../math/floor_div/docs/aclnnFloorDivides&aclnnInplaceFloorDivides.md) | 完成除法计算，对余数向下取整。                               | 默认确定性实现| 默认确定性实现 |
| [aclnnFmodScalar&aclnnInplaceFmodScalar](../../math/mod/docs/aclnnFmodScalar&aclnnInplaceFmodScalar.md) | 返回self除以other的余数。                                   | 默认确定性实现| 默认确定性实现 |
| [aclnnFmodTensor&aclnnInplaceFmodTensor](../../math/mod/docs/aclnnFmodTensor&aclnnInplaceFmodTensor.md) | 返回self除以other的余数。                                    | 默认确定性实现| 默认确定性实现 |
| [aclnnFrac&aclnnInplaceFrac](../../math/sub/docs/aclnnFrac&aclnnInplaceFrac.md) | 计算输入Tensor中每个元素的小数部分后输出。                   | 默认确定性实现|  默认确定性实现 |
| [aclnnGcd](../../math/gcd/docs/aclnnGcd.md)                     | 对给定的self和other计算element-wise维度的最大公约数，其中self和other都需要为整数。 | 默认确定性实现| 默认确定性实现 |
| [aclnnGer](../../math/ger/docs/aclnnGer.md)                     | 实现self和vec2的外积。                                       | 默认确定性实现| 默认确定性实现 |
| [aclnnGeScalar&aclnnInplaceGeScalar](../../math/greater_equal/docs/aclnnGeScalar&aclnnInplaceGeScalar.md) | 判断输入Tensor中的每个元素是否大于等于other Scalar的值，返回一个Bool类型的Tensor，对应输入Tensor中每个位置的大于等于判断是否成立。 | 默认确定性实现| 默认确定性实现 |
| [aclnnGeTensor&aclnnInplaceGeTensor](../../math/greater_equal/docs/aclnnGeTensor&aclnnInplaceGeTensor.md) | 判断输入Tensor中的每个元素是否大于等于other Tensor的值，返回一个Bool类型的Tensor，对应输入Tensor中每个位置的大于等于判断是否成立。 | 默认确定性实现| 默认确定性实现 |
| [aclnnGlobalAveragePool](../../math/reduce_mean/docs/aclnnGlobalAveragePool.md) | 传入一个输入张量X，并在同一通道中的值上应用平均池化。        | 默认确定性实现| 默认确定性实现|
| [aclnnGlobalMaxPool](../../math/reduce_max/docs/aclnnGlobalMaxPool.md) | 输入一个张量，并对同一通道中的值取最大值。                   | 默认确定性实现| 默认确定性实现|
| [aclnnGroupedBiasAddGrad](../../math/grouped_bias_add_grad/docs/aclnnGroupedBiasAddGrad.md) | 分组偏置加法（GroupedBiasAdd）的反向传播。                   | 默认确定性实现| 默认确定性实现|
| [aclnnGroupedBiasAddGradV2](../../math/grouped_bias_add_grad/docs/aclnnGroupedBiasAddGradV2.md) | 实现groupBiasAdd的反向计算。本接口是对[aclnnGroupedBiasAddGrad](../../math/grouped_bias_add_grad/docs/aclnnGroupedBiasAddGrad.md)接口的功能扩展，增加了groupIdxType属性，支持指定groupIdx的类型。                   | 默认确定性实现| 默认确定性实现  |
| [aclnnGtScalar&aclnnInplaceGtScalar](../../math/greater/docs/aclnnGtScalar&aclnnInplaceGtScalar.md) | 判断输入Tensor中的每个元素是否大于other Scalar的值，返回一个Bool类型的Tensor，对应输入Tensor中每个位置的大于判断是否成立。 | 默认确定性实现| 默认确定性实现 |
| [aclnnGtTensor&aclnnInplaceGtTensor](../../math/greater/docs/aclnnGtTensor&aclnnInplaceGtTensor.md)                                               | 判断self Tensor中的元素是否大于other Tensor中的元素。返回一个Tensor，若self>other，为True(1)；否则为False(0)。 | 默认确定性实现| 默认确定性实现 |
| [aclnnHansDecode](../../math/hans_decode/docs/aclnnHansDecode.md) | 对压缩后的张量基于PDF进行解码，同时基于mantissa（尾数）重组恢复张量。 | 默认确定性实现| - |
| [aclnnHansEncode](../../math/hans_encode/docs/aclnnHansEncode.md) | 对张量的指数位所在字节实现PDF统计，按PDF分布统计进行无损压缩。 | 默认确定性实现| - |
| [aclnnHardtanh&aclnnInplaceHardtanh](../../conversion/clip_by_value_v2/docs/aclnnHardtanh&aclnnInplaceHardtanh.md) | 将输入的所有元素限制在[clipValueMin,clipValueMax]范围内，若元素大于clipValueMax则限制为clipValueMax，若元素小于clipValueMin则限制为clipValueMin，否则等于元素本身。 | 默认确定性实现| 默认确定性实现 |
| [aclnnHistc](../../math/histogram_v2/docs/aclnnHistc.md) | 以min和max作为统计上下限，在min和max之间划出等宽的数量为bins的区间，统计张量self中元素在各个区间的数量。如果min和max相等，则使用张量中所有元素的最小值和最大值作为统计的上下限。小于min和大于max的元素不会被统计。 | 默认确定性实现| 默认非确定性实现，支持配置开启 |
| [aclnnIm2col](../../conversion/im2col/docs/aclnnIm2col.md)      | 图像到列，滑动局部窗口数据转为列向量，拼接为大张量。从批处理输入张量中提取滑动窗口。 | 默认确定性实现| 默认确定性实现|
| [aclnnInplaceAdds](../../math/add/docs/aclnnInplaceAdds.md) | 完成加法计算。 | 默认确定性实现|  默认确定性实现 |
| [aclnnInplaceBitwiseAndScalar](../../math/bitwise_and/docs/aclnnInplaceBitwiseAndScalar.md) | 计算输入tensor中每个元素和输入标量other的按位与结果。输入的tensor与other必须是整型或者BOOL型变量。当输入为BOOL型时，计算逻辑与的结果。             | 默认确定性实现| 默认确定性实现 |
| [aclnnInplaceBitwiseAndTensor](../../math/bitwise_and/docs/aclnnInplaceBitwiseAndTensor.md) | 输入为BOOL型tensor时，进行逻辑与运算；输入为INT型时，进行位与运算。 | 默认确定性实现| 默认确定性实现 |
| [aclnnInplaceCopy](../../conversion/view_copy/docs/aclnnInplaceCopy.md) | 将src中的元素复制到selfRef张量中并返回selfRef。              | 默认确定性实现| 默认确定性实现|
| [aclnnInplaceFillDiagonal](../../conversion/fill_diagonal_v2/docs/aclnnInplaceFillDiagonal.md) | 以fillValue填充tensor对角线。                                | 默认确定性实现| - |
| [aclnnInplaceFillScalar](../../conversion/fill/docs/aclnnInplaceFillScalar.md) | 对tensor填充指定标量。                                       | 默认确定性实现| 默认确定性实现|
| [aclnnInplaceFillTensor](../../conversion/fill/docs/aclnnInplaceFillTensor.md) | 对selfRef张量填充value， value是张量。                       | 默认确定性实现| 默认确定性实现|
| [aclnnInplaceMaskedFillScalar](../../conversion/masked_fill/docs/aclnnInplaceMaskedFillScalar.md) | 用value填充selfRef里面与mask矩阵中值为true的位置相对应的元素。 | 默认确定性实现| 默认确定性实现|
| [aclnnInplaceMaskedFillTensor](../../conversion/masked_fill/docs/aclnnInplaceMaskedFillTensor.md) | 用value填充selfRef里面与mask矩阵中值为true的位置相对应的元素。 | 默认确定性实现| 默认确定性实现|
| [aclnnInplaceNormal](../../random/dsa_random_normal/docs/aclnnInplaceNormal.md) | 使用mean和std的参数正态分布的元素进行填充张量。从给定的均值mean和标准差std的离散正态分布中抽取随机数，用于填充selfRef张量。其中，均值mean是一个基础类型float，包含每个输出元素相关的正态分布均值。std是一个基础类型float，包含每个输出元素相关的正态分布的标准差。                | 默认确定性实现|  默认确定性实现 |
| [aclnnInplaceNormalTensor](../../random/dsa_random_normal/docs/aclnnInplaceNormalTensor.md) | 使用mean和std的参数正态分布的元素进行填充张量。                | 默认确定性实现| 默认确定性实现  |
| [aclnnInplaceOne](../../math/ones_like/docs/aclnnInplaceOne.md) | 返回形状和类型相同的张量，所有元素都设置为1。                | 默认确定性实现| 默认确定性实现|
| [aclnnInplaceRandom](../../random/dsa_random_uniform/docs/aclnnInplaceRandom.md) | 从[from, to-1]的离散均匀分布中采样的数填充selfRef张量。                | 默认确定性实现| 默认确定性实现  |
| [aclnnInplaceRandomTensor](../../random/dsa_random_uniform/docs/aclnnInplaceRandomTensor.md) | 从[from, to-1]的离散均匀分布中采样的数填充selfRef张量。                | 默认确定性实现| 默认确定性实现  |
| [aclnnInplaceUniform](../../random/dsa_random_uniform/docs/aclnnInplaceUniform.md) | 生成[from, to)区间内离散均匀分布的随机数，并将其填充到selfRef张量中。(该接口的BOOL类型已废弃，如需使用，建议采用aclnnBernoulli或者aclnnInplaceBernoulli接口)     | 默认确定性实现|  默认确定性实现 |
| [aclnnInplaceUniformTensor](../../random/dsa_random_uniform/docs/aclnnInplaceUniformTensor.md) | 生成[from, to)区间内离散均匀分布的随机数，并将其填充到selfRef张量中。                | 默认确定性实现| 默认确定性实现  |
| [aclnnInplaceZero](../../math/zero_op/docs/aclnnInplaceZero.md) | 将selfRef张量填充为全零。                                    | 默认确定性实现| 默认确定性实现|
| [aclnnIsClose](../../math/is_close/docs/aclnnIsClose.md)        | 返回一个带有布尔元素的新张量，判断给定的self和other是否彼此接近，如果值接近，则返回True，否则返回False。 | 默认确定性实现| 默认确定性实现|
| [aclnnIsFinite](../../math/is_finite/docs/aclnnIsFinite.md)     | 判断张量中哪些元素是有限数值，即不是inf、-inf或nan。         | 默认确定性实现| 默认确定性实现|
| [aclnnIsInf](../../math/is_inf/docs/aclnnIsInf.md)              | 判断张量中哪些元素是无限大值，即是inf、-inf。                | 默认确定性实现| 默认确定性实现|
| [aclnnIsInScalarTensor](../../math/equal/docs/aclnnIsInScalarTensor.md) | 检查element中的元素是否等于testElement。                     | 默认确定性实现| 默认确定性实现 |
| [aclnnIsInTensorScalar](../../math/equal/docs/aclnnIsInTensorScalar.md) | 检查element中的元素是否等于testElement。                     | 默认确定性实现| 默认确定性实现 |
| [aclnnIsNegInf](../../math/is_neg_inf/docs/aclnnIsNegInf.md)    | 判断输入张量的元素是否为负无穷。                             | 默认确定性实现| 默认确定性实现|
| [aclnnIsPosInf](../../math/is_pos_inf/docs/aclnnIsPosInf.md)    | 判断输入张量的元素是否为正无穷。                             | 默认确定性实现| 默认确定性实现|
| [aclnnKlDiv](../../math/kl_div_v2/docs/aclnnKlDiv.md)           | 计算KL散度。                                                 | 默认确定性实现| 默认确定性实现|
| [aclnnLeftShift](../../math/left_shift/docs/aclnnLeftShift.md)           | 对于输入张量self中每个元素，根据输入张量shiftBits对应位置的参数，按位进行左移。           | 默认确定性实现| 默认确定性实现|
| [aclnnLeftShifts](../../math/left_shift/docs/aclnnLeftShifts.md)           | 对于输入张量self中每个元素，根据输入标量shiftBits，按位进行左移。   | 默认确定性实现| 默认确定性实现|
| [aclnnLerp&aclnnInplaceLerp](../../math/lerp/docs/aclnnLerp&aclnnInplaceLerp.md) | 根据给定的权重，在起始和结束Tensor之间进行线性插值，返回插值后的Tensor。 | 默认确定性实现| 默认确定性实现 |
| [aclnnLerps&aclnnInplaceLerps](../../math/lerp/docs/aclnnLerps&aclnnInplaceLerps.md) | 根据给定的权重，在起始和结束Tensor之间进行线性插值，返回插值后的Tensor。 | 默认确定性实现| 默认确定性实现 |
| [aclnnLeScalar&aclnnInplaceLeScalar](../../math/less_equal/docs/aclnnLeScalar&aclnnInplaceLeScalar.md) | 判断输入self中的元素值是否小于等于other的值，并将self的每个元素的值与other值的比较结果写入out中。 | 默认确定性实现| 默认确定性实现 |
| [aclnnLeTensor&aclnnInplaceLeTensor](../../math/less_equal/docs/aclnnLeTensor&aclnnInplaceLeTensor.md) | 计算self中的元素值是否小于等于other的值，将self每个元素与other值的比较结果写入out中。 | 默认确定性实现| 默认确定性实现 |
| [aclnnLgamma](../../math/lgamma/docs/aclnnLgamma.md)           | 为输入张量的每一个元素进行lgamma数学运算。           | 默认确定性实现| 默认确定性实现 |
| [aclnnLinalgCholesky](../../math/cholesky/docs/aclnnLinalgCholesky.md)     | 计算实数对称正定矩阵的Cholesky分解。             | 默认确定性实现| - |
| [aclnnLinalgCross](../../math/cross/docs/aclnnLinalgCross.md)     | 对输入Tensor完成linear_cross运算。             | 默认确定性实现| 默认确定性实现 |
| [aclnnLinalgQr](../../math/linalg_qr/docs/aclnnLinalgQr.md)     | 对输入Tensor进行正交分解。             | 默认确定性实现| 默认确定性实现 |
| [aclnnLinspace](../../math/lin_space/docs/aclnnLinspace.md)     | 生成一个等间隔数值序列。                                     | 默认确定性实现| 默认确定性实现 |
| [aclnnLog&aclnnInplaceLog](../../math/log/docs/aclnnLog&aclnnInplaceLog.md) | 完成自然对数的计算                                           | 默认确定性实现| 默认确定性实现|
| [aclnnLog2&aclnnInplaceLog2](../../math/log/docs/aclnnLog2&aclnnInplaceLog2.md) | 完成以2为底的对数计算                                        | 默认确定性实现| 默认确定性实现|
| [aclnnLog10&aclnnInplaceLog10](../../math/log/docs/aclnnLog10&aclnnInplaceLog10.md) | 完成输入以10为底的对数计算                                   | 默认确定性实现| 默认确定性实现|
| [aclnnLog1p&aclnnInplaceLog1p](../../math/log1p/docs/aclnnLog1p&aclnnInplaceLog1p.md) | 对输入Tensor完成log1p运算。                                  | 默认确定性实现| 默认确定性实现|
| [aclnnLogAddExp](../../math/log_add_exp/docs/aclnnLogAddExp.md) | 对输入的tensor计算幂和的对数（以e为底）。                                           | 默认确定性实现| 默认确定性实现 |
| [aclnnLogAddExp2](../../math/log_add_exp/docs/aclnnLogAddExp2.md) | 对输入的tensor计算幂和的对数（以2为底）。                                           | 默认确定性实现| 默认确定性实现 |
| [aclnnLogdet](../../math/logdet/docs/aclnnLogdet.md) | 计算输入self的行列式的自然对数。                                         | 默认确定性实现|  默认确定性实现 |
| [aclnnLogicalAnd&aclnnInplaceLogicalAnd](../../math/logical_and/docs/aclnnLogicalAnd&aclnnInplaceLogicalAnd.md) | 完成给定输入张量元素的逻辑与运算。0被视为False，非0被视为True。 | 默认确定性实现| 默认确定性实现 |
| [aclnnLogicalNot&aclnnInplaceLogicalNot](../../math/logical_not/docs/aclnnLogicalNot&aclnnInplaceLogicalNot.md) | 计算给定输入Tensor的逐元素逻辑非。如果未指定输出类型，输出Tensor是bool类型。 | 默认确定性实现| 默认确定性实现|
| [aclnnLogicalOr&aclnnInplaceLogicalOr](../../math/logical_or/docs/aclnnLogicalOr&aclnnInplaceLogicalOr.md) | 完成给定输入张量元素的逻辑或运算。当两个输入张量为非bool类型时，0被视为False，非0被视为True。 | 默认确定性实现| 默认确定性实现 |
| [aclnnLogicalXor](../../math/not_equal/docs/aclnnLogicalXor.md) | 完成给定输入张量元素的逻辑异或运算。当self和other为非bool类型时，0被视为False，非0被视为True。 | 默认确定性实现| 默认确定性实现 |
| [aclnnLogSpace](../../math/logspace/docs/aclnnLogSpace.md) | 创建一个大小为$\text{steps}$的一维张量，其值在$\text{base}^\text{start}$到$\text{base}^\text{end}$上对数尺度上均匀间隔，包含端点，以$\text{base}$为底。                                      | 默认确定性实现| 默认确定性实现 |
| [aclnnLogSumExp](../../math/reduce_log_sum_exp/docs/aclnnLogSumExp.md) | 返回输入tensor指定维度上的指数之和的对数。                   | 默认确定性实现| 默认确定性实现|
| [aclnnLtScalar&aclnnInplaceLtScalar](../../math/less/docs/aclnnLtScalar&aclnnInplaceLtScalar.md) | 判断输入self中的每个元素是否小于输入other的值，返回一个Bool类型的Tensor。 | 默认确定性实现| 默认确定性实现 |
| [aclnnLtTensor&aclnnInplaceLtTensor](../../math/less/docs/aclnnLtTensor&aclnnInplaceLtTensor.md) | 判断输入self中的每个元素是否小于输入other中的元素，返回一个Bool类型的Tensor。 | 默认确定性实现| 默认确定性实现 |
| [aclnnMaskedScale](../../math/masked_scale/docs/aclnnMaskedScale.md) | 完成elementwise计算。                                        | 默认确定性实现| 默认确定性实现|
| [aclnnMaskedSelect](../../conversion/masked_select_v3/docs/aclnnMaskedSelect.md) | 根据一个布尔掩码张量（mask）中的值选择输入张量（self）中的元素作为输出，形成一个新的一维张量。 | 默认确定性实现| 默认确定性实现|
| [aclnnMax](../../math/reduce_max/docs/aclnnMax.md)              | 返回Tensor所有元素中的最大值。                               | 默认确定性实现| 默认确定性实现|
| [aclnnMaxDim](../../math/arg_max_with_value/docs/aclnnMaxDim.md) | 返回Tensor指定维度的最大值及其索引位置。                     | 默认确定性实现| 默认确定性实现|
| [aclnnMaximum](../../math/maximum/docs/aclnnMaximum.md)         | 计算两个张量中每个元素的最大值，并返回一个新的张量。         | 默认确定性实现| 默认确定性实现 |
| [aclnnMaxN](../../math/maximum/docs/aclnnMaxN.md)               | 对输入tensor列表中每个输入tensor对应元素求max。              | 默认确定性实现| 默认确定性实现 |
| [aclnnMaxV2](../../math/reduce_max/docs/aclnnMaxV2.md)          | 按指定维度对输入tensor求元素最大值。                         | 默认确定性实现| 默认确定性实现|
| [aclnnMean](../../math/reduce_mean/docs/aclnnMean.md)           | 按指定维度对Tensor求均值。                                   | 默认确定性实现| 默认确定性实现|
| [aclnnMeanV2](../../math/reduce_mean/docs/aclnnMeanV2.md)       | 按指定维度对Tensor求均值。                                   | 默认确定性实现| 默认确定性实现|
| [aclnnMin](../../math/reduce_min/docs/aclnnMin.md)              | 返回Tensor所有元素中的最小值。                               | 默认确定性实现| 默认确定性实现|
| [aclnnMinDim](../../math/arg_min_with_value/docs/aclnnMinDim.md) | 返回self中指定维度的最小值及其索引位置。                     | 默认确定性实现| 默认确定性实现|
| [aclnnMinimum](../../math/minimum/docs/aclnnMinimum.md)         | 计算两个张量中每个元素的最小值，并返回一个新的张量。         | 默认确定性实现| 默认确定性实现 |
| [aclnnMinN](../../math/minimum/docs/aclnnMinN.md)               | 对输入tensor列表中每个输入tensor对应元素求min。              | 默认确定性实现| 默认确定性实现 |
| [aclnnMul&aclnnInplaceMul](../../math/mul/docs/aclnnMul&aclnnInplaceMul.md) | 完成tensor与tensor间的乘法计算。                             | 默认确定性实现| 默认确定性实现 |
| [aclnnMuls&aclnnInplaceMuls](../../math/muls/docs/aclnnMuls&aclnnInplaceMuls.md) | 完成tensor与scalar间的乘法计算。                             | 默认确定性实现| 默认确定性实现 |
| [aclnnMultinomial](../../random/stateless_sample_multinomial/docs/aclnnMultinomial.md) | 在输入张量中根据每个对象分布的概率，抽取numsamples个样本，并将这些样本的索引存储在输出张量中。                | 默认确定性实现| 默认确定性实现  |
| [aclnnMultinomialTensor](../../random/stateless_sample_multinomial/docs/aclnnMultinomialTensor.md) | 在输入张量中根据每个对象分布的概率，抽取numsamples个样本，并将这些样本的索引存储在输出张量中。                | 默认确定性实现| 默认确定性实现 |
| [aclnnNanToNum&aclnnInplaceNanToNum](../../math/nan_to_num/docs/aclnnNanToNum&aclnnInplaceNanToNum.md) | 将输入中的NaN、正无穷大和负无穷大值分别替换为nan、posinf、neginf指定的值。 | 默认确定性实现| 默认确定性实现|
| [aclnnNeg&aclnnInplaceNeg](../../math/neg/docs/aclnnNeg&aclnnInplaceNeg.md) | 对输入的每个元素完成相反数计算。                             | 默认确定性实现| 默认确定性实现|
| [aclnnNeScalar&aclnnInplaceNeScalar](../../math/not_equal/docs/aclnnNeScalar&aclnnInplaceNeScalar.md) | 计算selfRef中的元素的值与other的值是否不相等。               | 默认确定性实现| 默认确定性实现 |
| [aclnnNeTensor&aclnnInplaceNeTensor](../../math/not_equal/docs/aclnnNeTensor&aclnnInplaceNeTensor.md) | 计算self（selfRef）中的元素的值与other的值是否不相等。       | 默认确定性实现| 默认确定性实现 |
| [aclnnNormalFloatFloat](../../random/stateless_random_normal_v2/docs/aclnnNormalFloatFloat.md)         | 返回一个随机数张量，该随机数是从给定的均值(float)和标准差(float)的独立正态分布中获取。 | - |  默认确定性实现 |
| [aclnnNormalFloatTensor](../../random/stateless_random_normal_v2/docs/aclnnNormalFloatTensor.md)         | 返回一个随机数张量，该随机数是从给定的均值(float)和标准差(tensor)的独立正态分布中获取。 | 默认确定性实现| 默认确定性实现  |
| [aclnnNormalTensorFloat](../../random/stateless_random_normal_v2/docs/aclnnNormalTensorFloat.md)         | 返回一个随机数张量，该随机数是从给定的均值(tensor)和标准差(float)的独立正态分布中获取。 | 默认确定性实现| 默认确定性实现  |
| [aclnnNormalTensorTensor](../../random/stateless_random_normal_v2/docs/aclnnNormalTensorTensor.md)         | 返回一个随机数，该随机数是从给定的均值(tensor)和标准差(tensor)的独立正态分布中获取。 | 默认确定性实现| 默认确定性实现  |
| [aclnnNpuFormatCast](../../conversion/npu_format_cast/docs/aclnnNpuFormatCast.md) | 完成ND←→NZ的转换功能。                    | 默认确定性实现| 默认确定性实现  |
| [aclnnOneHot](../../math/one_hot/docs/aclnnOneHot.md)           | 对长度为n的输入self，经过one_hot的计算后得到一个元素数量为n\*k的输出out。           | 默认确定性实现| 默认确定性实现 |
| [aclnnPdist](../../math/pdist/docs/aclnnPdist.md)               | 计算输入self中每对行向量的p范数距离。                        | 默认确定性实现| - |
| [aclnnPdistForward](../../math/pdist/docs/aclnnPdistForward.md) | 计算输入self中每对行向量的p范数距离。                        | 默认确定性实现| - |
| [aclnnPermute](../../conversion/transpose/docs/aclnnPermute.md) | 对tensor的任意维度进行调换。如输入self是shape为[2, 3, 5]的tensor，dims为(2, 0, 1)，则输出是shape为[5, 2, 3]的tensor。 | 默认确定性实现| 默认确定性实现|
| [aclnnPolar](../../math/polar/docs/aclnnPolar.md) | 对输入input做绝对值和angle对应的极坐标求对应的笛卡尔坐标，得到一个复数张量out。              | 默认确定性实现| 默认确定性实现 |
| [aclnnPowScalarTensor](../../math/pow/docs/aclnnPowScalarTensor.md) | exponent每个元素作为input对应元素的幂完成计算。              | 默认确定性实现| 默认确定性实现 |
| [aclnnPowTensorScalar&aclnnInplacePowTensorScalar](../../math/pow/docs/aclnnPowTensorScalar&aclnnInplacePowTensorScalar.md) | exponent每个元素作为input对应元素的幂完成计算。              | 默认确定性实现| 默认确定性实现 |
| [aclnnPowTensorTensor&aclnnInplacePowTensorTensor](../../math/pow/docs/aclnnPowTensorTensor&aclnnInplacePowTensorTensor.md) | exponent每个元素作为input对应元素的幂完成计算。              | 默认确定性实现| 默认确定性实现 |
| [aclnnPrecisionCompare](../../math/precision_compare/docs/aclnnPrecisionCompare.md)           | 计算两个Tensor元素是否相同，进行精度比对，返回算子执行的状态码，不会触发管理面故障上报。                               | 默认确定性实现| 默认确定性实现  |
| [aclnnProd](../../math/reduce_prod/docs/aclnnProd.md)           | 回输入tensor中所有元素的乘积。                               | 默认确定性实现| 默认确定性实现|
| [aclnnProdDim](../../math/reduce_prod/docs/aclnnProdDim.md)     | 返回输入tensor给定维度上每行的乘积。                         | 默认确定性实现| 默认确定性实现|
| [aclnnQr](../../math/q_r/docs/aclnnQr.md)     | 对输入Tensor进行正交分解。                         | 默认确定性实现| - |
| [aclnnInplaceRandomWithoutFromTo](../../random/dsa_random_uniform/docs/aclnnInplaceRandomWithoutFromTo.md)     | 给定seed/offset(int64),默认从 [0, DTYPE_MAX_VALUE] 的离散均匀分布中采样整数填充selfRef张量。                         | -|  默认确定性实现 |
| [aclnnInplaceRandomWithoutFromToTensor](../../random/dsa_random_uniform/docs/aclnnInplaceRandomWithoutFromToTensor.md)     | 给定seed/offset(tensor),默认从 [0, DTYPE_MAX_VALUE] 的离散均匀分布中采样整数填充selfRef张量。                         | -|  默认确定性实现 |
| [aclnnRandperm](../../random/stateless_randperm/docs/aclnnRandperm.md)     | 返回从0到n-1的整数随机排列。                         | 默认确定性实现|  默认确定性实现 |
| [aclnnRange](../../math/range/docs/aclnnRange.md)               | 从start起始到end结束按照step的间隔取值。                     | 默认确定性实现| 默认确定性实现 |
| [aclnnReal](../../math/real/docs/aclnnReal.md)                  | 为输入张量的每一个元素取实数部分。                           | 默认确定性实现| - |
| [aclnnReciprocal&aclnnInplaceReciprocal](../../math/reciprocal/docs/aclnnReciprocal&aclnnInplaceReciprocal.md) | 返回一个具有每个输入元素倒数的新张量。                       | 默认确定性实现| 默认确定性实现|
| [aclnnReduceLogSum](../../math/reduce_log_sum/docs/aclnnReduceLogSum.md) | 返回给定维度中输入张量每行的和再取对数。                     | 默认确定性实现| - |
| [aclnnReduceNansum](../../math/reduce_nansum/docs/aclnnReduceNansum.md) | 将tensor中NaN处理为0后，返回输入tensor给定维度上的和。       | 默认确定性实现| - |
| [aclnnReduceSum](../../math/reduce_sum/docs/aclnnReduceSum.md) | 返回给定维度中输入张量每行的和。                             | 默认确定性实现| 默认确定性实现|
| [aclnnReflectionPad1d](../../conversion/mirror_pad/docs/aclnnReflectionPad1d.md) | 使用输入边界的反射填充输入tensor。                           | 默认确定性实现| 默认确定性实现|
| [aclnnReflectionPad1dBackward](../../conversion/pad_v4_grad/docs/aclnnReflectionPad1dBackward.md) | reflection_pad1d的反向传播，前向计算参考aclnnReflectionPad1d。 | 默认确定性实现| 默认确定性实现|
| [aclnnReflectionPad2d](../../conversion/mirror_pad/docs/aclnnReflectionPad2d.md) | 使用输入边界的反射填充输入tensor。                           | 默认确定性实现| 默认确定性实现|
| [aclnnReflectionPad2dBackward](../../conversion/pad_v4_grad/docs/aclnnReflectionPad2dBackward.md) | reflection_pad2d的反向传播，前向计算参考aclnnReflectionPad2d。 | 默认确定性实现| 默认确定性实现|
| [aclnnReflectionPad3d](../../conversion/mirror_pad/docs/aclnnReflectionPad3d.md) | 3D反射填充。             | 默认确定性实现| 默认确定性实现|
| [aclnnReflectionPad3dBackward](../../conversion/reflection_pad3d_grad/docs/aclnnReflectionPad3dBackward.md) | 计算aclnnReflectionPad3d api的反向传播。                     | 默认确定性实现| 默认确定性实现|
| [aclnnRemainderScalarTensor](../../math/floor_mod/docs/aclnnRemainderScalarTensor.md) | 将scalar self进行broadcast成和tensor other一样shape的tensor以后，其中的每个元素都转换为除以other的对应元素以后得到的余数。 | 默认确定性实现| 默认确定性实现 |
| [aclnnRemainderTensorScalar&aclnnInplaceRemainderTensorScalar](../../math/floor_mod/docs/aclnnRemainderTensorScalar&aclnnInplaceRemainderTensorScalar.md) | 将tensor self中的每个元素都转换为除以scalar other以后得到的余数。 | 默认确定性实现| 默认确定性实现 |
| [aclnnRemainderTensorTensor&aclnnInplaceRemainderTensorTensor](../../math/floor_mod/docs/aclnnRemainderTensorTensor&aclnnInplaceRemainderTensorTensor.md) | 将tensor self和tensor other进行broadcast成一致的shape后，其中的每个元素都转换为除以other的对应元素以后得到的余数。 | 默认确定性实现| 默认确定性实现 |
| [aclnnRepeat](../../math/tile/docs/aclnnRepeat.md)              | 对输入tensor沿着repeats中对每个维度指定的复制次数进行复制。  | 默认确定性实现| 默认确定性实现|
| [aclnnReplicationPad1d](../../conversion/pad_v3/docs/aclnnReplicationPad1d.md) | 使用输入边界填充输入tensor的最后一维。                       | 默认确定性实现| 默认确定性实现|
| [aclnnReplicationPad1dBackward](../../conversion/pad_v3_grad_replicate/docs/aclnnReplicationPad1dBackward.md) | replication_pad1d的反向传播。                                | 默认确定性实现| 默认确定性实现|
| [aclnnReplicationPad2d](../../conversion/pad_v3/docs/aclnnReplicationPad2d.md) | 使用输入边界填充输入tensor的最后两维。                       | 默认确定性实现| 默认确定性实现|
| [aclnnReplicationPad2dBackward](../../conversion/pad_v3_grad_replicate/docs/aclnnReplicationPad2dBackward.md) | replication_pad2d的反向传播，前向计算参考aclnnReplicationPad2d。 | 默认确定性实现| 默认确定性实现|
| [aclnnReplicationPad3d](../../conversion/pad_v3/docs/aclnnReplicationPad3d.md) | 使用输入边界填充输入tensor的最后三维。                       | 默认确定性实现| 默认确定性实现|
| [aclnnReplicationPad3dBackward](../../conversion/pad_v3_grad_replication/docs/aclnnReplicationPad3dBackward.md) | 计算aclnnReplicationPad3d的反向传播。                        | 默认确定性实现| 默认确定性实现|
| [aclnnRightShift](../../math/right_shift/docs/aclnnRightShift.md)         | 对于输入张量input中每个元素，根据输入张量shiftBits对应位置的参数，按位进行右移。 | 默认确定性实现| 默认确定性实现|
| [aclnnRoll](../../conversion/roll/docs/aclnnRoll.md)            | 沿给定尺寸和维度移动Tensor中的数据。                         | 默认确定性实现| 默认确定性实现|
| [aclnnRound&aclnnInplaceRound](../../math/round/docs/aclnnRound&aclnnInplaceRound.md) | 将输入的值舍入到最接近的整数，若该值与两个整数距离一样则向偶数取整。 | 默认确定性实现| 默认确定性实现|
| [aclnnRoundDecimals&aclnnInplaceRoundDecimals](../../math/round/docs/aclnnRoundDecimals&aclnnInplaceRoundDecimals.md) | 将输入Tensor的元素四舍五入到指定的位数。                     | 默认确定性实现| 默认确定性实现|
| [aclnnRsqrt&aclnnInplaceRsqrt](../../math/rsqrt/docs/aclnnRsqrt&aclnnInplaceRsqrt.md) | 求input(Tensor)每个元素的平方根的倒数。                      | 默认确定性实现| 默认确定性实现|
| [aclnnRsub](../../math/sub/docs/aclnnRsub.md)                   | 完成减法计算。                                               | 默认确定性实现| 默认确定性实现 |
| [aclnnRsubs](../../math/sub/docs/aclnnRsubs.md)         | 完成减法计算。    | 默认确定性实现| 默认确定性实现 |
| [aclnnScale](../../math/scale/docs/aclnnScale.md)               | 对输入Tensor进行scale和bias计算。若不输入bias，则 $y = x \cdot scale$；若输入bias，则 $y = x \cdot scale + bias$。                                               | 默认确定性实现| 默认确定性实现 |
| [aclnnSearchSorted](../../math/search_sorted/docs/aclnnSearchSorted.md) | 在一个已排序的张量（sortedSequence）中查找给定tensor值（self）应该插入的位置。 | 默认确定性实现| 默认确定性实现 |
| [aclnnSearchSorteds](../../math/search_sorted/docs/aclnnSearchSorteds.md) | 在一个已排序的一维张量（sortedSequence）中查找给定Scalar值（self）应该插入的位置。 | 默认确定性实现| - |
| [aclnnSign](../../math/sign/docs/aclnnSign.md)                  | 对输入的tensor逐元素进行Sign符号函数的运算并输出结果tensor。 | 默认确定性实现| 默认确定性实现|
| [aclnnSignbit](../../math/signbit/docs/aclnnSignbit.md)            | 判断输入中的每个元素符号位是否为1，返回一个tensor。 | 默认确定性实现| 默认确定性实现 |
| [aclnnSignBitsPack](../../math/sign_bits_pack/docs/aclnnSignBitsPack.md) | 将float16类型或者float32类型的1位Adam打包为uint8。           | 默认确定性实现| - |
| [aclnnSignBitsUnpack](../../math/sign_bits_unpack/docs/aclnnSignBitsUnpack.md) | 将uint8类型1位Adam拆包为float32或者float16。                 | 默认确定性实现| - |
| [aclnnSilentCheck](../../math/silent_check/docs/aclnnSilentCheck.md) | SilentCheckV2算子功能主要根据输入特征值（val），与绝对阈值、相对阈值比较，来识别是否触发静默检测故障。 | 默认确定性实现| - |
| [aclnnSilentCheckV2](../../math/silent_check_v2/docs/aclnnSilentCheckV2.md) | 根据stepRef参数与马尔可夫不等式阈值来识别是否触发静默检测故障。 | 默认确定性实现| - |
| [aclnnSimThreadExponential](../../random/sim_thread_exponential/docs/aclnnSimThreadExponential.md) | 生成服从参数为lambda的指数分布随机数，并将其填充到selfRef张量中。                                    | 默认确定性实现| 默认确定性实现 |
| [aclnnSin&aclnnInplaceSin](../../math/sin/docs/aclnnSin&aclnnInplaceSin.md) | 对输入Tensor完成sin运算。                                    | 默认确定性实现| 默认确定性实现|
| [aclnnSinc&aclnnInplaceSinc](../../math/sinc/docs/aclnnSinc&aclnnInplaceSinc.md) | 对输入Tensor完成sinc运算。                                   | 默认确定性实现| 默认确定性实现 |
| [aclnnSinh&aclnnInplaceSinh](../../math/sinh/docs/aclnnSinh&aclnnInplaceSinh.md) | 对输入Tensor完成sinh运算。                                   | 默认确定性实现| - |
| [aclnnSinkhorn](../../math/sinkhorn/docs/aclnnSinkhorn.md)      | 计算Sinkhorn距离，可以用于MoE模型中的专家路由。              | 默认确定性实现| - |
| [aclnnSlice](../../conversion/slice/docs/aclnnSlice.md) | 在指定维度dim上，根据给定的范围[start,end]和步长step，从输入张量self中提取子张量out。 | 默认确定性实现| 默认确定性实现|
| [aclnnSliceV2](../../conversion/strided_slice_v3/docs/aclnnSliceV2.md) | 根据给定的维度axes、范围[starts,ends]和步长steps，从输入张量self中提取张量out。 | 默认确定性实现| 默认确定性实现|
| [aclnnSlogdet](../../math/slogdet/docs/aclnnSlogdet.md) | 计算输入self的行列式的符号和自然对数。 | 默认确定性实现|  默认确定性实现 |
| [aclnnSort](../../math/sort/docs/aclnnSort.md)                  | 将输入tensor中的元素根据指定维度进行升序/降序，并且返回对应的index值。 | 默认确定性实现| 默认确定性实现 |
| [aclnnSplitTensor](../../conversion/split_v/docs/aclnnSplitTensor.md) | 将输入self沿dim轴按照splitSections大小均匀切分。             | 默认确定性实现| 默认确定性实现|
| [aclnnSplitWithSize](../../conversion/split_v/docs/aclnnSplitWithSize.md) | 将输入self沿dim轴切分至splitSize中每个元素的大小。           | 默认确定性实现| 默认确定性实现|
| [aclnnSqrt&aclnnInplaceSqrt](../../math/sqrt/docs/aclnnSqrt&aclnnInplaceSqrt.md) | 完成非负数平方根计算，负数情况返回nan。                      | 默认确定性实现| 默认确定性实现|
| [aclnnSquare](../../math/square/docs/aclnnSquare.md) | 为输入张量的每一个元素计算平方值。                      | 默认确定性实现| 默认确定性实现|
| [aclnnStack](../../conversion/pack/docs/aclnnStack.md)          | 沿着新维度连接张量序列。                                     | 默认确定性实现| 默认确定性实现  |
| [aclnnStd](../../math/reduce_std_v2/docs/aclnnStd.md)           | 计算指定维度(dim)的标准差，这个dim可以是单个维度，维度列表或者None。 | 默认确定性实现| 默认确定性实现|
| [aclnnStdMeanCorrection](../../math/reduce_std_with_mean/docs/aclnnStdMeanCorrection.md) | 计算样本标准差和均值。                                       | 默认确定性实现| 默认确定性实现|
| [aclnnStridedSlice](../../conversion/strided_slice/docs/aclnnStridedSlice.md) | 按照指定的起始、结束位置和步长，从输入张量中提取一个子张量。 | - | 默认确定性实现 |
| [aclnnStridedSliceAssignV2](../../conversion/strided_slice_assign_v2/docs/aclnnStridedSliceAssignV2.md) | StridedSliceAssign是一种张量切片赋值操作，它可以将张量inputValue的内容，赋值给目标张量varRef中的指定位置。 | 默认确定性实现| - |
| [aclnnSub&aclnnInplaceSub](../../math/sub/docs/aclnnSub&aclnnInplaceSub.md) | 完成减法计算，被减数按alpha进行缩放。                        | 默认确定性实现| 默认确定性实现 |
| [aclnnSubs&aclnnInplaceSubs](../../math/sub/docs/aclnnSubs&aclnnInplaceSubs.md) | 完成减法计算，被减数按alpha进行缩放。                        | 默认确定性实现| 默认确定性实现 |
| [aclnnSum](../../math/accumulate_nv2/docs/aclnnSum.md)          | 返回输入tensors列表中每个输入tensor依次做add求和。           | 默认确定性实现| - |
| [aclnnSvd](../../math/svd/docs/aclnnSvd.md)          | 计算一个或多个矩阵的奇异值分解。           | 默认确定性实现| 默认确定性实现 |
| [aclnnSWhere](../../math/select/docs/aclnnSWhere.md)            | 根据条件选取self或other中元素并返回（支持广播）。            | 默认确定性实现| 默认确定性实现 |
| [aclnnTan&aclnnInplaceTan](../../math/tan/docs/aclnnTan&aclnnInplaceTan.md) | 它计算输入张量self中每个元素的正切值，并将结果存储在输出张量out中。 | 默认确定性实现| 默认确定性实现 |
| [aclnnTanh&aclnnInplaceTanh](../../math/tanh/docs/aclnnTanh&aclnnInplaceTanh.md) | 激活函数。返回与输入tensor shape相同的tensor，对输入tensor进行elementwise的计算。 | 默认确定性实现| 默认确定性实现|
| [aclnnTanhBackward](../../math/tanh_grad/docs/aclnnTanhBackward.md) | aclnnTanh的反向。                                            | 默认确定性实现| 默认确定性实现 |
| [aclnnTopk](../../math/topk/docs/aclnnTopk.md) | 返回输入Tensor在指定维度上的k个极值及索引。                                            | 默认确定性实现| 默认确定性实现 |
| [aclnnTrace](../../math/trace/docs/aclnnTrace.md) | 计算矩阵从左上角开始的主对角线元素的和。                                            | 默认确定性实现| 默认确定性实现 |
| [aclnnTransformBiasRescaleQkv](../../math/transform_bias_rescale_qkv/docs/aclnnTransformBiasRescaleQkv.md) | TransformBiasRescaleQkv算子是一个用于处理多头注意力机制中查询（Query）、键（Key）、值（Value）向量的接口。 | 默认确定性实现| 默认确定性实现   |
| [aclnnTransConvolutionWeight](../../conversion/trans_data/docs/aclnnTransConvolutionWeight.md) | 需要和aclnnCalculateConvolutionWeightSize接口配套使用，用于创建一个对于Convolution算子计算性能亲和的weight Tensor。 | 默认确定性实现| - |
| [aclnnTransMatmulWeight](../../conversion/trans_data/docs/aclnnTransMatmulWeight.md) | 需要和aclnnCalculateMatmulWeightSize、aclnnCalculateMatmulWeightSizeV2接口配套使用，用于创建一个对于Matmul算子计算性能亲和的weight Tensor。 | 默认确定性实现| 默认确定性实现  |
| [aclnnTriangularSolve](../../math/triangular_solve/docs/aclnnTriangularSolve.md) | 求解一个具有方形上或下三角形可逆矩阵A和多个右侧b的方程组。   | 默认确定性实现| 默认确定性实现|
| [aclnnTril&aclnnInplaceTril](../../conversion/tril/docs/aclnnTril&aclnnInplaceTril.md) | 将输入的self张量的最后二维（按shape从左向右数）沿对角线的右上部分置零。   | 默认确定性实现| 默认确定性实现  |
| [aclnnTriu&aclnnInplaceTriu](../../conversion/triu/docs/aclnnTriu&aclnnInplaceTriu.md) | 将输入的self张量的最后二维（按shape从左向右数）沿对角线的左下部分置零。   | 默认确定性实现| 默认确定性实现  |
| [aclnnTrunc&aclnnInplaceTrunc](../../math/trunc/docs/aclnnTrunc&aclnnInplaceTrunc.md) | 对输入Tensor完成trunc运算（将数字的小数部分截去，返回整数部分）。 | 默认确定性实现| 默认确定性实现|
| [aclnnUnfoldGrad](../../conversion/unfold_grad/docs/aclnnUnfoldGrad.md) | 实现Unfold算子的反向功能，计算相应的梯度。                   | 默认确定性实现| 默认确定性实现  |
| [aclnnVar](../../math/reduce_var/docs/aclnnVar.md)  | 返回输入Tensor指定维度的值求得的方差。                       | 默认确定性实现| 默认确定性实现|
| [aclnnVarCorrection](../../math/reduce_var/docs/aclnnVarCorrection.md)              | 返回输入Tensor指定维度的值求得的方差。                      | 默认确定性实现| 默认确定性实现  |
| [aclnnVarMean](../../math/reduce_var/docs/aclnnVarMean.md)      | 返回输入Tensor指定维度的值求得的均值及方差。                 | 默认确定性实现| 默认确定性实现|
| [aclnnWeightQuantPreprocess](../../conversion/weight_quant_preprocess/docs/aclnnWeightQuantPreprocess.md) | 完成伪量化Matmul的参数预处理。 | - | 默认确定性实现 |
| [aclnnXlog1py](../../math/xlog1py/docs/aclnnXlog1py.md) | 计算x * log(1 + y)，当x == 0时结果为0。支持broadcast。 | 默认确定性实现| 默认确定性实现 |
| [aclnnAmpUpdateScale](../../math/amp_update_scale/docs/aclnnAmpUpdateScale.md) | 实现AMP训练中的动态Scale更新逻辑。 | 默认确定性实现 | 默认确定性实现 |
| [aclnnXLogYScalarOther&aclnnInplaceXLogYScalarOther](../../math/xlogy/docs/aclnnXLogYScalarOther&aclnnInplaceXLogYScalarOther.md) | 计算self * log(other)的结果。                                | 默认确定性实现| - |
| [aclnnXLogYScalarSelf](../../math/xlogy/docs/aclnnXLogYScalarSelf.md) | 计算self * log(other)的结果。                                | 默认确定性实现| - |
| [aclnnXLogYTensor&aclnnInplaceXLogYTensor](../../math/xlogy/docs/aclnnXLogYTensor&aclnnInplaceXLogYTensor.md) | 计算self * log(other)的结果。                                | 默认确定性实现| - |
| [aclRfft1D](../../math/rfft1_d/docs/aclRfft1D.md)               | 对输入张量self进行RFFT（傅里叶变换）计算，输出是一个包含非负频率的复数张量。 | 默认确定性实现| - |
| [aclStft](../../math/stft/docs/aclStft.md)                      | 计算输入在滑动窗口内的傅里叶变换。                           | 默认确定性实现| - |
