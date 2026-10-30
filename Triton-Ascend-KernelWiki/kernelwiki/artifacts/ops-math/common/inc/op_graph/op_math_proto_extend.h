/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file op_math_proto_extend.h
 * \brief
 */
#ifndef OPS_OP_MATH_PROTO_EXTEND_H_
#define OPS_OP_MATH_PROTO_EXTEND_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief Inserts a dimension of 1 into a tensor's shape. Only the tensor shape is changed, without changing the data. \n

*@par Inputs:
*@li x: A tensor.
*@li axis: The dimension index at which to expand. \n

*@par Outputs:
*y: A tensor with the same data as input, with an additional dimension inserted at the index specified by axis. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ExpandDims.
*/
REG_OP(ExpandDims)
    .INPUT(x, TensorType::ALL())
    .INPUT(axis, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(ExpandDims)

    /**
    * @brief Draws binary random numbers (0 or 1) from a Bernoulli distribution.The input tensor
    * should be a tensor containing probabilities p (a value in the range [0, 1]) to be used for
    * drawing the binary random number, where an output of 1 is produced with ptobability p and
    * an output of 0 is produced with probability (1-p). \n

    * @par Inputs:
    include:
    * @li x: All values in input have to be in the range:[0, 1].
    * @li seed: If seed is set to be -1, and offset is set to be 0, the random number
    * generator is seeded by a random seed. Otherwise, it is seeded by the given seed.
    * @li offset: To avoid seed collision. \n

    * @par Attributes:
    * @li dtype: The data type for the elements of the output tensor. if not specifed,
    * we will use the data type of the input tensor. \n

    * @par Outputs:
    * y: The returned output tensor only has values 0 or 1, same shape as input tensor. \n

    * @par Third-party framework compatibility
    * Compatible with the Onnx operator Bernoulli
    */
    REG_OP(StatelessBernoulliV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(seed, TensorType({DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL,
                           DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .ATTR(dtype, Type, DT_UNDEFINED)
    .OP_END_FACTORY_REG(StatelessBernoulliV2)

    /**
    *@brief Element-wise computes the bitwise left-shift of x and y . \n

    *@par Inputs:
    *Input "x" is a k-dimensional tensor. Inputs "num_lower" and "num_upper"
    are 0D scalars.
    * @li x: A Tensor. Must be one of the following types: int8, int16, int32,
    int64, uint8, uint16, uint32, uint64.
    * @li y: A Tensor. Has the same type as "x".  \n

    *@par Outputs:
    * z: A Tensor. Has the same type as "x".  \n

    *@attention Constraints:
    *Unique runs on the Ascend AI CPU, which delivers poor performance.  \n

    *@par Third-party framework compatibility
    *Compatible with the TensorFlow operator LeftShift.
    */

    REG_OP(LeftShift)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .INPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(z, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OP_END_FACTORY_REG(LeftShift)

    /**
    *@brief Computes a 2D Correlation given 4D "x" and "filter" tensors.
    *
    *@par Inputs:
    * @li filter: A 4D tensor of filters.
    * @li x: A 4D tensor of input images, batch number must equal to batch
    * number of "filter", and channel must equal to channel of "filter".
    *
    *@par Attributes:
    * @li groups: set correlation mode, must be 1 or channel.
    *
    *@par Outputs:
    *y: A Tensor. Has the same type as "x".

    *@par Third-party framework compatibility
    * Compatible with caffe correlation custom operator.
    */
    REG_OP(Correlation)
    .INPUT(filter, TensorType({DT_FLOAT16, DT_INT8}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32}))
    .ATTR(groups, Int, 1)
    .OP_END_FACTORY_REG(Correlation)

    /**
    * @brief Returns a one-hot tensor. The locations represented by index in "x" take value "on_value",
    *         while all other locations take value "off_value" .

    * @par Inputs:
    * Three inputs, including:
    * @li x: A Tensor of indices. Must be one of the following types: uint8, int32.
    * @li on_value: A scalar. The value to fill in output when indices[j] = i,
    *     Must be one of the following types: float16, float32, int32, int8, uint8.
    * @li off_value: A scalar. The value to fill in output when indices[j] != i,
    *     Has the same type as "on_value" . \n

    * @par Attributes:
    * @li depth: An required int to specify the depth of the one hot dimension.
    * @li axis: An int. The axis to fill. Defaults to "-1" . \n

    * @par Outputs:
    * y: A Tensor. Has the same type as "on_value" . \n

    * @par Third-party framework compatibility:
    * Compatible with the TensorFlow operator OneHot.
    *
    * @par Restrictions:
    * Warning: THIS FUNCTION IS DEPRECATED. Please use OneHot instead.
    */
    REG_OP(OneHotD)
    .INPUT(x, TensorType({DT_UINT8, DT_INT32}))
    .INPUT(on_value, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT8, DT_INT8}))
    .INPUT(off_value, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT8, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT8, DT_INT8}))
    .REQUIRED_ATTR(depth, Int)
    .ATTR(axis, Int, -1)
    .OP_END_FACTORY_REG(OneHotD)

    /**
    * @brief Returns a batched diagonal tensor with given batched diagonal values .

    * @par Inputs:
    * Five inputs, including:
    * @li diagonal: Rank `r`, where `r >= 1`. \n

    * @li k:
    * Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
    * diagonal, and negative value means subdiagonals. `k` can be a single integer
    * (for a single diagonal) or a pair of integers specifying the low and high ends
    * of a matrix band. `k[0]` must not be larger than `k[1]`. \n

    * @li num_rows:
    * The number of rows of the output matrix. If it is not provided, the op assumes
    * the output matrix is a square matrix and infers the matrix size from k and the
    * innermost dimension of `diagonal`. \n

    * @li num_cols: An NCHW, NHWC, or ND Tensor.
    * The number of columns of the output matrix. If it is not provided, the op
    * assumes the output matrix is a square matrix and infers the matrix size from
    * k and the innermost dimension of `diagonal`. \n

    * @li padding_value: The number to fill the area outside the specified diagonal band with. \n

    * @par Outputs:
    * output: Has rank `r+1` when `k` is an integer or `k[0] == k[1]`, rank `r` otherwise . \n

    * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator ScatterUpdate.
    */
    REG_OP(MatrixDiagV2)
    .INPUT(diagonal, TensorType::BasicType())
    .INPUT(k, TensorType({DT_INT32}))
    .INPUT(num_rows, TensorType({DT_INT32}))
    .INPUT(num_cols, TensorType({DT_INT32}))
    .INPUT(padding_value, TensorType::BasicType())
    .OUTPUT(output, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagV2)

    /**
    *@brief Add tensor with value.

    *@par Inputs:
    *One input, including: \n
    * x: A ND Tensor. Must be one of the following types:int32,int16, float16, float32, bfloat16, int64. \n

    *@par Attributes:
    *value: A scale. Must be float. \n

    *@par Outputs:
    *y: A ND Tensor. Has the same dtype and shape as "x1". \n

    *@par Third-party framework compatibility:
    * Compatible with the PyTorch operator adds.
    *@attention Constraints:
    * For parameters of the float32 type, there is no precision loss. For INT32 and INT64 parameters,
    * precision loss occurs when the parameter value exceeds 2^24. it is recommended to use Add.
    */
    REG_OP(Adds)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_FLOAT16, DT_BF16, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_FLOAT16, DT_BF16, DT_INT64}))
    .REQUIRED_ATTR(value, Float)
    .OP_END_FACTORY_REG(Adds)

    /**
    *@brief Draws samples from a multinomial distribution .

    *@par Inputs:
    *Inputs include:
    * @li logits: A Tensor. Must be one of the following types: float16, float, double.
    2-D Tensor with shape [batch_size, num_classes].
    * @li num_samples: A Tensor of type int32. 0-D. Number of independent samples to draw for each row slice . \n

    *@par Attributes:
    *@li output_dtype: An optional type from: int32, int64. Defaults to int64.
    *@li seed: An optional int. Defaults to 0.
    *@li seed2: An optional int. Defaults to 0 . \n

    *@par Outputs:
    *y_indices: A Tensor of type output_dtype . \n

    *@attention Constraints:
    *The implementation for Multinomial on Ascend uses AICPU, with bad performance.

    *@par Third-party framework compatibility
    *@li compatible with tensorflow Multinomial operator.
    */
    REG_OP(Multinomial)
    .INPUT(logits, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(num_samples, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Type, DT_INT64)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(Multinomial)

    /**
    * @brief Concatenates a list of "N" tensors along "concat_dim".
            All input of the PhonyConcat are allocated with the same memory block.
            GE calculates the memory offset of each input,
            and the custom operators before write the memory by offset.
    * Warning: This operator is used only to identify that the GE allocates continuous memory and does not perform any
    calculation.

    * @par Inputs:
    * Dynamic input: A list of input tensors. Has the same type and format as "x" .
    * @li x:An ND Tensor.
    * Must be one of the following types: float16, float32, int32, int8, int16,
      int64, uint8, uint16, uint32, uint64, bool, bfloat16.

    * @par Attributes:
    * @li concat_dim: A required list of int32. Specifies the dimensions along which to concat. No default value.
    * @li N: A required list of int32. Specifies the number of concat tensors. No default value .
    * @li keep_input_offset: A optional Bool. Specifies whether calculate the memory offset of input tensors. Default
    True .

    * @par Outputs:
    * @li y:One output.

    * @attention Constraints:
    * @li "concat_dim" is in the range [-len(x.shape), (x.shape)-1] .
    * @li "N" is greater than or equals to 1.

    * @par Third-party framework support
    * Support ONNX  framework.
    */
    REG_OP(PhonyConcat)
    .DYNAMIC_INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(concat_dim, ListInt)
    .REQUIRED_ATTR(N, ListInt)
    .ATTR(keep_input_offset, Bool, true)
    .OP_END_FACTORY_REG(PhonyConcat)

    /**
    * @brief Splits a tensor along dimension "split_dim" into "num_split" smaller tensors.
            All outputs of the PhonySplit are allocated with the same memory block.
            GE calculates the memory offset of each output,
            and the custom operators next read the memory by offset.
    * Warning: This operator is used only to identify that the GE allocates continuous memory and does not perform any
    calculation.

    * @par Inputs:
    * One input:
    * @li x:An ND Tensor.
    * Must be one of the following types: float16, float32, int32, int8, int16,
      int64, uint8, uint16, uint32, uint64, bool, bfloat16.

    * @par Attributes:
    * @li split_dim: A required int32. Specifies the dimension along which to split. No default value.
    * @li num_split: A required int32. Specifies the number of output tensors. No default value .
    * @li keep_output_offset: A optional Bool. Specifies whether calculate the memory offset of outpute tensors. Default
    True .

    * @par Outputs:
    * @li y:Dynamic output. A list of output tensors. Has the same type and format as "x" .

    * @attention Constraints:
    * @li "num_split" is greater than or equals to 1.
    * @li "num_split" is divisible by the size of dimension "split_dim".
    * @li "split_dim" is in the range [-len(x.shape), (x.shape)-1] .

    * @par Third-party framework support
    * Support ONNX  framework.
    */
    REG_OP(PhonySplit)
    .INPUT(x, TensorType::ALL())
    .DYNAMIC_OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(split_dim, ListInt)
    .REQUIRED_ATTR(num_split, ListInt)
    .ATTR(keep_output_offset, Bool, true)
    .OP_END_FACTORY_REG(PhonySplit)

    /**
    *@brief Outputs random integers from a uniform distribution. \n

    *@par Inputs:
    *Inputs include:
    * @li shape: A Tensor. Must be one of the following types: int32, int64. The shape of the output tensor.
    * @li min: A Tensor. Must be one of the following types: int32, int64. 0-D.
    * @li max: A Tensor. Must have the same type as min. 0-D . \n

    *@par Attributes:
    *@li seed: An optional int. Defaults to 0. If either seed or seed2 are set to be non-zero,
    the random number generator is seeded by the given seed. Otherwise, it is seeded by a random seed.
    *@li seed2: An optional int. Defaults to 0 . A second seed to avoid seed collision. \n

    *@par Outputs:
    *y: A Tensor. Has the same type as min. \n

    *@attention Constraints:
    *The implementation for RandomUniformInt on Ascend uses AICPU, with bad performance.

    *@par Third-party framework compatibility
    *@li compatible with tensorflow RandomUniformInt operator.
    */
    REG_OP(RandomUniformInt)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(min, TensorType({DT_INT32, DT_INT64}))
    .INPUT(max, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomUniformInt)

    /**
    *@brief Outputs random values from a uniform distribution. \n

    *@par Inputs:
    *Inputs include:
    *shape: A Tensor. Must be one of the following types: int32, int64. The shape of the output tensor. \n

    *@par Attributes:
    *@li dtype: A type from: float16, float32, double, bfloat16. The type of the output.
    *@li seed: An optional int. Defaults to 0. If either seed or seed2 are set to be non-zero,
    the random number generator is seeded by the given seed. Otherwise, it is seeded by a random seed.
    *@li seed2: An optional int. Defaults to 0 . A second seed to avoid seed collision. \n

    *@par Outputs:
    *y: A Tensor of type float32, float16, double, bfloat16. \n

    *@attention Constraints:
    *The implementation for RandomUniform on Ascend uses AICPU, with bad performance.

    *@par Third-party framework compatibility
    *@li compatible with tensorflow RandomUniform operator.
    */
    REG_OP(RandomUniform)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomUniform)

    /**
    * @brief Returns the lower or upper triangular part of a matrix (2-D tensor) or batch of matrices input \n

    *@par Inputs:
    * @li x: A tensor, which supports 2-8 dimensions or be empty. Must be one of the following types:
    * float16, bfloat16, float32, double, int32, uint8, int16, int8, int64,
    * qint8, quint8, qint32, quint16, qint16, uint16, uint32, uint64, bool. \n

    * @li k: An optional input tensor indicates the diagonal to consider. Must be int32 or int64 type. If no input,
    * will be considerer as 0.

    * @par Attributes:
    * upper: An attribute indicates which part of the triangular matrix to be returned. Must be int32 type.
    * If upper is 1, returns the upper triangular matrix,
    * else if upper is 0, returns the lower triangular matrix. Default is 0. \n

    * @par Outputs:
    * y: A tensor. Has the same type and shape as "x" . \n

    */
    REG_OP(Trilu)
    .INPUT(x, "T")
    .OPTIONAL_INPUT(k, "T_K")
    .ATTR(upper, Int, 0)
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, DT_INT8, DT_INT64,
                             DT_QINT8, DT_QUINT8, DT_QINT32, DT_QUINT16, DT_QINT16, DT_UINT16, DT_UINT32, DT_UINT64,
                             DT_BOOL}))
    .DATATYPE(T_K, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(Trilu)

    /**
    *@brief Outputs random values from a normal distribution. \n

    *@par Inputs:
    *Inputs include:
    *shape: A Tensor. Must be one of the following types: int32, int64. The shape of the output tensor. \n

    *@par Attributes:
    *@li dtype: A type from: float16, float32, double, bfloat16. The type of the output.
    *@li seed: An optional int. Defaults to 0. If either seed or seed2 are set to be non-zero,
    the random number generator is seeded by the given seed. Otherwise, it is seeded by a random seed.
    *@li seed2: An optional int. Defaults to 0 . A second seed to avoid seed collision. \n

    *@par Outputs:
    *y: A Tensor of type float32, float16, double, bfloat16. \n

    *@attention Constraints:
    *The implementation for RandomStandardNormal on Ascend uses AICPU, with bad performance.

    *@par Third-party framework compatibility
    *@li compatible with tensorflow RandomStandardNormal operator.
    */
    REG_OP(RandomStandardNormal)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .REQUIRED_ATTR(dtype, Type)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomStandardNormal)

    /**
    *@brief This operation convert output dataType and shape.

    *@par Inputs:
    *The input handle must have the resource type. Inputs include:
    *x:A list of Tensor objects. One or more tensors from which
    the enqueued tensors should be taken . \n

    *@par Outputs:
    *y:A list of Tensor objects. One or more tensors from which
    the enqueued tensors should be taken . \n

    *@par Attributes:
    *type: An optional ge::DataType. It refers to the target data type of outputs . \n
    *keepdim: If True, output dims equal intput dims. otherwise output dims can append. default value is false.  \n

    *@par Third-party framework compatibility
    *Compatible with tensorflow QueueIsClosed operator.
    */
    REG_OP(Bitcast)
    .INPUT(x, TensorType({DT_BOOL,       DT_FLOAT16, DT_FLOAT,  DT_INT4,   DT_INT8,    DT_INT32,  DT_UINT32,
                          DT_UINT8,      DT_INT64,   DT_UINT64, DT_INT16,  DT_UINT16,  DT_DOUBLE, DT_COMPLEX64,
                          DT_COMPLEX128, DT_QINT8,   DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32}))
    .OUTPUT(y, TensorType({DT_BOOL,       DT_FLOAT16, DT_FLOAT,  DT_INT4,   DT_INT8,    DT_INT32,  DT_UINT32,
                           DT_UINT8,      DT_INT64,   DT_UINT64, DT_INT16,  DT_UINT16,  DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128, DT_QINT8,   DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32}))
    .REQUIRED_ATTR(type, Type)
    .ATTR(keep_dim, Bool, false)
    .OP_END_FACTORY_REG(Bitcast)

    /**
    *@brief Computes the Cholesky decomposition of one or more square matrices . \n

    *@par Inputs:
    *The input x has to be symmetric and positive definite.Inputs include:
    *x:A Tensor. Must be one of the following types: double, float32, float16,
    complex64, complex128. Shape is [..., M, M] . \n

    *@par Outputs:
    *y:A Tensor. Has the same type as x . \n

    *@attention Constraints:
    *The input x is a tensor of shape [..., M, M] whose inner-most 2 dimensions
    form square matrices.

    *@par Third-party framework compatibility
    *Compatible with tensorflow Cholesky operator.
    */
    REG_OP(Cholesky)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Cholesky)

    /**
    * @brief Computes the confusion matrix from predictions and labels .

    * @par Inputs:
    * Three inputs, including:
    * @li labels: A Tensor. Must be one of the following types: float16, float32,
    * int32, int8, uint8. 1D. Has format ND.
    * @li predictions: A Tensor. Must be one of the following types: float16,
    * float32, int32, int8, uint8. 1D. Has format ND.
    * @li weights: A optional Tensor. Must be one of the following types: float16, float32,
    * int32, int8, uint8. 1D. Has format ND. \n

    * @par Attributes:
    * @li num_classes: An integer for the shape of the output matrix.
    * @li dtype: Data type of the confusion matrix. \n

    * @par Outputs:
    * y: A Tensor. 1D. Has format ND. Has the same type and format as input "labels" . \n

    * @attention Constraints:
    * @li "weights", "labels", and "predictions" are 1D tensors.
    * @li The output is with shape (num_classes, num_classes),
    * where, 1 <= num_classes <= 4096 . \n

    * @see Region()

    * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator ConfusionMatrix.
    */
    REG_OP(ConfusionMatrix)
    .INPUT(labels, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_INT8, DT_UINT8}))
    .INPUT(predictions, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_INT8, DT_UINT8}))
    .OPTIONAL_INPUT(weights, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_INT8, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_FLOAT16, DT_INT8, DT_UINT8}))
    .REQUIRED_ATTR(num_classes, Int)
    .REQUIRED_ATTR(dtype, String)
    .OP_END_FACTORY_REG(ConfusionMatrix)

    /**
    * @brief Computes the cumulative product of the tensor "x" along "axis" .

    * @par Inputs:
    * Two inputs, including:
    * @li x: A Tensor. Must be one of the following types:
    * double, float32, float16, bfloat16, complex32, complex64, complex128,
    * int8, uint8, int16, uint16, int32, uint32, int64, uint64, qint8, quint8, qint32.
    * @li axis: A Tensor of type int32 or int64. Range is [-rank(x),rank(x)). Defaults to "0".
    *
    * @par Attributes:
    * @li exclusive: If "False", performs inclusive cumprod, which means that the first element of the input
    * is identical to the first element of the output. If "True", performs exclusive cumprod.
    * @li reverse: A bool. Defaults to "False".
    *
    * @par Outputs:
    * y: A Tensor. Has the same type as "x".
    * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator Cumprod.
    */
    REG_OP(Cumprod)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axis, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(exclusive, Bool, false)
    .ATTR(reverse, Bool, false)
    .OP_END_FACTORY_REG(Cumprod)

    /**
    * @brief Computes the cumulative log sum exp of the tensor "x" along "axis" .

    * @par Inputs:
    * Two inputs, including:
    * @li x: A Tensor. Must be one of the following types: float32, float16.
    * @li axis A Tensor of type int32 or int16. Defaults to "0".
    *
    * @par Attributes:
    * @li exclusive: If "False", performs inclusive CumulativeLogsumexp,
    * which means that the first element of the input is identical to the first element of the output.
    * If "True", performs exclusive CumulativeLogsumexp.
    * @li reverse: A bool. Defaults to "False".
    *
    * @par Outputs:
    * y: A Tensor. Has the same type as "x".
    * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator Cumsum.
    */
    REG_OP(CumulativeLogsumexp)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
    .INPUT(axis, TensorType({DT_INT32, DT_INT16}))
    .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
    .ATTR(exclusive, Bool, false)
    .ATTR(reverse, Bool, false)
    .OP_END_FACTORY_REG(CumulativeLogsumexp)

    /**
    * @brief Generate DSA random bit mask for dropout. \n

    * @par Inputs:
    * @li count:The shape of the input tensor. Must be int64.
    * @li seed:If seed is set to be non-zero, the random number
    * generator is seeded by the given seed. Otherwise, it is seeded by a random seed. Must be int64.
    * @li dropout:0-D. Number of bit 1. Must be one of the following dtypes:float16, float32, bf16. Must in [0,1). \n

    * @par Attributes:
    * @li random_algorithm:The default value is "Philox".
    * @li output_dtype:The dtype of output. The default value is "uint1". \n

    * @par Outputs:
    * y:If the dtype of y is uint8, output (1-D) random number using uint8 data format. Else if the dtype of y is uint1,
    * output random number with the shape of count using uint1 data format.
    * Must be one of the following types:uint1, uint8.\n

    * @see DSAGenBitMask()
    */
    REG_OP(DSAGenBitMask)
    .INPUT(count, TensorType({DT_INT64}))
    .INPUT(seed, TensorType({DT_UINT64}))
    .INPUT(dropout, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(out, TensorType({DT_UINT1, DT_UINT8}))
    .ATTR(random_algorithm, String, "Philox")
    .ATTR(output_dtype, String, "uint8")
    .OP_END_FACTORY_REG(DSAGenBitMask)

    /**
    * @brief Generate DSA normal data in random. \n

    * @par Inputs:
    include:
    * @li count: The shape of the input tensor.
    * @li seed: If seed is set to be non-zero, the random number
    * generator is seeded by the given seed. Otherwise, it is seeded by a random seed
    * @li mean: A Tensor. Must be one of the following types: float16, float32, double
    * @li stdev: A Tensor. Must be one of the following types: float16, float32, double.
    * @li Parameters mean and stdev must be configured at the same time. \n
    * @par Attributes:
    * @li random_algorithm:The default value is "Philox". \n

    * @par Outputs:
    * y:Output (1-D) random number using float and bf data format . \n

    * @see DSARandomNormal()
    */
    REG_OP(DSARandomNormal)
    .INPUT(count, TensorType({DT_INT64}))
    .INPUT(seed, TensorType({DT_UINT64}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(stdev, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(out, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .ATTR(random_algorithm, String, "Philox")
    .OP_END_FACTORY_REG(DSARandomNormal)

    /**
    * @brief Generate DSA uniform data in random. \n

    * @par Inputs:
    include:
    * @li count: The shape of the input tensor.
    * @li seed: If seed is set to be non-zero, the random number
    * generator is seeded by the given seed. Otherwise, it is seeded by a random seed
    * @li low: A Tensor. Must be one of the following types: int, float, bf
    * @li high: A Tensor. Must be one of the following types: int, float, bf.
    * @li Parameters low and high must be configured at the same time, and low < high. \n

    * @par Attributes:
    * @li random_algorithm:The default value is "Philox". \n

    * @par Outputs:
    * y:Output (1-D) random number using float int and bf data format . \n

    * @see DSARandomUniform()
    */
    REG_OP(DSARandomUniform)
    .INPUT(count, TensorType({DT_INT64}))
    .INPUT(seed, TensorType({DT_UINT64}))
    .INPUT(low, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64}))
    .INPUT(high, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64}))
    .OUTPUT(out, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64}))
    .ATTR(random_algorithm, String, "Philox")
    .OP_END_FACTORY_REG(DSARandomUniform)

    /**
    * @brief Counts the number of occurrences of each value in an integer array.

    * @par Inputs:
    * @li input: A Tensor of type int32, int64. 1D or 2D int Tensor.
    * @li size: A Tensor. Must have the same type as input. non-negative int scalar Tensor.
    * @li weights: A Tensor. Must be one of the following types: int32, int64, float32, float64.
                   with the same shape as input,
                   or a length-0 Tensor, in which case it acts as all weights equal to 1. \n

    * @par Outputs:
    * @li output: A Tensor with length "size" for each stride and has the same dtype as weights. \n

    * @par Attributes:
    * binary_output: An optional bool. Defaults to False. bool;
                     Whether the kernel should count the appearance or number of occurrences. \n

    * @attention Constraints:
    * The operator will use the interface set_atomic_add(), therefore weights and output should be float32 only. \n

    * @par Third-party framework compatibility
    * Compatible with tensorflow DenseBincount operator.
    */
    REG_OP(DenseBincount)
    .INPUT(input, TensorType({DT_INT32, DT_INT64}))
    .INPUT(size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(weights, TensorType(DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE))
    .OUTPUT(output, TensorType(DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE))
    .ATTR(binary_output, Bool, false)
    .OP_END_FACTORY_REG(DenseBincount)

    /**
    * @brief Flattens the inputs tensor into a 2D matrix. If input tensor has shape (d_0, d_1,..., d_n),
    *        then the output will have shape (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis + 1)...X d_n)\n

    * @par Inputs:
    * One input:
    * x: A multi-dimensional tensor. All data types are supported.

    * @par Outputs:
    * y: A 2D flattened tensor with the contents of the input tensor, with input dimensions up to axis flattened
    * to the outer dimension of the output and remaining input dimensions flattened into the inner dimension of the
    output.
    * Has the same type as "x".

    * @par Attributes:
    * axis: A optional int32, default value is 1. Indicate up to which input dimensions (exclusive) should be flattened
    * to the outer dimension of the output. The value for axis must be in the range [-r, r], where r is the rank of
    * the input tensor. Negative value means counting dimensions from the back. When axis = 0, the shape of
    * the output tensor is (1, (d_0 X d_1 ... d_n), where the shape of the input tensor is (d_0, d_1, ... d_n).

    * @par Third-party framework compatibility
    * Compatible with TensorFlow / ONNX operator Flatten.
    */
    REG_OP(Flatten)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, Int, 1)
    .OP_END_FACTORY_REG(Flatten)

    /**
    * @brief Says whether the targets are in the top "k" predictions . \n

    * @par Inputs:
    * @li x1: A 2D Tensor of type float32. A "batch_size * classes" tensor.
    * @li x2: A 1D Tensor of type IndexNumberType. A batch_size tensor of class ids.
    * @li k: A 1D Tensor of the same type as "x2".
    * Specifies the number of top elements to look at for computing precision . \n

    * @par Outputs:
    * y: A Tensor of type bool . \n

    * @attention Constraints:
    * @li x2 must be non-negative tensor.

    * @par Third-party framework compatibility
    * @li Compatible with the TensorFlow operator InTopKV2.
    */
    REG_OP(InTopK)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType(IndexNumberType))
    .INPUT(k, TensorType({IndexNumberType}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(InTopK)

    /**
    * @brief Adds "v" into specified rows of "x".
    * Computes y = x; y[i, :] += v.
    * @par Inputs:
    * Three inputs, including:
    * @li x: A Tensor.
    *     TensorType::BasicType(), Format is ND.
    * @li indices: A vector of type int32.
    *     Indices into the left-most dimension of "x".
    * @li v: A Tensor of the same type as "x".
    *     Same dimension sizes as x except the first dimension,
    *     which must be the same as the size of "indices" . \n

    * @par Outputs:
    * y: A Tensor of the same type as "x".
    *  An alias of "x". The content of "y" is undefined if there are duplicates in indices.
    * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator InplaceAdd.
    */
    REG_OP(InplaceAdd)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(v, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(InplaceAdd)

    /**
    * @brief Subtracts "v" into specified rows of "x".
    * Computes y = x; y[i, :] -= v; return y.
    * @par Inputs:
    ** Three inputs, including:
    * @li x: A Tensor. TensorType::BasicType(), Format is ND.
    * @li indices: A vector of type int32, Format is ND. Indices into the left-most dimension of x.
    * @li v: A Tensor of the same type as "x", Format is ND.
    * Same dimension sizes as "x" except the first dimension, which must be the same as the size of "indices" . \n

    * @par Outputs:
    * y: A Tensor. Has the same type as "x", Format is ND.
    *  An alias of "x". The content of "y" is undefined if there are duplicates in indices . \n

    * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator InplaceSub.
    */
    REG_OP(InplaceSub)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(v, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(InplaceSub)

    /**
    * @brief Performs transform mask to argmax .

    * @par Inputs:
    * Two inputs:
    * @li x: A 4D Tensor of type float16. supported format list ["NC1HWC0"]
    * @li mask: A 4D Tensor of type uint16. supported format list ["NC1HWC0"]. \n

    * @par Attributes:
    * @li ksize: A required list of int8, int16, int32, or int64 values, specifying the size of the window for each
    dimension of the input tensor.
    * @li strides: A required list of int8, int16, int32, or int64 values, specifying the stride of the sliding window
    for each dimension of the input tensor.
    * @li padding: A required string.
    * @li originshape:A required list of int8, int16, int32, or int64 values. \n

    * @par Outputs:
    * argmax: A 4D Tensor of type int32. supported format list ["NC1HWC0"]. \n

    * @attention Constraints:
    * @li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1, ksize[1] * ksize[2] <= 255.
    * @li "strides" is a list that has length 4: strides[0] = 1 or strides[3] = 1, strides[1] <= 63, strides[0] >= 1,
    strides[2] <= 63, strides[2] >= 1.
    * @li "padding" is either "SAME" or "VALID" . \n

    * @par Third-party framework compatibility
    * Compatible with the TensorFlow operator Mask2Argmax.
    */
    REG_OP(Mask2Argmax)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(mask, TensorType::IndexNumberType())
    .OUTPUT(argmax, TensorType::IndexNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .REQUIRED_ATTR(originshape, ListInt)
    .OP_END_FACTORY_REG(Mask2Argmax)

    /**
    *@brief Shuffle index of no-zero element . \n

    *@par Inputs:
    include:
    *x:A tensor <= 5-D . \n

    *@par Attributes:
    *@li count:the count of output, if 0, out all no-zero elements.
    *@li seed:If either seed or seed2 are set to be non-zero, the random number generator is seeded by the given seed.
              Otherwise, it is seeded by a random seed.
    *@li seed2:A second seed to avoid seed collision . \n

    *@par Outputs:
    *@li y:2-D tensor, no-zero element index.
    *@li mask:1-D, whether the corresponding index is valid . \n

    *@see RandomChoiceWithMask()
    */
    REG_OP(RandomChoiceWithMask)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OUTPUT(mask, TensorType({DT_BOOL}))
    .ATTR(count, Int, 0)
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .OP_END_FACTORY_REG(RandomChoiceWithMask)

    /**
    * @brief Calculates the standard deviation or the variance of Tensors with the average value.

    * @par Inputs:
    * Two inputs, including:
    * @li x: A Tensor. Must be one of the following types: float16, float32, bfloat16. \n
    * @li mean: A Tensor. It's the mean of X. Has the same shape and type as "x" \n

    * @par Attributes:
    * Five Attributes, including:
    * @li dim: The dimensions to reduce. A required listint.
    *     Must be in the range [-rank(x), rank(x)).
    * @li if_std: An optional bool. Defaults to "False".
    *     If "True", Calculate the standard deviation
    *     If "False", Calculate the variance
    * @li unbiased: An optional bool. Defaults to "True".
    *     If "True", Use Bessel Correction.
    *     If "False", Do not use Bessel Correction. \n
    * @li keepdim: An optional bool. Defaults to "False".
    *     If "True", Keep the original tensor dimension.
    *     If "False", Do not keep the original tensor dimension. \n
    * @li correction: An optional int. Defaults to 1.
    *     If unbiased is "True", use Bessel Correction. \n

    * @par Outputs:
    * @li output_var: A Tensor. It's the standard deviation or the variance of X. Has the same type as "x".

    * @par Third-party framework compatibility
    * Compatible with the Pytorch operator Var_mean.
    */
    REG_OP(ReduceStdV2Update)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(output_var, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(dim, ListInt)
    .ATTR(if_std, Bool, false)
    .ATTR(unbiased, Bool, true)
    .ATTR(keepdim, Bool, false)
    .ATTR(correction, Int, 1)
    .OP_END_FACTORY_REG(ReduceStdV2Update)

    /**
    * @brief Calculates the standard deviation of Tensors.

    * @par Inputs:
    * include:
    * @li x: A tensor. Must be one of the following types: float16, float32, bfloat16.
          The format must be NCHW, NHWC, or ND. \n
    * @li mean: A tensor. It's the mean of X. Has the same shape, format and type as "x". \n


    * @par Attributes:
    * Six Attributes, including:
    * @li dim: The dimensions to reduce. An optional listint, Defaults to "None".
    *     If None (the default), reduces all dimensions.
    *     Must be in the range [-rank(x), rank(x)). \n
    * @li unbiased: An optional bool. Defaults to "True".
    *     If "True", Use Bessel Correction.
    *     If "False", Do not use Bessel Correction. \n
    * @li keepdim: An optional bool. Defaults to "False".
    *     If "True", Keep the original tensor dimension.
    *     If "False", Do not keep the original tensor dimension. \n
    * @li invert: An optional bool, Defaults to "False".
    *     If "True", the output is inverse of variance.
    *     If "False", the output is variance.
    * @li epsilon: An optional float, Defaults to 0.001.
    *     Prevent division by 0.
    * @li correction: An optional int. Defaults to 1.
    *     If unbiased is "True", use Bessel Correction. \n

    * @par Outputs:
    * y: A tensor. It's the variance of X or reciprocal of vaiance of X. Has the same type and format as "x".

    * @par Third-party framework compatibility
    * Compatible with the Pytorch operator ReduceStdWithMean.
    */
    REG_OP(ReduceStdWithMean)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(dim, ListInt, {})
    .ATTR(unbiased, Bool, true)
    .ATTR(keepdim, Bool, false)
    .ATTR(invert, Bool, false)
    .ATTR(epsilon, Float, 0.001)
    .ATTR(correction, Int, 1)
    .OP_END_FACTORY_REG(ReduceStdWithMean)

    /**
     * @brief SignBitsPack.

     * @par Inputs:
     * one input, including:
     * x: A 1D Tensor of float32 or float16.
     *
     * @par Attributes:
     * size: first dim value of output tensor. Must be uint8 type.
     *
     * @par Outputs:
     * y: A 2D Tensor of type uint8 with shape (size, N)
     */
    REG_OP(SignBitsPack)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_UINT8}))
    .REQUIRED_ATTR(size, Int)
    .OP_END_FACTORY_REG(SignBitsPack)

    /**
    * @brief SignBitsUnpack.

    * @par Inputs:
    * one input, including:
    * x: A 1D Tensor of uint8.

    * @par Attributes:
    * @li size: dim of out put tensor, defaults to 1. Must be int type.
    * @li dtype: dtype of out put tensor: DT_FLOAT(0) or DT_FLOAT16(1).

    * @par Outputs:
    * y: A 2D Tensor of type float32 (float16) with shape (size, (x.shape * 8) / size),
    */
    REG_OP(SignBitsUnpack)
    .INPUT(x, TensorType({DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(size, Int)
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(SignBitsUnpack)

    /**
    * @brief Assigns "value" to the sliced l-value reference of "var".
    * The values of "value" are assigned to the positions in the variable. "var"
    * that are selected by the slice parameters. The slice parameters "begin, "end",
    * "strides", etc. work exactly as in "StridedSlice" . \n

    * @par Inputs:
    * Five inputs, including:
    * @li var: A mutable ND Tensor of type BasicType.
    * Support Dtype: [float16,float32,int32,int16,bfloat16], Support format: [ND].
    * @li begin: A mutable ND Tensor of type IndexNumberType. Support dtype: [int64], support format: [ND].
    * Specifies the index of the first value to select.
    * @li end: A mutable ND Tensor of type IndexNumberType. Support dtype: [int64], support format: [ND].
    * Specifies the index of the last value to select.
    * @li strides: A mutable ND Tensor of type IndexNumberType. Support dtype: [int64], support format: [ND].
    * Specifies the stride to select.
    * @li input_value: A mutable ND Tensor of type BasicType .
    * Support Dtype: [float16,float32,int32,int16,bfloat16], Support format: [ND]. \n

    * @par Attributes:
    * @li begin_mask: An optional int. Defaults to "0".
    * @li end_mask: An optional int. Defaults to "0".
    * @li ellipsis_mask: An optional int. Defaults to "0".
    * @li new_axis_mask: An optional int. Defaults to "0".
    * @li shrink_axis_mask: An optional int. Defaults to "0" . \n

    * @par Outputs:
    * var: A mutable Tensor. Has the same type as "var" . \n

    * @attention Constraints:
    * This operator currently does not support broadcasting. Therefore, the shape
    * of "value" must be exactly the shape produced by the slice of "var" . \n

    * @see StridedSlice()

    * @par Third-party framework compatibility
    * @li Compatible with the TensorFlow operator StridedSlice.
    */
    REG_OP(StridedSliceAssign)
    .INPUT(var, TensorType(BasicType))
    .INPUT(begin, TensorType(IndexNumberType))
    .INPUT(end, TensorType(IndexNumberType))
    .INPUT(strides, TensorType(IndexNumberType))
    .INPUT(input_value, TensorType(BasicType))
    .OUTPUT(var, TensorType(BasicType))
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OP_END_FACTORY_REG(StridedSliceAssign)

    /**
    *@brief Outputs random values from a truncated normal distribution . \n

    *@par Inputs:
    *Inputs include:
    *shape: A Tensor. Must be one of the following types: int32, int64 . \n

    *@par Attributes:
    *@li seed: An optional int. Defaults to 0.If either `seed` or `seed2`
    are set to be non-zero, the random number generator is seeded by the given
    seed. Otherwise, it is seeded by a random seed.
    *@li seed2: An optional int. Defaults to 0 . A second seed to avoid seed collision. \n
    * @li dtype: An optional attribute, a type from: float16, float32, double, bfloat16. The default type is float32.
    * The corresponding relationshape between the enumeration values and real output type is :
    * 0(float32), 1(float16), 11(double), 27(bfloat16).

    *@par Outputs:
    *y: A Tensor of types: float16, float32, double, bfloat16 . A tensor of the specified shape
    filled with random truncated normal values. \n

    *@attention Constraints:
    *The implementation for TruncatedNormal on Ascend uses AICPU, with bad performance.

    *@par Third-party framework compatibility
    *@li compatible with tensorflow TruncatedNormal operator.
    */
    REG_OP(TruncatedNormal)
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_DOUBLE}))
    .ATTR(seed, Int, 0)
    .ATTR(seed2, Int, 0)
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(TruncatedNormal)

    /**
    *@brief Returns x1 + x2 element-wise.

    *@par Inputs:
    * @li x1: A tensor. Must be one of the following types: bfloat16, float16, float32, float64,
    *     uint8, int8, int16, int32, int64, complex64, complex128.
    * @li x2: A tensor of the same dtype as "x1".
    *
    *@attention Constraints:
    * AddV2 supports broadcasting.
    *
    *@par Outputs:
    * y: A tensor. Has the same dtype as "x1".
    *
    *@par Third-party framework compatibility
    *Compatible with the TensorFlow operator AddV2.
    *
    */
    REG_OP(AddV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX64, DT_BF16, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX64, DT_BF16, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE,
                           DT_COMPLEX64, DT_BF16, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(AddV2)

    /**
    *@brief Calculate the P-norm distance between vectors  function. \n

    *@par Inputs:
    *One inputs, including:
    * x: A tensor. Must be one of the following types:
    *     float16, float32. \n

    *@par Attributes:
    *p: An optional float.Defaults to 2. \n

    *@par Outputs:
    *y: A Tensor with the same type and shape of x. \n

    *@par Third-party framework compatibility
    *Compatible with the Pytorch operator Pdist. \n
    */
    REG_OP(Pdist)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(p, Float, 2.0)
    .OP_END_FACTORY_REG(Pdist)

} // namespace ge

#endif
