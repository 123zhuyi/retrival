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
 * \file view_copy_proto.h
 * \brief
 */
#ifndef OP_PROTO_VIEW_COPY_PROTO_H_
#define OP_PROTO_VIEW_COPY_PROTO_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
*@brief Copy the src tensor to the dst tensor according the special parameter.
*The function of ViewCopy operator is to copy the non-contiguous src tensor to the non-contiguous dst tensor, each defined respectively 
*by a quadruple of parameters: (src, src_storage_offset, src_stride, src_size) and (dst, dst_storage_offset, dst_stride, dst_size) 
*which both correspond to the term (tensor, storage_offset, view_stride, view_shape) and of course must satisfy the constraints regarding view_shape, 
*view_stride, and storage_offset for a valid tensor, details are as follows:
*@li The length of view_shape must equal the length of view_stride: len(view_shape) == len(view_stride).
*@li Each element in the tensor view_stride is non-negative.
*@li The storage_offset must be non-negative and not exceed the bounds of the tensor's size.
*@li All memory accesses must be legal; any element addressed by the view_shape and view_stride parameters must lie within the valid range of 
*the original tensor, in other words, for each axis i of view_stride, storage_offset + (view_shape[i] -1 ) * view_stride[i] <= total element num of tensor.

*@par Inputs:
*Eight inputs, including:
*@li dst: A tensor. Must be one of the following types:
*float32, float16, bfloat16, int8, uint8, int16, uint16,
*int32, uint32, int64, uint64, bool, hifloat8,
*float8_e5m2, float8_e4m3fn.
*@li dst_size: A tensor, the view shape of dst tensor. Must be one of the following types: int32, int64, not negative.
*@li dst_stride: A tensor, the view stride of dst tensor. Must be one of the following types: int32, int64, not negative.
*@li dst_storage_offset: A tensor representing the storage offset of dst tensor. Must be one of the following types: int32, int64, not negative and can't out range of dst tensor.
*@li src: A tensor. Must be one of the following types:
*float32, float16, bfloat16, int8, uint8, int16, uint16,
*int32, uint32, int64, uint64, bool, hifloat8, float8_e5m2, float8_e4m3fn.
*@li src_size: A tensor, the view shape of src tensor. Must be one of the following types: int32, int64, not negative.
*@li src_stride: A tensor, the view stride of src tensor. Must be one of the following types: int32, int64, not negative.
*@li src_storage_offset: A tensor representing the storage offset of src tensor. Must be one of the following types:
*int32, int64, not negative and can't out range of src tensor.
*@par Outputs:
*dst: An ref tensor. Must be one of the following types:
*float32, float16, bfloat16, int8, uint8, int16, uint16,
*int32, uint32, int64, uint64, bool, hifloat8, float8_e5m2, float8_e4m3fn.
*
*@attention Constraints:
*@li dst_size and src_size must have same shape. For example: {'dst_size': (1, 199, 53, 3), 'src_size': (1, 199, 53, 3)}, for each axis i, dst_size[i] and src_size[i] is equal.
*@li The dst_size and dst_stride must have same dimension size. For example: {'dst_size': (192, 1, 16, 4), 'dst_stride': (80, 80, 5, 1)}, dst_size and dst_stride both have the same dimension size 4.
*@li The src_size and src_stride must have same dimension size. For example: {'src_size': (991, 1, 149), 'src_stride': (149, 149, 1)}, src_size and src_stride both have the same dimension size 3.
*@li For each axis i of dst_stride, the result: dst_storage_offset + (dst_size[i] - 1) * dst_stride[i] should less equal than dst tensor size.
*@li For each axis i of src_stride, the result: src_storage_offset + (src_size[i] - 1) * src_stride[i] should less equal than src tensor size.
*/

REG_OP(ViewCopy)
    .INPUT(dst, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(dst_size, TensorType::IndexNumberType())
    .INPUT(dst_stride, TensorType::IndexNumberType())
    .INPUT(dst_storage_offset, TensorType::IndexNumberType())
    .INPUT(src, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(src_size, TensorType::IndexNumberType())
    .INPUT(src_stride, TensorType::IndexNumberType())
    .INPUT(src_storage_offset, TensorType::IndexNumberType())
    .OUTPUT(dst, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(ViewCopy)

} // namespace ge
#endif // OP_PROTO_VIEW_COPY_PROTO_H_