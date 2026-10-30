/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_PROTO_IM2COL_PROTO_H_
#define OP_PROTO_IM2COL_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs Im2col for each batch entry.

* @par Inputs:
* x: A 4D tensor with shape [batch, in_rows, in_cols, depth] in NHWC format or [batch, depth, in_rows, in_cols] in NCHW
* format, must be one of the following types: bfloat16, float16, float32, double, int8, int16, int32, int64, uint8,
* uint16, uint32, uint64, complex32, complex64, bool.
* The following types are valid only for Ascend950 AI Processors and later products: complex32, complex64, bool.
* The inputs must have data_format with one of follows: NHWC, NCHW.

* @par Attributes:
* @li ksizes: A required list or tuple. The size of the sliding window for
* each dimension of images, shape:(2), value: (ksizes_h, ksizes_w).
* @li strides: An optional list or tuple. The distance between the centers of two consecutive patches in the images,
* the size of strides only support 1 or 2 which indicates: \n
* size of strides == 1 : strides_h = strides[0]; strides_w = strides[0]; \n
* size of strides == 2 : strides_h = strides[0]; strides_w = strides[1]; \n
* Defaults to "{1,1}".
* @li dilations: An optional list or tuple, the size of dilations only support 1 or 2 which indicates: \n
* size of dilations == 1 : dilations_h = dilations[0]; dilations_w = dilations[0]; \n
* size of dilations == 2 : dilations_h = dilations[0]; dilations_w = dilations[1]; \n
* Defaults to "{1,1}". \n
* This is the input stride, specifying how far two consecutive patch
* samples are in the input. Equivalent to extracting patches
* with patch_sizes_eff = patch_sizes + (patch_sizes - 1) *
* (dilations - 1), followed by subsampling them spatially
* by a factor of dilations.
* This is equivalent to rate in dilated (a.k.a. Atrous) convolutions.
* @li padding_mode: An optional String. The type of padding algorithm to use,
* support "SAME", "VALID", "CALCULATED". Among the three modes, only the
* "CALCULATED" means to use the pads below. Defaults to "CALCULATED".
* @li pads: An optional list or tuple. The pad distance.
* When padding_mode is equal to "CALCULATED", the size of pads only support 1, 2 or 4 which indicates: \n
* size of pads == 1 : pad_h_top = pad[0]; pad_h_bottom = pad[0]; pad_w_before = pad[0]; pad_w_after = pad[0]; \n
* size of pads == 2 : pad_h_top = pad[0]; pad_h_bottom = pad[0]; pad_w_before = pad[1]; pad_w_after = pad[1]; \n
* size of pads == 4 : pad_h_top = pad[0]; pad_h_bottom = pad[1]; pad_w_before = pad[2]; pad_w_after = pad[3]; \n
* pad_h_top and pad_h_bottom should be added to h-dimension of outputshape. \n
* pad_w_before and pad_w_after should be added to w-dimension of outputshape. \n
* Defaults to "{0,0}".
* @par Outputs:
* y: A 4D tensor has same dtype as "x", with shape
* [batch, out_rows, out_cols, ksize_rows * ksize_cols * depth] in NHWC format or
* [batch, ksize_rows * ksize_cols * depth, out_rows, out_cols] in NCHW format.
* containing image patches with size ksize_rows x
* ksize_cols x depth vectorized in the "depth" dimension.
* Note "out_rows" and "out_cols" are the dimensions of the output patches. \n

* @attention Constraints:
* "ksizes", "strides", "dilations" and "pads" are lists of integers. \n

* @par Third-party framework compatibility
* Compatible with Pytorch Im2col operator.
*/
REG_OP(Im2col)
    .INPUT(x, TensorType({RealNumberType(), DT_BOOL, DT_COMPLEX32, DT_COMPLEX64}))
    .OUTPUT(y, TensorType({RealNumberType(), DT_BOOL, DT_COMPLEX32, DT_COMPLEX64}))
    .REQUIRED_ATTR(ksizes, ListInt)
    .ATTR(strides, ListInt, {1})
    .ATTR(dilations, ListInt, {1})
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0})
    .OP_END_FACTORY_REG(Im2col)
} // namespace ge

#endif
