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
 * \file allto_allv_grouped_mat_mul_proto.h
 * \brief
 */
#ifndef ALLTO_ALLV_GROUPED_MAT_MUL_PROTO_H_
#define ALLTO_ALLV_GROUPED_MAT_MUL_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {


/**
* @brief Fusion of alltoallv and grouped matmul.
* @par Inputs:
* @li gmm_x: A matrix Tensor of shape [BSK, H1]. The data type of elements supports float16, bfloat16 or hifloat8; the format supports ND.
* @li gmm_weight: A matrix Tensor of shape [e, H1, N1]. The data type of elements supports float16, bfloat16 or hifloat8 and should match that of gmm_x; the format supports ND.
* @li send_counts_tensor: A Tensor of shape [e * ep]. The data type of elements supports int32 or int64; the format supports ND.
* @li recv_counts_tensor: A Tensor of shape [e * ep]. The data type of elements supports int32 or int64; the format supports ND.
* @li mm_x: A matrix Tensor of shape [BS, H1]. The data type of elements supports float16, bfloat16 or hifloat8; the format supports ND.
* @li mm_weight: A matrix Tensor of shape [H2, N2]. The data type of elements supports float16, bfloat16 or hifloat8 and should match that of mm_x; the format supports ND.
* @li gmm_x_scale: A matrix Tensor. The type support float32. The format supports ND.
* @li gmm_weight_scale: A matrix Tensor. The type support float32. The format supports ND.
* @li gmm_x_offset: A matrix Tensor. The type support float32. The format supports ND.
* @li gmm_weight_offset: A matrix Tensor. The type support float32. The format supports ND.
* @li mm_x_scale: A matrix Tensor. The type support float32. The format supports ND.
* @li mm_weight_scale: A matrix Tensor. The type support float32. The format supports ND.
* @li mm_x_offset: A matrix Tensor. The type support float32. The format supports ND.
* @li mm_weight_offset: A matrix Tensor. The type support float32. The format supports ND.
*
* @par Attributes:
* @li group: A required String identifying the expert group of ranks.
* @li ep_world_size: A required int identifying the number of expert parallel group rank num.
* @li send_counts: An int list. A list containing amount of data to be sent.
* @li recv_counts: An int list. A list containing amount of data to be received.
* @li trans_gmm_weight: A boolean value. Whether gmm_weight is transposed. True indicates transposition. Default: false.
* @li trans_mm_weight: A boolean value. Whether mm_weight is transposed. True indicates transposition. Default: false.
* @li permute_out_flag: A boolean value. Whether to output permute_out. True indicates that output permute_out is required. Default: false.
* @li gmm_x_quant_mode: An int. Quantization mode of gmm_x. Default: 0.
*        - 0：No Quantization
*        - 1：PerTensor Quantization
*        - 2：PerChannel Quantization
*        - 3：PerToken Quantization
*        - 4：PerGroup Quantization
*        - 5：PerBlock Quantization
*        - 6：Mx Quant Quantization
* @li gmm_weight_quant_mode: An int. Quantization mode of gmm_weight. Default: 0.
* @li mm_x_quant_mode: An int. Quantization mode of mm_x. Default: 0.
* @li mm_weight_quant_mode: An int. Quantization mode of mm_weight. Default: 0.
* @li group_size: An int. Default: 0.
* @li y_dtype: An int. Declare the gmm_y dtype. Default: static_cast<int64_t>(ge::DT_UNDEFINED) is 28.
* @li mm_dtype: An int. Declare the mm_y dtype. Default: static_cast<int64_t>(ge::DT_UNDEFINED) is 28.
*
* @par Outputs:
* @li gmm_y: A matrix tensor of shape [A, N1] containing result of grouped matmul. The data type of elements supports float16 or bfloat16; the format supports ND.
* @li mm_y: A matrix tensor of shape [BS, N2] containing result of matmul. The data type of elements supports float16 or bfloat16; the format supports ND.
* @li permute_out: A matrix tensor of shape [BSK, H1] containing result of permutation if permute_out_flag == true. The data type of elements supports float16, bfloat16 or hifloat8; the format supports ND.
*/
REG_OP(AlltoAllvGroupedMatMul)
      .INPUT(gmm_x, TensorType({DT_FLOAT16, DT_BF16, DT_HIFLOAT8}))
      .INPUT(gmm_weight, TensorType({DT_FLOAT16, DT_BF16, DT_HIFLOAT8}))
      .OPTIONAL_INPUT(send_counts_tensor, TensorType({DT_INT32, DT_INT64}))
      .OPTIONAL_INPUT(recv_counts_tensor, TensorType({DT_INT32, DT_INT64}))
      .OPTIONAL_INPUT(mm_x, TensorType({DT_FLOAT16, DT_BF16, DT_HIFLOAT8}))
      .OPTIONAL_INPUT(mm_weight, TensorType({DT_FLOAT16, DT_BF16, DT_HIFLOAT8}))
      .OPTIONAL_INPUT(gmm_x_scale, TensorType({DT_FLOAT}))
      .OPTIONAL_INPUT(gmm_weight_scale, TensorType({DT_FLOAT}))
      .OPTIONAL_INPUT(gmm_x_offset, TensorType({DT_FLOAT}))
      .OPTIONAL_INPUT(gmm_weight_offset, TensorType({DT_FLOAT}))
      .OPTIONAL_INPUT(mm_x_scale, TensorType({DT_FLOAT}))
      .OPTIONAL_INPUT(mm_weight_scale, TensorType({DT_FLOAT}))
      .OPTIONAL_INPUT(mm_x_offset, TensorType({DT_FLOAT}))
      .OPTIONAL_INPUT(mm_weight_offset, TensorType({DT_FLOAT}))
      .OUTPUT(gmm_y, TensorType({DT_FLOAT16, DT_BF16}))
      .OUTPUT(mm_y, TensorType({DT_FLOAT16, DT_BF16}))
      .OUTPUT(permute_out, TensorType({DT_FLOAT16, DT_BF16, DT_HIFLOAT8}))
      .REQUIRED_ATTR(group, String)
      .REQUIRED_ATTR(ep_world_size, Int)
      .REQUIRED_ATTR(send_counts, ListInt)
      .REQUIRED_ATTR(recv_counts, ListInt)
      .ATTR(trans_gmm_weight, Bool, false)
      .ATTR(trans_mm_weight, Bool, false)
      .ATTR(permute_out_flag, Bool, false)
      .ATTR(gmm_x_quant_mode, Int, 0)
      .ATTR(gmm_weight_quant_mode, Int, 0)
      .ATTR(mm_x_quant_mode, Int, 0)
      .ATTR(mm_weight_quant_mode, Int, 0)
      .ATTR(group_size, Int, 0)
      .ATTR(y_dtype, Int, 28)
      .ATTR(mm_dtype, Int, 28)
      .OP_END_FACTORY_REG(AlltoAllvGroupedMatMul)

}  // namespace ge


#endif  // ALLTO_ALLV_GROUPED_MAT_MUL_PROTO_H_
