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
 * \file moe_distribute_dispatch_v3_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief MoeDistributeDispatchV3 operator interface implementation.

* @par Inputs
* @li context: A tensor. Support dtype: int32, dimension must be 1. Shape supports (2052, ), support format: ND.
* @li x: A tensor. Support dtype: float16,bfloat16, dimension must be 2. Shape supports (BS, H), support format: ND.
* @li expertIds: A tensor. Support dtype: int32, indicates top k experts of each token, dimension must be 2. Shape supports (BS, K), support format: ND.
* @li scales: An optional tensor. Support dtype: float32, dimension must be 2, support format: ND.
* @li x_active_mask: An optional tensor. Support dtype: bool, Support Shape: (Bs, ) or (Bs, K), support format: ND.
* @li expert_scales: An optional tensor. Support dtype: float32. Shape supports (BS, K), support format: ND.
* @li elastic_info: An optional tensor. Support dtype: int32, Support Shape: (4 + 2 * ep_world_size, ) support format: ND.
* @li performance_info: An optional tensor. Support dtype: int64, Support Shape: (ep_world_size) support format: ND.

* @par Attributes
* @li ep_world_size: Required. Input ep comm world size, Support Range: [2, 768], dtype: int64.
* @li ep_rank_id: Required. Input ep comm rank Id, dtype: int64.
* @li moe_expert_num: Required. Input moe expert num, Support Range: (0, 1024], dtype: int64.
* @li ccl_buffer_size: Required, Input ccl buffer size, Support Range: [0, MAX_INT32)，MAX_INT32 = 2^31 - 1, dtype: Int64.
* @li tp_world_size: Input tp comm world size, dtype: int64. Support Range: [0, 2], Default: 0.
* @li tp_rank_id: Input tp comm rank Id, dtype: int64. Support Range: [0, 2], Default: 0.
* @li expert_shard_type: Input moe shard type, dtype: int64. Support Range: [0], Default: 0.
* @li shared_expert_num: Input shared expert num, dtype: int64. Support Range: [0, 4], Default: 0.
* @li shared_expert_rank_num: Input shared expert rank num, dtype: int64. Support Range: [0, ep_world_size), Default: 0.
* @li quant_mode: Input quant mode. The options are 0 (non-quantization), 1 (static quantization), and 2 (dynamic quantization). dtype: int64.
* @li global_bs: Input global batch size, dtype: int64. The options are : 0 or (Bs * ep_world_size) when Bs is uniform across ranks; (maxBs * ep_world_size) when Bs is different across ranks, maxBs is the largest possible Bs. Default: 0.
* @li expert_token_nums_type: Input expert token nums type, dtype: int64. Support:0, Default: 0.
* @li comm_alg: Input comm alg type, dtype: String. Support: ("", "fullmesh_v1", "fullmesh_v2"). Default: "".
* @li zero_expert_num: Input zero expert num, dtype: int64. Support Range: [0, MAX_INT32)，MAX_INT32 = 2^31 - 1. Default: 0.
* @li copy_expert_num: Input copy expert num, dtype: int64. Support Range: [0, MAX_INT32)，MAX_INT32 = 2^31 - 1, Default: 0.
* @li const_expert_num: Input const expert num, dtype: int64. Support: 0, Default: 0.
* @li y_dtype: Input y dtype, dtype: int64. Support: 28, Default: 28.

* @par Outputs
* Seven outputs, including:
* @li expand_x: A tensor. Result of each expert after dispatching. Support dtype: float16,bfloat16,int8,float8_e4m3,float8_e5m2，hifloat8. Shape supports (A, H), support format: ND.
* @li dynamic_scales: If quant is enabled, scale value of each token. A tensor. Support dtype: float32,float8_e8m0. Shape supports (A, ), support format: ND.
* @li assist_info_for_combine: A tensor. Support dtype: int32. Shape supports (A * 128), support format: ND.
* @li expert_token_nums: A tensor. Tokens nums of expand_x. Support dtype: int64, support format: ND.
* @li ep_recv_count: A tensor. Received token nums after dispatching. Support dtype: int32, support format: ND.
* @li tp_recv_count: A tensor. Received token nums after allgather. Support dtype: int32, support format: ND.
* @li expand_scales: A tensor. Scales of each token to sum for combine. Support dtype: float32. Shape supports (A, ), support format: ND.
*/
REG_OP(MoeDistributeDispatchV3)
    .INPUT(context, TensorType({DT_INT32}))
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(expert_ids, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x_active_mask, TensorType({DT_BOOL}))
    .OPTIONAL_INPUT(expert_scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(elastic_info, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(performance_info, TensorType({DT_INT64}))
    .OUTPUT(expand_x, TensorType({DT_BF16, DT_INT8, DT_FLOAT16, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .OUTPUT(dynamic_scales, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(assist_info_for_combine, TensorType({DT_INT32}))
    .OUTPUT(expert_token_nums, TensorType({DT_INT64}))
    .OUTPUT(ep_recv_count, TensorType({DT_INT32}))
    .OUTPUT(tp_recv_count, TensorType({DT_INT32}))
    .OUTPUT(expand_scales, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .REQUIRED_ATTR(ccl_buffer_size, Int)
    .ATTR(tp_world_size, Int, 0)
    .ATTR(tp_rank_id, Int, 0)
    .ATTR(expert_shard_type, Int, 0)
    .ATTR(shared_expert_num, Int, 1)
    .ATTR(shared_expert_rank_num, Int, 0)
    .ATTR(quant_mode, Int, 0)
    .ATTR(global_bs, Int, 0)
    .ATTR(expert_token_nums_type, Int, 1)
    .ATTR(comm_alg, String, "")
    .ATTR(zero_expert_num, Int, 0)
    .ATTR(copy_expert_num, Int, 0)
    .ATTR(const_expert_num, Int, 0)
    .ATTR(y_dtype, Int, 28)
    .OP_END_FACTORY_REG(MoeDistributeDispatchV3)

}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
