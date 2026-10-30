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
 * \file moe_distribute_combine_v3.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
const int DIM_ONE = 1;
const int DIM_TWO = 2;

/**
 * @brief Warpper for moe_distribute_combine_v5
 */
at::Tensor npu_moe_distribute_combine_v3(const at::Tensor &context, const at::Tensor &expand_x,
                                         const at::Tensor &expert_ids,
                                         const at::Tensor &assist_info_for_combine,
                                         const at::Tensor &ep_send_counts, const at::Tensor &expert_scales,
                                         int64_t ep_world_size, int64_t ep_rank_id,
                                         int64_t moe_expert_num, int64_t ccl_buffer_size,
                                         const c10::optional<at::Tensor> &tp_send_counts,
                                         const c10::optional<at::Tensor> &x_active_mask,
                                         const c10::optional<at::Tensor> &expand_scales,
                                         const c10::optional<at::Tensor> &shared_expert_x,
                                         const c10::optional<at::Tensor> &elastic_info,
                                         const c10::optional<at::Tensor> &ori_x,
                                         const c10::optional<at::Tensor> &const_expert_alpha_1,
                                         const c10::optional<at::Tensor> &const_expert_alpha_2,
                                         const c10::optional<at::Tensor> &const_expert_v,
                                         const c10::optional<at::Tensor> &performance_info,
                                         int64_t tp_world_size, int64_t tp_rank_id,
                                         int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num,
                                         int64_t global_bs, int64_t comm_quant_mode,
                                         std::string comm_alg, int64_t zero_expert_num, int64_t copy_expert_num, int64_t const_expert_num)
{
    TORCH_CHECK((expand_x.dim() == DIM_TWO) && (expert_ids.dim() == DIM_TWO), "The x and expert_ids should be 2D");
    TORCH_CHECK((expand_x.scalar_type() == at::kBFloat16) || (expand_x.scalar_type() == at::kHalf) || (expand_x.scalar_type() == at::kInt),
                "dtype of expand_x should be BFloat16, Float16 or Int, but got " + std::string(c10::toString(expand_x.scalar_type())));
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
                "dtype of expert_ids should be Int, but got " + std::string(c10::toString(expert_ids.scalar_type())));
    auto expand_x_size = expand_x.sizes();
    auto expert_ids_size = expert_ids.sizes();

    int64_t bs = expert_ids_size[0];
    int64_t h = expand_x_size[1];
    int64_t global_bs_real = (global_bs == 0) ? (bs * ep_world_size) : global_bs;

    at::Tensor output;
    {
        auto localDevice = c10::Device(expand_x.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        if (expand_x.scalar_type() != at::kInt) {
            output = at::empty({bs, h}, expand_x.options().dtype(expand_x.scalar_type()));
        } else {
            output = at::empty({bs, h}, expand_x.options().dtype(at::kBFloat16));
        }
    }
    c10::optional<at::Tensor> nulltensor = c10::nullopt;
    int64_t out_dtype = 0;
    int64_t group_list_type = 0;

    std::string comm_alg_str = std::string(comm_alg);
    char *comm_alg_ptr = const_cast<char *>(comm_alg_str.c_str());

    ACLNN_CMD(aclnnMoeDistributeCombineV5, context, expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales, tp_send_counts, x_active_mask,
              nulltensor, nulltensor, nulltensor, expand_scales, shared_expert_x, elastic_info, ori_x, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
              performance_info, ep_world_size, ep_rank_id, moe_expert_num, ccl_buffer_size, tp_world_size, tp_rank_id,
              expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs_real, out_dtype, comm_quant_mode, group_list_type,
              comm_alg_ptr, zero_expert_num, copy_expert_num, const_expert_num, output);

    return output;
}
// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_distribute_combine_v3", &npu_moe_distribute_combine_v3, "moe_distribute_combine_v3");
}
} // op_api