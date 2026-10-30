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
 * \file moe_distribute_dispatch_v3.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
const int DIM_ONE = 1;
const int DIM_TWO = 2;

/**
 * @brief Warpper for moe_distribute_dispatch_v5
 */
tensor_list npu_moe_distribute_dispatch_v3(const at::Tensor &context, const at::Tensor &x, 
                                           const at::Tensor &expert_ids,
                                           int64_t ep_world_size, int64_t ep_rank_id,
                                           int64_t moe_expert_num, int64_t ccl_buffer_size,
                                           const c10::optional<at::Tensor> &scales,
                                           const c10::optional<at::Tensor> &x_active_mask,
                                           const c10::optional<at::Tensor> &expert_scales,
                                           const c10::optional<at::Tensor> &elastic_info,
                                           const c10::optional<at::Tensor> &performance_info,
                                           int64_t tp_world_size, int64_t tp_rank_id,
                                           int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num,
                                           int64_t quant_mode, int64_t global_bs, int64_t expert_token_nums_type,
                                           std::string comm_alg, int64_t zero_expert_num, int64_t copy_expert_num, int64_t const_expert_num,
                                           c10::optional<int64_t> y_dtype, c10::optional<int64_t> x_dtype,
                                           c10::optional<int64_t> scales_dtype)
{
    TORCH_CHECK((x.dim() == DIM_TWO) && (expert_ids.dim() == DIM_TWO), "The x and expert_ids should be 2D");
    TORCH_CHECK((ep_world_size > 0), "The ep_world_sizes should be greater than 0, current is: ", ep_world_size);
    TORCH_CHECK((ep_rank_id >= 0) && (ep_rank_id < ep_world_size),
                "ep_rank_id should be in [0, ep_world_size), but got",
                " ep_world_size: ", ep_world_size,
                ", ep_rank_id: ", ep_rank_id,
                ". ");
    TORCH_CHECK((shared_expert_rank_num >= 0) && (shared_expert_rank_num < ep_world_size),
                "shared_expert_rank_num should be in [0, ep_world_size), but got",
                " ep_world_size: ", ep_world_size,
                ", shared_expert_rank_num: ", shared_expert_rank_num,
                ". ");
    bool is_shared_default = ((shared_expert_num == 1) && (shared_expert_rank_num == 0));
    bool is_no_shared = ((shared_expert_num == 0) && (shared_expert_rank_num == 0));
    bool is_valid_shared = ((shared_expert_num > 0)
        && ((shared_expert_rank_num / shared_expert_num) > 0)
        && ((shared_expert_rank_num % shared_expert_num) == 0));
    TORCH_CHECK(is_shared_default || is_no_shared || is_valid_shared,
                "shared_expert_num and shared_expertrank_num have obvious value situations: "
                "1. shared_expert_num is 1, shared_expert_rank_num is 0; 2. shared_expert num is 0, "
                "shared_expert_rank_num is 0; 3. shared_expert_num in (0, shared_expert_rank_num] and "
                "shared_expert_rank_num % shared_expert_num = 0. but the current input value is ",
                " shared_expert_num: ", shared_expert_num,
                ", shared_expert_rank_num: ", shared_expert_rank_num,
                ". ");
    TORCH_CHECK((expert_token_nums_type == 0) || (expert_token_nums_type == 1),
                "The expert_token_nums_type should be 0 or 1.");
    auto x_size = x.sizes();
    auto expert_ids_size = expert_ids.sizes();

    int64_t bs = x_size[0];
    int64_t h = x_size[1];
    int64_t k = expert_ids_size[1];

    // a2 expert_shard_type、shared_expert_rank_num 应为0
    bool shared_front = (expert_shard_type == 0);
    int64_t local_moe_expert_num = 1;
    int64_t global_bs_real = (global_bs == 0) ? (bs * ep_world_size) : global_bs;
    int64_t a = 0;
    int64_t ep_recv_cnt_num = 0;
    bool is_shared_expert = (shared_front && ep_rank_id < shared_expert_rank_num);
    if (is_shared_expert) {
        local_moe_expert_num = 1;
        int64_t max_bs = global_bs_real / ep_world_size;  // 前面已有拦截，保证ep_world_size > 0
        int64_t rank_num_per_shared_expert = shared_expert_rank_num / shared_expert_num;  // 前面已有拦截, 保证进入该分支时shared_expert_num > 0
        int64_t max_shared_group_num = (ep_world_size + rank_num_per_shared_expert - 1) / rank_num_per_shared_expert;
        a = max_bs * max_shared_group_num;
    } else {
        local_moe_expert_num = moe_expert_num / (ep_world_size - shared_expert_rank_num);
        a = global_bs_real * std::min(local_moe_expert_num, k);
    }
    if (shared_front && elastic_info.has_value()) {
        if ((is_shared_default) || (is_no_shared)) {
            local_moe_expert_num = std::max(local_moe_expert_num, moe_expert_num / (ep_world_size - shared_expert_rank_num));
            a = global_bs_real * std::min(local_moe_expert_num, k);
        } else {
            int64_t max_bs = global_bs_real / ep_world_size;
            int64_t rank_num_per_shared_expert = shared_expert_rank_num / shared_expert_num;
            int64_t max_shared_group_num = (ep_world_size + rank_num_per_shared_expert - 1) / rank_num_per_shared_expert;
            a = std::max(max_bs * max_shared_group_num, global_bs_real * std::min(moe_expert_num / (ep_world_size - shared_expert_rank_num), k));
            local_moe_expert_num = std::max(local_moe_expert_num, moe_expert_num / (ep_world_size - shared_expert_rank_num));
        }
    }
    if (tp_world_size == DIM_TWO) {
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num * tp_world_size;
    } else {
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num;
    }

    auto output_dtype = at::kChar;
    if (quant_mode == QuantMode::QUANT_MODE_NO_QUANT) {
        output_dtype = x.scalar_type();
    }
    at::Tensor expand_x{nullptr};
    at::Tensor dynamic_scales{nullptr};
    at::Tensor assist_info_forcombine{nullptr};
    at::Tensor expert_token_nums{nullptr};
    at::Tensor ep_recv_counts{nullptr};
    at::Tensor tp_recv_counts{nullptr};
    at::Tensor expand_scales{nullptr};
    {
        auto localDevice = c10::Device(x.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        expand_x = at::empty({std::max(a, a * tp_world_size), h}, x.options().dtype(output_dtype));  
        if (tp_world_size == 0) {
            dynamic_scales = at::empty({a}, x.options().dtype(at::kFloat));
        } else {
            dynamic_scales = at::empty({a * tp_world_size}, x.options().dtype(at::kFloat));
        }

        expert_token_nums = at::empty({local_moe_expert_num}, x.options().dtype(at::kLong));
        if (expert_scales.has_value() && expert_scales.value().defined()) {
            // 2: 2 buffer, 8 ranknum per server
            ep_recv_cnt_num = ep_world_size * local_moe_expert_num + 2 * global_bs_real * k * (ep_world_size / 8);
        }
        ep_recv_counts = at::empty({ep_recv_cnt_num}, x.options().dtype(at::kInt));
        tp_recv_counts = at::empty({tp_world_size}, x.options().dtype(at::kInt));
        assist_info_forcombine = at::empty({std::max(bs * k, a * 128)}, x.options().dtype(at::kInt));
        expand_scales = at::empty({a}, x.options().dtype(at::kFloat));
    }


    std::string comm_alg_str = std::string(comm_alg);
    char *comm_alg_ptr = const_cast<char *>(comm_alg_str.c_str());

    ACLNN_CMD(aclnnMoeDistributeDispatchV5, context, x, expert_ids, scales, x_active_mask, expert_scales,
              elastic_info, performance_info, ep_world_size, ep_rank_id, moe_expert_num, ccl_buffer_size,
              tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num,
              quant_mode, global_bs_real, expert_token_nums_type, comm_alg_ptr, zero_expert_num, copy_expert_num,
              const_expert_num, expand_x, dynamic_scales, assist_info_forcombine, expert_token_nums,
              ep_recv_counts, tp_recv_counts, expand_scales);

    return std::tie(expand_x, dynamic_scales, assist_info_forcombine, expert_token_nums, ep_recv_counts, tp_recv_counts,
        expand_scales);
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_distribute_dispatch_v3", &npu_moe_distribute_dispatch_v3, "moe_distribute_dispatch_v3");
}
} // op_api