/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file one_hot_tiling.h
 * \brief
 */
#ifndef OPS_OP_TILING_ONE_HOT_TILING_H_
#define OPS_OP_TILING_ONE_HOT_TILING_H_

#include <cstdint>

namespace optiling {
struct OneHotCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is_regbase_soc_version;
};

struct OneHotTilingParams {
    int64_t is_zero_off_value;
    int64_t not_last_core_numel;
    int64_t mode_of_cal_with_axis;
    int64_t core_used;
    int64_t numel_shape_x;
    int64_t first_dim_x;
    int64_t last_dim_x;
    int64_t numel_shape_off_value_tensor;
    int64_t last_core_numel;
    int64_t not_last_core_index;
    int64_t last_core_index;
    int64_t tiling_core_num;
};
} // namespace optiling
#endif // OPS_OP_TILING_ONE_HOT_TILING_H_
