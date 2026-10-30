/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file xdivy_tiling_arch35.h
 * \brief Xdivy Tiling — arch35 头文件
 */
#pragma once
#include <cstdint>
#include <vector>
#include <exe_graph/runtime/tiling_context.h>
#include "math/xdivy/op_kernel/arch35/xdivy_tiling_struct.h"

namespace xdivy {

bool PadAndSqueeze(
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::vector<int64_t>>& output_shapes,
    std::vector<int64_t>&                    maximum_bro_shape,
    std::vector<std::vector<int64_t>>&       normal_input_shapes,
    std::vector<std::vector<int64_t>>&       normal_output_shapes);

bool FindSplitAxis(
    const std::vector<int64_t>& max_bro_shape,
    int64_t dtype_size, int64_t ub_per_core, int64_t phys_nodes,
    SplitResult& out);

bool MultiCoreSplit(
    const std::vector<int64_t>& max_bro_shape,
    const SplitResult& ub_split, int64_t max_cores,
    MultiCoreResult& out);

inline void GetCoreRange(int64_t core_id, int64_t tiles_main, int64_t cores_tail,
    int64_t& start, int64_t& end)
{
    if (core_id < cores_tail) {
        start = core_id * (tiles_main + 1);
        end = start + tiles_main + 1;
    } else {
        start = cores_tail * (tiles_main + 1) + (core_id - cores_tail) * tiles_main;
        end = start + tiles_main;
    }
}

inline int64_t GetUBSplitRange(int64_t a_o_off, int64_t a_o, int64_t a_i, int64_t a_i_tail)
{
    return (a_o_off == a_o - 1) ? a_i_tail : a_i;
}

bool PrecomputeInputStrides(const std::vector<int64_t>& s, std::vector<int64_t>& strides);
bool PrecomputeOutputStrides(const std::vector<int64_t>& s, std::vector<int64_t>& strides);

} // namespace xdivy

namespace optiling {

struct XdivyCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class XdivyTiling {
public:
    explicit XdivyTiling(gert::TilingContext* ctx);
    ge::graphStatus RunTiling();

private:
    ge::graphStatus GetShapeInfo();
    template<int64_t R> ge::graphStatus DoTilingAndSet();

    gert::TilingContext* ctx_;
    std::vector<std::vector<int64_t>> raw_input_shapes_;
    std::vector<std::vector<int64_t>> raw_output_shapes_;
    std::vector<int64_t>              max_bro_shape_;
    std::vector<std::vector<int64_t>> normal_input_shapes_;
    std::vector<std::vector<int64_t>> normal_output_shapes_;
    int64_t dtype_size_ = 0;
    int64_t phys_nodes_ = 3;   // P: 3 for FP32, 4 for FP16/BF16
    int64_t rank_ = 0;
};

} // namespace optiling
