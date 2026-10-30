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
 * \file fused_mul_add_n_tiling_data.h
 * \brief A2 (DAV_2201 / ascend910b) TilingData definition for FusedMulAddN.
 *        Plain POD struct shared by host (op_host/fused_mul_add_n_tiling.cpp, written via
 *        context->GetTilingData<FusedMulAddNTilingData>()) and device kernel
 *        (op_kernel/fused_mul_add_n.cpp, read via GET_TILING_DATA_WITH_STRUCT). Placed under
 *        op_kernel/ per the experimental directory convention (TilingData header lives with the kernel).
 */

#ifndef FUSED_MUL_ADD_N_TILING_DATA_H
#define FUSED_MUL_ADD_N_TILING_DATA_H

#include <cstdint>

struct FusedMulAddNTilingData {
    int64_t totalNum;            // 总元素数（x1 全部元素，= shapeSize）
    int64_t blockNum;            // 实际使用核数（= SetBlockDim 值）
    int64_t blockFormer;         // 每个 former 核分到的元素数
    int64_t blockTail;           // 尾核（最后一个核）分到的元素数
    int64_t ubFormer;            // 单次 UB tile 元素数（former tile）
    int64_t ubLoopOfFormerBlock; // former 核的 ub 循环次数
    int64_t ubLoopOfTailBlock;   // 尾核的 ub 循环次数
    int64_t ubTailOfFormerBlock; // former 核最后一个 tile 的元素数
    int64_t ubTailOfTailBlock;   // 尾核最后一个 tile 的元素数
};

namespace optiling {
struct FusedMulAddNCompileInfo {
    int64_t coreNum{0};
    int64_t ubSize{0};
};
} // namespace optiling

#endif // FUSED_MUL_ADD_N_TILING_DATA_H
