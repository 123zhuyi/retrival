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
 * \file dynamic_partition_tiling_data_struct.h
 * \brief define tiling data of DynamicPartition
 */

#ifndef OP_KERNEL_DYNAMIC_PARTITION_TILING_DATA_STRUCT_H_
#define OP_KERNEL_DYNAMIC_PARTITION_TILING_DATA_STRUCT_H_

#include <cstdint>

namespace DynPart {
constexpr int8_t MAX_DIM_NUM = 7;
constexpr int32_t NUM_PARTITION_UNIT = 4096;
constexpr uint64_t TILING_H_MC_UB_CAN_HOLD_SPLIT_W = 50000UL;
constexpr uint64_t TILING_W_MC_UB_CAN_HOLD_SPLIT_W = 50001UL;
constexpr uint64_t TILING_H_MC_UB_CANNOT_HOLD_SPLIT_W = 50002UL;
constexpr uint64_t TILING_W_MC_UB_CANNOT_HOLD_SPLIT_W = 50003UL;
constexpr uint64_t TILING_XP_EMPTY = 50004UL;
constexpr uint64_t TILING_XP_SCALAR = 50005UL;
constexpr uint64_t TILING_X_EMPTY = 50006UL;
constexpr uint64_t TILING_H_MC_WITH_SMALL_W = 50007UL;
constexpr uint64_t TILING_NUM_PART_ONE = 50008UL;

#define KEY_H_MC_UB_CAN_HOLD_SPLIT_W 50000
#define KEY_W_MC_UB_CAN_HOLD_SPLIT_W 50001
#define KEY_H_MC_UB_CANNOT_HOLD_SPLIT_W 50002
#define KEY_W_MC_UB_CANNOT_HOLD_SPLIT_W 50003
#define KEY_XP_EMPTY 50004
#define KEY_XP_SCALAR 50005
#define KEY_X_EMPTY 50006
#define KEY_H_MC_WITH_SMALL_W 50007
#define KEY_NUM_PART_ONE 50008

struct DynPartTilingData {
    uint64_t tilingKey{0};
    int64_t usedCoreCnt{0};
    uint32_t dataUBSize{32}; // hLpUnit * wLpUnit * sizeof(T)
    uint32_t partUBSize{32}; // hLpUnit * sizeof(int32_t), partitionUB
    uint32_t totalUBSize{0};
    int32_t numPartitions{1};
    int64_t hMSize{1};
    int64_t hTSize{1};
    int64_t hLpUnit{1};
    int64_t hOffset{1};
    int64_t wMSize{1};
    int64_t wTSize{1};
    int64_t wLpUnit{1};
    int64_t dimNumExtFirst{0};
    int64_t outDimsExtFirst[MAX_DIM_NUM]{0, 0, 0, 0, 0, 0, 0}; // except dim 0
};
struct DynPartNumPartOneTilingData {
    uint64_t tilingKey{0};                                     // = TILING_NUM_PART_ONE
    int64_t usedCoreCnt{0};                                    // 实际使用核数
    int64_t totalElements{0};                                  // x 元素总数
    int64_t mainSize{0};                                       // 主块：每核处理的元素数
    int64_t tailSize{0};                                       // 尾块：最后一核处理的元素数
    int64_t ubFactor{0};                                       // UB 内循环单元（元素数）
    uint64_t totalUBSize{0};                                   // 总 UB 大小（字节）
    int64_t dim0{0};                                           // y[0] 第 0 维大小（= partitions 元素总数）
    int64_t dimNumExtFirst{0};                                 // partitions 维之外的额外维数
    int64_t outDimsExtFirst[MAX_DIM_NUM]{0, 0, 0, 0, 0, 0, 0}; // 额外维大小
};
} // namespace DynPart

#endif // OP_KERNEL_DYNAMIC_PARTITION_TILING_DATA_STRUCT_H_
