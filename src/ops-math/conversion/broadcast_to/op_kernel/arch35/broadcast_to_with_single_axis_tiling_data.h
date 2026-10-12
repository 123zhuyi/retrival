/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file broadcast_to_with_single_axis_tiling_data.h
 * \brief 单轴特化 broadcast tiling 数据结构 (host/kernel 共享)
 */

#ifndef BROADCAST_TO_WITH_SINGLE_AXIS_TILING_DATA_H_
#define BROADCAST_TO_WITH_SINGLE_AXIS_TILING_DATA_H_

#include <cstdint>

namespace BrcSA {

struct SingleAxisBrcTilingData {
    uint64_t shapeSize;   // 输出tensor总元素个数
    uint32_t tileSize;    // 每块tile处理的元素个数
    uint32_t blockNum;    // 实际使用的核数
    uint64_t blockFactor; // 主核处理的tile块数, 尾核处理blockFactor-1块
};

} // namespace BrcSA

#endif // BROADCAST_TO_WITH_SINGLE_AXIS_TILING_DATA_H_
