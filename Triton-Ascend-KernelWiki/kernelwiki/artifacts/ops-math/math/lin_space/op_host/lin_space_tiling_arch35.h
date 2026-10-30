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
 * \file lin_space_tiling_arch35.h
 * \brief
 */
#ifndef LIN_SPACE_REGBASE_TILING_H_
#define LIN_SPACE_REGBASE_TILING_H_

#include "lin_space_tiling.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(LinSpaceRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);            // 物理总核数
TILING_DATA_FIELD_DEF(int64_t, totalUbSize);             // 总UB可用大小
TILING_DATA_FIELD_DEF(int64_t, num);                     // 总共处理的元素个数
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);             // 使用的总核数
TILING_DATA_FIELD_DEF(int64_t, ubOneLoopNum);            // 单次循环UB可以处理元素个数上限
TILING_DATA_FIELD_DEF(int64_t, numOfPerCore);            // 非尾核处理元素个数
TILING_DATA_FIELD_DEF(int64_t, loopOfPerCore);           // 非尾核中loop次数
TILING_DATA_FIELD_DEF(int64_t, perOfPerCore);            // 非尾核中非尾loop每次处理元素个数
TILING_DATA_FIELD_DEF(int64_t, tailOfPerCore);           // 非尾核中尾loop处理元素个数
TILING_DATA_FIELD_DEF(int64_t, numOfTailCore);           // 尾核处理元素个数
TILING_DATA_FIELD_DEF(int64_t, loopOfTailCore);          // 尾核中loop次数
TILING_DATA_FIELD_DEF(int64_t, perOfTailCore);           // 尾核中非尾loop处理元素个数
TILING_DATA_FIELD_DEF(int64_t, tailOfTailCore);          // 尾核中尾loop处理的元素个数
TILING_DATA_FIELD_DEF(int64_t, halfWayCoreIdx);          // 界限核所在的核，该核上存在递增/递减双逻辑
TILING_DATA_FIELD_DEF(int64_t, forwardNumOfHalfWayCore); // 界限核，正向循环处理数据个数
TILING_DATA_FIELD_DEF(int64_t, loopOfForward);           // 界限核，正向循环次数
TILING_DATA_FIELD_DEF(int64_t, tailOfForward);           // 界限核，正向循环， 尾loop处理元素个数
TILING_DATA_FIELD_DEF(int64_t, loopOfBackward);          // 界限核，反向循环次数
TILING_DATA_FIELD_DEF(int64_t, tailOfBackward);          // 界限核, 反向循环， 尾loop处理元素个数
TILING_DATA_FIELD_DEF(float, start);
TILING_DATA_FIELD_DEF(float, stop);
TILING_DATA_FIELD_DEF(float, step);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(LinSpace_1002, LinSpaceRegbaseTilingData)

struct LinSpaceRegbaseTilingParam {
    int64_t totalCoreNum;
    int64_t totalUbSize;
    int64_t num;
    int64_t usedCoreNum;
    int64_t ubOneLoopNum;
    int64_t numOfPerCore;
    int64_t loopOfPerCore;
    int64_t perOfPerCore;
    int64_t tailOfPerCore;
    int64_t numOfTailCore;
    int64_t loopOfTailCore;
    int64_t perOfTailCore;
    int64_t tailOfTailCore;
    int64_t halfWayCoreIdx;
    int64_t forwardNumOfHalfWayCore;
    int64_t loopOfForward;
    int64_t tailOfForward;
    int64_t loopOfBackward;
    int64_t tailOfBackward;
    float start;
    float stop;
    float step;
};

constexpr uint64_t DOUBLE_CAST_TILING_KEY = 1002;

constexpr size_t RESERVED_WORKSPACE = static_cast<size_t>(16 * 1024 * 1024);

} // namespace optiling
#endif // LIN_SPACE_REGBASE_TILING_H_
