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
 * \file data_compare_tiling_data.h
 * \brief DataCompare TilingData 定义
 *
 * All Reduce 算子，MAX_PATTERN_RANK=2。
 * 字段分 6 组：pattern 描述 / 多核切分 / UB 切分 / 外层 R loop / UB buffer 字节数 / group + data_compare 扩展
 */
#ifndef OPS_DATA_COMPARE_TILING_DATA_H_
#define OPS_DATA_COMPARE_TILING_DATA_H_

#include <cstdint>

constexpr int32_t MAX_PATTERN_RANK = 2;

struct DataCompareTilingData {
    // ─── pattern 描述（去 1 / 合轴 / 补 leading A / 补 R 增广之后）───
    int32_t axisNum = 0;
    int64_t axisShape[MAX_PATTERN_RANK] = {0};
    int64_t axisStride[MAX_PATTERN_RANK] = {0};

    // ─── 多核切分（fused aLoop）───
    int64_t aLoopCntTotal = 0;
    int64_t aSplitChunkCnt = 0;
    int64_t aBigCoreLoopCnt = 0;
    int64_t aSmallCoreLoopCnt = 0;
    int32_t aBigCoreCnt = 0;
    int32_t usedCoreNum = 0;

    // ─── UB 切分 ───
    int32_t aSplitAxisIdx = 0;
    int32_t rSplitAxisIdx = 0;
    int64_t aUbFactor = 0;
    int64_t aUbFactorAlign = 0;
    int64_t rUbFactor = 0;
    int64_t rUbFactorAlign = 0;
    int64_t innerAProd = 0;
    int64_t innerAProdAlign = 0;
    int64_t innerRProd = 0;
    int64_t innerRProdAlign = 0;

    // ─── 外层 R loop 扁平化 ───
    int64_t rLoopCntTotal = 0;

    // ─── UB buffer 字节数 ───
    int64_t preReduceUbSize = 0;
    int64_t postReduceUbSize = 0;
    int64_t tmpBufUbSize = 0;
    int64_t cacheBufUbSize = 0; // 固定 16 KB

    // ─── group 模板 ───
    int64_t rGroupCnt = 0;

    // ─── data_compare 扩展字段 ───
    float atol = 1e-5f;
    float rtol = 1e-3f;
};

#endif // OPS_DATA_COMPARE_TILING_DATA_H_
