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
 * \file matrix_set_diag_tilingdata.h
 * \brief
 */

#ifndef MATRIX_SET_DIAG_TILINGDATA_H_
#define MATRIX_SET_DIAG_TILINGDATA_H_

#include <cstdint>

struct MatrixSetDiagCompileInfo {};

struct MatrixSetDiagInputInfo {
    uint64_t mergeDimSize{1}; // 非尾轴合轴后的大小，D0*D1*...*Dn-3
    uint64_t xRowNum{0};      // x的行数，Dn-2
    uint64_t xColNum{0};      // x的列数，Dn-1
    uint64_t diagNum{1};      // 对角线条数 = k1 - k0 + 1
    uint32_t maxDiagLen{0};   // 最大对角线长度
    int32_t k0{0};            // 下对角线偏移
    int32_t k1{0};            // 上对角线偏移
    int32_t dSize{0};         // 数据类型大小
};

struct MatrixSetDiagV2TilingData {
    uint64_t mergeDimSize; // 非尾轴合轴后的大小，D0*D1*...*Dn-3
    uint64_t xRowNum;      // x的行数，Dn-2
    uint64_t xColNum;      // x的列数，Dn-1
    uint64_t diagNum;      // 对角线条数 = k1 - k0 + 1
    uint32_t maxDiagLen;   // 最大对角线长度
    int32_t k0;            // 下对角线偏移
    int32_t k1;            // 上对角线偏移
    uint32_t coreNum;      // 核数
};

struct MSDV2CutTailTilingData {
    MatrixSetDiagV2TilingData input;
    uint32_t xRowFactor;      // 一次处理的行数
    uint32_t xColFactor;      // 一次处理的列数
    uint64_t totalCntPerCore; // 每个核上处理总块数
};

struct MSDV2NoCutTailTilingData {
    MatrixSetDiagV2TilingData input;
    uint64_t mergeDimNumPerCore; // 每个核上处理总块数
    uint64_t ubFactor;
};

struct MatrixSetDiagTilingData {
    uint32_t coreNum;
    uint64_t mergeDimSize;
    uint64_t xRowNum;
    uint64_t xColNum;
    uint64_t diagLen;
    uint64_t ubPerCore;
    uint64_t ubPerTail;
    uint64_t ubFactor;
    uint64_t ubTotalCount;
    uint64_t tailAxisDataSize;
};

#endif
