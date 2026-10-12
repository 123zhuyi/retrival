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
 * \file matrix_diag_tiling_data_struct.h
 * \brief define tiling data of MatrixDiagAsc
 */

#ifndef OP_KERNEL_MATRIX_DIAG_TILING_DATA_STRUCT_H_
#define OP_KERNEL_MATRIX_DIAG_TILING_DATA_STRUCT_H_

#include <cstdint>

namespace MatrixDiagAsc
{
constexpr uint64_t TILING_SCATTER_LOW = 100UL;
constexpr uint64_t TILING_SCATTER_HIGH = 101UL;
constexpr uint64_t TILING_SIMT = 102UL;
constexpr uint64_t TILING_PURE_COPY = 103UL;
constexpr uint64_t TILING_SCATTER_LARGE = 104UL;

#define TILING_KEY_SCATTER_LOW 100
#define TILING_KEY_SCATTER_HIGH 101
#define TILING_KEY_SIMT 102
#define TILING_KEY_PURE_COPY 103
#define TILING_KEY_SCATTER_LARGE 104

struct MatrixDiagScatterTilingData {
    int64_t realCoreNum{0};
    int64_t batchSize{0};
    int64_t nSize{0};
    int64_t batchUbFactor{0};
    int64_t batchUbTailFactor{0};
    int64_t batchUbCount{0};
    int64_t blockFactor{0};
    int64_t blockTailFactor{0};
    int64_t blockMainCount{0};
};

struct MatrixDiagScatterLargeTilingData {
    int64_t realCoreNum{0};
    int64_t batchSize{0};
    int64_t nSize{0};
    int64_t nUbFactor{0};
    int64_t nUbTailFactor{0};
    int64_t nUbCount{0};
    int64_t blockFactor{0};
    int64_t blockTailFactor{0};
    int64_t blockMainCount{0};
};

struct MatrixDiagSimtTilingData {
    int64_t batchSize{0};
    int64_t nSize{0};
    int64_t ubSize{0};
    int64_t realCoreNum{0};
    int64_t mainBlockCount{0};
    int64_t mainBlockFactor{0};
    int64_t tailBlockFactor{0};
};

struct MatrixDiagPureCopyTilingData {
    int64_t size{0};
    int64_t ubSize{0};
    int64_t realCoreNum{0};
    int64_t mainBlockCount{0};
    int64_t mainBlockFactor{0};
    int64_t tailBlockFactor{0};
};
}
#endif  // OP_KERNEL_MATRIX_DIAG_TILING_DATA_STRUCT_H_