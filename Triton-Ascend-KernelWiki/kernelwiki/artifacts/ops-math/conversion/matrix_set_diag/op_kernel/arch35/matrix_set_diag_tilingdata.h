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

#endif //