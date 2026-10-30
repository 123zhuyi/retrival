/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file log_add_exp_tiling_data.h
 * \brief LogAddExp TilingData structure definition
 */

#ifndef _LOG_ADD_EXP_TILING_DATA_H_
#define _LOG_ADD_EXP_TILING_DATA_H_

struct LogAddExpTilingData {
    int64_t totalLength = 0;     // Total number of output elements (after broadcast)
    int64_t blockFactor = 0;     // Number of elements each core processes
    int64_t ubFactor = 0;        // Number of elements processed per UB iteration
    int64_t needBroadcast = 0;   // Whether broadcast is needed (0=no, 1=yes)
    int64_t dimNum = 0;          // Number of dimensions (used for broadcast)
    int64_t xShape[8] = {0};     // x shape (right-aligned, broadcast dims padded with 1)
    int64_t yShape[8] = {0};     // y shape (right-aligned, broadcast dims padded with 1)
    int64_t outShape[8] = {0};   // Output shape (after broadcast)
    int64_t xStrides[8] = {0};   // x strides (broadcast dim stride=0)
    int64_t yStrides[8] = {0};   // y strides (broadcast dim stride=0)
    int64_t outStrides[8] = {0}; // Output strides
    // Binary doubling broadcast optimization fields
    int64_t useBinaryDoubling = 0; // 1: use binary doubling to pre-expand broadcast dim
    int64_t expandSrcIsX = 0;      // 1: x is the (1,N) tensor; 0: y is the (1,N) tensor
    int64_t innerSize = 0;         // N: inner contiguous dimension size
    int64_t expandRows = 0;        // M: total rows after expansion
};

#endif
