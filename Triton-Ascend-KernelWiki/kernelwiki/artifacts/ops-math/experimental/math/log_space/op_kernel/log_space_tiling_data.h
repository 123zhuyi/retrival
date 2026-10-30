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

/*!
 * \file log_space_tiling_data.h
 * \brief LogSpace TilingData 结构体定义
 */
#ifndef _LOG_SPACE_TILING_DATA_H_
#define _LOG_SPACE_TILING_DATA_H_

struct LogSpaceTilingData {
    uint64_t totalLen = 0;    // steps
    uint32_t coreNum = 1;     // 参与计算的 AI Core 数
    uint32_t tileLen = 0;     // 每个常规核处理的元素数
    uint32_t tailCoreIdx = 0; // 处理尾块的核 id
    uint32_t tailTileLen = 0; // 尾核处理的元素数
    uint32_t ubChunk = 0;     // 单次搬运粒度（fp32 元素数）
    uint32_t reserved = 0;    // 对齐占位（保证 float 字段 4B 对齐）
    float startF = 0.0f;
    float stepF = 0.0f;   // (end-start)/(steps-1), steps==1 时为 0
    float logBase = 0.0f; // ln(base)
};
#endif
