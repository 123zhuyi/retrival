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
 * \file ndtri_tiling_data.h
 * \brief Ndtri Tiling 数据结构定义
 */

#ifndef NDTRI_TILING_DATA_H_
#define NDTRI_TILING_DATA_H_

struct NdtriTilingData {
    int64_t totalNum = 0;     // self 元素总数
    int64_t blockFactor = 0;  // 每核主体处理元素数（按 alignElem 向上对齐）
    int64_t ubFactor = 0;     // 单次 UB 循环处理元素数（对齐 256）
};

#endif  // NDTRI_TILING_DATA_H_
