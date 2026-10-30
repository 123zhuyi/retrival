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
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file asinh_with_agent_tiling_data.h
 * \brief AsinhWithAgent TilingData 结构体定义（arch32）
 */

#ifndef ASINH_WITH_AGENT_TILING_DATA_H
#define ASINH_WITH_AGENT_TILING_DATA_H

struct AsinhWithAgentTilingData {
    int64_t totalNum = 0;    // 输入总元素数量
    int64_t blockFactor = 0; // 每个 AI Core 处理的元素数量
    int64_t ubFactor = 0;    // 每次 UB 循环处理的元素数量
    int64_t tmpBufSize = 0;  // sharedTmpBuffer 大小（字节），由 GetAsinhMaxMinTmpSize 计算
};

#endif // ASINH_WITH_AGENT_TILING_DATA_H
