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
 * \file asin_with_agent_tiling_data.h
 * \brief AsinWithAgent TilingData 结构体定义（arch32）
 *
 * 使用标准 C++ struct（禁止 BEGIN_TILING_DATA_DEF 宏）
 *
 * 迭代二新增字段：midBufferSize
 *   - INT8/UINT8/BOOL：half 中间 buffer，大小 = tileLength * sizeof(half)
 *   - INT64：int32 中间 buffer，大小 = tileLength * sizeof(int32_t)
 *   - 其他 dtype：0（不需要中间 buffer）
 */

#ifndef _ASIN_WITH_AGENT_TILING_DATA_H_
#define _ASIN_WITH_AGENT_TILING_DATA_H_

#include <cstdint>

struct AsinWithAgentTilingData {
    uint32_t totalLength;    // 输入总元素数
    uint32_t tileLength;     // 每个 tile 的元素数（UB切分单位，对齐后）
    uint32_t loopCount;      // 每个 core 的主循环次数
    uint32_t tailTileLength; // 每个 core 的 tail tile 元素数（可能为0）
    uint32_t usedCoreNum;    // 实际使用的 core 数
    uint32_t tmpBufferSize;  // AscendC::Asin 高阶 API 所需 tmpBuffer 字节数
    uint32_t tilingKey;      // TilingKey 值（0=fp32, 1=fp16, 2=fp64, 3-8=整数/BOOL）
    uint32_t midBufferSize;  // Group C 中间 Cast buffer 大小（字节）
                             // INT8/UINT8/BOOL: tileLength * sizeof(half)
                             // INT64: tileLength * sizeof(int32_t)
                             // 其他: 0
};

#endif // _ASIN_WITH_AGENT_TILING_DATA_H_
