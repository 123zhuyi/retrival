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
 * \file kernel_utils.h
 * \brief kernel_utils
 */

/**
    grouped_bias_add_grad kernel common function
 */

#ifndef OPS_BUILT_IN_OP_GROUP_KERNEL_UTILS_H_
#define OPS_BUILT_IN_OP_GROUP_KERNEL_UTILS_H_

namespace utils {

constexpr static int64_t CONST1 = 1;
constexpr static int64_t CONST2 = 2;                                                                                                                  
constexpr static int64_t CONST4 = 4;                                                                                                                  
constexpr static int64_t CONST_SIXTY_THREE = 63;
constexpr static uint64_t ALIGN_BYTE= 32;

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline T FloorAlign(T a, T b)
{
    return a / b * b;
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template<typename T, typename PromoteDataT>
__aicore__ inline void FindProcessedHLen(int64_t maxPerHLen, int64_t usedUbSize, int64_t g, int64_t curH, int64_t& hMainLen, int64_t& hTailLen, int64_t& loops)
{
/*
    思路：由于单个块除了尾块已经等于128B了，因此，为了避免最后一次处理的量较小，选择均分
    @param: 
        input：
                maxPerHLen: 单次处理的最大H长度
                usedUbSize: 可用的ub空间，用以计算单次处理最大H长度
                g: 当前group的高度
                curH: 当前需要处理的H长度
        output:
                loops: 循环次数
                hMainLen： 主循环处理的h长度
                hTailLen： 最后一次需要处理的h长度
*/
    int64_t perMaxH = FloorAlign(usedUbSize / sizeof(T) / g, ALIGN_BYTE / sizeof(T));
    perMaxH = perMaxH / (sizeof(PromoteDataT) / sizeof(T));
    if (maxPerHLen < perMaxH) {
        perMaxH = maxPerHLen;
    }
    perMaxH = FloorAlign<int64_t>(perMaxH, static_cast<int64_t>(ALIGN_BYTE / sizeof(T)));
    
    loops = CeilDiv<int64_t>(curH, perMaxH); // 循环次数
    hMainLen = perMaxH;
    hTailLen = curH - (loops - 1) * hMainLen;
}

__aicore__ inline uint64_t FindNearestPower2(uint64_t value) 
{
    if (value == 0) {
        return 0;
    } else if (value <= CONST2) {
        return CONST1;
    } else if (value <= CONST4) {
        return CONST2;
    } else {
        const uint64_t num = value - CONST1;
        const uint64_t pow = CONST_SIXTY_THREE - AscendC::ScalarCountLeadingZero(num);
        return (CONST1 << pow);
    }
}

__aicore__ inline uint64_t CalLog2(uint64_t value)
{
    uint64_t res = 0;
    while (value > 1) {
        value = value >> 1;
        res++;
    }
    return res;
}

__aicore__ inline uint64_t GetCacheID(const int64_t idx)
{
    return AscendC::ScalarGetCountOfValue<CONST1>(idx ^ (idx + CONST1)) - CONST1;
}

}
#endif  // OPS_BUILT_IN_OP_GROUP_KERNEL_UTILS_H_
