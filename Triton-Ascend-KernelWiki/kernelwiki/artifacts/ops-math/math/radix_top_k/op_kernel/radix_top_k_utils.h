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
 * \file radix_top_k_utils.h
 * \brief Radix TopK 通用工具函数（Copy / CreateVecIndex / Sync 同步原语）
 */

#ifndef RADIX_TOPK_UTILS_H
#define RADIX_TOPK_UTILS_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"

namespace RadixTopK {
using namespace AscendC;

static constexpr uint32_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
static constexpr uint32_t REPEAT_SIZE = 256;
static constexpr uint32_t BYTE_SIZE = 8;
static constexpr uint32_t TWO_BYTE = 2;
static constexpr uint32_t NUM_VALUE_2BIT = 4;
static constexpr uint32_t INT32_NUM_PER_REPEAT = 64;
static constexpr int16_t XOR_OP_VALUE_16 = 0x8000;
static constexpr uint32_t MAX_TILE_NUM_IN_UB = 6144;
static constexpr uint32_t MAX_TILE_NUM_IN_UB_BY2 = 3072;
static constexpr uint32_t MAX_TILE_NUM_IN_UB_BY3 = 2048;
static constexpr uint32_t FLOAT32_SAFE_INT = 16777216;
static constexpr uint32_t BITS_PER_ROUND = 2;

/**
 * @brief 从 srcTensor 复制 dataNum 个元素到 dstTensor
 * @param dstTensor 目标 Tensor
 * @param srcTensor 源 Tensor
 * @param dataNum 复制数据量
 */
template <typename IT>
__aicore__ inline void CopyData(const LocalTensor<IT>& dstTensor, const LocalTensor<IT>& srcTensor,
                                const uint64_t& dataNum)
{
    uint64_t numPerRepeat = REPEAT_SIZE / sizeof(IT);
    uint64_t repeatTimes = dataNum / numPerRepeat;
    uint64_t remain = dataNum % numPerRepeat;
    if (repeatTimes > 0) {
        Copy(dstTensor, srcTensor, numPerRepeat, repeatTimes, {1, 1, 8, 8});
    }
    if (remain > 0) {
        Copy(dstTensor[repeatTimes * numPerRepeat], srcTensor[repeatTimes * numPerRepeat], remain, 1, {1, 1, 8, 8});
    }
}

/**
 * @brief 通过分段 Adds 生成索引序列
 * @param dstTensor 目标索引 Tensor
 * @param srcTensor 源索引 Tensor（包含 [0, 1, ..., baseDataNum-1]）
 * @param dataNum 需生成的索引总数
 * @param baseDataNum 每段基础索引长度
 */
template <typename IT>
__aicore__ inline void CreateVecIndexDataByAdds(const LocalTensor<IT>& dstTensor, const LocalTensor<IT>& srcTensor,
                                                const uint32_t& dataNum, const uint32_t& baseDataNum)
{
    uint32_t repeatTimes = dataNum / baseDataNum;
    uint32_t remain = dataNum % baseDataNum;
    for (int i = 1; i < repeatTimes; i++) {
        Adds(dstTensor[i * baseDataNum], srcTensor, static_cast<IT>(i * baseDataNum),
             static_cast<int32_t>(baseDataNum));
    }
    if (remain > 0) {
        Adds(dstTensor[repeatTimes * baseDataNum], srcTensor, static_cast<IT>(repeatTimes * baseDataNum),
             static_cast<int32_t>(remain));
    }
}

/**
 * @brief 创建索引：先用 CreateVecIndex 生成基索引，不足部分由 CreateVecIndexDataByAdds 补充
 * @param dstTensor 目标索引 Tensor
 * @param firstValue 索引起始值
 * @param dataNum 需生成的索引总数
 * @param baseDataNum 基索引长度
 */
template <typename IT>
__aicore__ inline void CreateVecIndexData(const LocalTensor<IT>& dstTensor, const IT& firstValue,
                                          const uint32_t& dataNum, const uint32_t& baseDataNum)
{
    CreateVecIndex(dstTensor, firstValue, baseDataNum);
    if (dataNum == baseDataNum)
        return;
    CreateVecIndexDataByAdds(dstTensor, dstTensor, dataNum, baseDataNum);
}

/**
 * @brief 性能优化的索引生成，根据 dataNum 大小选择最优 baseDataNum
 * @param dstTensor 目标索引 Tensor
 * @param firstValue 索引起始值
 * @param dataNum 需生成的索引总数
 */
template <typename IT>
__aicore__ inline void CreateVecIndexPerf(const LocalTensor<IT>& dstTensor, const IT& firstValue,
                                          const uint32_t& dataNum)
{
    if (dataNum == 0)
        return;
    if (dataNum < 32) {
        CreateVecIndexData<IT>(dstTensor, firstValue, dataNum, dataNum);
    } else if (dataNum < 256) {
        CreateVecIndexData<IT>(dstTensor, firstValue, dataNum, 32);
    } else if (dataNum < 2048) {
        CreateVecIndexData<IT>(dstTensor, firstValue, dataNum, 256);
    } else if (dataNum < 7680) {
        CreateVecIndexData<IT>(dstTensor, firstValue, dataNum, 512);
    } else {
        CreateVecIndexData<IT>(dstTensor, firstValue, dataNum, 1024);
    }
}

/**
 * @brief MTE3 → MTE2 同步
 */
__aicore__ inline void MTE3ToMTE2Sync()
{
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

/**
 * @brief MTE3 → S 同步
 */
__aicore__ inline void MTE3ToSSync()
{
    event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
}

/**
 * @brief MTE3 → V 同步
 */
__aicore__ inline void MTE3ToVSync()
{
    event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
}

/**
 * @brief V → MTE2 同步
 */
__aicore__ inline void VToMTE2Sync()
{
    event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
}

/**
 * @brief V → MTE3 同步
 */
__aicore__ inline void VToMTE3Sync()
{
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}

/**
 * @brief V → S 同步
 */
__aicore__ inline void VToSSync()
{
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}

/**
 * @brief S → V 同步
 */
__aicore__ inline void SToVSync()
{
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
}

/**
 * @brief S → MTE3 同步
 */
__aicore__ inline void SToMTE3Sync()
{
    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
}

/**
 * @brief S → MTE2 同步
 */
__aicore__ inline void SToMTE2Sync()
{
    event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
    WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
}

/**
 * @brief MTE2 → MTE3 同步
 */
__aicore__ inline void MTE2ToMTE3Sync()
{
    event_t eventIDMTE2ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
}

/**
 * @brief MTE2 → V 同步
 */
__aicore__ inline void MTE2ToVSync()
{
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
}

/**
 * @brief MTE2 → S 同步
 */
__aicore__ inline void MTE2ToSSync()
{
    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
}

} // namespace RadixTopK
#endif // RADIX_TOPK_UTILS_H
