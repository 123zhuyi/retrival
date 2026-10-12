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
 * \file isin_part_v1.h
 * \brief
 */
#ifndef __ISIN_PART_V1_H__
#define __ISIN_PART_V1_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "isin_part_v1_tiling_data.h"
#include "isin_part_v1_tiling_key.h"
namespace NsIsinPartV1 {

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t OUT_ELEMENTS_SIZE = 1 << 15; // 输出参数占用空间 32KB
constexpr uint32_t IN_ELEMENTS_SIZE = 1 << 16;  // 输入参数占用UB的空间64KB

template <typename T>
class GmInCache {
public:
    __aicore__ inline GmInCache() {}
    // 构造函数：直接传入已分配的 UB 缓存
    __aicore__ inline void Init(
        GlobalTensor<T>& gm, TQue<QuePosition::VECIN, BUFFER_NUM>& q, uint32_t totalSize, uint32_t cacheSize)
    {
        this->gmTensor = gm;
        this->queue = q;
        this->totalSize = totalSize;
        this->cacheSize = cacheSize;
        LocalTensor<T> local = queue.AllocTensor<T>();
        localTensor = local;
        startIdx = 0;
        CopyIn();
        queue.EnQue(localTensor);
        localTensor = queue.DeQue<T>();
    }

    __aicore__ inline T operator[](uint32_t idx)
    {
        // 判断索引是否在当前缓存范围内
        if (!isInCache(idx)) {
            startIdx = (idx / cacheSize) * cacheSize;
            CopyIn();
            queue.EnQue(localTensor);
            localTensor = queue.DeQue<T>();
        }
        uint32_t localOffset = idx - startIdx;
        return localTensor.GetValue(localOffset);
    }

    __aicore__ inline void CopyIn() const
    {
        uint32_t num = (startIdx + cacheSize <= totalSize) ? cacheSize : (totalSize - startIdx);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(num * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        DataCopyPad(localTensor, gmTensor[startIdx], copyParams, padParams);
    }

private:
    uint32_t startIdx = 0;
    uint32_t totalSize = 0; // 所有元素的个数，譬如UB只能放1024个元素，总共有5000个元素，totalSize=5000
    uint32_t cacheSize = 0; // 当前缓存的元素个数
    TQue<QuePosition::VECIN, BUFFER_NUM> queue;
    GlobalTensor<T> gmTensor;
    LocalTensor<T> localTensor;
    __aicore__ bool inline isInCache(uint32_t idx) const { return (idx >= startIdx && idx < startIdx + cacheSize); }
};

template <typename T>
class GmOutCache {
public:
    __aicore__ inline GmOutCache() {}
    // 构造函数：直接传入已分配的 UB 缓存
    __aicore__ inline void Init(
        GlobalTensor<T>& gm, TQue<QuePosition::VECOUT, BUFFER_NUM>& q, uint32_t totalSize, uint32_t cacheSize)
    {
        this->gmTensor = gm;
        this->queue = q;
        this->totalSize = totalSize;
        this->cacheSize = cacheSize;
        LocalTensor<T> local = queue.AllocTensor<T>();
        localTensor = local;
        startIdx = 0;
        DuplicateForUB(localTensor, cacheSize);
    }

    // 为 int8_t 类型做特殊处理
    __aicore__ inline void DuplicateForUB(LocalTensor<T> src, uint32_t elementsNum)
    {
        if constexpr (std::is_same_v<T, std::int8_t>) {
            int32_t count = elementsNum / sizeof(int16_t);
            LocalTensor<int16_t> dstLocal = src.template ReinterpretCast<int16_t>();
            Duplicate(dstLocal, (int16_t)0, count);
            if (elementsNum % sizeof(int16_t)) {
                src.SetValue(elementsNum - 1, 0);
            }
        } else {
            Duplicate(src, (T)0, elementsNum);
        }
    }

    __aicore__ inline void SetValue(uint32_t idx, T value)
    {
        // 判断索引是否在当前缓存范围内
        if (!isInCache(idx)) {
            Flush();
            event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);

            startIdx = (idx / cacheSize) * cacheSize;
            DuplicateForUB(localTensor, cacheSize);
        }
        uint32_t localOffset = idx - startIdx;
        localTensor.SetValue(localOffset, value);
    }

    __aicore__ inline void Flush() const
    {
        uint32_t num = (startIdx + cacheSize <= totalSize) ? cacheSize : (totalSize - startIdx);
        DataCopyExtParams copyParams{1, num, 0, 0, 0};

        SetAtomicAdd<T>();
        DataCopyPad(gmTensor[startIdx], localTensor, copyParams);
        SetAtomicNone();
    }

private:
    uint32_t startIdx = 0;
    uint32_t totalSize = 0; // 所有元素的个数，譬如UB只能放1024个元素，总共有5000个元素，totalSize=5000
    uint32_t cacheSize = 0; // 当前缓存的元素个数
    TQue<QuePosition::VECOUT, BUFFER_NUM> queue;
    GlobalTensor<T> gmTensor;
    LocalTensor<T> localTensor;
    __aicore__ bool inline isInCache(uint32_t idx) const { return (idx >= startIdx && idx < startIdx + cacheSize); }
};

template <typename T>
class IsinPartV1 {
public:
    __aicore__ inline IsinPartV1(){};
    __aicore__ inline void Init(
        GM_ADDR value, GM_ADDR index, GM_ADDR elementsNum, GM_ADDR z, const IsinPartV1TilingData* tilingData);
    __aicore__ inline void Process();

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueValue;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueIndex;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    AscendC::GlobalTensor<T> valueGm;
    AscendC::GlobalTensor<int32_t> indexGm;
    AscendC::GlobalTensor<int8_t> zGm;
    AscendC::LocalTensor<int8_t> zLocal;
    GmOutCache<int8_t> zGmCache;
    GmInCache<T> valueGmCache;
    GmInCache<int32_t> indexGmCache;

    int32_t elementsNum;
    uint32_t totalLength;
    uint32_t blockStart;
    uint32_t blockEnd;
    uint32_t blockLength;

    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void GMout(int32_t start, int32_t end);
};

template <typename T>
__aicore__ inline void IsinPartV1<T>::Init(
    GM_ADDR value, GM_ADDR index, GM_ADDR elementsNumAddr, GM_ADDR z, const IsinPartV1TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = AscendC::GetBlockIdx();

    AscendC::GlobalTensor<int32_t> elementsNumGm;
    elementsNumGm.SetGlobalBuffer((__gm__ int32_t*)elementsNumAddr, 1);
    this->elementsNum = elementsNumGm.GetValue(0);
    ASSERT(elementsNum != 0 && "elementsNum can not be zero!");

    this->totalLength = tilingData->totalLength;
    this->blockLength = totalLength / AscendC::GetBlockNum();
    this->blockStart = this->blockLength * AscendC::GetBlockIdx();
    this->blockEnd = (AscendC::GetBlockIdx() != AscendC::GetBlockNum() - 1) ?
                         this->blockLength * (AscendC::GetBlockIdx() + 1) :
                         totalLength;

    valueGm.SetGlobalBuffer((__gm__ T*)value, totalLength);
    indexGm.SetGlobalBuffer((__gm__ int32_t*)index, totalLength);
    zGm.SetGlobalBuffer((__gm__ int8_t*)z, elementsNum);
    pipe.InitBuffer(inQueueValue, BUFFER_NUM, IN_ELEMENTS_SIZE);
    pipe.InitBuffer(inQueueIndex, BUFFER_NUM, IN_ELEMENTS_SIZE);
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, OUT_ELEMENTS_SIZE * sizeof(int8_t));
}

template <typename T>
__aicore__ inline void IsinPartV1<T>::GMout(int32_t start, int32_t end)
{
    for (int i = start; i <= end; i++) {
        auto v = indexGmCache[i];
        if (v < elementsNum) {
            zGmCache.SetValue(v, 1);
        }
    }
}

template <typename T>
__aicore__ inline void IsinPartV1<T>::CopyIn()
{
    // 初始化缓存类
    valueGmCache.Init(valueGm, inQueueValue, totalLength, IN_ELEMENTS_SIZE / sizeof(T));
    indexGmCache.Init(indexGm, inQueueIndex, totalLength, IN_ELEMENTS_SIZE / sizeof(int32_t));
    zGmCache.Init(zGm, outQueueZ, elementsNum, OUT_ELEMENTS_SIZE);

    // modify blockStart
    if (this->blockStart) {
        T startValue = valueGmCache[this->blockStart - 1];
        while (startValue == valueGmCache[this->blockStart] && this->blockStart != this->blockEnd)
            this->blockStart++;
    }

    // modify blockEnd
    if (this->blockEnd != totalLength && this->blockStart != this->blockEnd) {
        T endValue = valueGmCache[this->blockEnd - 1];
        while (endValue == valueGmCache[this->blockEnd])
            this->blockEnd++;
    }
    // 区间[blockStart, blockEnd)，左闭右开
    this->blockLength = this->blockEnd - this->blockStart;
}

template <typename T>
__aicore__ inline void IsinPartV1<T>::Compute()
{
    int32_t pre = this->blockStart;
    int8_t gmOutTriggered = 0;
    int8_t oneceTriggered = 0; // 有一次触发
    for (int j = blockStart; j < blockEnd; ++j) {
        if (indexGmCache[j] >= elementsNum && indexGmCache[pre] < elementsNum && !gmOutTriggered &&
            valueGmCache[j] == valueGmCache[pre]) {
            GMout(pre, j - 1);
            gmOutTriggered = 1;
            oneceTriggered = 1;
        } else if (j > pre && valueGmCache[j] != valueGmCache[pre]) {
            gmOutTriggered = 0;
            pre = j;
        }
    }
    if (oneceTriggered) {
        zGmCache.Flush();
    }
}

template <typename T>
__aicore__ inline void IsinPartV1<T>::Process()
{
    CopyIn();
    if (blockEnd > blockStart) {
        Compute();
    }
}

} // namespace NsIsinPartV1
#endif // Isin_Part_V1_H