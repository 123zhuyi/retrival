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
 * \file gather_pa_kv_cache_nd.h
 * \brief
 */

#ifndef GATHER_PA_KV_CACHE_ND_H_
#define GATHER_PA_KV_CACHE_ND_H_

#include "kernel_operator.h"

namespace GatherPaKvCacheV35 {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t DOUBLE_BUFFER = 2;

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
class GatherPaKvCacheNd {
public:
    __aicore__ inline GatherPaKvCacheNd(TPipe *pipe, const GatherPaKvCacheTilingDataV35 *__restrict tiling)
        : pipe_(pipe), tl_(tiling){};
    __aicore__ inline void Init(GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables, GM_ADDR seq_lens,
                                GM_ADDR key_in, GM_ADDR value_in, GM_ADDR seq_offset, GM_ADDR key_out,
                                GM_ADDR value_out);
    __aicore__ inline void Process();
    __aicore__ inline void InitParams();
    __aicore__ inline void GatherKvCache(GlobalTensor<T> dstCacheGm, GlobalTensor<T> srcCacheGm, uint64_t curLen,
                                         bool isFilledWithZero);
    __aicore__ inline T_INDEX CalcKvCoreOffset(int64_t reduceLen);

private:
private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> cacheQueue_;
    TQue<QuePosition::VECIN, 1> seqLensQueue_;
    TBuf<TPosition::VECCALC> prefixSumBuffer_;

    DataCopyPadExtParams<T> padExtParams_;

    const GatherPaKvCacheTilingDataV35 *tl_;

    GlobalTensor<T> keyCacheGm_;
    GlobalTensor<T> valueCacheGm_;
    GlobalTensor<T_INDEX> blockTablesGm_;
    GlobalTensor<T_INDEX> seqLensGm_;
    GlobalTensor<T_INDEX> seqOffsetGm_;
    GlobalTensor<T> outKeyGm_;
    GlobalTensor<T> outValueGm_;

    uint32_t batchPerCore_;
    uint32_t needCoreNum_;
    int64_t batchCount_;
    uint32_t seqLenAccSize_;
    int64_t cacheBlockSize_;
    int64_t blockTableWidth_;
    uint64_t maxUbHiddenSizeK_;
    uint64_t maxUbHiddenSizeV_;
    uint64_t maxUbHiddenSize_;
    int64_t numBlocks_;
    uint64_t hiddenSizeK_;
    uint64_t hiddenSizeV_;
    int64_t numTokens_;
};

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::Init(
    GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables, GM_ADDR seq_lens, GM_ADDR key_in, GM_ADDR value_in,
    GM_ADDR seq_offset, GM_ADDR key_out, GM_ADDR value_out)
{
    InitParams();

    keyCacheGm_.SetGlobalBuffer((__gm__ T *)(key_cache));
    valueCacheGm_.SetGlobalBuffer((__gm__ T *)(value_cache));
    blockTablesGm_.SetGlobalBuffer((__gm__ T_INDEX *)block_tables);
    seqLensGm_.SetGlobalBuffer((__gm__ T_INDEX *)(seq_lens));
    seqOffsetGm_.SetGlobalBuffer((__gm__ T_INDEX *)(seq_offset));

    outKeyGm_.SetGlobalBuffer((__gm__ T *)(key_out));
    outValueGm_.SetGlobalBuffer((__gm__ T *)(value_out));

    pipe_->InitBuffer(cacheQueue_, DOUBLE_BUFFER, (maxUbHiddenSize_) * sizeof(T));
    pipe_->InitBuffer(seqLensQueue_, DOUBLE_BUFFER, (seqLenAccSize_) * sizeof(T_INDEX));
    pipe_->InitBuffer(prefixSumBuffer_, BLOCK_SIZE);
}

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::InitParams()
{
    cacheBlockSize_ = tl_->kvCacheBlockSize;
    batchPerCore_ = tl_->batchPerCore;
    needCoreNum_ = tl_->needCoreNum;
    batchCount_ = tl_->batchCount;
    blockTableWidth_ = tl_->blockTableWidth;
    // UB放得下的kv Cache Block大小
    maxUbHiddenSizeK_ = tl_->maxUbHiddenSizeK;
    maxUbHiddenSizeV_ = tl_->maxUbHiddenSizeV;
    maxUbHiddenSize_ = tl_->maxUbHiddenSize;
    seqLenAccSize_ = tl_->seqLenAccumSize;
    numBlocks_ = tl_->numBlocks;
    hiddenSizeK_ = tl_->hiddenSizeK;
    hiddenSizeV_ = tl_->hiddenSizeV;
    numTokens_ = tl_->numTokens;

    padExtParams_.isPad = false;
    padExtParams_.leftPadding = 0;
    padExtParams_.rightPadding = 0;
    padExtParams_.paddingValue = 0;
}

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::Process()
{
    int64_t batchStart = GetBlockIdx() * batchPerCore_;
    int64_t batchEnd = batchStart + batchPerCore_;
    if (GetBlockIdx() == needCoreNum_ - 1) {
        batchEnd = batchCount_;
    }

    // 如果isSeqLensCumsum为false，在此处计算前缀和，即每个核的偏移
    int64_t coreOffset;
    if constexpr (isSeqLensCumsum) {
        coreOffset = seqLensGm_.GetValue(batchStart);
    } else {
        coreOffset = CalcKvCoreOffset(batchStart);
    }

    for (uint32_t i = batchStart; i < batchEnd; i++) {
        // 读取cache的数量
        // 累加模式
        T_INDEX seqLen;
        int64_t batchOffset, accumSeqLen;
        if constexpr (isSeqLensCumsum) {
            // 当前batch对应的seqLen
            seqLen = seqLensGm_.GetValue(i + 1) - seqLensGm_.GetValue(i);
            batchOffset = seqLensGm_.GetValue(i);
            accumSeqLen = seqLensGm_.GetValue(i + 1);
        } else {
            seqLen = seqLensGm_.GetValue(i);
            batchOffset = coreOffset;
            accumSeqLen = coreOffset + seqLensGm_.GetValue(i);
            coreOffset = accumSeqLen;
        }

        if (batchOffset >= numTokens_) {
            break;
        }

        // 如果numTokens小于seqLen总和，尾块需要截断
        if (numTokens_ > batchOffset && numTokens_ <= accumSeqLen) {
            seqLen = numTokens_ - batchOffset;
        }

        // block起点 + 偏移
        uint64_t seqOffset = 0;
        if constexpr (hasSeqOffset) {
            seqOffset = seqOffsetGm_.GetValue(i) / cacheBlockSize_; // 除以blocksize表示在block_tables中的偏移
        }

        // 当前batch关联到几个block
        uint64_t blockCount = CeilDivision(seqLen, cacheBlockSize_);
        uint64_t keyOffset = batchOffset * hiddenSizeK_;
        uint64_t valueOffset = batchOffset * hiddenSizeV_;
        // 对blockIdx（用来从block_tables中取blockId）循环
        for (uint32_t blockIdx = 0; blockIdx < blockCount; blockIdx++) {
            uint64_t blockTableOffset = blockIdx + seqOffset;
            bool isFilledWithZero;
            int64_t blockId;
            if ((blockTableOffset >= blockTableWidth_) || (blockTableOffset < 0)) {
                isFilledWithZero = true;
                blockId = 0;
            } else {
                isFilledWithZero = false;
                blockId = blockTablesGm_.GetValue(blockTableWidth_ * i + blockTableOffset);
            }

            uint64_t curBlockSize = cacheBlockSize_;
            if (blockIdx == blockCount - 1) {
                curBlockSize = seqLen - (blockCount - 1) * cacheBlockSize_; // 尾块处理
            }

            uint64_t keyCacheStart = blockId * cacheBlockSize_ * hiddenSizeK_;
            uint64_t valueCacheStart = blockId * cacheBlockSize_ * hiddenSizeV_;
            // 切分当前block的n个slot，每个tile长度为fracCacheBlockSize_
            uint32_t fracBlockCount = CeilDivision(curBlockSize * hiddenSizeK_, maxUbHiddenSizeK_);
            for (uint32_t fracBlockId = 0; fracBlockId < fracBlockCount; fracBlockId++) {
                uint64_t curFracBlockLen = maxUbHiddenSizeK_;
                if (fracBlockId == fracBlockCount - 1) {
                    curFracBlockLen =
                        curBlockSize * hiddenSizeK_ - (fracBlockCount - 1) * maxUbHiddenSizeK_; // 尾块处理
                }
                uint64_t keyCacheOffset = keyCacheStart + fracBlockId * maxUbHiddenSizeK_;
                GatherKvCache(outKeyGm_[keyOffset], keyCacheGm_[keyCacheOffset], curFracBlockLen, isFilledWithZero);
                keyOffset += curFracBlockLen;
            }
            fracBlockCount = CeilDivision(curBlockSize * hiddenSizeV_, maxUbHiddenSizeV_);
            for (uint32_t fracBlockId = 0; fracBlockId < fracBlockCount; fracBlockId++) {
                uint64_t curFracBlockLen = maxUbHiddenSizeV_;
                if (fracBlockId == fracBlockCount - 1) {
                    curFracBlockLen =
                        curBlockSize * hiddenSizeV_ - (fracBlockCount - 1) * maxUbHiddenSizeV_; // 尾块处理
                }
                uint64_t valueCacheOffset = valueCacheStart + fracBlockId * maxUbHiddenSizeV_;
                GatherKvCache(outValueGm_[valueOffset], valueCacheGm_[valueCacheOffset], curFracBlockLen,
                              isFilledWithZero);
                valueOffset += curFracBlockLen;
            }
        }
    }
}

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline T_INDEX
GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::CalcKvCoreOffset(int64_t reduceLen)
{
    LocalTensor<T_INDEX> prefixSumLocal = prefixSumBuffer_.Get<T_INDEX>();
    uint64_t loopTimes = CeilDivision(reduceLen, seqLenAccSize_);
    DataCopyPadExtParams<T_INDEX> padParams = {false, 0, 0, 0};
    DataCopyExtParams seqLensCopyParams;
    seqLensCopyParams.blockCount = 1;
    seqLensCopyParams.srcStride = 0;
    seqLensCopyParams.dstStride = 0; // 通用设置
    uint32_t seqOffset, seqLength;

    T_INDEX coreOffset = 0;
    for (uint64_t i = 0; i < loopTimes; i++) {
        seqOffset = i * seqLenAccSize_;
        seqLength = seqLenAccSize_;
        if (i == loopTimes - 1) {
            seqLength = reduceLen - (loopTimes - 1) * seqLenAccSize_;
        }
        seqLensCopyParams.blockLen = static_cast<uint32_t>(seqLength * sizeof(T_INDEX));
        LocalTensor<T_INDEX> seqLenLocal = seqLensQueue_.AllocTensor<T_INDEX>();
        DataCopyPad(seqLenLocal, seqLensGm_[seqOffset], seqLensCopyParams, padParams); // 把数据放入UB
        seqLensQueue_.EnQue<T_INDEX>(seqLenLocal);

        seqLenLocal = seqLensQueue_.DeQue<T_INDEX>();
        uint32_t srcShape[2] = {uint32_t(1), seqLength};
        AscendC::ReduceSum<T_INDEX, Pattern::Reduce::AR, true>(prefixSumLocal, seqLenLocal, srcShape, false);
        AscendC::TEventID eventIdVecToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventIdVecToS);
        WaitFlag<HardEvent::V_S>(eventIdVecToS);
        seqLensQueue_.FreeTensor<T_INDEX>(seqLenLocal);
        coreOffset += prefixSumLocal.GetValue(0);
    }

    return coreOffset;
}

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
__aicore__ inline void GatherPaKvCacheNd<T, T_INDEX, isSeqLensCumsum, hasSeqOffset>::GatherKvCache(
    GlobalTensor<T> dstCacheGm, GlobalTensor<T> srcCacheGm, uint64_t curLen, bool isFilledWithZero)
{
    LocalTensor<T> cacheLocal = cacheQueue_.AllocTensor<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = curLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    if (isFilledWithZero) {
        AscendC::TEventID eventIdMTE3ToVec = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToVec);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToVec);
        Duplicate<T>(cacheLocal, 0, curLen);
        AscendC::TEventID eventIdVecToMTE3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(eventIdVecToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVecToMTE3);
    } else {
        DataCopyPad<T, PaddingMode::Normal>(cacheLocal, srcCacheGm, dataCopyParams, padExtParams_);
        cacheQueue_.EnQue<T>(cacheLocal);
        cacheLocal = cacheQueue_.DeQue<T>();
    }

    DataCopyPad<T, PaddingMode::Normal>(dstCacheGm, cacheLocal, dataCopyParams);
    event_t eventIdMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
    cacheQueue_.FreeTensor(cacheLocal);
}

} // namespace GatherPaKvCacheV35
#endif