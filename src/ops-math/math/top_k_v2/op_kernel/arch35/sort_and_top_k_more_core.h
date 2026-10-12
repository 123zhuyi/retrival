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
 * \file sort_and_top_k_more_core.h
 * \brief sort_and_top_k morecore mode impl
 */

#ifndef SORT_AND_TOP_K_MORE_CORE_H
#define SORT_AND_TOP_K_MORE_CORE_H

#include <cmath>
#include "kernel_operator.h"
#include "../../sort/arch35/sort_radix_sort_more_core.h"
#include "../../sort/arch35/sort_tiling_data.h"      // sort_radix_sort_more_core.h 里面引用了 sort_tiling_data.h
#include "../../sort/arch35/common/util_type_simd.h" // 使用 ROUND_UP_AGLIN , DoubleBufferSimd
#include "top_k_small_size_bitonic_sort.h"
#include "top_k_util_type_simd.h"

namespace SortAndTopK {
const uint64_t AGLIN_FACTOR = 32;

using namespace AscendC;

// T1-输入x dtype,T2-输出Idx dtype, UT-无符号的数据类型
template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend, bool IS_BITONIC_SORT = false>
class SortAndTopKMoreCore : public Sort::SortRadixMoreCore<T1, T2, UT, T3, isDescend> {
public:
    __aicore__ inline SortAndTopKMoreCore(){};
    __aicore__ inline void InitParam(GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex, GM_ADDR workspace,
                                     const TopKV2TilingDataSimd* tilingData, TPipe* pipe);
    __aicore__ inline void ProcessTopK();

private:
    GlobalTensor<T1> sortOutValueGM_;
    GlobalTensor<uint32_t> sortOutIdxGM_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> topkResQueue_;
    TBuf<TPosition::VECCALC> bitonicSmallIndexTBuf_;

    const TopKV2TilingDataSimd* tilingData_;

    int64_t topKRealValue_ = 0;
    uint32_t tileDataSize_ = 0;
    uint32_t blockTileNum_ = 0;
    uint32_t tailTileNum_ = 0;

    __aicore__ inline void ProcessSortAndTopK(GlobalTensor<T1> inputXGm, int64_t gmOffset, uint32_t sortLoopRound);
    __aicore__ inline void ParserTilingData();
    __aicore__ inline void GetValueResData(uint32_t dataCopyLoopTimes, bool hasLastTile, int64_t outGmOffset,
                                           int64_t sortOutGmOffset);
    __aicore__ inline void GetIndexResData(uint32_t dataCopyLoopTimes, bool hasLastTile, int64_t outGmOffset,
                                           int64_t sortOutGmOffset);
    __aicore__ inline void GetTopKRes(uint32_t sortLoopRound);
    __aicore__ inline void GetBitonicSmallTopKRes(uint32_t sortLoopRound);
    __aicore__ inline void InitSortBuffers();
};

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend, bool IS_BITONIC_SORT>
__aicore__ inline void SortAndTopKMoreCore<T1, T2, UT, T3, isDescend, IS_BITONIC_SORT>::InitParam(
    GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex, GM_ADDR workspace, const TopKV2TilingDataSimd* tilingData, TPipe* pipe)
{
    this->blockIdx_ = GetBlockIdx();
    this->pipe_ = pipe;
    tilingData_ = tilingData;
    ParserTilingData();
    this->realCoreNum_ = GetBlockNum();
    if constexpr (sizeof(T3) == sizeof(int64_t)) {
        this->factor_ = 2;
    }

    this->inputXGm_.SetGlobalBuffer((__gm__ T1*)x);
    this->outValueGm_.SetGlobalBuffer((__gm__ T1*)value);
    this->outIdxGm_.SetGlobalBuffer((__gm__ uint32_t*)sortIndex);
    uint64_t wkOffset = this->clearCoreSize0_ * this->clearCore0_;
    uint64_t oneBlockNumB32 = this->oneBlock_ / sizeof(int32_t);
    if constexpr (sizeof(T3) == sizeof(int64_t)) {
        wkOffset = wkOffset * 2;
    }
    wkOffset = Ops::Base::CeilAlign(wkOffset, oneBlockNumB32);
    this->excusiveBinsGmWk_.SetGlobalBuffer((__gm__ uint32_t*)workspace, wkOffset);
    wkOffset = wkOffset * sizeof(uint32_t);

    uint64_t histOffset = this->clearCout_ * this->clearSize_ * this->clearCore1_;
    if constexpr (sizeof(T3) == sizeof(int64_t)) {
        histOffset = histOffset * 2;
    }
    histOffset = Ops::Base::CeilAlign(histOffset, oneBlockNumB32);
    this->globalHistGmWk_.SetGlobalBuffer((__gm__ uint32_t*)(workspace + wkOffset), histOffset);
    wkOffset = wkOffset + histOffset * sizeof(uint32_t);

    uint64_t dbOffset = this->totalDataNum_ * this->unsortedDimParallel_;
    if constexpr (sizeof(T3) == sizeof(int64_t)) {
        dbOffset = dbOffset * 2;
    }
    dbOffset = Ops::Base::CeilAlign(dbOffset, oneBlockNumB32);
    this->outIdxDbWK_.SetGlobalBuffer((__gm__ uint32_t*)(workspace + wkOffset), dbOffset);
    wkOffset = wkOffset + dbOffset * sizeof(uint32_t);

    dbOffset = this->totalDataNum_ * this->unsortedDimParallel_;
    if constexpr (sizeof(T2) == sizeof(int64_t)) {
        dbOffset = dbOffset * 2;
    }
    dbOffset = Ops::Base::CeilAlign(dbOffset, oneBlockNumB32);
    sortOutIdxGM_.SetGlobalBuffer((__gm__ uint32_t*)(workspace + wkOffset), dbOffset);
    wkOffset = wkOffset + dbOffset * sizeof(uint32_t);

    uint64_t histTileOffset = this->lastDimTileNum_ * Sort::RADIX_SORT_NUM * this->unsortedDimParallel_;
    this->histTileGmWk_.SetGlobalBuffer((__gm__ uint16_t*)(workspace + wkOffset), histTileOffset);
    wkOffset = wkOffset + histTileOffset * sizeof(uint16_t);
    this->histCumsumTileGmWk_.SetGlobalBuffer((__gm__ uint16_t*)(workspace + wkOffset), histTileOffset);
    wkOffset = wkOffset + histTileOffset * sizeof(uint16_t);

    uint64_t xB8Offset = this->lastDimTileNum_ * this->numTileData_ * this->unsortedDimParallel_;
    xB8Offset = Ops::Base::CeilAlign(xB8Offset, this->oneBlock_);
    this->xB8GmWk_.SetGlobalBuffer((__gm__ uint8_t*)(workspace + wkOffset), xB8Offset);
    wkOffset = wkOffset + xB8Offset * sizeof(uint8_t);

    dbOffset = this->totalDataNum_ * this->unsortedDimParallel_;
    dbOffset = Ops::Base::CeilAlign(dbOffset * sizeof(T1), this->oneBlock_) / sizeof(T1);
    this->outValueDbWK_.SetGlobalBuffer((__gm__ T1*)(workspace + wkOffset), dbOffset);
    wkOffset = wkOffset + dbOffset * sizeof(T1);
    sortOutValueGM_.SetGlobalBuffer((__gm__ T1*)(workspace + wkOffset), dbOffset);

    InitSortBuffers();

    this->globalHistGmWkTmp_ = this->globalHistGmWk_.template ReinterpretCast<T3>();
    this->excusiveBinsGmWkTmp_ = this->excusiveBinsGmWk_.template ReinterpretCast<T3>();
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend, bool IS_BITONIC_SORT>
__aicore__ inline void SortAndTopKMoreCore<T1, T2, UT, T3, isDescend, IS_BITONIC_SORT>::InitSortBuffers()
{
    this->pipe_->InitBuffer(this->inQueueX_, 1, this->numTileData_ * sizeof(T1));
    this->pipe_->InitBuffer(this->inQueueIndex_, 1, this->numTileData_ * sizeof(T3));
    this->pipe_->InitBuffer(this->inQueueGlobalHist_, 1, Sort::RADIX_SORT_NUM * sizeof(T3));
    this->pipe_->InitBuffer(this->outValueQueue_, 1, this->numTileData_);
    this->pipe_->InitBuffer(this->blockExcusiveInQue_, 1, Sort::RADIX_SORT_NUM * sizeof(uint16_t));
    this->pipe_->InitBuffer(this->blockHistInQue_, 1, Sort::RADIX_SORT_NUM * sizeof(uint16_t));
    this->pipe_->InitBuffer(this->blockUbFlagQue_, 1, Sort::RADIX_SORT_NUM * sizeof(T3));
    this->pipe_->InitBuffer(this->inputB8Que_, 1, this->numTileData_);
    this->pipe_->InitBuffer(this->outIdxQueue_, 1, this->numTileData_ * sizeof(uint32_t));
    this->pipe_->InitBuffer(this->tmpUb_, this->tmpUbSize_);
    this->pipe_->InitBuffer(this->blockHistFlagUbQue_, 1, Sort::RADIX_SORT_NUM * sizeof(T3));
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend, bool IS_BITONIC_SORT>
__aicore__ inline void SortAndTopKMoreCore<T1, T2, UT, T3, isDescend, IS_BITONIC_SORT>::ParserTilingData()
{
    this->totalDataNum_ = tilingData_->lastAxisNum;                // h轴大小
    this->numTileData_ = tilingData_->numTileDataSize;             // ub循环块大小
    this->unsortedDimNum_ = tilingData_->unsortedDimNum;           // b轴大小
    this->unsortedDimParallel_ = tilingData_->unsortedDimParallel; // b轴使用的核数
    this->lastDimTileNum_ = tilingData_->lastDimTileNum;           // h轴循环次数
    this->sortLoopTimes_ = tilingData_->sortLoopTimes;             // b轴循环次数
    this->lastDimRealCore_ = tilingData_->lastDimNeedCore;         // h轴需要的核数
    this->tmpUbSize_ = tilingData_->tmpUbSize;                     // 高级api需要用的ub大小
    topKRealValue_ = tilingData_->topKRealValue;                   // k值
    tileDataSize_ = tilingData_->sortAndTopkTileDataSize;          // 获取topk时tile大小
    blockTileNum_ = tilingData_->sortAndTopkBlockTileNum;          // 获取topk时单核分配tile数量
    tailTileNum_ = tilingData_->sortAndTopkTailTileNum;            // 获取topk时尾部tile数量

    this->clearCore1_ = tilingData_->keyParams0;     // 用于清零的globalHistGmWk_的核
    this->clearCore0_ = tilingData_->keyParams1;     // 用于清零excusiveBinsGmWk_的核
    this->clearSize_ = tilingData_->keyParams2;      // 每次清零的ub大小，按照大的globalHistGmWk_所需ub算
    this->clearCout_ = tilingData_->keyParams3;      // 清零globalHistGmWk_ ub循环次数
    this->clearCoreSize0_ = tilingData_->keyParams4; // 清零excusiveBinsGmWk_,每个核处理多少个数
    this->clearCoreSize1_ = tilingData_->keyParams5; // 清零globalHistGmWk_，每个核处理多少
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend, bool IS_BITONIC_SORT>
__aicore__ inline void SortAndTopKMoreCore<T1, T2, UT, T3, isDescend, IS_BITONIC_SORT>::GetValueResData(
    uint32_t dataCopyLoopTimes, bool hasLastTile, int64_t outGmOffset, int64_t sortOutGmOffset)
{
    if (dataCopyLoopTimes == 0) {
        return;
    }
    int64_t sortOutValueGMOffset;
    int64_t outValueGmOffset;
    uint32_t tileDataSize = tileDataSize_;
    for (uint32_t i = 0; i < dataCopyLoopTimes; i++) {
        LocalTensor<T1> topkResTensor = topkResQueue_.AllocTensor<T1>();
        if (hasLastTile && i == dataCopyLoopTimes - 1) {
            tileDataSize = topKRealValue_ % tileDataSize_;
        }
        sortOutValueGMOffset = sortOutGmOffset + i * tileDataSize_;
        outValueGmOffset = outGmOffset + i * tileDataSize_;
        DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParam;
        dataCopyParam.blockCount = 1;
        dataCopyParam.blockLen = tileDataSize * sizeof(T1);
        dataCopyParam.srcStride = 0;
        dataCopyParam.dstStride = 0;
        DataCopyPad(topkResTensor, sortOutValueGM_[sortOutValueGMOffset], dataCopyParam, padParams);
        topkResQueue_.EnQue(topkResTensor);
        topkResTensor = topkResQueue_.DeQue<T1>();
        DataCopyPad(this->outValueGm_[outValueGmOffset], topkResTensor, dataCopyParam);
        topkResQueue_.FreeTensor(topkResTensor);
    }
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend, bool IS_BITONIC_SORT>
__aicore__ inline void SortAndTopKMoreCore<T1, T2, UT, T3, isDescend, IS_BITONIC_SORT>::GetIndexResData(
    uint32_t dataCopyLoopTimes, bool hasLastTile, int64_t outGmOffset, int64_t sortOutGmOffset)
{
    if (dataCopyLoopTimes == 0) {
        return;
    }
    GlobalTensor<T2> sortOutIdxTmpGM_ = sortOutIdxGM_.template ReinterpretCast<T2>();
    GlobalTensor<T2> outIdxTmpGm_ = this->outIdxGm_.template ReinterpretCast<T2>();
    int64_t sortOutIdxGMOffset;
    int64_t outIdxTmpGmOffset;
    uint32_t tileDataSize = tileDataSize_;
    for (uint32_t i = 0; i < dataCopyLoopTimes; i++) {
        LocalTensor<T2> topkResTensor = topkResQueue_.AllocTensor<T2>();
        if (hasLastTile && i == dataCopyLoopTimes - 1) {
            tileDataSize = topKRealValue_ % tileDataSize_;
        }
        sortOutIdxGMOffset = sortOutGmOffset + i * tileDataSize_;
        outIdxTmpGmOffset = outGmOffset + i * tileDataSize_;
        DataCopyPadExtParams<T2> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParam;
        dataCopyParam.blockCount = 1;
        dataCopyParam.blockLen = tileDataSize * sizeof(T2);
        dataCopyParam.srcStride = 0;
        dataCopyParam.dstStride = 0;
        DataCopyPad(topkResTensor, sortOutIdxTmpGM_[sortOutIdxGMOffset], dataCopyParam, padParams);
        topkResQueue_.EnQue(topkResTensor);
        topkResTensor = topkResQueue_.DeQue<T2>();
        DataCopyPad(outIdxTmpGm_[outIdxTmpGmOffset], topkResTensor, dataCopyParam);
        topkResQueue_.FreeTensor(topkResTensor);
    }
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend, bool IS_BITONIC_SORT>
__aicore__ inline void SortAndTopKMoreCore<T1, T2, UT, T3, isDescend, IS_BITONIC_SORT>::GetBitonicSmallTopKRes(
    uint32_t sortLoopRound)
{
    uint32_t k = static_cast<uint32_t>(topKRealValue_);
    uint32_t valueAlignCount = Ops::Base::CeilAlign(k * sizeof(T1), AGLIN_FACTOR) / sizeof(T1);
    uint32_t indexAlignCount = Ops::Base::CeilAlign(k * sizeof(T2), AGLIN_FACTOR) / sizeof(T2);
    this->pipe_->InitBuffer(topkResQueue_, 1, valueAlignCount * sizeof(T1));
    this->pipe_->InitBuffer(bitonicSmallIndexTBuf_, indexAlignCount * sizeof(T2));
    if (this->blockIdx_ == 0U) {
        GlobalTensor<T2> sortOutIndex = sortOutIdxGM_.template ReinterpretCast<T2>();
        GlobalTensor<T2> outputIndex = this->outIdxGm_.template ReinterpretCast<T2>();
        for (uint32_t row = 0U; row < this->unsortedDimParallel_; ++row) {
            int64_t sortOffset = static_cast<int64_t>(row) * this->totalDataNum_;
            int64_t outputOffset = static_cast<int64_t>(sortLoopRound) * this->unsortedDimParallel_ * k + row * k;
            LocalTensor<T1> valuesLocal = topkResQueue_.AllocTensor<T1>();
            LocalTensor<T2> indicesLocal = bitonicSmallIndexTBuf_.Get<T2>();
            DataCopyExtParams valueCopyParams{1U, static_cast<uint32_t>(k * sizeof(T1)), 0U, 0U, 0U};
            DataCopyExtParams indexCopyParams{1U, static_cast<uint32_t>(k * sizeof(T2)), 0U, 0U, 0U};
            DataCopyPadExtParams<T1> valuePadParams{true, 0U, static_cast<uint8_t>(valueAlignCount - k),
                                                    static_cast<T1>(0)};
            DataCopyPadExtParams<T2> indexPadParams{true, 0U, static_cast<uint8_t>(indexAlignCount - k),
                                                    static_cast<T2>(0)};
            DataCopyPad(valuesLocal, sortOutValueGM_[sortOffset], valueCopyParams, valuePadParams);
            DataCopyPad(indicesLocal, sortOutIndex[sortOffset], indexCopyParams, indexPadParams);
            event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);
            if constexpr (IS_BITONIC_SORT) {
                topkV2::RunBitonicSmallTopKFinalize<T1, T2, static_cast<bool>(isDescend)>(valuesLocal, indicesLocal, k,
                                                                                          1U, k, k);
            }
            event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(vToMte3);
            WaitFlag<HardEvent::V_MTE3>(vToMte3);
            DataCopyPad(this->outValueGm_[outputOffset], valuesLocal, valueCopyParams);
            DataCopyPad(outputIndex[outputOffset], indicesLocal, indexCopyParams);
            event_t mte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
            topkResQueue_.FreeTensor(valuesLocal);
        }
    }
    SyncAll();
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend, bool IS_BITONIC_SORT>
__aicore__ inline void SortAndTopKMoreCore<T1, T2, UT, T3, isDescend, IS_BITONIC_SORT>::GetTopKRes(
    uint32_t sortLoopRound)
{
    // 重置TPipe,申请新的UB空间
    this->pipe_->Reset();
    if constexpr (IS_BITONIC_SORT) {
        GetBitonicSmallTopKRes(sortLoopRound);
        return;
    }
    uint32_t maxDTypeSize = sizeof(T1) >= sizeof(T2) ? sizeof(T1) : sizeof(T2);
    this->pipe_->InitBuffer(topkResQueue_, 1, maxDTypeSize * tileDataSize_);

    // 当前block处理的tile数
    uint32_t dataCopyLoopTimes = (tailTileNum_ > 0 && this->blockIdx_ < tailTileNum_) ? blockTileNum_ + 1 :
                                                                                        blockTileNum_;
    // 判断当前block是否有尾块需要处理
    bool hasLastTile = false;
    if (blockTileNum_ > 0 && this->blockIdx_ == this->realCoreNum_ - 1) {
        hasLastTile = true;
    } else if (blockTileNum_ == 0 && this->blockIdx_ == tailTileNum_ - 1) {
        hasLastTile = true;
    }

    // 当前block处理的tile的起始索引
    int64_t blockStartIdx = 0;
    if (this->blockIdx_ < tailTileNum_) {
        blockStartIdx = (blockTileNum_ + 1) * this->blockIdx_;
    } else {
        blockStartIdx = (blockTileNum_ + 1) * tailTileNum_ + (this->blockIdx_ - tailTileNum_) * blockTileNum_;
    }

    // 输出GM地址偏移量
    int64_t outGmOffset = sortLoopRound * this->unsortedDimParallel_ * topKRealValue_ + blockStartIdx * tileDataSize_;
    // sortOutGM地址偏移量
    int64_t sortOutGmOffset = blockStartIdx * tileDataSize_;
    // 超大排序轴场景下，unsortedDimParallel_一般等于1
    for (uint32_t i = 0; i < this->unsortedDimParallel_; i++) {
        outGmOffset += i * topKRealValue_;
        sortOutGmOffset += i * this->totalDataNum_;
        GetValueResData(dataCopyLoopTimes, hasLastTile, outGmOffset, sortOutGmOffset);
        SyncAll();
        GetIndexResData(dataCopyLoopTimes, hasLastTile, outGmOffset, sortOutGmOffset);
        SyncAll();
    }
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend, bool IS_BITONIC_SORT>
__aicore__ inline void SortAndTopKMoreCore<T1, T2, UT, T3, isDescend, IS_BITONIC_SORT>::ProcessSortAndTopK(
    GlobalTensor<T1> inputXGm, int64_t gmOffset, uint32_t sortLoopRound)
{
    this->pipe_->Reset();
    InitSortBuffers();
    this->ClearWorkSapce();
    SyncAll();

    if constexpr (sizeof(T2) == sizeof(uint32_t)) {
        if constexpr (sizeof(T1) == sizeof(int8_t)) {
            this->inputXDbGm_.SetDoubleBuffer(this->outValueDbWK_, sortOutValueGM_);
            this->idxDbGm_.SetDoubleBuffer(this->outIdxDbWK_, sortOutIdxGM_);
        } else {
            this->inputXDbGm_.SetDoubleBuffer(sortOutValueGM_, this->outValueDbWK_);
            this->idxDbGm_.SetDoubleBuffer(sortOutIdxGM_, this->outIdxDbWK_);
        }
    } else {
        if constexpr (sizeof(T1) == sizeof(int8_t)) {
            this->inputXDbGm_.SetDoubleBuffer(this->outValueDbWK_, sortOutValueGM_);
            this->idxDbGm_.SetDoubleBuffer(this->outIdxDbWK_, sortOutIdxGM_);
        } else {
            this->inputXDbGm_.SetDoubleBuffer(sortOutValueGM_, this->outValueDbWK_);
            this->idxDbGm_.SetDoubleBuffer(sortOutIdxGM_, this->outIdxDbWK_);
        }
    }
    for (uint32_t round = 0; round < static_cast<uint32_t>(sizeof(T1)); round++) {
        this->GetGlobalExcusiveSum(round, sortLoopRound, inputXGm);
        SyncAll();
        this->ComputeOnePass(round, sortLoopRound, inputXGm);
        SyncAll();
    }
    GetTopKRes(sortLoopRound);
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend, bool IS_BITONIC_SORT>
__aicore__ inline void SortAndTopKMoreCore<T1, T2, UT, T3, isDescend, IS_BITONIC_SORT>::ProcessTopK()
{
    if (this->blockIdx_ >= this->realCoreNum_) {
        return;
    }
    for (uint32_t i = 0; i < this->sortLoopTimes_; i++) {
        int64_t loopOffset = i * this->unsortedDimParallel_ * this->totalDataNum_;
        ProcessSortAndTopK(this->inputXGm_[loopOffset], loopOffset, i);
    }
}

} // namespace SortAndTopK

#endif // SORT_AND_TOP_K_MORE_CORE_H
