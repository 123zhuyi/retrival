/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KTH_VALUE_RADIX_MORE_CORE_H
#define KTH_VALUE_RADIX_MORE_CORE_H

#include <cmath>
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "kth_value_tiling_data.h"
#include "../../sort/arch35/common/radix_more_core_base.h"
#include "../../sort/arch35/common/util_type_simd.h"

namespace KthValue {
using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;
using namespace RadixSortCommon;

struct KthValueRadixMoreInnerTilingData {
    uint32_t numTileDataSize;
    uint32_t unsortedDimParallel;
    uint32_t lastDimTileNum;
    uint32_t sortLoopTimes;
    uint32_t lastDimNeedCore;
    uint32_t keyParams0;
    uint32_t keyParams1;
    uint32_t keyParams2;
    uint32_t keyParams3;
    uint32_t keyParams4;
    uint32_t keyParams5;
    uint32_t tmpUbSize;
    int64_t lastAxisNum;
    int64_t unsortedDimNum;
};

// T1输入x dtype T2输出Idx dtype UT无符号的数据类型
template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
class KthValueRadixMoreInnerCore
    : public RadixSortCommon::RadixMoreCoreBase<KthValueRadixMoreInnerCore<T1, T2, UT, T3, isDescend>, T1, T2, UT, T3,
                                                isDescend> {
    using Base = RadixSortCommon::RadixMoreCoreBase<KthValueRadixMoreInnerCore<T1, T2, UT, T3, isDescend>, T1, T2, UT,
                                                    T3, isDescend>;
    friend Base;

public:
    __aicore__ inline KthValueRadixMoreInnerCore(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex, GM_ADDR workspace,
                                const KthValueRadixMoreInnerTilingData* __restrict tilingData, TPipe* pipe);
    __aicore__ inline void SetKthOutput(GM_ADDR value, GM_ADDR index, T3 kthIndex);

protected:
    __aicore__ inline void ParserTilingData();
    __aicore__ inline void ScatterKeysGlobal(LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal,
                                             LocalTensor<uint32_t> xInputIndexLocal,
                                             LocalTensor<uint8_t> sortedValueLocal,
                                             LocalTensor<uint16_t> blockExcusiveSum,
                                             LocalTensor<T3> blockDataInGlobalPos, LocalTensor<uint32_t> blockHistFlag,
                                             LocalTensor<uint16_t> blockHist, uint32_t round, T3 tileDataStart,
                                             uint32_t cureTileSize, uint32_t sortLoopRound);
    __aicore__ inline void ScatterOutB8Int32(LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal,
                                             LocalTensor<uint32_t> xInputIndexLocal,
                                             LocalTensor<uint8_t> sortedValueLocal,
                                             LocalTensor<uint16_t> blockExcusiveSum,
                                             LocalTensor<T3> blockDataInGlobalPos, LocalTensor<uint32_t> blockHistFlag,
                                             LocalTensor<uint16_t> blockHist, uint32_t round, T3 tileDataStart,
                                             uint32_t cureTileSize, uint32_t sortLoopRound);
    __aicore__ inline void ScatterOutB8Int64(LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal,
                                             LocalTensor<uint32_t> xInputIndexLocal,
                                             LocalTensor<uint8_t> sortedValueLocal,
                                             LocalTensor<uint16_t> blockExcusiveSum,
                                             LocalTensor<T3> blockDataInGlobalPos, LocalTensor<uint32_t> blockHistFlag,
                                             LocalTensor<uint16_t> blockHist, uint32_t round, T3 tileDataStart,
                                             uint32_t cureTileSize, uint32_t sortLoopRound);
    __aicore__ inline void ScatterOutInt32(LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal,
                                           LocalTensor<uint32_t> xInputIndexLocal,
                                           LocalTensor<uint8_t> sortedValueLocal,
                                           LocalTensor<uint16_t> blockExcusiveSum, LocalTensor<T3> blockDataInGlobalPos,
                                           LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist,
                                           uint32_t round, T3 tileDataStart, uint32_t cureTileSize,
                                           uint32_t sortLoopRound);
    __aicore__ inline void ScatterOutInt32ToInt64(
        LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal,
        LocalTensor<uint32_t> xInputIndexLocal, LocalTensor<uint8_t> sortedValueLocal,
        LocalTensor<uint16_t> blockExcusiveSum, LocalTensor<uint32_t> blockDataInGlobalPos,
        LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist, uint32_t round, T3 tileDataStart,
        uint32_t cureTileSize, uint32_t sortLoopRound);
    __aicore__ inline void ScatterOutInt64(LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal,
                                           LocalTensor<uint32_t> xInputIndexLocal,
                                           LocalTensor<uint8_t> sortedValueLocal,
                                           LocalTensor<uint16_t> blockExcusiveSum, LocalTensor<T3> blockDataInGlobalPos,
                                           LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist,
                                           uint32_t round, T3 tileDataStart, uint32_t cureTileSize,
                                           uint32_t sortLoopRound);

    const KthValueRadixMoreInnerTilingData* tilingData_;
    GlobalTensor<T1> kthValueGm_;
    GlobalTensor<T2> kthIndexGm_;
    T3 kthIndex_ = 0;
    bool writeKthOutput_ = false;
};

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void KthValueRadixMoreInnerCore<T1, T2, UT, T3, isDescend>::Init(
    GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex, GM_ADDR workspace,
    const KthValueRadixMoreInnerTilingData* __restrict tilingData, TPipe* pipe)
{
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
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

    uint64_t dbOffset = static_cast<uint64_t>(this->totalDataNum_) * this->unsortedDimParallel_;
    if constexpr (sizeof(T3) == sizeof(int64_t)) {
        dbOffset = dbOffset * 2;
    }
    dbOffset = Ops::Base::CeilAlign(dbOffset, oneBlockNumB32);
    this->outIdxDbWK_.SetGlobalBuffer((__gm__ uint32_t*)(workspace + wkOffset), dbOffset);
    wkOffset = wkOffset + dbOffset * static_cast<uint64_t>(sizeof(uint32_t));

    uint64_t histTileOffset = this->lastDimTileNum_ * RADIX_SORT_NUM * this->unsortedDimParallel_;
    this->histTileGmWk_.SetGlobalBuffer((__gm__ uint16_t*)(workspace + wkOffset), histTileOffset);
    wkOffset = wkOffset + histTileOffset * sizeof(uint16_t);
    this->histCumsumTileGmWk_.SetGlobalBuffer((__gm__ uint16_t*)(workspace + wkOffset), histTileOffset);
    wkOffset = wkOffset + histTileOffset * sizeof(uint16_t);

    uint64_t xB8Offset = static_cast<uint64_t>(this->lastDimTileNum_) * this->numTileData_ * this->unsortedDimParallel_;
    xB8Offset = Ops::Base::CeilAlign(xB8Offset, this->oneBlock_);
    this->xB8GmWk_.SetGlobalBuffer((__gm__ uint8_t*)(workspace + wkOffset), xB8Offset);
    wkOffset = wkOffset + xB8Offset * sizeof(uint8_t);

    dbOffset = static_cast<uint64_t>(this->totalDataNum_) * this->unsortedDimParallel_;
    dbOffset = Ops::Base::CeilAlign(dbOffset * sizeof(T1), this->oneBlock_) / sizeof(T1);
    this->outValueDbWK_.SetGlobalBuffer((__gm__ T1*)(workspace + wkOffset), dbOffset);

    this->pipe_->InitBuffer(this->inQueueX_, 1, this->numTileData_ * sizeof(T1));
    this->pipe_->InitBuffer(this->inQueueIndex_, 1, this->numTileData_ * sizeof(T3));
    this->pipe_->InitBuffer(this->inQueueGlobalHist_, 1, RADIX_SORT_NUM * sizeof(T3));
    this->pipe_->InitBuffer(this->outValueQueue_, 1, this->numTileData_);
    this->pipe_->InitBuffer(this->blockExcusiveInQue_, 1, RADIX_SORT_NUM * sizeof(uint16_t));
    this->pipe_->InitBuffer(this->blockHistInQue_, 1, RADIX_SORT_NUM * sizeof(uint16_t));
    this->pipe_->InitBuffer(this->blockUbFlagQue_, 1, RADIX_SORT_NUM * sizeof(T3));
    this->pipe_->InitBuffer(this->inputB8Que_, 1, this->numTileData_);
    this->pipe_->InitBuffer(this->outIdxQueue_, 1, this->numTileData_ * sizeof(uint32_t));
    this->pipe_->InitBuffer(this->tmpUb_, this->tmpUbSize_);
    this->pipe_->InitBuffer(this->blockHistFlagUbQue_, 1, RADIX_SORT_NUM * sizeof(T3));

    this->globalHistGmWkTmp_ = this->globalHistGmWk_.template ReinterpretCast<T3>();
    this->excusiveBinsGmWkTmp_ = this->excusiveBinsGmWk_.template ReinterpretCast<T3>();
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void KthValueRadixMoreInnerCore<T1, T2, UT, T3, isDescend>::SetKthOutput(GM_ADDR value, GM_ADDR index,
                                                                                           T3 kthIndex)
{
    kthValueGm_.SetGlobalBuffer((__gm__ T1*)value);
    kthIndexGm_.SetGlobalBuffer((__gm__ T2*)index);
    kthIndex_ = kthIndex;
    writeKthOutput_ = true;
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void KthValueRadixMoreInnerCore<T1, T2, UT, T3, isDescend>::ParserTilingData()
{
    this->totalDataNum_ = tilingData_->lastAxisNum;                // h轴大小
    this->numTileData_ = tilingData_->numTileDataSize;             // ub循环块大小
    this->unsortedDimNum_ = tilingData_->unsortedDimNum;           // b轴大小
    this->unsortedDimParallel_ = tilingData_->unsortedDimParallel; // b轴使用的核数
    this->lastDimTileNum_ = tilingData_->lastDimTileNum;           // h轴循环次数
    this->sortLoopTimes_ = tilingData_->sortLoopTimes;             // b轴循环次数
    this->lastDimRealCore_ = tilingData_->lastDimNeedCore;         // h轴需要的核数
    this->tmpUbSize_ = tilingData_->tmpUbSize;                     // 高级api需要用的ub大小

    this->clearCore1_ = tilingData_->keyParams0;     // 用于清零的globalHistGmWk_的核
    this->clearCore0_ = tilingData_->keyParams1;     // 用于清零excusiveBinsGmWk_的核
    this->clearSize_ = tilingData_->keyParams2;      // 每次清零的ub大小，按照大的globalHistGmWk_所需ub算
    this->clearCout_ = tilingData_->keyParams3;      // 清零globalHistGmWk_ ub循环次数
    this->clearCoreSize0_ = tilingData_->keyParams4; // 清零excusiveBinsGmWk_,每个核处理多少个数
    this->clearCoreSize1_ = tilingData_->keyParams5; // 清零globalHistGmWk_，每个核处理多少
}

/**
 * @brief Extract the kth value and its original index directly from the final radix scatter state.
 *
 * The first loop computes the global scatter base of every radix bucket for the current tile. After all SIMT
 * threads publish those bases, the second loop locates the bucket interval containing kthIndex and writes only
 * that element to the compact [B, 1] outputs. This avoids materializing the complete final sorted row.
 *
 * blockHistFlagAddr contains lookback counts with state bits in the high bits; the state bits must be removed
 * before calculating the scatter base. blockExcusiveSumAddr is the bucket start inside the tile, while
 * blockHistAddr is the number of elements in that bucket. Their combination defines the half-open global range
 * [globalBucketStart, globalBucketEnd) owned by this tile and bucket.
 *
 * @param tileDataStart Original row-local index of the first element in this tile.
 * @param unSortIdOffset Row/pass offset of the 256-bin global prefix table.
 * @param kthIndex Zero-based target position in the fully sorted row.
 * @param outputRow Row index in the compact KthValue outputs.
 * @param blockDataInGlobalPosAddr Shared UB scratch used to publish per-bucket global scatter bases.
 * @param sortedIndexLocalAddr Mapping from the radix-sorted local position to the source position in this tile.
 * @param xInputIndexLocalAddr Original indices carried from the previous radix round.
 */
template <typename T1, typename T2, typename T3, typename KthIdxT, int32_t round>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_NUM) __aicore__
    void CopyOutKthGm(T3 tileDataStart, uint64_t unSortIdOffset, T3 kthIndex, uint32_t outputRow,
                      __ubuf__ uint16_t* blockExcusiveSumAddr, __gm__ volatile T3* excusiveBinsGmAddr,
                      __ubuf__ T3* blockDataInGlobalPosAddr, __ubuf__ uint32_t* sortedIndexLocalAddr,
                      __ubuf__ T3* xInputIndexLocalAddr, __ubuf__ T1* xInputValueLocalAddr,
                      __ubuf__ T3* blockHistFlagAddr, __ubuf__ uint16_t* blockHistAddr,
                      __gm__ volatile T1* kthValueGmAddr, __gm__ volatile KthIdxT* kthIndexGmAddr)
{
    // Phase 1: compute the global scatter base for each bucket in this tile. The formula is identical to the
    // regular radix scatter path, but only the bucket containing kthIndex will be consumed below.
    for (int i = threadIdx.x; i < RADIX_SORT_NUM; i += THREAD_DIM_NUM) {
        T3 blockHistCumsumVal = blockHistFlagAddr[i];
        // Lookback stores its readiness state in the high bits; retain only the accumulated element count.
        if constexpr (IsSameType<T3, uint32_t>::value) {
            blockHistCumsumVal = blockHistCumsumVal & VALUE_MASK;
        } else {
            blockHistCumsumVal = blockHistCumsumVal & VALUE_MASK_B64;
        }
        uint32_t blockHistVal = blockHistAddr[i];
        uint32_t blockExcusiveSumVal = blockExcusiveSumAddr[i];
        T3 globalKeyOffsetVal = excusiveBinsGmAddr[unSortIdOffset + i];
        T3 finalpos = globalKeyOffsetVal + blockHistCumsumVal - blockHistVal - blockExcusiveSumVal;
        blockDataInGlobalPosAddr[i] = finalpos;
    }
    // All lanes in phase 2 may read bases written by other lanes in phase 1.
    asc_syncthreads();

    // Phase 2: find the unique tile/bucket interval containing kthIndex and emit one value/index pair.
    for (int i = threadIdx.x; i < RADIX_SORT_NUM; i += THREAD_DIM_NUM) {
        T3 localBucketStart = static_cast<T3>(blockExcusiveSumAddr[i]);
        T3 localBucketEnd = localBucketStart + static_cast<T3>(blockHistAddr[i]);
        T3 globalBucketStart = blockDataInGlobalPosAddr[i] + localBucketStart;
        T3 globalBucketEnd = blockDataInGlobalPosAddr[i] + localBucketEnd;
        if (kthIndex >= globalBucketStart && kthIndex < globalBucketEnd) {
            // Convert the row-global kth position to a position in the locally radix-sorted index array.
            T3 sortedLocalPos = kthIndex - blockDataInGlobalPosAddr[i];
            T3 localDataIndex = static_cast<T3>(sortedIndexLocalAddr[static_cast<uint32_t>(sortedLocalPos)]);
            T3 dataInitIndex = 0;
            // Round 0 still refers directly to the input tile. Later rounds carry the original index through the
            // double-buffered index workspace.
            if constexpr (round != 0) {
                dataInitIndex = xInputIndexLocalAddr[localDataIndex];
            } else {
                dataInitIndex = tileDataStart + localDataIndex;
            }
            kthValueGmAddr[outputRow] = xInputValueLocalAddr[localDataIndex];
            kthIndexGmAddr[outputRow] = static_cast<KthIdxT>(dataInitIndex);
        }
    }
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void KthValueRadixMoreInnerCore<T1, T2, UT, T3, isDescend>::ScatterOutInt32(
    LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal, LocalTensor<uint32_t> xInputIndexLocal,
    LocalTensor<uint8_t> sortedValueLocal, LocalTensor<uint16_t> blockExcusiveSum, LocalTensor<T3> blockDataInGlobalPos,
    LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist, uint32_t round, T3 tileDataStart,
    uint32_t cureTileSize, uint32_t sortLoopRound)
{
    uint32_t unSortId = this->blockIdx_ / this->lastDimRealCore_;
    uint64_t outputXUnsortedAxisOffset = static_cast<uint64_t>(unSortId) * static_cast<uint64_t>(this->totalDataNum_);
    uint64_t unSortIdOffset = static_cast<uint64_t>(unSortId) * RADIX_SORT_NUM * sizeof(T1) + round * RADIX_SORT_NUM;
    uint64_t outputRow = static_cast<uint64_t>(sortLoopRound) * this->unsortedDimParallel_ + unSortId;
    bool writeKthOutput = writeKthOutput_ && round == static_cast<uint32_t>(sizeof(T1) - 1) &&
                          outputRow < static_cast<uint64_t>(this->unsortedDimNum_);
    if (round == 0) {
        if (writeKthOutput) {
            asc_vf_call<CopyOutKthGm<T1, uint32_t, T3, uint32_t, 0>>(
                dim3(THREAD_DIM_NUM), tileDataStart, unSortIdOffset, kthIndex_, outputRow,
                (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
                (__ubuf__ T3*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
                (__ubuf__ T3*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()),
                (__ubuf__ T3*)(blockHistFlag.GetPhyAddr()), (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()),
                (__gm__ T1*)(kthValueGm_.GetPhyAddr()), (__gm__ uint32_t*)(kthIndexGm_.GetPhyAddr()));
        } else {
            asc_vf_call<CopyOutGm<T1, uint32_t, T3, uint32_t, 0>>(
                dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize, outputXUnsortedAxisOffset, unSortIdOffset,
                (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
                (__ubuf__ T3*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
                (__ubuf__ T3*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(sortedValueLocal.GetPhyAddr()),
                (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ T3*)(blockHistFlag.GetPhyAddr()),
                (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()),
                (__gm__ uint32_t*)(this->idxDbGm_.Alternate().GetPhyAddr()),
                (__gm__ T1*)(this->inputXDbGm_.Alternate().GetPhyAddr()));
        }
    } else {
        if (writeKthOutput) {
            asc_vf_call<CopyOutKthGm<T1, uint32_t, T3, uint32_t, 1>>(
                dim3(THREAD_DIM_NUM), tileDataStart, unSortIdOffset, kthIndex_, outputRow,
                (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
                (__ubuf__ T3*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
                (__ubuf__ T3*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()),
                (__ubuf__ T3*)(blockHistFlag.GetPhyAddr()), (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()),
                (__gm__ T1*)(kthValueGm_.GetPhyAddr()), (__gm__ uint32_t*)(kthIndexGm_.GetPhyAddr()));
        } else {
            asc_vf_call<CopyOutGm<T1, uint32_t, T3, uint32_t, 1>>(
                dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize, outputXUnsortedAxisOffset, unSortIdOffset,
                (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
                (__ubuf__ T3*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
                (__ubuf__ T3*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(sortedValueLocal.GetPhyAddr()),
                (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ T3*)(blockHistFlag.GetPhyAddr()),
                (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()),
                (__gm__ uint32_t*)(this->idxDbGm_.Alternate().GetPhyAddr()),
                (__gm__ T1*)(this->inputXDbGm_.Alternate().GetPhyAddr()));
        }
    }
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void KthValueRadixMoreInnerCore<T1, T2, UT, T3, isDescend>::ScatterOutInt32ToInt64(
    LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal, LocalTensor<uint32_t> xInputIndexLocal,
    LocalTensor<uint8_t> sortedValueLocal, LocalTensor<uint16_t> blockExcusiveSum,
    LocalTensor<uint32_t> blockDataInGlobalPos, LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist,
    uint32_t round, T3 tileDataStart, uint32_t cureTileSize, uint32_t sortLoopRound)
{
    uint32_t unSortId = this->blockIdx_ / this->lastDimRealCore_;
    uint64_t outputXUnsortedAxisOffset = static_cast<uint64_t>(unSortId) * static_cast<uint64_t>(this->totalDataNum_);
    uint64_t unSortIdOffset = static_cast<uint64_t>(unSortId) * RADIX_SORT_NUM * sizeof(T1) + round * RADIX_SORT_NUM;
    uint64_t outputRow = static_cast<uint64_t>(sortLoopRound) * this->unsortedDimParallel_ + unSortId;
    bool writeKthOutput = writeKthOutput_ && round == static_cast<uint32_t>(sizeof(T1) - 1) &&
                          outputRow < static_cast<uint64_t>(this->unsortedDimNum_);
    if (round == 0) {
        asc_vf_call<CopyOutGm<T1, uint32_t, T3, uint32_t, 0>>(
            dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize, outputXUnsortedAxisOffset, unSortIdOffset,
            (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
            (__ubuf__ T3*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
            (__ubuf__ uint32_t*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(sortedValueLocal.GetPhyAddr()),
            (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ uint32_t*)(blockHistFlag.GetPhyAddr()),
            (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()), (__gm__ uint32_t*)(this->idxDbGm_.Alternate().GetPhyAddr()),
            (__gm__ T1*)(this->inputXDbGm_.Alternate().GetPhyAddr()));
    } else if (round < static_cast<uint32_t>(sizeof(T1) - 1)) {
        asc_vf_call<CopyOutGm<T1, uint32_t, T3, uint32_t, 1>>(
            dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize, outputXUnsortedAxisOffset, unSortIdOffset,
            (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
            (__ubuf__ T3*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
            (__ubuf__ uint32_t*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(sortedValueLocal.GetPhyAddr()),
            (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ uint32_t*)(blockHistFlag.GetPhyAddr()),
            (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()), (__gm__ uint32_t*)(this->idxDbGm_.Alternate().GetPhyAddr()),
            (__gm__ T1*)(this->inputXDbGm_.Alternate().GetPhyAddr()));
    } else {
        if (writeKthOutput) {
            asc_vf_call<CopyOutKthGm<T1, uint32_t, T3, int64_t, 1>>(
                dim3(THREAD_DIM_NUM), tileDataStart, unSortIdOffset, kthIndex_, outputRow,
                (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
                (__ubuf__ T3*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
                (__ubuf__ uint32_t*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()),
                (__ubuf__ uint32_t*)(blockHistFlag.GetPhyAddr()), (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()),
                (__gm__ T1*)(kthValueGm_.GetPhyAddr()), (__gm__ int64_t*)(kthIndexGm_.GetPhyAddr()));
        } else {
            asc_vf_call<CopyOutGm<T1, uint32_t, T3, uint32_t, 1>>(
                dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize, outputXUnsortedAxisOffset, unSortIdOffset,
                (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
                (__ubuf__ T3*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
                (__ubuf__ uint32_t*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(sortedValueLocal.GetPhyAddr()),
                (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ uint32_t*)(blockHistFlag.GetPhyAddr()),
                (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()),
                (__gm__ uint32_t*)(this->idxDbGm_.Alternate().GetPhyAddr()),
                (__gm__ T1*)(this->inputXDbGm_.Alternate().GetPhyAddr()));
        }
    }
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void KthValueRadixMoreInnerCore<T1, T2, UT, T3, isDescend>::ScatterOutInt64(
    LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal, LocalTensor<uint32_t> xInputIndexLocal,
    LocalTensor<uint8_t> sortedValueLocal, LocalTensor<uint16_t> blockExcusiveSum, LocalTensor<T3> blockDataInGlobalPos,
    LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist, uint32_t round, T3 tileDataStart,
    uint32_t cureTileSize, uint32_t sortLoopRound)
{
    uint32_t unSortId = this->blockIdx_ / this->lastDimRealCore_;
    uint64_t outputXUnsortedAxisOffset = static_cast<uint64_t>(unSortId) * static_cast<uint64_t>(this->totalDataNum_);
    uint64_t unSortIdOffset = static_cast<uint64_t>(unSortId) * RADIX_SORT_NUM * sizeof(T1) + round * RADIX_SORT_NUM;
    uint64_t outputRow = static_cast<uint64_t>(sortLoopRound) * this->unsortedDimParallel_ + unSortId;
    bool writeKthOutput = writeKthOutput_ && round == static_cast<uint32_t>(sizeof(T1) - 1) &&
                          outputRow < static_cast<uint64_t>(this->unsortedDimNum_);

    if (round == 0) {
        if (writeKthOutput) {
            asc_vf_call<CopyOutKthGm<T1, T2, T3, T2, 0>>(
                dim3(THREAD_DIM_NUM), tileDataStart, unSortIdOffset, kthIndex_, outputRow,
                (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
                (__ubuf__ T3*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
                (__ubuf__ T3*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()),
                (__ubuf__ T3*)(blockHistFlag.GetPhyAddr()), (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()),
                (__gm__ T1*)(kthValueGm_.GetPhyAddr()), (__gm__ T2*)(kthIndexGm_.GetPhyAddr()));
        } else {
            asc_vf_call<CopyOutGm<T1, T2, T3, T2, 0>>(
                dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize, outputXUnsortedAxisOffset, unSortIdOffset,
                (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
                (__ubuf__ T3*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
                (__ubuf__ T3*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(sortedValueLocal.GetPhyAddr()),
                (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ T3*)(blockHistFlag.GetPhyAddr()),
                (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()), (__gm__ T2*)(this->idxDbGm_.Alternate().GetPhyAddr()),
                (__gm__ T1*)(this->inputXDbGm_.Alternate().GetPhyAddr()));
        }
    } else {
        if (writeKthOutput) {
            asc_vf_call<CopyOutKthGm<T1, T2, T3, T2, 1>>(
                dim3(THREAD_DIM_NUM), tileDataStart, unSortIdOffset, kthIndex_, outputRow,
                (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
                (__ubuf__ T3*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
                (__ubuf__ T3*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()),
                (__ubuf__ T3*)(blockHistFlag.GetPhyAddr()), (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()),
                (__gm__ T1*)(kthValueGm_.GetPhyAddr()), (__gm__ T2*)(kthIndexGm_.GetPhyAddr()));
        } else {
            asc_vf_call<CopyOutGm<T1, T2, T3, T2, 1>>(
                dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize, outputXUnsortedAxisOffset, unSortIdOffset,
                (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
                (__ubuf__ T3*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
                (__ubuf__ T3*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(sortedValueLocal.GetPhyAddr()),
                (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ T3*)(blockHistFlag.GetPhyAddr()),
                (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()), (__gm__ T2*)(this->idxDbGm_.Alternate().GetPhyAddr()),
                (__gm__ T1*)(this->inputXDbGm_.Alternate().GetPhyAddr()));
        }
    }
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void KthValueRadixMoreInnerCore<T1, T2, UT, T3, isDescend>::ScatterOutB8Int32(
    LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal, LocalTensor<uint32_t> xInputIndexLocal,
    LocalTensor<uint8_t> sortedValueLocal, LocalTensor<uint16_t> blockExcusiveSum, LocalTensor<T3> blockDataInGlobalPos,
    LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist, uint32_t round, T3 tileDataStart,
    uint32_t cureTileSize, uint32_t sortLoopRound)
{
    uint32_t unSortId = this->blockIdx_ / this->lastDimRealCore_;
    uint64_t outputXUnsortedAxisOffset = static_cast<uint64_t>(unSortId) * static_cast<uint64_t>(this->totalDataNum_);
    uint64_t unSortIdOffset = static_cast<uint64_t>(unSortId) * RADIX_SORT_NUM * sizeof(T1) + round * RADIX_SORT_NUM;
    uint64_t outputRow = static_cast<uint64_t>(sortLoopRound) * this->unsortedDimParallel_ + unSortId;
    bool writeKthOutput = writeKthOutput_ && outputRow < static_cast<uint64_t>(this->unsortedDimNum_);
    if (writeKthOutput) {
        asc_vf_call<CopyOutKthGm<T1, uint32_t, T3, int64_t, 0>>(
            dim3(THREAD_DIM_NUM), tileDataStart, unSortIdOffset, kthIndex_, outputRow,
            (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
            (__ubuf__ T3*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
            (__ubuf__ uint32_t*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()),
            (__ubuf__ T3*)(blockHistFlag.GetPhyAddr()), (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()),
            (__gm__ T1*)(kthValueGm_.GetPhyAddr()), (__gm__ int64_t*)(kthIndexGm_.GetPhyAddr()));
    } else {
        asc_vf_call<CopyOutGm<T1, uint32_t, T3, uint32_t, 0>>(
            dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize, outputXUnsortedAxisOffset, unSortIdOffset,
            (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
            (__ubuf__ T3*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
            (__ubuf__ uint32_t*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(sortedValueLocal.GetPhyAddr()),
            (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ T3*)(blockHistFlag.GetPhyAddr()),
            (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()), (__gm__ uint32_t*)(this->idxDbGm_.Alternate().GetPhyAddr()),
            (__gm__ T1*)(this->inputXDbGm_.Alternate().GetPhyAddr()));
    }
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void KthValueRadixMoreInnerCore<T1, T2, UT, T3, isDescend>::ScatterOutB8Int64(
    LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal, LocalTensor<uint32_t> xInputIndexLocal,
    LocalTensor<uint8_t> sortedValueLocal, LocalTensor<uint16_t> blockExcusiveSum, LocalTensor<T3> blockDataInGlobalPos,
    LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist, uint32_t round, T3 tileDataStart,
    uint32_t cureTileSize, uint32_t sortLoopRound)
{
    uint32_t unSortId = this->blockIdx_ / this->lastDimRealCore_;
    uint64_t outputXUnsortedAxisOffset = static_cast<uint64_t>(unSortId) * static_cast<uint64_t>(this->totalDataNum_);
    uint64_t unSortIdOffset = static_cast<uint64_t>(unSortId) * RADIX_SORT_NUM * sizeof(T1) + round * RADIX_SORT_NUM;
    uint64_t outputRow = static_cast<uint64_t>(sortLoopRound) * this->unsortedDimParallel_ + unSortId;
    bool writeKthOutput = writeKthOutput_ && outputRow < static_cast<uint64_t>(this->unsortedDimNum_);
    if (writeKthOutput) {
        asc_vf_call<CopyOutKthGm<T1, T2, T3, T2, 0>>(
            dim3(THREAD_DIM_NUM), tileDataStart, unSortIdOffset, kthIndex_, outputRow,
            (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
            (__ubuf__ T3*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
            (__ubuf__ T3*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()),
            (__ubuf__ T3*)(blockHistFlag.GetPhyAddr()), (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()),
            (__gm__ T1*)(kthValueGm_.GetPhyAddr()), (__gm__ T2*)(kthIndexGm_.GetPhyAddr()));
    } else {
        asc_vf_call<CopyOutGm<T1, T2, T3, T2, 0>>(
            dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize, outputXUnsortedAxisOffset, unSortIdOffset,
            (__ubuf__ uint16_t*)(blockExcusiveSum.GetPhyAddr()), (__gm__ T3*)(this->excusiveBinsGmWk_.GetPhyAddr()),
            (__ubuf__ T3*)(blockDataInGlobalPos.GetPhyAddr()), (__ubuf__ uint32_t*)(sortedIndexLocal.GetPhyAddr()),
            (__ubuf__ T3*)(xInputIndexLocal.GetPhyAddr()), (__ubuf__ uint8_t*)(sortedValueLocal.GetPhyAddr()),
            (__ubuf__ T1*)(xInputValueLocal.GetPhyAddr()), (__ubuf__ T3*)(blockHistFlag.GetPhyAddr()),
            (__ubuf__ uint16_t*)(blockHist.GetPhyAddr()), (__gm__ T2*)(this->idxDbGm_.Alternate().GetPhyAddr()),
            (__gm__ T1*)(this->inputXDbGm_.Alternate().GetPhyAddr()));
    }
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void KthValueRadixMoreInnerCore<T1, T2, UT, T3, isDescend>::ScatterKeysGlobal(
    LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal, LocalTensor<uint32_t> xInputIndexLocal,
    LocalTensor<uint8_t> sortedValueLocal, LocalTensor<uint16_t> blockExcusiveSum, LocalTensor<T3> blockDataInGlobalPos,
    LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist, uint32_t round, T3 tileDataStart,
    uint32_t cureTileSize, uint32_t sortLoopRound)
{
    if constexpr (sizeof(T1) == sizeof(int8_t)) {
        // One-byte input has a single radix pass, so scatter directly with the selected index width.
        if constexpr (IsSameType<T3, uint32_t>::value) {
            ScatterOutB8Int32(xInputValueLocal, sortedIndexLocal, xInputIndexLocal, sortedValueLocal, blockExcusiveSum,
                              blockDataInGlobalPos, blockHistFlag, blockHist, round, tileDataStart, cureTileSize,
                              sortLoopRound);
        } else {
            ScatterOutB8Int64(xInputValueLocal, sortedIndexLocal, xInputIndexLocal, sortedValueLocal, blockExcusiveSum,
                              blockDataInGlobalPos, blockHistFlag, blockHist, round, tileDataStart, cureTileSize,
                              sortLoopRound);
        }
    } else if constexpr (sizeof(T2) == sizeof(int32_t)) {
        // 输出idx本省就是int32，无需cast
        ScatterOutInt32(xInputValueLocal, sortedIndexLocal, xInputIndexLocal, sortedValueLocal, blockExcusiveSum,
                        blockDataInGlobalPos, blockHistFlag, blockHist, round, tileDataStart, cureTileSize,
                        sortLoopRound);
    } else if constexpr (IsSameType<T3, uint32_t>::value) {
        // 输出idx是int64，需要在最后一次scatter时cast为int64
        ScatterOutInt32ToInt64(xInputValueLocal, sortedIndexLocal, xInputIndexLocal, sortedValueLocal, blockExcusiveSum,
                               blockDataInGlobalPos, blockHistFlag, blockHist, round, tileDataStart, cureTileSize,
                               sortLoopRound);
    } else {
        // 计算过程中idx使用int64
        ScatterOutInt64(xInputValueLocal, sortedIndexLocal, xInputIndexLocal, sortedValueLocal, blockExcusiveSum,
                        blockDataInGlobalPos, blockHistFlag, blockHist, round, tileDataStart, cureTileSize,
                        sortLoopRound);
    }
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
class KthValueRadixMoreBatchCore : public KthValueRadixMoreInnerCore<T1, T2, UT, T3, isDescend> {
public:
    __aicore__ inline void ProcessBatch(uint64_t inputOffset, uint32_t sortLoopRound)
    {
        this->ProcessRadix(this->inputXGm_[inputOffset], 0, sortLoopRound);
    }
};

template <typename T, typename T3, typename KeyT>
class KthValueRadixMoreCore {
public:
    __aicore__ inline KthValueRadixMoreCore(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace,
                                const KthValueTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ParserTilingData();
    __aicore__ inline void FillRadixTilingData();

    GM_ADDR x_ = nullptr;
    GM_ADDR values_ = nullptr;
    GM_ADDR indices_ = nullptr;
    GM_ADDR sortedValueWorkspace_ = nullptr;
    GM_ADDR sortedIndexWorkspace_ = nullptr;
    GM_ADDR sortWorkspace_ = nullptr;

    GlobalTensor<T> outValueGm_;
    GlobalTensor<int64_t> outIdxGm_;

    TPipe* pipe_ = nullptr;
    const KthValueTilingData* tilingData_ = nullptr;
    KthValueRadixMoreInnerTilingData radixTilingData_;

    int64_t totalDataNum_ = 0;
    int64_t unsortedDimNum_ = 0;
    uint32_t unsortedDimParallel_ = 0;
    uint32_t lastDimNeedCore_ = 0;
    uint32_t sortLoopTimes_ = 0;
    T3 kthIndex_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t realCoreNum_ = 0;
};

template <typename T, typename T3, typename KeyT>
__aicore__ inline void KthValueRadixMoreCore<T, T3, KeyT>::Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                                                                GM_ADDR workspace, const KthValueTilingData* tilingData,
                                                                TPipe* pipe)
{
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
    x_ = x;
    values_ = values;
    indices_ = indices;
    pipe_ = pipe;
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    realCoreNum_ = GetBlockNum();
    ParserTilingData();
    FillRadixTilingData();

    uint64_t fullValueBytes = static_cast<uint64_t>(totalDataNum_) * static_cast<uint64_t>(unsortedDimParallel_) *
                              static_cast<uint64_t>(sizeof(T));
    uint64_t fullIndexBytes = static_cast<uint64_t>(totalDataNum_) * static_cast<uint64_t>(unsortedDimParallel_) *
                              static_cast<uint64_t>(sizeof(T3));
    uint64_t valueWorkspaceBytes = ROUND_UP_AGLIN_UINT64(fullValueBytes);
    uint64_t indexWorkspaceBytes = ROUND_UP_AGLIN_UINT64(fullIndexBytes);

    // Workspace is split into one batch-sized sorted value area, one batch-sized sorted index area,
    // and the inner radix workspace. It is reused for every sortLoopTimes_ batch.
    sortedValueWorkspace_ = workspace;
    sortedIndexWorkspace_ = workspace + valueWorkspaceBytes;
    sortWorkspace_ = workspace + valueWorkspaceBytes + indexWorkspaceBytes;

    outValueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(values));
    outIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices));
}

template <typename T, typename T3, typename KeyT>
__aicore__ inline void KthValueRadixMoreCore<T, T3, KeyT>::ParserTilingData()
{
    totalDataNum_ = tilingData_->lastAxisNum;
    unsortedDimNum_ = tilingData_->unsortedDimNum;
    unsortedDimParallel_ = tilingData_->unsortedDimParallel;
    lastDimNeedCore_ = tilingData_->lastDimNeedCore;
    sortLoopTimes_ = tilingData_->sortLoopTimes;
    kthIndex_ = static_cast<T3>(tilingData_->kthIndex);
}

template <typename T, typename T3, typename KeyT>
__aicore__ inline void KthValueRadixMoreCore<T, T3, KeyT>::FillRadixTilingData()
{
    // Repackage KthValue tiling data for the local radix sorter. Kth output state is not part of
    // this struct because SetKthOutput owns the final [B, 1] writeback.
    radixTilingData_.numTileDataSize = tilingData_->numTileDataSize;
    radixTilingData_.unsortedDimParallel = tilingData_->unsortedDimParallel;
    radixTilingData_.lastDimTileNum = tilingData_->lastDimTileNum;
    radixTilingData_.sortLoopTimes = tilingData_->sortLoopTimes;
    radixTilingData_.lastDimNeedCore = tilingData_->lastDimNeedCore;
    radixTilingData_.keyParams0 = tilingData_->keyParams0;
    radixTilingData_.keyParams1 = tilingData_->keyParams1;
    radixTilingData_.keyParams2 = tilingData_->keyParams2;
    radixTilingData_.keyParams3 = tilingData_->keyParams3;
    radixTilingData_.keyParams4 = tilingData_->keyParams4;
    radixTilingData_.keyParams5 = tilingData_->keyParams5;
    radixTilingData_.tmpUbSize = tilingData_->tmpUbSize;
    radixTilingData_.lastAxisNum = tilingData_->lastAxisNum;
    radixTilingData_.unsortedDimNum = tilingData_->unsortedDimNum;
}

template <typename T, typename T3, typename KeyT>
__aicore__ inline void KthValueRadixMoreCore<T, T3, KeyT>::Process()
{
    KthValueRadixMoreBatchCore<T, int64_t, KeyT, T3, 0> radixOp;
    radixOp.Init(x_, sortedValueWorkspace_, sortedIndexWorkspace_, sortWorkspace_, &radixTilingData_, pipe_);
    radixOp.SetKthOutput(values_, indices_, kthIndex_);
    for (uint32_t loopIdx = 0; loopIdx < sortLoopTimes_; ++loopIdx) {
        // Process one unsortedDimParallel_ batch per loop. The inner radix core sorts into the
        // reusable workspace and only row-leading h-cores emit kth results.
        uint64_t inputOffset = static_cast<uint64_t>(loopIdx) * static_cast<uint64_t>(unsortedDimParallel_) *
                               static_cast<uint64_t>(totalDataNum_);
        radixOp.ProcessBatch(inputOffset, loopIdx);
    }
}

} // namespace KthValue

#endif
