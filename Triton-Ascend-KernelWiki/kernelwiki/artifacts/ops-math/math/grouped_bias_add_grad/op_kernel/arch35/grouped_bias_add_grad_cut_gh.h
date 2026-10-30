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
 * \file group_bias_add_grad_cut_gh.h
 * \brief group_bias_add_grad_cut_gh template
 */

#ifndef GROUPED_BIAS_ADD_GRAD_CUT_GH_H
#define GROUPED_BIAS_ADD_GRAD_CUT_GH_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "grouped_bias_add_grad_struct.h"
#include "group_utils.h"

namespace GroupedBiasAddGrad {
using namespace AscendC;

template <typename T, typename U>
class GroupedBiasAddGradCutGH {
    using PromoteDataT = float;

    constexpr static int64_t ALIGN_BYTE_32 = 32;
    constexpr static int32_t BUFFER_NUM = 4;
    constexpr static float ZERO_VALUE = 0.0;
    constexpr static uint64_t H_CUT_BLOCK_SIZE = 512;
    constexpr static uint64_t CS_BUF_SIZE = 32;
    constexpr static uint64_t BLOCK_UB_SIZE = ALIGN_BYTE_32 / sizeof(T);
    constexpr static int32_t ELEMENT_ONE_REPEAT_COMPUTE = Ops::Base::GetVRegSize() / sizeof(PromoteDataT);

#ifdef __CCE_AICORE__
    constexpr static AscendC::Reg::CastTrait castTrait0 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
#endif

public:
    __aicore__ inline GroupedBiasAddGradCutGH(TPipe& pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR groupIdx, GM_ADDR y,
                                const GroupedBiasAddGradCutGTilingData* __restrict tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcMaxHLen(int64_t currRowWidth, int64_t currHIdx, int64_t maxProcessBlocks,
                                       int64_t& processHBlocks, int64_t& processHLen);
    __aicore__ inline void CopyGroupIdx();
    __aicore__ inline void UpdateCurParam(int64_t currRowIdx, int64_t& currRowCumSum, int64_t& currRowWidth);
    __aicore__ inline void ProcessZeroG(int64_t currRowIdx, int64_t currHIdx, int64_t dimH);
    __aicore__ inline void ProcessOneG(int64_t currRowIdx, int64_t currHIdx, int64_t currRowWidth,
                                       int64_t currRowCumSum, int64_t loop);
    __aicore__ inline void ProcessMultiHBlock(int64_t currRowIdx, int64_t currHIdx, int64_t currRowWidth,
                                              int64_t currRowCumSum, int64_t loop);
    __aicore__ inline void ProcessSingleHBlock(int64_t currHLen, int64_t currRowIdx, int64_t currHIdx,
                                               int64_t currRowWidth, int64_t currRowCumSum);
    __aicore__ inline void CopyDataIn(int64_t burstLen, int64_t burstNum, int64_t srcGmOffset);
    __aicore__ inline void CopyDataInForOne(int64_t burstLen, int64_t burstNum, int64_t srcGmOffset);
    __aicore__ inline void CopyDataOut(int64_t outputDataOffset, int64_t dimH);
    __aicore__ inline void CopyDataOutForBuf(int64_t outputDataOffset, LocalTensor<PromoteDataT>& ubRes, int64_t dimH);
    __aicore__ inline void CopyDataOutUnCast(int64_t outputDataOffset, int64_t dimH, bool isOne);
    __aicore__ inline void BinaryReduceSum(int64_t currRowCumSum, int64_t currRowIdx, int64_t currHIdx,
                                           int64_t currRowWidth, int64_t singleH);
    __aicore__ inline void UpdateCacheAux(const int64_t cacheID, const int64_t stride, const int64_t count);
    __aicore__ inline void SetGParam(int64_t g, int64_t singleH);

private:
    const GroupedBiasAddGradCutGTilingData* tiling_;
    TPipe& pipe_;

    int32_t blockIdx_ = 0;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<U> groupIdxGm_;

    TQue<QuePosition::VECIN, 1> xQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> outQue_;
    TQue<TPosition::VECOUT, 1> zeroOutQue_;
    TQue<QuePosition::VECIN, 1> groupIdxQue_;

    TBuf<QuePosition::VECCALC> groupIdxCastBuf_;
    TBuf<QuePosition::VECCALC> tempBufBuf_;
    TBuf<QuePosition::VECCALC> tempResBuf_;
    TBuf<QuePosition::VECCALC> computeBuf_;
    TBuf<QuePosition::VECCALC> cumSumBuf_;

    LocalTensor<U> groupIdxLocal_;
    LocalTensor<int64_t> groupIdxCastLocal_;
    LocalTensor<PromoteDataT> tempLocal_;
    LocalTensor<int64_t> cumSumLocal_;

    int64_t BlockStartIdx_;
    int64_t currBlockFactor_;
    int64_t groupedIdxSize_;
    int64_t inputCol_;
    int64_t inputRow_;
    int64_t inUbSize_;
    bool groupIdxType_;
    int64_t cutHDim_;
    int64_t cutGDim_;
    int64_t hTailFactor_;
    int64_t maxPerHLen_;

    int64_t blockGNum_ = 0;
    int64_t mainG_ = 0;
    int64_t tailG_ = 0;
    int64_t bisectionPos_ = 0;
    int64_t cacheCount_ = 0;
    int64_t bisectionTail_ = 0;
    int64_t curProcessHBlocks_ = 0;

    LocalTensor<PromoteDataT> tempRes_;
    LocalTensor<PromoteDataT> computeRes_;
    LocalTensor<PromoteDataT> tempBuf_;
    LocalTensor<PromoteDataT> xTensor_;
};

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::Init(GM_ADDR x, GM_ADDR groupIdx, GM_ADDR y,
                                                           const GroupedBiasAddGradCutGTilingData* __restrict tiling)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tiling;
    cutHDim_ = tiling_->cutHDim;
    cutGDim_ = tiling_->cutGDim;
    hTailFactor_ = tiling_->ubHTailFactor;
    groupedIdxSize_ = tiling_->groupedIdxSize;
    inputRow_ = tiling_->inputShape[0];
    inputCol_ = tiling_->inputShape[1];
    groupIdxType_ = tiling_->groupIdxType;
    maxPerHLen_ = tiling_->maxOutputElements;

    if (groupIdxType_) {
        if constexpr (!IsSameType<U, int64_t>::value) {
            // U 为 int32 且需要前缀和：扣除 cast 空间 + cumSum 空间
            inUbSize_ = tiling_->useUbSize - CS_BUF_SIZE - groupedIdxSize_ / 2;
            pipe_.InitBuffer(groupIdxCastBuf_, 2 * groupedIdxSize_); // int32→int64，空间翻倍
            pipe_.InitBuffer(cumSumBuf_, CS_BUF_SIZE);
            groupIdxCastLocal_ = groupIdxCastBuf_.Get<int64_t>();
            cumSumLocal_ = cumSumBuf_.Get<int64_t>();
        } else {
            // U 为 int64 且需要前缀和：只扣除 cumSum 空间
            inUbSize_ = tiling_->useUbSize - CS_BUF_SIZE;
            pipe_.InitBuffer(cumSumBuf_, CS_BUF_SIZE);
            cumSumLocal_ = cumSumBuf_.Get<U>();
        }
    } else {
        inUbSize_ = tiling_->useUbSize;
    }

    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    groupIdxGm_.SetGlobalBuffer((__gm__ U*)groupIdx);

    pipe_.InitBuffer(xQue_, BUFFER_NUM, inUbSize_);
    pipe_.InitBuffer(outQue_, BUFFER_NUM, tiling_->outputSize);
    pipe_.InitBuffer(zeroOutQue_, BUFFER_NUM, tiling_->outputSize);
    pipe_.InitBuffer(groupIdxQue_, 1, groupedIdxSize_);

    pipe_.InitBuffer(tempBufBuf_, tiling_->useTempBuf);
    pipe_.InitBuffer(tempResBuf_, H_CUT_BLOCK_SIZE * 2);
    pipe_.InitBuffer(computeBuf_, H_CUT_BLOCK_SIZE * 2);

    if (blockIdx_ >= tiling_->blockTailStartIndex) {
        BlockStartIdx_ = tiling_->blockTailStartIndex * tiling_->blockFactor +
                         (blockIdx_ - tiling_->blockTailStartIndex) * tiling_->blockTailFactor;
        currBlockFactor_ = tiling_->blockTailFactor;
    } else {
        BlockStartIdx_ = blockIdx_ * tiling_->blockFactor;
        currBlockFactor_ = tiling_->blockFactor;
    }

    tempBuf_ = tempBufBuf_.template Get<PromoteDataT>();
    computeRes_ = computeBuf_.template Get<PromoteDataT>();
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::SetGParam(int64_t g, int64_t singleH)
{
    int64_t singleG = inUbSize_ /
                      (utils::CeilAlign<int64_t>(static_cast<int64_t>(singleH), static_cast<int64_t>(BLOCK_UB_SIZE)) *
                       sizeof(PromoteDataT));
    blockGNum_ = utils::CeilDiv<int64_t>(g, singleG);
    mainG_ = singleG;
    tailG_ = g - (blockGNum_ - 1) * mainG_;
    if (blockGNum_ == 1) {
        mainG_ = tailG_;
    }

    bisectionPos_ = utils::FindNearestPower2(blockGNum_);
    cacheCount_ = utils::CalLog2(bisectionPos_) + 1;
    bisectionTail_ = blockGNum_ - bisectionPos_;
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::UpdateCurParam(int64_t currRowIdx, int64_t& currRowCumSum,
                                                                     int64_t& currRowWidth)
{
    if (currRowIdx > 0) {
        if (groupIdxType_) {
            // groupIdx 存的是每行的长度，需要 reduceSum
            uint32_t shape[] = {static_cast<uint32_t>(currRowIdx), 1};
            if constexpr (!IsSameType<U, int64_t>::value) {
                ReduceSum<int64_t, Pattern::Reduce::RA, true>(cumSumLocal_, groupIdxCastLocal_, shape, false);
                event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventId);
                WaitFlag<HardEvent::V_S>(eventId);
                currRowWidth = groupIdxCastLocal_.GetValue(currRowIdx);
            } else {
                ReduceSum<U, Pattern::Reduce::RA, true>(cumSumLocal_, groupIdxLocal_, shape, false);
                event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventId);
                WaitFlag<HardEvent::V_S>(eventId);
                currRowWidth = groupIdxLocal_.GetValue(currRowIdx);
            }
            currRowCumSum = cumSumLocal_.GetValue(0);
        } else {
            // groupIdx 存的是累积索引，直接取值
            currRowCumSum = groupIdxLocal_.GetValue(currRowIdx - 1);
            currRowWidth = (currRowIdx == 0) ?
                               groupIdxLocal_.GetValue(0) :
                               (groupIdxLocal_.GetValue(currRowIdx) - groupIdxLocal_.GetValue(currRowIdx - 1));
        }
    }
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::Process()
{
    // Step 1: mte2 groupIdx
    CopyGroupIdx();

    // Step 2: 计算初始位置的前缀和
    int64_t currBlockIdx = BlockStartIdx_;
    int64_t currRowIdx = currBlockIdx / cutHDim_;               // 初始行号
    int64_t currRowCumSum = 0;                                  // 初始前缀和
    int64_t currRowWidth = groupIdxLocal_.GetValue(currRowIdx); // 当前行宽度

    // 计算初始前缀和（前 currRowIdx 个 groupIdx 的和）
    UpdateCurParam(currRowIdx, currRowCumSum, currRowWidth);

    // Step 3: 主循环处理
    int64_t prevRowIdx = currRowIdx;

    for (int64_t i = 0; i < currBlockFactor_;) {
        currRowIdx = currBlockIdx / cutHDim_;
        int64_t currHIdx = currBlockIdx % cutHDim_; // 当前 H 块索引
        // 3.1 行切换时更新前缀和 和 行宽度
        if (currRowIdx != prevRowIdx) {
            if (groupIdxType_) {
                currRowCumSum += groupIdxLocal_.GetValue(prevRowIdx);
                currRowWidth = groupIdxLocal_.GetValue(currRowIdx);
            } else {
                currRowCumSum = groupIdxLocal_.GetValue(currRowIdx - 1);
                currRowWidth = groupIdxLocal_.GetValue(currRowIdx) - groupIdxLocal_.GetValue(currRowIdx - 1);
            }
            prevRowIdx = currRowIdx;
        }
        // 3.2 判断当前 H 块是否为尾块，计算当前 H 块长度（元素数）
        bool isHTail = (currHIdx == cutHDim_ - 1);
        int64_t currHLen = isHTail ? hTailFactor_ : (H_CUT_BLOCK_SIZE / sizeof(T));

        // 3.3 计算当前块大小（元素数）
        int64_t currBlockSize = currRowWidth * utils::CeilAlign<int64_t>(currHLen * sizeof(T), ALIGN_BYTE_32) /
                                sizeof(T);
        // 3.4 根据 UB 容量选择处理方式
        if (currRowWidth == 0) {
            ProcessZeroG(currRowIdx, currHIdx, currHLen);
            currBlockIdx += 1;
            i += 1;
        } else if (currRowWidth == 1) {
            ProcessOneG(currRowIdx, currHIdx, currRowWidth, currRowCumSum, i);
            currBlockIdx += curProcessHBlocks_;
            i += curProcessHBlocks_;
        } else if (currBlockSize * sizeof(PromoteDataT) > inUbSize_) {
            // ===== 分支2: 二分累加 =====
            SetGParam(currRowWidth, currHLen);
            BinaryReduceSum(currRowCumSum, currRowIdx, currHIdx, currRowWidth, currHLen);
            currBlockIdx += 1;
            i += 1;
        } else if (currBlockSize * sizeof(PromoteDataT) <= inUbSize_ / 2 && !isHTail) {
            // ===== 分支3: 扩展 H 轴处理多块 =====
            ProcessMultiHBlock(currRowIdx, currHIdx, currRowWidth, currRowCumSum, i);
            currBlockIdx += curProcessHBlocks_;
            i += curProcessHBlocks_;
        } else {
            // ===== 分支4: 单块处理 =====
            ProcessSingleHBlock(currHLen, currRowIdx, currHIdx, currRowWidth, currRowCumSum);
            currBlockIdx += 1;
            i += 1;
        }
    }
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::ProcessZeroG(int64_t currRowIdx, int64_t currHIdx, int64_t dimH)
{
    int64_t outputDataOffset = currRowIdx * inputCol_ + currHIdx * H_CUT_BLOCK_SIZE / sizeof(T);
    LocalTensor<T> outputTensor = zeroOutQue_.AllocTensor<T>();
    Duplicate<T>(outputTensor, static_cast<T>(ZERO_VALUE), dimH);

    zeroOutQue_.EnQue<T>(outputTensor);
    CopyDataOutUnCast(outputDataOffset, dimH, 0);
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::ProcessOneG(int64_t currRowIdx, int64_t currHIdx,
                                                                  int64_t currRowWidth, int64_t currRowCumSum,
                                                                  int64_t loop)
{
    int64_t remainHBlocks = cutHDim_ - currHIdx;
    int64_t remainTaskBlocks = currBlockFactor_ - loop;
    int64_t maxProcessBlocks = remainHBlocks < remainTaskBlocks ? remainHBlocks : remainTaskBlocks;
    if (maxProcessBlocks * H_CUT_BLOCK_SIZE > maxPerHLen_ * sizeof(T)) {
        maxProcessBlocks = maxPerHLen_ * sizeof(T) / H_CUT_BLOCK_SIZE;
    }

    int64_t processHBlocks = 0;
    int64_t processHLen = 0;
    CalcMaxHLen(currRowWidth, currHIdx, maxProcessBlocks, processHBlocks, processHLen);
    curProcessHBlocks_ = processHBlocks;
    int64_t dataCopyOffset = currRowCumSum * inputCol_ + currHIdx * H_CUT_BLOCK_SIZE / sizeof(T);
    int64_t outputDataOffset = currRowIdx * inputCol_ + currHIdx * H_CUT_BLOCK_SIZE / sizeof(T);
    CopyDataInForOne(processHLen * sizeof(T), currRowWidth, dataCopyOffset);
    CopyDataOutUnCast(outputDataOffset, processHLen, 1);
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::ProcessMultiHBlock(int64_t currRowIdx, int64_t currHIdx,
                                                                         int64_t currRowWidth, int64_t currRowCumSum,
                                                                         int64_t loop)
{
    int64_t remainHBlocks = cutHDim_ - currHIdx;        // 当前行剩余 H 块数
    int64_t remainTaskBlocks = currBlockFactor_ - loop; // 当前核剩余任务块数
    int64_t maxProcessBlocks = remainHBlocks < remainTaskBlocks ? remainHBlocks : remainTaskBlocks;
    if (maxProcessBlocks * H_CUT_BLOCK_SIZE > maxPerHLen_ * sizeof(T)) {
        maxProcessBlocks = maxPerHLen_ * sizeof(T) / H_CUT_BLOCK_SIZE;
    }

    int64_t processHBlocks = 0;
    int64_t processHLen = 0;
    CalcMaxHLen(currRowWidth, currHIdx, maxProcessBlocks, processHBlocks, processHLen);
    curProcessHBlocks_ = processHBlocks;
    int64_t processHLenAlign = utils::CeilAlign<int64_t>(processHLen, BLOCK_UB_SIZE);
    int64_t dataCopyOffset = currRowCumSum * inputCol_ + currHIdx * H_CUT_BLOCK_SIZE / sizeof(T);

    CopyDataIn(processHLen * sizeof(T), currRowWidth, dataCopyOffset);
    uint32_t srcShape[2] = {static_cast<uint32_t>(currRowWidth), static_cast<uint32_t>(processHLenAlign)};

    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventId);
    WaitFlag<HardEvent::MTE3_V>(eventId);

    LocalTensor<PromoteDataT> outputTensor = outQue_.AllocTensor<PromoteDataT>();
    ReduceSum<PromoteDataT, AscendC::Pattern::Reduce::RA, true>(outputTensor, xTensor_, srcShape, false);
    outQue_.EnQue<TPosition::VECOUT, TPosition::GM, PromoteDataT>(outputTensor);

    int64_t outputDataOffset = currRowIdx * inputCol_ + currHIdx * H_CUT_BLOCK_SIZE / sizeof(T);
    CopyDataOut(outputDataOffset, processHLen);
    xQue_.FreeTensor(xTensor_);
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::ProcessSingleHBlock(int64_t currHLen, int64_t currRowIdx,
                                                                          int64_t currHIdx, int64_t currRowWidth,
                                                                          int64_t currRowCumSum)
{
    int64_t currHLenAlign = utils::CeilAlign<int64_t>(currHLen, BLOCK_UB_SIZE);
    int64_t dataCopyOffset = currRowCumSum * inputCol_ + currHIdx * H_CUT_BLOCK_SIZE / sizeof(T);

    CopyDataIn(currHLen * sizeof(T), currRowWidth, dataCopyOffset);
    uint32_t srcShape[2] = {static_cast<uint32_t>(currRowWidth), static_cast<uint32_t>(currHLenAlign)};

    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventId);
    WaitFlag<HardEvent::MTE3_V>(eventId);

    LocalTensor<PromoteDataT> outputTensor = outQue_.AllocTensor<PromoteDataT>();
    ReduceSum<PromoteDataT, AscendC::Pattern::Reduce::RA, true>(outputTensor, xTensor_, srcShape, false);
    outQue_.EnQue<TPosition::VECOUT, TPosition::GM, PromoteDataT>(outputTensor);

    int64_t outputDataOffset = currRowIdx * inputCol_ + currHIdx * H_CUT_BLOCK_SIZE / sizeof(T);
    CopyDataOut(outputDataOffset, currHLen);
    xQue_.FreeTensor(xTensor_);
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::BinaryReduceSum(int64_t currRowCumSum, int64_t currRowIdx,
                                                                      int64_t currHIdx, int64_t currRowWidth,
                                                                      int64_t singleH)
{
    int64_t singleHAlign = utils::CeilAlign<int64_t>(singleH, BLOCK_UB_SIZE); // 32字节对齐
    int64_t i = 0;
    int64_t copyAddr = currRowCumSum * inputCol_ + currHIdx * H_CUT_BLOCK_SIZE / sizeof(T);
    int64_t cacheStride = 0;

    // 1.前bisectionTail_块切分的G相同
    for (; i < bisectionTail_; i++) {
        int64_t offsetLeft = copyAddr + (i + bisectionPos_) * mainG_ * inputCol_;
        if (i == bisectionTail_ - 1) {
            CopyDataIn(singleH * sizeof(T), tailG_, offsetLeft);
        } else {
            CopyDataIn(singleH * sizeof(T), mainG_, offsetLeft);
        }
        uint32_t srcShape[2] = {
            (i == bisectionTail_ - 1) ? static_cast<uint32_t>(tailG_) : static_cast<uint32_t>(mainG_),
            static_cast<uint32_t>(singleHAlign)};
        tempRes_ = tempResBuf_.template Get<PromoteDataT>();
        ReduceSum<PromoteDataT, AscendC::Pattern::Reduce::RA, true>(tempRes_, xTensor_, srcShape, false);
        xQue_.FreeTensor(xTensor_);

        int64_t offsetRight = copyAddr + i * mainG_ * inputCol_;
        CopyDataIn(singleH * sizeof(T), mainG_, offsetRight);
        srcShape[0] = static_cast<uint32_t>(mainG_);
        ReduceSum<PromoteDataT, AscendC::Pattern::Reduce::RA, true>(computeRes_, xTensor_, srcShape, false);
        xQue_.FreeTensor(xTensor_);
        Add(computeRes_, computeRes_, tempRes_, 1 * singleH);
        tempResBuf_.FreeTensor(tempRes_);
        int64_t cacheID = utils::GetCacheID(i);
        cacheStride = utils::CeilDiv<int64_t>(static_cast<int64_t>(singleH),
                                              static_cast<int64_t>(ELEMENT_ONE_REPEAT_COMPUTE)) *
                      ELEMENT_ONE_REPEAT_COMPUTE;
        UpdateCacheAux(cacheID, cacheStride, singleH);
    }

    // 2.剩下的块
    for (; i < bisectionPos_; i++) {
        int64_t offset = copyAddr + i * mainG_ * inputCol_;
        CopyDataIn(singleH * sizeof(T), mainG_, offset);
        uint32_t srcShape[2] = {static_cast<uint32_t>(mainG_), static_cast<uint32_t>(singleHAlign)};
        ReduceSum<PromoteDataT, AscendC::Pattern::Reduce::RA, true>(computeRes_, xTensor_, srcShape, false);
        xQue_.FreeTensor(xTensor_);
        int64_t cacheID = utils::GetCacheID(i);
        cacheStride = utils::CeilDiv<int64_t>(static_cast<int64_t>(singleH),
                                              static_cast<int64_t>(ELEMENT_ONE_REPEAT_COMPUTE)) *
                      ELEMENT_ONE_REPEAT_COMPUTE;
        UpdateCacheAux(cacheID, cacheStride, singleH);
    }
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);

    int64_t tmpBufOffest = (cacheCount_ - 1) * cacheStride;
    LocalTensor<PromoteDataT> ubRes = tempBuf_[tmpBufOffest];
    int64_t outputDataOffset = currRowIdx * inputCol_ + currHIdx * H_CUT_BLOCK_SIZE / sizeof(T);
    CopyDataOutForBuf(outputDataOffset, ubRes, singleH);
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::UpdateCacheAux(const int64_t cacheID, const int64_t stride,
                                                                     const int64_t count)
{
    uint16_t outerLoopTimes = utils::CeilDiv<int64_t>(static_cast<int64_t>(count * sizeof(PromoteDataT)),
                                                      static_cast<int64_t>(Ops::Base::GetVRegSize()));
    uint16_t innerLoopTimes = cacheID;
    uint32_t outerLoopStride = ELEMENT_ONE_REPEAT_COMPUTE;
    uint32_t innerLoopStride = stride;

    LocalTensor<PromoteDataT> dstTensor = tempBuf_;
    LocalTensor<PromoteDataT> srcTensor = computeRes_;

    __VEC_SCOPE__
    {
        __local_mem__ PromoteDataT* dst = (__local_mem__ PromoteDataT*)dstTensor.GetPhyAddr();
        __local_mem__ PromoteDataT* cah = (__local_mem__ PromoteDataT*)dstTensor.GetPhyAddr() + cacheID * stride;
        __local_mem__ PromoteDataT* src = (__local_mem__ PromoteDataT*)srcTensor.GetPhyAddr();

        uint32_t sreg = static_cast<uint32_t>(count);
        AscendC::Reg::RegTensor<PromoteDataT> aReg, bReg;
        AscendC::Reg::MaskReg pMask;

        for (uint16_t i = 0; i < outerLoopTimes; ++i) {
            pMask = AscendC::Reg::UpdateMask<PromoteDataT>(sreg);
            DataCopy(aReg, (__local_mem__ PromoteDataT*)src + i * outerLoopStride);
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                DataCopy(bReg, (__local_mem__ PromoteDataT*)dst + i * outerLoopStride + j * innerLoopStride);
                Add<PromoteDataT, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
            }
            DataCopy((__local_mem__ PromoteDataT*)cah + i * outerLoopStride, aReg, pMask);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::CalcMaxHLen(int64_t currRowWidth, int64_t currHIdx,
                                                                  int64_t maxProcessBlocks, int64_t& processHBlocks,
                                                                  int64_t& processHLen)
{
    constexpr int64_t hBlockLen = H_CUT_BLOCK_SIZE / sizeof(T);
    int64_t maxUbBlocks = inUbSize_ / (currRowWidth * hBlockLen * sizeof(PromoteDataT));
    processHBlocks = maxUbBlocks < maxProcessBlocks ? maxUbBlocks : maxProcessBlocks;
    bool includesTail = (currHIdx + processHBlocks == cutHDim_);

    if (includesTail) {
        // 包含尾块：检查 UB 是否能容纳 (processHBlocks-1)*hBlockLen + hTailFactor_
        int64_t totalLen = (processHBlocks - 1) * hBlockLen + hTailFactor_;
        if (currRowWidth * totalLen * sizeof(PromoteDataT) <= inUbSize_) {
            processHLen = totalLen;
        } else {
            processHBlocks -= 1;
            processHLen = processHBlocks * hBlockLen;
        }
    } else {
        // 不包含尾块：全部是完整块
        processHLen = processHBlocks * hBlockLen;
    }
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::CopyGroupIdx()
{
    groupIdxLocal_ = groupIdxQue_.AllocTensor<U>();
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(cutGDim_ * sizeof(U)), 0, 0, 0};
    DataCopyPadExtParams<U> dataCopyPadExtParams = {false, 0, 0, 0};
    DataCopyPad(groupIdxLocal_, groupIdxGm_, copyInParams, dataCopyPadExtParams);

    groupIdxQue_.EnQue<U>(groupIdxLocal_);
    groupIdxLocal_ = groupIdxQue_.DeQue<U>();

    if (groupIdxType_ == 1) {
        if constexpr (!IsSameType<U, int64_t>::value) {
            Cast(groupIdxCastLocal_, groupIdxLocal_, RoundMode::CAST_NONE, cutGDim_);

            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventId);
            WaitFlag<HardEvent::V_S>(eventId);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::CopyDataInForOne(int64_t burstLen, int64_t burstNum,
                                                                       int64_t srcGmOffset)
{
    LocalTensor<T> xTensorT = outQue_.AllocTensor<T>();
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = burstNum;
    copyInParams.blockLen = burstLen;
    copyInParams.srcStride = (inputCol_ - burstLen / sizeof(T)) * sizeof(T);
    copyInParams.dstStride = 0;
    DataCopyPadExtParams<T> dataCopyPadExtParams = {false, 0, 0, 0};

    DataCopyPad(xTensorT, xGm_[srcGmOffset], copyInParams, dataCopyPadExtParams);
    outQue_.EnQue<T>(xTensorT);
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::CopyDataIn(int64_t burstLen, int64_t burstNum,
                                                                 int64_t srcGmOffset)
{
    xTensor_ = xQue_.AllocTensor<PromoteDataT>();
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = burstNum;
    copyInParams.blockLen = burstLen;
    copyInParams.srcStride = (inputCol_ - burstLen / sizeof(T)) * sizeof(T);
    copyInParams.dstStride = 0;
    DataCopyPadExtParams<T> dataCopyPadExtParams = {false, 0, 0, 0};

    if constexpr (IsSameType<PromoteDataT, T>::value) {
        DataCopyPad(xTensor_, xGm_[srcGmOffset], copyInParams, dataCopyPadExtParams);
        xQue_.EnQue<T>(xTensor_);
    } else {
        LocalTensor<T> inputLocal = xTensor_.template ReinterpretCast<T>();
        int64_t inputOffset = burstNum * utils::CeilAlign(burstLen, ALIGN_BYTE_32) / sizeof(T);
        DataCopyPad(inputLocal[inputOffset], xGm_[srcGmOffset], copyInParams, dataCopyPadExtParams);

        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        int64_t count = burstNum * utils::CeilAlign<int64_t>(burstLen / sizeof(T), BLOCK_UB_SIZE);
        uint16_t loops = utils::CeilDiv<int64_t>(static_cast<int64_t>(count * sizeof(PromoteDataT)),
                                                 static_cast<int64_t>(Ops::Base::GetVRegSize()));
        uint32_t loopsStride = Ops::Base::GetVRegSize() / sizeof(PromoteDataT);

        __VEC_SCOPE__
        {
            uint32_t inputOffsetReg = inputOffset;
            __local_mem__ PromoteDataT* dst = (__local_mem__ PromoteDataT*)xTensor_.GetPhyAddr();
            __local_mem__ T* src = (__local_mem__ T*)xTensor_.GetPhyAddr() + inputOffsetReg;

            uint32_t sreg = static_cast<uint32_t>(count);
            AscendC::Reg::MaskReg mask;
            AscendC::Reg::RegTensor<T> aReg;
            AscendC::Reg::RegTensor<PromoteDataT> bReg;

            for (uint16_t i = 0; i < loops; i++) {
                mask = AscendC::Reg::UpdateMask<PromoteDataT>(sreg);
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                    aReg, (__local_mem__ T*)src + i * loopsStride);
                AscendC::Reg::Cast<PromoteDataT, T, castTrait0>(bReg, aReg, mask);
                AscendC::Reg::DataCopy(dst + i * loopsStride, bReg, mask);
            }
        }
        xQue_.EnQue<PromoteDataT>(xTensor_);
    }
    xTensor_ = xQue_.DeQue<PromoteDataT>();
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::CopyDataOutUnCast(int64_t outputDataOffset, int64_t dimH,
                                                                        bool isOne)
{
    DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};
    copyOutParams.blockLen = dimH * sizeof(T);

    if (isOne) {
        LocalTensor<T> outputTensor = outQue_.DeQue<T>();
        DataCopyPad(yGm_[outputDataOffset], outputTensor, copyOutParams);
        outQue_.FreeTensor(outputTensor);
    } else {
        LocalTensor<T> outputTensor = zeroOutQue_.DeQue<T>();
        DataCopyPad(yGm_[outputDataOffset], outputTensor, copyOutParams);
        zeroOutQue_.FreeTensor(outputTensor);
    }
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::CopyDataOutForBuf(int64_t outputDataOffset,
                                                                        LocalTensor<PromoteDataT>& ubRes, int64_t dimH)
{
    DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};
    copyOutParams.blockLen = dimH * sizeof(T);
    if constexpr (IsSameType<PromoteDataT, T>::value) {
        DataCopyPad(yGm_[outputDataOffset], ubRes, copyOutParams);
    } else {
        LocalTensor<T> outputLocal = ubRes.template ReinterpretCast<T>();
        Cast(outputLocal, ubRes, RoundMode::CAST_RINT, dimH);

        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        DataCopyPad(yGm_[outputDataOffset], outputLocal, copyOutParams);
    }
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventId);
    WaitFlag<HardEvent::MTE3_V>(eventId);
}

template <typename T, typename U>
__aicore__ inline void GroupedBiasAddGradCutGH<T, U>::CopyDataOut(int64_t outputDataOffset, int64_t dimH)
{
    DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};
    copyOutParams.blockLen = dimH * sizeof(T);

    LocalTensor<PromoteDataT> outputTensor = outQue_.DeQue<TPosition::VECOUT, TPosition::GM, PromoteDataT>();
    if constexpr (IsSameType<PromoteDataT, T>::value) {
        DataCopyPad(yGm_[outputDataOffset], outputTensor, copyOutParams);
    } else {
        LocalTensor<T> outputLocal = outputTensor.template ReinterpretCast<T>();
        Cast(outputLocal, outputTensor, RoundMode::CAST_RINT, dimH);

        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        DataCopyPad(yGm_[outputDataOffset], outputLocal, copyOutParams);
    }
    outQue_.FreeTensor(outputTensor);
}

} // namespace GroupedBiasAddGrad

#endif
