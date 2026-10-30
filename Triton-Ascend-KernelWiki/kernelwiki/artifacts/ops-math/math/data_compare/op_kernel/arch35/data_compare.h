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
 * \file data_compare.h
 * \brief DataCompare 算子 kernel 类实现（Normal + Group 模板）
 *
 * All Reduce 简化：isTailR 恒为 true，kernel 类不带 isTailR 模板参数。
 * 双输入：CopyIn 加载 x1 和 x2 两个张量。
 * pre_elewise 7 步 VF 链：Cast(仅整数)→AbsSub→Abs→Muls→Adds→Compare<GT>→Select+Duplicate
 * Reducer = ReduceSum（identity=0, combine=+, needs_bisection=true, is_fast_path=true）
 */
#ifndef OPS_DATA_COMPARE_H_
#define OPS_DATA_COMPARE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/reduce/reduce.h"
#include "data_compare_tiling_data.h"
#include "data_compare_tiling_key.h"

namespace NsDataCompare {

using namespace AscendC;

constexpr uint32_t kVlBytes = 256;
constexpr uint32_t kRepF32 = kVlBytes / sizeof(float);
constexpr uint16_t kRepF32U = static_cast<uint16_t>(kRepF32);
constexpr uint32_t kBlockBytes = 32;
constexpr uint32_t kBlockF32 = kBlockBytes / sizeof(float);

struct UBAxisDesc {
    int32_t gmIdx;
    int64_t ubSize;
    int64_t paddedSize;
    int64_t gmStride;
};

// CastTrait 定义
static constexpr AscendC::Reg::CastTrait kCastTraitB16ToF32{
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE};

static constexpr AscendC::Reg::CastTrait kCastTraitB8ToB16{
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

static constexpr AscendC::Reg::CastTrait kCastTraitB16ToF32Int{
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC};

static constexpr AscendC::Reg::CastTrait kCastTraitI32ToF32{
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

template <typename DType>
class DataCompareKernel {
public:
    using D_T = DType;

    static constexpr bool kIsFp32 = AscendC::IsSameType<D_T, float>::value;
    static constexpr bool kIsB16 = (sizeof(D_T) == 2);
    static constexpr bool kIsB8 = (sizeof(D_T) == 1);
    static constexpr bool kIsI32 = AscendC::IsSameType<D_T, int32_t>::value;

    __aicore__ inline DataCompareKernel() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const DataCompareTilingData* td);
    __aicore__ inline void Process();
    __aicore__ inline void InitGroup(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, const DataCompareTilingData* td);
    __aicore__ inline void ProcessGroup();

private:
    __aicore__ inline void DoOneAChunk(int64_t outerGmOff, int64_t aLen);
    __aicore__ inline void PostElewiseAndCopyOut(int64_t outerOutOff, int64_t aLen);
    __aicore__ inline int32_t BuildUBAxes(int64_t aLen, int64_t rLen, UBAxisDesc out[]);
    __aicore__ inline void DoCopyInTile(
        int64_t baseGmOff, int64_t aLen, int64_t rLen, AscendC::LocalTensor<D_T>& preInLocal, GlobalTensor<D_T>& srcGm);
    __aicore__ inline void PreElewiseVf(
        AscendC::LocalTensor<D_T>& x1Buf, AscendC::LocalTensor<D_T>& x2Buf, uint32_t slot);
    __aicore__ inline void ClearChunkExtensionVf(uint32_t slot, int64_t rLen);
    __aicore__ inline void MergeTmpBufVf();
    __aicore__ inline void ReduceRPattern();
    __aicore__ inline void ClearCacheTreeVf();
    __aicore__ inline void DoCachingVf(uint16_t cacheID);
    __aicore__ inline int32_t LastAAxis() const;
    __aicore__ inline int32_t LastRAxis() const;
    __aicore__ inline int64_t RLenOfChunk(int64_t rChunkIdx) const;
    __aicore__ inline int64_t UnravelOuterR(int64_t rIdx, int64_t& rChunkIdxOut, int64_t& rLenOut) const;
    __aicore__ inline uint16_t GetCacheID(int64_t idx) const;
    __aicore__ inline uint64_t FindNearestPower2(uint64_t v) const;
    __aicore__ inline uint64_t CalLog2(uint64_t v) const;

    // Group helpers
    __aicore__ inline void DoOneAChunkGroup(int64_t outerGmOff, int64_t aLen, int64_t rStart, int64_t rEnd);
    __aicore__ inline void Phase1OutputToWorkspace(int64_t wsColOff, int64_t aLen, int64_t rChunkIdx, int64_t rCount);
    __aicore__ inline void Phase2Process();

    // tilingdata mirrors
    int32_t axisNum_ = 0;
    int64_t axisShape_[MAX_PATTERN_RANK] = {0};
    int64_t axisStride_[MAX_PATTERN_RANK] = {0};
    int32_t aSplit_ = 0;
    int32_t rSplit_ = 0;
    int64_t aLoopCntTotal_ = 0;
    int64_t aSplitChunkCnt_ = 0;
    int64_t aBigCoreLoopCnt_ = 0;
    int64_t aSmallCoreLoopCnt_ = 0;
    int32_t aBigCoreCnt_ = 0;
    int32_t usedCoreNum_ = 0;
    int64_t aUbFactor_ = 0;
    int64_t aUbFactorAlign_ = 0;
    int64_t rUbFactor_ = 0;
    int64_t rUbFactorAlign_ = 0;
    int64_t innerAProd_ = 0;
    int64_t innerAProdAlign_ = 0;
    int64_t innerRProd_ = 0;
    int64_t innerRProdAlign_ = 0;
    int64_t rLoopCntTotal_ = 0;
    int64_t bisectionPos_ = 0;
    int64_t bisectionTail_ = 0;
    int64_t cacheCount_ = 0;
    int64_t preReduceUbSize_ = 0;
    int64_t postReduceUbSize_ = 0;
    int64_t tmpSlotElems_ = 0;
    int64_t cacheBufElems_ = 0;
    float atol_ = 1e-5f;
    float rtol_ = 1e-3f;
    int64_t outStride_[MAX_PATTERN_RANK] = {0};

    GlobalTensor<D_T> x1Gm_;
    GlobalTensor<D_T> x2Gm_;
    GlobalTensor<float> yGm_;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> preInQueX1_;
    TQue<QuePosition::VECIN, 1> preInQueX2_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    TBuf<QuePosition::VECCALC> cacheBuf_;
    TQue<QuePosition::VECOUT, 1> outQue_;

    GlobalTensor<float> wsGm_;
    int64_t rGroupCnt_ = 0;
    int64_t aTotal_ = 0;
};

// ════════════════════════════════════════════════════════════════════════════
// Utility functions
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline int32_t DataCompareKernel<DType>::LastAAxis() const
{
    for (int32_t i = axisNum_ - 1; i >= 0; --i) {
        if (i % 2 == 0)
            return i;
    }
    return 0;
}
template <typename DType>
__aicore__ inline int32_t DataCompareKernel<DType>::LastRAxis() const
{
    for (int32_t i = axisNum_ - 1; i >= 0; --i) {
        if (i % 2 == 1)
            return i;
    }
    return 1;
}
template <typename DType>
__aicore__ inline uint64_t DataCompareKernel<DType>::FindNearestPower2(uint64_t v) const
{
    if (v == 0)
        return 0;
    if (v <= 2)
        return 1;
    if (v <= 4)
        return 2;
    const uint64_t num = v - 1;
    const uint64_t pow = 63 - AscendC::ScalarCountLeadingZero(num);
    return static_cast<uint64_t>(1) << pow;
}
template <typename DType>
__aicore__ inline uint64_t DataCompareKernel<DType>::CalLog2(uint64_t v) const
{
    uint64_t res = 0;
    while (v > 1) {
        v >>= 1;
        res++;
    }
    return res;
}
template <typename DType>
__aicore__ inline uint16_t DataCompareKernel<DType>::GetCacheID(int64_t idx) const
{
    const uint64_t v = static_cast<uint64_t>(idx);
    return static_cast<uint16_t>(AscendC::ScalarGetCountOfValue<1>(v ^ (v + 1)) - 1);
}
template <typename DType>
__aicore__ inline int64_t DataCompareKernel<DType>::RLenOfChunk(int64_t rChunkIdx) const
{
    const int64_t rAxisSize = axisShape_[rSplit_];
    const int64_t start = rChunkIdx * rUbFactor_;
    return (start + rUbFactor_ > rAxisSize) ? (rAxisSize - start) : rUbFactor_;
}
template <typename DType>
__aicore__ inline int64_t DataCompareKernel<DType>::UnravelOuterR(
    int64_t rIdx, int64_t& rChunkIdxOut, int64_t& rLenOut) const
{
    const int64_t rChunksOnSplit = (axisShape_[rSplit_] + rUbFactor_ - 1) / rUbFactor_;
    rChunkIdxOut = rIdx % rChunksOnSplit;
    int64_t cur = rIdx / rChunksOnSplit;
    int64_t gmOff = 0;
    for (int32_t i = rSplit_ - 1; i >= 0; --i) {
        if (i % 2 == 1) {
            const int64_t sz = axisShape_[i];
            const int64_t ix = cur % sz;
            cur /= sz;
            gmOff += ix * axisStride_[i];
        }
    }
    rLenOut = RLenOfChunk(rChunkIdxOut);
    return gmOff + rChunkIdxOut * rUbFactor_ * axisStride_[rSplit_];
}

// ════════════════════════════════════════════════════════════════════════════
// Init
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void DataCompareKernel<DType>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const DataCompareTilingData* td)
{
    axisNum_ = td->axisNum;
    for (int32_t i = 0; i < MAX_PATTERN_RANK; ++i) {
        axisShape_[i] = td->axisShape[i];
        axisStride_[i] = td->axisStride[i];
    }
    aSplit_ = td->aSplitAxisIdx;
    rSplit_ = td->rSplitAxisIdx;
    aLoopCntTotal_ = td->aLoopCntTotal;
    aSplitChunkCnt_ = td->aSplitChunkCnt;
    aBigCoreLoopCnt_ = td->aBigCoreLoopCnt;
    aSmallCoreLoopCnt_ = td->aSmallCoreLoopCnt;
    aBigCoreCnt_ = td->aBigCoreCnt;
    usedCoreNum_ = td->usedCoreNum;
    aUbFactor_ = td->aUbFactor;
    aUbFactorAlign_ = td->aUbFactorAlign;
    rUbFactor_ = td->rUbFactor;
    rUbFactorAlign_ = td->rUbFactorAlign;
    innerAProd_ = td->innerAProd;
    innerAProdAlign_ = td->innerAProdAlign;
    innerRProd_ = td->innerRProd;
    innerRProdAlign_ = td->innerRProdAlign;
    rLoopCntTotal_ = td->rLoopCntTotal;
    preReduceUbSize_ = td->preReduceUbSize;
    postReduceUbSize_ = td->postReduceUbSize;
    tmpSlotElems_ = td->tmpBufUbSize / static_cast<int64_t>(sizeof(float));
    cacheBufElems_ = td->cacheBufUbSize / static_cast<int64_t>(sizeof(float));
    atol_ = td->atol;
    rtol_ = td->rtol;

    bisectionPos_ = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(rLoopCntTotal_)));
    bisectionTail_ = rLoopCntTotal_ - bisectionPos_;
    cacheCount_ = static_cast<int64_t>(CalLog2(static_cast<uint64_t>(bisectionPos_))) + 1;

    // Output stride (A axes only, dense)
    {
        int64_t acc = 1;
        for (int32_t i = axisNum_ - 1; i >= 0; --i) {
            if (i % 2 == 0) {
                outStride_[i] = acc;
                acc *= axisShape_[i];
            }
        }
    }

    x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(x1));
    x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(x2));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    pipe_.InitBuffer(preInQueX1_, 2, td->preReduceUbSize);
    pipe_.InitBuffer(preInQueX2_, 1, td->preReduceUbSize);
    pipe_.InitBuffer(tmpBuf_, td->tmpBufUbSize * 2);
    pipe_.InitBuffer(cacheBuf_, td->cacheBufUbSize);
    pipe_.InitBuffer(outQue_, 1, td->postReduceUbSize);
}

// ════════════════════════════════════════════════════════════════════════════
// Process (Normal)
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void DataCompareKernel<DType>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(usedCoreNum_))
        return;

    int64_t aLoopStart, aLoopEnd;
    if (blockIdx < static_cast<int64_t>(aBigCoreCnt_)) {
        aLoopStart = blockIdx * aBigCoreLoopCnt_;
        aLoopEnd = aLoopStart + aBigCoreLoopCnt_;
    } else {
        aLoopStart = static_cast<int64_t>(aBigCoreCnt_) * aBigCoreLoopCnt_ +
                     (blockIdx - static_cast<int64_t>(aBigCoreCnt_)) * aSmallCoreLoopCnt_;
        aLoopEnd = aLoopStart + aSmallCoreLoopCnt_;
    }

    for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
        int64_t rem = aLoopIdx;
        const int64_t aSplitChunkIdx = rem % aSplitChunkCnt_;
        rem /= aSplitChunkCnt_;

        int64_t chunkGmOff = 0;
        int64_t chunkOutOff = 0;
        for (int32_t k = aSplit_ - 2; k >= 0; k -= 2) {
            const int64_t sz = axisShape_[k];
            const int64_t ix = rem % sz;
            rem /= sz;
            chunkGmOff += ix * axisStride_[k];
            chunkOutOff += ix * outStride_[k];
        }

        const int64_t aChunkStart = aSplitChunkIdx * aUbFactor_;
        const int64_t aEnd = aChunkStart + aUbFactor_;
        const int64_t aLen = (aEnd > axisShape_[aSplit_]) ? (axisShape_[aSplit_] - aChunkStart) : aUbFactor_;
        if (aLen <= 0)
            continue;

        chunkGmOff += aChunkStart * axisStride_[aSplit_];
        chunkOutOff += aChunkStart * outStride_[aSplit_];

        DoOneAChunk(chunkGmOff, aLen);
        PostElewiseAndCopyOut(chunkOutOff, aLen);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// DoOneAChunk: clear cache tree → Phase A pairing → Phase B single blocks
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void DataCompareKernel<DType>::DoOneAChunk(int64_t outerGmOff, int64_t aLen)
{
    ClearCacheTreeVf();

    // Phase A: main-tail pairing
    for (int64_t i = 0; i < bisectionTail_; ++i) {
        // Main block slot=0
        {
            int64_t rChunk = 0, rLen = 0;
            const int64_t rOff = UnravelOuterR(i, rChunk, rLen);
            auto x1Local = preInQueX1_.AllocTensor<D_T>();
            DoCopyInTile(outerGmOff + rOff, aLen, rLen, x1Local, x1Gm_);
            preInQueX1_.EnQue(x1Local);
            auto x2Local = preInQueX2_.AllocTensor<D_T>();
            DoCopyInTile(outerGmOff + rOff, aLen, rLen, x2Local, x2Gm_);
            preInQueX2_.EnQue(x2Local);
            auto x1Deq = preInQueX1_.DeQue<D_T>();
            auto x2Deq = preInQueX2_.DeQue<D_T>();
            PreElewiseVf(x1Deq, x2Deq, 0U);
            preInQueX1_.FreeTensor(x1Deq);
            preInQueX2_.FreeTensor(x2Deq);
            if (rLen < rUbFactor_)
                ClearChunkExtensionVf(0U, rLen);
        }
        // Tail block slot=1
        {
            int64_t rChunk = 0, rLen = 0;
            const int64_t rOff = UnravelOuterR(i + bisectionPos_, rChunk, rLen);
            auto x1Local = preInQueX1_.AllocTensor<D_T>();
            DoCopyInTile(outerGmOff + rOff, aLen, rLen, x1Local, x1Gm_);
            preInQueX1_.EnQue(x1Local);
            auto x2Local = preInQueX2_.AllocTensor<D_T>();
            DoCopyInTile(outerGmOff + rOff, aLen, rLen, x2Local, x2Gm_);
            preInQueX2_.EnQue(x2Local);
            auto x1Deq = preInQueX1_.DeQue<D_T>();
            auto x2Deq = preInQueX2_.DeQue<D_T>();
            PreElewiseVf(x1Deq, x2Deq, 1U);
            preInQueX1_.FreeTensor(x1Deq);
            preInQueX2_.FreeTensor(x2Deq);
            if (rLen < rUbFactor_)
                ClearChunkExtensionVf(1U, rLen);
        }
        MergeTmpBufVf();
        ReduceRPattern();
        DoCachingVf(GetCacheID(i));
    }
    // Phase B: single blocks
    for (int64_t i = bisectionTail_; i < bisectionPos_; ++i) {
        int64_t rChunk = 0, rLen = 0;
        const int64_t rOff = UnravelOuterR(i, rChunk, rLen);
        auto x1Local = preInQueX1_.AllocTensor<D_T>();
        DoCopyInTile(outerGmOff + rOff, aLen, rLen, x1Local, x1Gm_);
        preInQueX1_.EnQue(x1Local);
        auto x2Local = preInQueX2_.AllocTensor<D_T>();
        DoCopyInTile(outerGmOff + rOff, aLen, rLen, x2Local, x2Gm_);
        preInQueX2_.EnQue(x2Local);
        auto x1Deq = preInQueX1_.DeQue<D_T>();
        auto x2Deq = preInQueX2_.DeQue<D_T>();
        PreElewiseVf(x1Deq, x2Deq, 0U);
        preInQueX1_.FreeTensor(x1Deq);
        preInQueX2_.FreeTensor(x2Deq);
        if (rLen < rUbFactor_)
            ClearChunkExtensionVf(0U, rLen);
        ReduceRPattern();
        DoCachingVf(GetCacheID(i));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// BuildUBAxes (isTailR=true always for All Reduce)
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline int32_t DataCompareKernel<DType>::BuildUBAxes(int64_t aLen, int64_t rLen, UBAxisDesc out[])
{
    int32_t k = 0;
    const int32_t lastR = LastRAxis();
    const int64_t bsElem = static_cast<int64_t>(kBlockBytes) / static_cast<int64_t>(sizeof(D_T));
    // Inner bundle = R bundle (isTailR=true)
    for (int32_t i = axisNum_ - 1; i >= rSplit_; --i) {
        if (i % 2 != 1)
            continue;
        int64_t actual, padded;
        if (i == rSplit_) {
            actual = rLen;
            padded = rUbFactorAlign_;
        } else if (i == lastR) {
            actual = axisShape_[i];
            padded = (actual + bsElem - 1) / bsElem * bsElem;
        } else {
            actual = axisShape_[i];
            padded = actual;
        }
        out[k] = {i, actual, padded, axisStride_[i]};
        ++k;
    }
    // Outer bundle = A bundle
    for (int32_t i = axisNum_ - 1; i >= aSplit_; --i) {
        if (i % 2 != 0)
            continue;
        out[k] = {
            i, (i == aSplit_) ? aLen : axisShape_[i], (i == aSplit_) ? aUbFactor_ : axisShape_[i], axisStride_[i]};
        ++k;
    }
    return k;
}

// ════════════════════════════════════════════════════════════════════════════
// DoCopyInTile
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void DataCompareKernel<DType>::DoCopyInTile(
    int64_t baseGmOff, int64_t aLen, int64_t rLen, AscendC::LocalTensor<D_T>& preInLocal, GlobalTensor<D_T>& srcGm)
{
    UBAxisDesc ubAxes[MAX_PATTERN_RANK];
    const int32_t K = BuildUBAxes(aLen, rLen, ubAxes);
    const int64_t dtBytes = static_cast<int64_t>(sizeof(D_T));

    DataCopyExtParams extParams;
    LoopModeParams loopParams;
    loopParams.loop1Size = 0;
    loopParams.loop1SrcStride = 0;
    loopParams.loop1DstStride = 0;
    loopParams.loop2Size = 0;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;

    extParams.blockLen = static_cast<uint32_t>(ubAxes[0].ubSize * dtBytes);
    uint32_t misalign = extParams.blockLen & (kBlockBytes - 1u);
    uint8_t rPad = 0;
    if (misalign != 0)
        rPad = static_cast<uint8_t>((kBlockBytes - misalign) / static_cast<uint32_t>(dtBytes));
    DataCopyPadExtParams<D_T> padParams{true, 0, rPad, static_cast<D_T>(0)};

    const int64_t copyPadBytes =
        (static_cast<int64_t>(extParams.blockLen) + kBlockBytes - 1) / kBlockBytes * kBlockBytes;
    extParams.dstStride =
        static_cast<uint32_t>((ubAxes[0].paddedSize * dtBytes - copyPadBytes) / static_cast<int64_t>(kBlockBytes));

    if (K >= 2) {
        extParams.blockCount = static_cast<uint16_t>(ubAxes[1].ubSize);
        extParams.srcStride =
            static_cast<uint32_t>(ubAxes[1].gmStride * dtBytes - static_cast<int64_t>(extParams.blockLen));
    } else {
        extParams.blockCount = 1;
        extParams.srcStride = 0;
    }

    int64_t ubStride[MAX_PATTERN_RANK];
    ubStride[0] = dtBytes;
    for (int32_t i = 1; i < K; ++i)
        ubStride[i] = ubStride[i - 1] * ubAxes[i - 1].paddedSize;

    if (K >= 3) {
        loopParams.loop1Size = static_cast<uint32_t>(ubAxes[2].ubSize);
        loopParams.loop1SrcStride = static_cast<uint64_t>(ubAxes[2].gmStride) * static_cast<uint64_t>(dtBytes);
        loopParams.loop1DstStride = static_cast<uint64_t>(ubStride[2]);
        loopParams.loop2Size = 1;
    }
    if (K >= 4) {
        loopParams.loop2Size = static_cast<uint32_t>(ubAxes[3].ubSize);
        loopParams.loop2SrcStride = static_cast<uint64_t>(ubAxes[3].gmStride) * static_cast<uint64_t>(dtBytes);
        loopParams.loop2DstStride = static_cast<uint64_t>(ubStride[3]);
    }

    const bool useLoopMode = (K >= 3);
    if (useLoopMode)
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);

    int64_t outerProd = 1;
    for (int32_t k = 4; k < K; ++k)
        outerProd *= ubAxes[k].ubSize;

    for (int64_t outerFlat = 0; outerFlat < outerProd; ++outerFlat) {
        int64_t addGmOffElem = 0, addUbOffBytes = 0, cur = outerFlat;
        for (int32_t k = 4; k < K; ++k) {
            const int64_t sz = ubAxes[k].ubSize;
            const int64_t ix = cur % sz;
            cur /= sz;
            addGmOffElem += ix * ubAxes[k].gmStride;
            addUbOffBytes += ix * ubStride[k];
        }
        DataCopyPad(preInLocal[addUbOffBytes / dtBytes], srcGm[baseGmOff + addGmOffElem], extParams, padParams);
    }
    if (useLoopMode)
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
}

// ════════════════════════════════════════════════════════════════════════════
// PreElewiseVf: dual-input comparison chain → tmpBuf[slot]
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void DataCompareKernel<DType>::PreElewiseVf(
    AscendC::LocalTensor<D_T>& x1Buf, AscendC::LocalTensor<D_T>& x2Buf, uint32_t slot)
{
    auto tmpAll = tmpBuf_.Get<float>();
    auto tmpSlot = tmpAll[static_cast<int32_t>(slot) * static_cast<int32_t>(tmpSlotElems_)];
    __ubuf__ D_T* x1Ptr = reinterpret_cast<__ubuf__ D_T*>(x1Buf.GetPhyAddr());
    __ubuf__ D_T* x2Ptr = reinterpret_cast<__ubuf__ D_T*>(x2Buf.GetPhyAddr());
    __ubuf__ float* dstPtr = reinterpret_cast<__ubuf__ float*>(tmpSlot.GetPhyAddr());

    const uint32_t totalElems =
        static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_ * rUbFactorAlign_ * innerRProdAlign_);
    const uint16_t repeatTime = static_cast<uint16_t>((totalElems + kRepF32U - 1) / kRepF32U);
    const float atolVal = atol_;
    const float rtolVal = rtol_;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> x1Reg, x2Reg, diffReg, absX2Reg, threshReg, resultReg;
        AscendC::Reg::RegTensor<float> oneReg, zeroReg;
        AscendC::Reg::MaskReg mask, cmpMask;
        AscendC::Reg::Duplicate(oneReg, 1.0f);
        AscendC::Reg::Duplicate(zeroReg, 0.0f);
        uint32_t remaining = totalElems;

        for (uint16_t i = 0; i < repeatTime; ++i) {
            int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
            mask = AscendC::Reg::UpdateMask<float>(remaining);

            // Load x1, x2 → fp32
            if constexpr (kIsB16) {
                AscendC::Reg::RegTensor<D_T> b16Reg1, b16Reg2;
                AscendC::Reg::LoadAlign<D_T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg1, x1Ptr + off);
                AscendC::Reg::LoadAlign<D_T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg2, x2Ptr + off);
                AscendC::Reg::Cast<float, D_T, kCastTraitB16ToF32>(x1Reg, b16Reg1, mask);
                AscendC::Reg::Cast<float, D_T, kCastTraitB16ToF32>(x2Reg, b16Reg2, mask);
            } else if constexpr (kIsB8) {
                AscendC::Reg::RegTensor<D_T> b8Reg1, b8Reg2;
                AscendC::Reg::LoadAlign<D_T, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(b8Reg1, x1Ptr + off);
                AscendC::Reg::LoadAlign<D_T, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(b8Reg2, x2Ptr + off);
                AscendC::Reg::RegTensor<half> h16Reg1, h16Reg2;
                AscendC::Reg::Cast<half, D_T, kCastTraitB8ToB16>(h16Reg1, b8Reg1, mask);
                AscendC::Reg::Cast<half, D_T, kCastTraitB8ToB16>(h16Reg2, b8Reg2, mask);
                AscendC::Reg::Cast<float, half, kCastTraitB16ToF32Int>(x1Reg, h16Reg1, mask);
                AscendC::Reg::Cast<float, half, kCastTraitB16ToF32Int>(x2Reg, h16Reg2, mask);
            } else if constexpr (kIsI32) {
                AscendC::Reg::RegTensor<int32_t> iReg1, iReg2;
                AscendC::Reg::LoadAlign(iReg1, x1Ptr + off);
                AscendC::Reg::LoadAlign(iReg2, x2Ptr + off);
                AscendC::Reg::Cast<float, int32_t, kCastTraitI32ToF32>(x1Reg, iReg1, mask);
                AscendC::Reg::Cast<float, int32_t, kCastTraitI32ToF32>(x2Reg, iReg2, mask);
            } else {
                AscendC::Reg::LoadAlign(x1Reg, reinterpret_cast<__ubuf__ float*>(x1Ptr) + off);
                AscendC::Reg::LoadAlign(x2Reg, reinterpret_cast<__ubuf__ float*>(x2Ptr) + off);
            }

            AscendC::Reg::AbsSub(diffReg, x1Reg, x2Reg, mask);
            AscendC::Reg::Abs(absX2Reg, x2Reg, mask);
            AscendC::Reg::Muls(threshReg, absX2Reg, rtolVal, mask);
            AscendC::Reg::Adds(threshReg, threshReg, atolVal, mask);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::GT>(cmpMask, diffReg, threshReg, mask);
            AscendC::Reg::Select(resultReg, oneReg, zeroReg, cmpMask);
            AscendC::Reg::StoreAlign(dstPtr + off, resultReg, mask);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ClearChunkExtensionVf (isTailR=true path only)
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void DataCompareKernel<DType>::ClearChunkExtensionVf(uint32_t slot, int64_t rLen)
{
    if (rLen >= rUbFactor_)
        return;
    auto tmpAll = tmpBuf_.Get<float>();
    auto tmpSlot = tmpAll[static_cast<int32_t>(slot) * static_cast<int32_t>(tmpSlotElems_)];
    __ubuf__ float* base = reinterpret_cast<__ubuf__ float*>(tmpSlot.GetPhyAddr());

    const uint32_t aBundleEntries = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t innerRPA = static_cast<uint32_t>(innerRProdAlign_);
    const uint32_t rLenInner = static_cast<uint32_t>(rLen) * innerRPA;
    const uint32_t extStart = (rLenInner + kBlockF32 - 1) / kBlockF32 * kBlockF32;
    const uint32_t aStride = static_cast<uint32_t>(rUbFactorAlign_) * innerRPA;
    if (extStart >= aStride)
        return;
    const uint32_t extLanes = aStride - extStart;
    const uint32_t repPerA = (extLanes + kRepF32 - 1) / kRepF32;
    const uint16_t aU16 = static_cast<uint16_t>(aBundleEntries);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> idReg;
        AscendC::Reg::Duplicate(idReg, 0.0f);
        for (uint16_t a = 0; a < aU16; ++a) {
            int32_t aOff = static_cast<int32_t>(a) * static_cast<int32_t>(aStride);
            uint32_t remaining = extLanes;
            for (uint16_t r = 0; r < static_cast<uint16_t>(repPerA); ++r) {
                int32_t off =
                    aOff + static_cast<int32_t>(extStart) + static_cast<int32_t>(r) * static_cast<int32_t>(kRepF32);
                auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                AscendC::Reg::StoreAlign(base + off, idReg, mask);
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// MergeTmpBufVf: tmpBuf[0] += tmpBuf[1]
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void DataCompareKernel<DType>::MergeTmpBufVf()
{
    auto tmpAll = tmpBuf_.Get<float>();
    __ubuf__ float* p0 = reinterpret_cast<__ubuf__ float*>(tmpAll.GetPhyAddr());
    __ubuf__ float* p1 = p0 + tmpSlotElems_;
    const uint32_t totalElems =
        static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_ * rUbFactorAlign_ * innerRProdAlign_);
    const uint16_t repeatTime = static_cast<uint16_t>((totalElems + kRepF32U - 1) / kRepF32U);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> aReg, bReg;
        AscendC::Reg::MaskReg mask;
        uint32_t remaining = totalElems;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
            mask = AscendC::Reg::UpdateMask<float>(remaining);
            AscendC::Reg::LoadAlign(aReg, p0 + off);
            AscendC::Reg::LoadAlign(bReg, p1 + off);
            AscendC::Reg::Add(aReg, aReg, bReg, mask);
            AscendC::Reg::StoreAlign(p0 + off, aReg, mask);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ReduceRPattern: ReduceSum AR
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void DataCompareKernel<DType>::ReduceRPattern()
{
    auto tmpAll = tmpBuf_.Get<float>();
    auto src = tmpAll;
    auto dst = tmpAll[static_cast<int32_t>(tmpSlotElems_)];
    const uint32_t aBundle = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t rBundle = static_cast<uint32_t>(rUbFactorAlign_ * innerRProdAlign_);
    uint32_t srcShape[2] = {aBundle, rBundle};
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dst, src, srcShape, true);
}

// ════════════════════════════════════════════════════════════════════════════
// ClearCacheTreeVf
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void DataCompareKernel<DType>::ClearCacheTreeVf()
{
    __ubuf__ float* base = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr());
    const uint32_t totalElems = static_cast<uint32_t>(cacheBufElems_);
    const uint16_t repeatTime = static_cast<uint16_t>((totalElems + kRepF32U - 1) / kRepF32U);
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> zReg;
        AscendC::Reg::Duplicate(zReg, 0.0f);
        AscendC::Reg::MaskReg mask;
        uint32_t remaining = totalElems;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
            mask = AscendC::Reg::UpdateMask<float>(remaining);
            AscendC::Reg::StoreAlign(base + off, zReg, mask);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// DoCachingVf
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void DataCompareKernel<DType>::DoCachingVf(uint16_t cacheID)
{
    auto tmpAll = tmpBuf_.Get<float>();
    __ubuf__ float* srcPtr =
        reinterpret_cast<__ubuf__ float*>(tmpAll[static_cast<int32_t>(tmpSlotElems_)].GetPhyAddr());
    __ubuf__ float* cachePtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr());

    const uint32_t laneN = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t levelStride = (laneN + kBlockF32 - 1) / kBlockF32 * kBlockF32;
    const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);
    const uint16_t repeatTime = static_cast<uint16_t>((laneN + kRepF32U - 1) / kRepF32U);
    const uint16_t cacheLvlU16 = cacheID;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> aReg, bReg;
        AscendC::Reg::MaskReg mask;
        uint32_t remaining = laneN;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
            mask = AscendC::Reg::UpdateMask<float>(remaining);
            AscendC::Reg::LoadAlign(aReg, srcPtr + off);
            for (uint16_t j = 0; j < cacheLvlU16; ++j) {
                int32_t jOff = static_cast<int32_t>(j) * static_cast<int32_t>(levelStride) + off;
                AscendC::Reg::LoadAlign(bReg, cachePtr + jOff);
                AscendC::Reg::Add(aReg, aReg, bReg, mask);
            }
            AscendC::Reg::StoreAlign(cachePtr + levelOff + off, aReg, mask);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PostElewiseAndCopyOut: read cache root → CopyOut (identity post_op, fp32 output)
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void DataCompareKernel<DType>::PostElewiseAndCopyOut(int64_t outerOutOff, int64_t aLen)
{
    const uint32_t laneN = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t levelStride = (laneN + kBlockF32 - 1) / kBlockF32 * kBlockF32;
    const int32_t rootOff = static_cast<int32_t>(cacheCount_ - 1) * static_cast<int32_t>(levelStride);
    __ubuf__ float* rootPtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr()) + rootOff;

    // Output is fp32, copy cache root → outBuf
    auto outLocal = outQue_.AllocTensor<float>();
    __ubuf__ float* outPtr = reinterpret_cast<__ubuf__ float*>(outLocal.GetPhyAddr());
    const uint16_t repeatTime = static_cast<uint16_t>((laneN + kRepF32U - 1) / kRepF32U);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> f32Reg;
        AscendC::Reg::MaskReg mask;
        uint32_t remaining = laneN;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
            mask = AscendC::Reg::UpdateMask<float>(remaining);
            AscendC::Reg::LoadAlign(f32Reg, rootPtr + off);
            AscendC::Reg::StoreAlign(outPtr + off, f32Reg, mask);
        }
    }
    outQue_.EnQue(outLocal);

    auto outDeq = outQue_.DeQue<float>();
    DataCopyExtParams outParams;
    outParams.blockLen = static_cast<uint32_t>(aLen * innerAProd_ * static_cast<int64_t>(sizeof(float)));
    outParams.blockCount = 1;
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    DataCopyPad(yGm_[outerOutOff], outDeq, outParams);
    outQue_.FreeTensor(outDeq);
}

// ════════════════════════════════════════════════════════════════════════════
// Group template
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void DataCompareKernel<DType>::InitGroup(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, const DataCompareTilingData* td)
{
    Init(x1, x2, y, td);
    wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    rGroupCnt_ = td->rGroupCnt;
    aTotal_ = 1;
    for (int32_t i = 0; i < axisNum_; i += 2)
        aTotal_ *= axisShape_[i];
}

template <typename DType>
__aicore__ inline void DataCompareKernel<DType>::ProcessGroup()
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(usedCoreNum_))
        return;

    const int64_t aOuter = aLoopCntTotal_;
    const int64_t rOuter = rLoopCntTotal_;
    int64_t aChunkIdx = blockIdx / rGroupCnt_;
    int64_t rChunkIdx = blockIdx % rGroupCnt_;

    int64_t rPerCore = (rOuter + rGroupCnt_ - 1) / rGroupCnt_;
    int64_t rStart = rChunkIdx * rPerCore;
    int64_t rEnd = rStart + rPerCore;
    if (rEnd > rOuter)
        rEnd = rOuter;
    if (rStart >= rOuter)
        return;

    int64_t aLoopIdx = aChunkIdx;
    int64_t rem = aLoopIdx;
    const int64_t aSplitChunkIdx = rem % aSplitChunkCnt_;
    rem /= aSplitChunkCnt_;
    int64_t chunkGmOff = 0, chunkOutOff = 0;
    for (int32_t k = aSplit_ - 2; k >= 0; k -= 2) {
        const int64_t sz = axisShape_[k];
        const int64_t ix = rem % sz;
        rem /= sz;
        chunkGmOff += ix * axisStride_[k];
        chunkOutOff += ix * outStride_[k];
    }
    const int64_t aChunkStart = aSplitChunkIdx * aUbFactor_;
    const int64_t aLen =
        (aChunkStart + aUbFactor_ > axisShape_[aSplit_]) ? (axisShape_[aSplit_] - aChunkStart) : aUbFactor_;
    if (aLen <= 0)
        return;
    chunkGmOff += aChunkStart * axisStride_[aSplit_];
    chunkOutOff += aChunkStart * outStride_[aSplit_];

    int64_t rCount = rEnd - rStart;
    DoOneAChunkGroup(chunkGmOff, aLen, rStart, rEnd);
    Phase1OutputToWorkspace(chunkOutOff, aLen, rChunkIdx, rCount);
    SyncAll();
    Phase2Process();
}

template <typename DType>
__aicore__ inline void DataCompareKernel<DType>::DoOneAChunkGroup(
    int64_t outerGmOff, int64_t aLen, int64_t rStart, int64_t rEnd)
{
    ClearCacheTreeVf();
    int64_t localRCnt = rEnd - rStart;
    int64_t localBisPos = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(localRCnt)));
    int64_t localBisTail = localRCnt - localBisPos;

    for (int64_t i = 0; i < localBisTail; ++i) {
        {
            int64_t rChunk = 0, rLen = 0;
            const int64_t rOff = UnravelOuterR(rStart + i, rChunk, rLen);
            auto x1L = preInQueX1_.AllocTensor<D_T>();
            DoCopyInTile(outerGmOff + rOff, aLen, rLen, x1L, x1Gm_);
            preInQueX1_.EnQue(x1L);
            auto x2L = preInQueX2_.AllocTensor<D_T>();
            DoCopyInTile(outerGmOff + rOff, aLen, rLen, x2L, x2Gm_);
            preInQueX2_.EnQue(x2L);
            auto x1D = preInQueX1_.DeQue<D_T>();
            auto x2D = preInQueX2_.DeQue<D_T>();
            PreElewiseVf(x1D, x2D, 0U);
            preInQueX1_.FreeTensor(x1D);
            preInQueX2_.FreeTensor(x2D);
            if (rLen < rUbFactor_)
                ClearChunkExtensionVf(0U, rLen);
        }
        {
            int64_t rChunk = 0, rLen = 0;
            const int64_t rOff = UnravelOuterR(rStart + i + localBisPos, rChunk, rLen);
            auto x1L = preInQueX1_.AllocTensor<D_T>();
            DoCopyInTile(outerGmOff + rOff, aLen, rLen, x1L, x1Gm_);
            preInQueX1_.EnQue(x1L);
            auto x2L = preInQueX2_.AllocTensor<D_T>();
            DoCopyInTile(outerGmOff + rOff, aLen, rLen, x2L, x2Gm_);
            preInQueX2_.EnQue(x2L);
            auto x1D = preInQueX1_.DeQue<D_T>();
            auto x2D = preInQueX2_.DeQue<D_T>();
            PreElewiseVf(x1D, x2D, 1U);
            preInQueX1_.FreeTensor(x1D);
            preInQueX2_.FreeTensor(x2D);
            if (rLen < rUbFactor_)
                ClearChunkExtensionVf(1U, rLen);
        }
        MergeTmpBufVf();
        ReduceRPattern();
        DoCachingVf(GetCacheID(i));
    }
    for (int64_t i = localBisTail; i < localBisPos; ++i) {
        int64_t rChunk = 0, rLen = 0;
        const int64_t rOff = UnravelOuterR(rStart + i, rChunk, rLen);
        auto x1L = preInQueX1_.AllocTensor<D_T>();
        DoCopyInTile(outerGmOff + rOff, aLen, rLen, x1L, x1Gm_);
        preInQueX1_.EnQue(x1L);
        auto x2L = preInQueX2_.AllocTensor<D_T>();
        DoCopyInTile(outerGmOff + rOff, aLen, rLen, x2L, x2Gm_);
        preInQueX2_.EnQue(x2L);
        auto x1D = preInQueX1_.DeQue<D_T>();
        auto x2D = preInQueX2_.DeQue<D_T>();
        PreElewiseVf(x1D, x2D, 0U);
        preInQueX1_.FreeTensor(x1D);
        preInQueX2_.FreeTensor(x2D);
        if (rLen < rUbFactor_)
            ClearChunkExtensionVf(0U, rLen);
        ReduceRPattern();
        DoCachingVf(GetCacheID(i));
    }
}

template <typename DType>
__aicore__ inline void DataCompareKernel<DType>::Phase1OutputToWorkspace(
    int64_t wsOff, int64_t aLen, int64_t rChunkIdx, int64_t rCount)
{
    // V → MTE3 fence
    AscendC::TEventID ev = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    SetFlag<HardEvent::V_MTE3>(ev);
    WaitFlag<HardEvent::V_MTE3>(ev);

    const uint32_t laneN = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t levelStride = (laneN + kBlockF32 - 1) / kBlockF32 * kBlockF32;
    const uint64_t localBisPos = FindNearestPower2(static_cast<uint64_t>(rCount));
    const int32_t localRootLevel = static_cast<int32_t>(CalLog2(localBisPos));
    const int32_t rootOff = localRootLevel * static_cast<int32_t>(levelStride);

    auto cacheLocal = cacheBuf_.Get<float>();

    // MTE3 CopyOut: cache 树根 → workspace（tail-R 路径，单 block 连续搬）
    DataCopyExtParams outParams;
    outParams.blockLen = static_cast<uint32_t>(aLen * innerAProd_ * static_cast<int64_t>(sizeof(float)));
    outParams.blockCount = 1;
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    DataCopyPad(wsGm_[rChunkIdx * aTotal_ + wsOff], cacheLocal[rootOff], outParams);
}

template <typename DType>
__aicore__ inline void DataCompareKernel<DType>::Phase2Process()
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t preInElems = preReduceUbSize_ / static_cast<int64_t>(sizeof(float));
    constexpr int64_t BS_FP32 = static_cast<int64_t>(kBlockF32);
    int64_t aUbFactor_p2 = preInElems / rGroupCnt_;
    if (aUbFactor_p2 >= BS_FP32)
        aUbFactor_p2 = (aUbFactor_p2 / BS_FP32) * BS_FP32;
    if (aUbFactor_p2 < 1)
        aUbFactor_p2 = 1;
    if (aUbFactor_p2 > aTotal_)
        aUbFactor_p2 = aTotal_;

    int64_t aSplitChunkCnt_p2 = (aTotal_ + aUbFactor_p2 - 1) / aUbFactor_p2;
    int64_t aLoopCntTotal_p2 = aSplitChunkCnt_p2;
    int64_t aSmallCoreLoopCnt_p2 = aLoopCntTotal_p2 / usedCoreNum_;
    int64_t aBigCoreCnt_p2 = aLoopCntTotal_p2 % usedCoreNum_;
    int64_t aBigCoreLoopCnt_p2 = aSmallCoreLoopCnt_p2 + (aBigCoreCnt_p2 > 0 ? 1 : 0);
    int64_t usedCoreNum_p2 = (aSmallCoreLoopCnt_p2 > 0) ? usedCoreNum_ : aBigCoreCnt_p2;
    if (blockIdx >= usedCoreNum_p2)
        return;

    int64_t aLoopStart, aLoopEnd;
    if (blockIdx < aBigCoreCnt_p2) {
        aLoopStart = blockIdx * aBigCoreLoopCnt_p2;
        aLoopEnd = aLoopStart + aBigCoreLoopCnt_p2;
    } else {
        aLoopStart = aBigCoreCnt_p2 * aBigCoreLoopCnt_p2 + (blockIdx - aBigCoreCnt_p2) * aSmallCoreLoopCnt_p2;
        aLoopEnd = aLoopStart + aSmallCoreLoopCnt_p2;
    }

    for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
        int64_t a_off = aLoopIdx * aUbFactor_p2;
        int64_t a_len = aUbFactor_p2;
        if (a_off + a_len > aTotal_)
            a_len = aTotal_ - a_off;
        int64_t a_len_ub = (a_len + BS_FP32 - 1) / BS_FP32 * BS_FP32;

        // CopyIn from workspace → preInQueX1_ (复用 Phase 1 的 DB 槽)
        auto wsLocal = preInQueX1_.AllocTensor<float>();
        DataCopyExtParams ext;
        ext.blockLen = static_cast<uint32_t>(a_len * sizeof(float));
        ext.blockCount = static_cast<uint16_t>(rGroupCnt_);
        ext.srcStride = static_cast<uint32_t>(aTotal_ * sizeof(float) - ext.blockLen);
        ext.dstStride = 0;
        uint32_t misalign = ext.blockLen & (kBlockBytes - 1u);
        uint8_t rPad = misalign == 0 ? 0 : static_cast<uint8_t>((kBlockBytes - misalign) / sizeof(float));
        DataCopyPadExtParams<float> padParams{true, 0, rPad, 0.0f};
        DataCopyPad(wsLocal, wsGm_[a_off], ext, padParams);
        preInQueX1_.EnQue(wsLocal);

        // Reduce: RA pattern
        auto wsDeq = preInQueX1_.DeQue<float>();
        uint32_t srcShape_p2[2] = {static_cast<uint32_t>(rGroupCnt_), static_cast<uint32_t>(a_len_ub)};
        auto tmpLocal = tmpBuf_.Get<float>();
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(tmpLocal, wsDeq, srcShape_p2, true);
        preInQueX1_.FreeTensor(wsDeq); // 用完立即释放

        // CopyOut: 通过 outQue_ 中转，EnQue/DeQue 自带 V→MTE3 同步
        auto outLocal = outQue_.AllocTensor<float>();
        __ubuf__ float* outPtr = reinterpret_cast<__ubuf__ float*>(outLocal.GetPhyAddr());
        __ubuf__ float* srcPtr = reinterpret_cast<__ubuf__ float*>(tmpBuf_.Get<float>().GetPhyAddr());
        const uint16_t rep = static_cast<uint16_t>((static_cast<uint32_t>(a_len) + kRepF32U - 1) / kRepF32U);
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> reg;
            AscendC::Reg::MaskReg mask;
            uint32_t remaining = static_cast<uint32_t>(a_len);
            for (uint16_t i = 0; i < rep; ++i) {
                int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
                mask = AscendC::Reg::UpdateMask<float>(remaining);
                AscendC::Reg::LoadAlign(reg, srcPtr + off);
                AscendC::Reg::StoreAlign(outPtr + off, reg, mask);
            }
        }
        outQue_.EnQue(outLocal);
        auto outDeq = outQue_.DeQue<float>();
        DataCopyExtParams cpExt;
        cpExt.blockLen = static_cast<uint32_t>(a_len * sizeof(float));
        cpExt.blockCount = 1;
        cpExt.srcStride = 0;
        cpExt.dstStride = 0;
        DataCopyPad(yGm_[a_off], outDeq, cpExt);
        outQue_.FreeTensor(outDeq);
    }
}

} // namespace NsDataCompare

#endif // OPS_DATA_COMPARE_H_
