/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cdist_reduce_hl.h
 * \brief 方案三 (reduce_highlevel): cdist M∈[2,256] 融合内核 —— broadcast + WholeReduceSum M轴归约.
 *
 *   用基础API WholeReduceSum替代方案VF Reg::ReduceSum手写scope。
 *   elementwise（Sub/Mul/Abs/Sqrt/Ln/Exp/Muls）全部走高层向量 API。
 *
 *   计算布局（每个 UB tile, mSize 列）:
 *     x1 UB-resident [bSize,pSize,MAlign], x2 UB-resident [bSize,rSize,MAlign]（Normal CopyIn 已搬入，
 *     每行 x1[b,p,:] / x2[b,r,:] 各只搬一次，p×r 网格在 UB 内复用 —— 即 broadcast+reuse，GM 流量 ÷(P·R)）。
 *     逐 (b,p): Broadcast x1[b,p,:M] [1,M]->[rSize,M] (rank2 axis0) 到 x1exp；
 *               Sub(diff, x1exp, x2[b], rSize*MAlign)；Abs/Mul（按 p）；
 *               WholeReduceSum(acc[b,p,0:rSize], diff, mask=mSize, repeat=rSize,
 *                              dstRepStride=1, srcBlkStride=1, srcRepStride=MAlign/blk) → 每 r 一个和；
 *               M-split(ubLoopNumM>1) 时把段和累加到 yFp32_（p!=inf: Add; p==inf: Max），末段再归一。
 */
#ifndef CDIST_REDUCE_HL_H
#define CDIST_REDUCE_HL_H

#include "kernel_operator.h"
#include "../cdist_tiling_data.h"

#if !defined(__NPU_HOST__)

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace NsCdist {

using namespace AscendC;

template <typename T>
class CdistReduceHL {
public:
    __aicore__ inline CdistReduceHL(){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const CdistTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const CdistTilingData* tilingData);
    __aicore__ inline void CopyInX1(uint64_t Offset);
    __aicore__ inline void CopyInX2(uint64_t Offset);
    __aicore__ inline void CopyOut(uint64_t Offset);
    __aicore__ inline void Compute();
    __aicore__ inline void ComputeSplitM();
    __aicore__ inline void ProcessSplitM(uint32_t bOffset, uint32_t pOffset, uint32_t rOffsetBlock,
                                         uint32_t blockFactorR);
    __aicore__ inline void ProcessNoSplitM(uint32_t bOffset, uint32_t pOffset, uint32_t rOffsetBlock,
                                           uint32_t blockFactorR);
    __aicore__ inline void CalSplitMResult(int32_t processNum);
    // High-level reduce compute: fill dst[b,p,0:rSize] with p-norm sum over the mSize M-columns.
    __aicore__ inline void ComputeOneSizeHL(const LocalTensor<float>& x1Local, const LocalTensor<float>& x2Local,
                                            const LocalTensor<float>& dst, bool finalize);
    __aicore__ inline void ReduceRowsSum(const LocalTensor<float>& acc, const LocalTensor<float>& src, uint32_t rows,
                                         uint32_t mLen, uint32_t mAlign, bool accumulate);
    __aicore__ inline void ReduceRowsMax(const LocalTensor<float>& acc, const LocalTensor<float>& src, uint32_t rows,
                                         uint32_t mLen, uint32_t mAlign, bool accumulate);
    // VF Cast T→float in-place: load T staging from upper half, cast to float, store to lower half.
    __aicore__ inline void VfCastTToFloat(const LocalTensor<float>& dst, const LocalTensor<T>& src, uint32_t count);

private:
    constexpr static int32_t BUFFER_NUM = 2;
    constexpr static int32_t BLOCK_SIZE = 32;
    constexpr static uint32_t BASE_ONE = 1;
    constexpr static int32_t REDUCE_MAX_MASK = 64; // fp32: 256B / 4 = 64 elems per repeat
    constexpr static int32_t REDUCE_MAX_REPEAT = 255;
    // Fixed UB budget for the two HL fp32 compute planes (x1exp + diff). Decouples compute-plane size
    // from the Normal tiling's ubFactorR/ubFactorM so the extra planes never overflow UB. R (and the
    // reduce repeat count) are chunked inside ComputeOneSizeHL to this cap.
    constexpr static int32_t HL_PLANE_ELEMS = 8192; // 32KB per fp32 plane, 64KB for both
    constexpr static ExpConfig expConfig = {ExpAlgo::PRECISION_1ULP_FTZ_FALSE};
    constexpr static LnConfig lnConfig = {LnAlgo::PRECISION_1ULP_FTZ_FALSE};
    constexpr static SqrtConfig sqrtConfig = {SqrtAlgo::PRECISION_0ULP_FTZ_FALSE};
    constexpr static Reg::CastTrait castTraitB16ToB32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    constexpr static uint32_t VREG_SIZE = 256;                       // arch35 vector register width in bytes
    constexpr static uint32_t VREG_FP32 = VREG_SIZE / sizeof(float); // 64 float elements per VReg
    int64_t blockIdx_;
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> x1Queue_;
    TQue<QuePosition::VECIN, 1> x2Queue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TQue<QuePosition::VECCALC, 1> tmpQueue_; // segment result buffer for M-split (like Normal tmpLocal_)
    TBuf<QuePosition::VECCALC> x1ExpBuf_;    // x1 broadcast plane [rSize, MAlign] fp32
    TBuf<QuePosition::VECCALC> diffBuf_;     // |diff|^p plane      [rSize, MAlign] fp32
    TBuf<QuePosition::VECCALC> partBuf_;     // WholeReduceSum partial scratch (M>64), <=hlRTile lanes
    GlobalTensor<T> x1GM_;
    GlobalTensor<T> x2GM_;
    GlobalTensor<T> yGM_;
    LocalTensor<float> yFp32_;
    LocalTensor<float> tmpLocal_; // segment accumulator [bSize*pSize*RAlign] fp32
    const CdistTilingData* tiling_;
    uint32_t realCoreNum_ = 0;
    uint32_t B_ = 0;
    uint32_t P_ = 0;
    uint32_t R_ = 0;
    uint32_t RAlign_ = 0;
    uint32_t M_ = 0;
    uint32_t MAlign_ = 0;
    uint32_t blockMainNumB_ = 0;
    uint32_t blockTailNumB_ = 0;
    uint32_t blockMainFactorB_ = 0;
    uint32_t blockTailFactorB_ = 0;
    uint32_t blockMainNumP_ = 0;
    uint32_t blockTailNumP_ = 0;
    uint32_t blockMainFactorP_ = 0;
    uint32_t blockTailFactorP_ = 0;
    uint32_t blockMainNumR_ = 0;
    uint32_t blockTailNumR_ = 0;
    uint32_t blockMainFactorR_ = 0;
    uint32_t blockTailFactorR_ = 0;
    uint32_t ubLoopNumB_ = 0;
    uint32_t ubFactorB_ = 0;
    uint32_t ubTailFactorB_ = 0;
    uint32_t ubLoopNumP_ = 0;
    uint32_t ubFactorP_ = 0;
    uint32_t ubTailFactorP_ = 0;
    uint32_t ubLoopNumR_ = 0;
    uint32_t ubFactorR_ = 0;
    uint32_t ubTailFactorR_ = 0;
    uint32_t ubLoopNumM_ = 0;
    uint32_t ubFactorM_ = 0;
    uint32_t ubTailFactorM_ = 0;
    float p_ = 0;
    uint32_t bSize_ = 0;
    uint32_t pSize_ = 0;
    uint32_t rSize_ = 0;
    uint32_t mSize_ = 0;
    uint32_t ubFactorMAlign_ = 0;
    uint32_t ubFactorRAlign_ = 0;
    // Datacopy params (verbatim from Normal)
    DataCopyExtParams copyInParamsX1_{1, 0, 0, 0, 0};
    DataCopyExtParams copyInParamsX2_{1, 0, 0, 0, 0};
    LoopModeParams loopParamX1_{1, 0, 0, 0, 0, 0};
    LoopModeParams loopParamX2_{1, 0, 0, 0, 0, 0};
    DataCopyExtParams copyOutParams_{1, 0, 0, 0, 0};
    LoopModeParams loopParamOut_{1, 0, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};
};

template <typename T>
__aicore__ inline void CdistReduceHL<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const CdistTilingData* tilingData,
                                              TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    pipe_ = pipe;
    tiling_ = tilingData;
    ParseTilingData(tiling_);
    x1GM_.SetGlobalBuffer((__gm__ T*)x1);
    x2GM_.SetGlobalBuffer((__gm__ T*)x2);
    yGM_.SetGlobalBuffer((__gm__ T*)y);
    if (ubLoopNumM_ == 1) {
        ubFactorMAlign_ = ((ubFactorM_ * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE) / sizeof(T);
    } else {
        ubFactorMAlign_ = ubFactorM_;
    }
    ubFactorRAlign_ = ((ubFactorR_ * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE) / sizeof(T);
    // x1/x2/y queues always allocated as sizeof(float). For fp16, the T-typed GM data is DataCopyPad'd
    // into the upper half of the float buffer via ReinterpretCast<T>, then VF Cast T→float in-place.
    // This eliminates separate x1CastQueue_/x2CastQueue_/yCastQueue_.
    pipe_->InitBuffer(x1Queue_, BUFFER_NUM, ubFactorB_ * ubFactorP_ * ubFactorMAlign_ * sizeof(float));
    pipe_->InitBuffer(x2Queue_, BUFFER_NUM, ubFactorB_ * ubFactorR_ * ubFactorMAlign_ * sizeof(float));
    pipe_->InitBuffer(yQueue_, BUFFER_NUM, ubFactorB_ * ubFactorP_ * ubFactorRAlign_ * sizeof(float));
    // segment accumulator (fp32) for M-split path — only large when M is actually split; otherwise a
    // 32B stub (Compute() reduces straight into yFp32_, tmpLocal_ unused).
    uint32_t tmpBytes = (ubLoopNumM_ > 1) ? (ubFactorB_ * ubFactorP_ * ubFactorRAlign_ * (uint32_t)sizeof(float)) :
                                            BLOCK_SIZE;
    pipe_->InitBuffer(tmpQueue_, 1, tmpBytes);
    // HL compute planes (fp32): x1 broadcast + diff, each capped at HL_PLANE_ELEMS. R is chunked inside
    // ComputeOneSizeHL to hlRTile_ = HL_PLANE_ELEMS / MAlign rows so the planes never overflow UB
    // (decoupled from the Normal tiling's ubFactorR/ubFactorM which sizes only x1/x2/y). x1ExpBuf_ head
    // is also reused as the WholeReduceSum partial scratch (>= hlRTile_ lanes, guaranteed since
    // hlRTile_ <= HL_PLANE_ELEMS/1).
    pipe_->InitBuffer(x1ExpBuf_, HL_PLANE_ELEMS * sizeof(float));
    pipe_->InitBuffer(diffBuf_, HL_PLANE_ELEMS * sizeof(float));
    // Dedicated WholeReduceSum partial scratch (one lane per reduced row, rep <= REDUCE_MAX_REPEAT=255
    // and ReduceRowsMax redOut <= 248 lanes; 256 for block alignment). Must NOT alias x1exp/diff.
    pipe_->InitBuffer(partBuf_, 256 * sizeof(float));
    // yQueue_ holds float compute results (BUFFER_NUM=2). yFp32_ is DeQue'd/EnQue'd per iteration
    // — no pre-allocation in Init. tmpLocal_ stays EnQue'd from Init (DeQue'd in ComputeSplitM only).
    tmpLocal_ = tmpQueue_.AllocTensor<float>();
    Duplicate<float>(
        tmpLocal_, (float)0,
        (ubLoopNumM_ > 1) ? (ubFactorB_ * ubFactorP_ * ubFactorRAlign_) : (BLOCK_SIZE / (int32_t)sizeof(float)));
    tmpQueue_.EnQue(tmpLocal_);
}

template <typename T>
__aicore__ inline void CdistReduceHL<T>::ParseTilingData(const CdistTilingData* tdPtr)
{
    B_ = tdPtr->B;
    P_ = tdPtr->P;
    R_ = tdPtr->R;
    M_ = tdPtr->M;
    blockMainNumB_ = tdPtr->blockMainNumB;
    blockTailNumB_ = tdPtr->blockTailNumB;
    blockMainFactorB_ = tdPtr->blockMainFactorB;
    blockTailFactorB_ = tdPtr->blockTailFactorB;
    blockMainNumP_ = tdPtr->blockMainNumP;
    blockTailNumP_ = tdPtr->blockTailNumP;
    blockMainFactorP_ = tdPtr->blockMainFactorP;
    blockTailFactorP_ = tdPtr->blockTailFactorP;
    blockMainNumR_ = tdPtr->blockMainNumR;
    blockTailNumR_ = tdPtr->blockTailNumR;
    blockMainFactorR_ = tdPtr->blockMainFactorR;
    blockTailFactorR_ = tdPtr->blockTailFactorR;
    ubLoopNumB_ = tdPtr->ubLoopNumB;
    ubFactorB_ = tdPtr->ubFactorB;
    ubTailFactorB_ = tdPtr->ubTailFactorB;
    ubLoopNumP_ = tdPtr->ubLoopNumP;
    ubFactorP_ = tdPtr->ubFactorP;
    ubTailFactorP_ = tdPtr->ubTailFactorP;
    ubLoopNumR_ = tdPtr->ubLoopNumR;
    ubFactorR_ = tdPtr->ubFactorR;
    ubTailFactorR_ = tdPtr->ubTailFactorR;
    ubLoopNumM_ = tdPtr->ubLoopNumM;
    ubFactorM_ = tdPtr->ubFactorM;
    ubTailFactorM_ = tdPtr->ubTailFactorM;
    p_ = tdPtr->p;
}

// VF Cast T→float: MicroAPI loop loading T via DIST_UNPACK_B16, casting to float, storing to dst.
// Used by CopyInX1/X2 for fp16→fp32 in-place cast within the same UB buffer (staging at upper half).
// Pattern follows inplace_index_add_with_sorted arch35 (A5) reference implementation.
template <typename T>
__aicore__ inline void CdistReduceHL<T>::VfCastTToFloat(const LocalTensor<float>& dst, const LocalTensor<T>& src,
                                                        uint32_t count)
{
    __VEC_SCOPE__
    {
        __local_mem__ float* dstPtr = reinterpret_cast<__local_mem__ float*>(dst.GetPhyAddr());
        __local_mem__ T* srcPtr = reinterpret_cast<__local_mem__ T*>(src.GetPhyAddr());
        uint32_t sreg = count;
        Reg::MaskReg mask;
        Reg::RegTensor<T> aReg;
        Reg::RegTensor<float> bReg;
        uint16_t loops = (count + VREG_FP32 - 1) / VREG_FP32;
        for (uint16_t i = 0; i < loops; i++) {
            mask = Reg::UpdateMask<float>(sreg);
            Reg::DataCopy<T, Reg::LoadDist::DIST_UNPACK_B16>(aReg, srcPtr + i * VREG_FP32);
            Reg::Cast<float, T, castTraitB16ToB32>(bReg, aReg, mask);
            Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(dstPtr + i * VREG_FP32, bReg, mask);
        }
    }
}

template <typename T>
__aicore__ inline void CdistReduceHL<T>::CopyInX1(uint64_t Offset)
{
    LocalTensor<float> x1Local = x1Queue_.AllocTensor<float>();
    if constexpr (sizeof(T) == sizeof(float)) {
        // fp32: direct DataCopyPad, no cast needed
        LocalTensor<T> x1T = x1Local.template ReinterpretCast<T>();
        copyInParamsX1_.blockCount = static_cast<uint16_t>(pSize_);
        copyInParamsX1_.blockLen = (ubLoopNumM_ == 1) ? static_cast<uint32_t>(M_ * sizeof(T)) :
                                                        static_cast<uint32_t>(mSize_ * sizeof(T));
        copyInParamsX1_.srcStride = (ubLoopNumM_ == 1) ? 0 : static_cast<uint32_t>((M_ - mSize_) * sizeof(T));
        copyInParamsX1_.dstStride = 0;
        loopParamX1_.loop1Size = static_cast<uint32_t>(bSize_);
        loopParamX1_.loop2Size = 1;
        loopParamX1_.loop1SrcStride = static_cast<uint64_t>((uint64_t)M_ * P_ * sizeof(T));
        loopParamX1_.loop2SrcStride = 0;
        loopParamX1_.loop1DstStride = static_cast<uint64_t>((pSize_ * MAlign_) * sizeof(T));
        loopParamX1_.loop2DstStride = 0;
        SetLoopModePara(loopParamX1_, DataCopyMVType::OUT_TO_UB);
        DataCopyPad(x1T, x1GM_[Offset], copyInParamsX1_, padParams_);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    } else {
        // fp16: DataCopyPad T into upper-half staging, then VF Cast T→float in-place
        LocalTensor<T> x1T = x1Local.template ReinterpretCast<T>();
        constexpr uint32_t blockUbs = BLOCK_SIZE / sizeof(T); // 16 for fp16
        uint32_t totalElems = bSize_ * pSize_ * MAlign_;
        uint32_t stageOffset = ((totalElems + blockUbs - 1) / blockUbs) * blockUbs;
        copyInParamsX1_.blockCount = static_cast<uint16_t>(pSize_);
        copyInParamsX1_.blockLen = (ubLoopNumM_ == 1) ? static_cast<uint32_t>(M_ * sizeof(T)) :
                                                        static_cast<uint32_t>(mSize_ * sizeof(T));
        copyInParamsX1_.srcStride = (ubLoopNumM_ == 1) ? 0 : static_cast<uint32_t>((M_ - mSize_) * sizeof(T));
        copyInParamsX1_.dstStride = 0;
        loopParamX1_.loop1Size = static_cast<uint32_t>(bSize_);
        loopParamX1_.loop2Size = 1;
        loopParamX1_.loop1SrcStride = static_cast<uint64_t>((uint64_t)M_ * P_ * sizeof(T));
        loopParamX1_.loop2SrcStride = 0;
        loopParamX1_.loop1DstStride = static_cast<uint64_t>((pSize_ * MAlign_) * sizeof(T));
        loopParamX1_.loop2DstStride = 0;
        SetLoopModePara(loopParamX1_, DataCopyMVType::OUT_TO_UB);
        DataCopyPad(x1T[stageOffset], x1GM_[Offset], copyInParamsX1_, padParams_);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        // sync MTE2 → V before VF cast
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        uint32_t count = ((totalElems + blockUbs - 1) / blockUbs) * blockUbs;
        VfCastTToFloat(x1Local, x1T[stageOffset], count);
    }
    x1Queue_.EnQue(x1Local);
}

template <typename T>
__aicore__ inline void CdistReduceHL<T>::CopyInX2(uint64_t Offset)
{
    LocalTensor<float> x2Local = x2Queue_.AllocTensor<float>();
    if constexpr (sizeof(T) == sizeof(float)) {
        LocalTensor<T> x2T = x2Local.template ReinterpretCast<T>();
        copyInParamsX2_.blockCount = static_cast<uint16_t>(rSize_);
        copyInParamsX2_.blockLen = (ubLoopNumM_ == 1) ? static_cast<uint32_t>(M_ * sizeof(T)) :
                                                        static_cast<uint32_t>(mSize_ * sizeof(T));
        copyInParamsX2_.srcStride = (ubLoopNumM_ == 1) ? 0 : static_cast<uint32_t>((M_ - mSize_) * sizeof(T));
        copyInParamsX2_.dstStride = 0;
        loopParamX2_.loop1Size = static_cast<uint32_t>(bSize_);
        loopParamX2_.loop2Size = 1;
        loopParamX2_.loop1SrcStride = static_cast<uint64_t>((uint64_t)M_ * R_ * sizeof(T));
        loopParamX2_.loop2SrcStride = 0;
        loopParamX2_.loop1DstStride = static_cast<uint64_t>((rSize_ * MAlign_) * sizeof(T));
        loopParamX2_.loop2DstStride = 0;
        SetLoopModePara(loopParamX2_, DataCopyMVType::OUT_TO_UB);
        DataCopyPad(x2T, x2GM_[Offset], copyInParamsX2_, padParams_);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    } else {
        LocalTensor<T> x2T = x2Local.template ReinterpretCast<T>();
        constexpr uint32_t blockUbs = BLOCK_SIZE / sizeof(T); // 16 for fp16
        uint32_t totalElems = bSize_ * rSize_ * MAlign_;
        uint32_t stageOffset = ((totalElems + blockUbs - 1) / blockUbs) * blockUbs;
        copyInParamsX2_.blockCount = static_cast<uint16_t>(rSize_);
        copyInParamsX2_.blockLen = (ubLoopNumM_ == 1) ? static_cast<uint32_t>(M_ * sizeof(T)) :
                                                        static_cast<uint32_t>(mSize_ * sizeof(T));
        copyInParamsX2_.srcStride = (ubLoopNumM_ == 1) ? 0 : static_cast<uint32_t>((M_ - mSize_) * sizeof(T));
        copyInParamsX2_.dstStride = 0;
        loopParamX2_.loop1Size = static_cast<uint32_t>(bSize_);
        loopParamX2_.loop2Size = 1;
        loopParamX2_.loop1SrcStride = static_cast<uint64_t>((uint64_t)M_ * R_ * sizeof(T));
        loopParamX2_.loop2SrcStride = 0;
        loopParamX2_.loop1DstStride = static_cast<uint64_t>((rSize_ * MAlign_) * sizeof(T));
        loopParamX2_.loop2DstStride = 0;
        SetLoopModePara(loopParamX2_, DataCopyMVType::OUT_TO_UB);
        DataCopyPad(x2T[stageOffset], x2GM_[Offset], copyInParamsX2_, padParams_);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        uint32_t count = ((totalElems + blockUbs - 1) / blockUbs) * blockUbs;
        VfCastTToFloat(x2Local, x2T[stageOffset], count);
    }
    x2Queue_.EnQue(x2Local);
}

template <typename T>
__aicore__ inline void CdistReduceHL<T>::CopyOut(uint64_t Offset)
{
    // yFp32_ was EnQue'd as float by Compute/CalSplitMResult. DeQue<float> here.
    // For fp16: cast float→T in-place (T fits in lower part of float buffer), then DataCopy T→GM.
    // For fp32: yFp32_ IS the T tensor, DataCopy directly.
    LocalTensor<float> yFp32 = yQueue_.DeQue<float>();
    LocalTensor<T> yLocal = yFp32.template ReinterpretCast<T>();
    if constexpr (sizeof(T) != sizeof(float)) {
        Cast(yLocal, yFp32, RoundMode::CAST_RINT, (uint32_t)(bSize_ * pSize_ * RAlign_));
        // sync V → MTE3 so cast completes before DataCopy
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }
    copyOutParams_.blockCount = static_cast<uint16_t>(pSize_);
    copyOutParams_.blockLen = static_cast<uint32_t>(rSize_ * sizeof(T));
    copyOutParams_.srcStride = 0;
    copyOutParams_.dstStride = static_cast<uint32_t>((R_ - rSize_) * sizeof(T));
    loopParamOut_.loop1Size = static_cast<uint32_t>(bSize_);
    loopParamOut_.loop2Size = 1;
    loopParamOut_.loop1SrcStride = static_cast<uint64_t>((pSize_ * RAlign_) * sizeof(T));
    loopParamOut_.loop2SrcStride = 0;
    loopParamOut_.loop1DstStride = static_cast<uint64_t>((uint64_t)P_ * R_ * sizeof(T));
    loopParamOut_.loop2DstStride = 0;
    SetLoopModePara(loopParamOut_, DataCopyMVType::UB_TO_OUT);
    DataCopyPad(yGM_[Offset], yLocal, copyOutParams_);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    yQueue_.FreeTensor(yFp32);
}

// Per-row (per output element) sum-reduce of `rows` rows of `mLen` fp32 columns (row stride = mAlign),
// using WholeReduceSum. Chunks mask over REDUCE_MAX_MASK (64) and rows over
// REDUCE_MAX_REPEAT (255). acc[row] gets the running total (accumulate=false: overwrite; true: add).
template <typename T>
__aicore__ inline void CdistReduceHL<T>::ReduceRowsSum(const LocalTensor<float>& acc, const LocalTensor<float>& src,
                                                       uint32_t rows, uint32_t mLen, uint32_t mAlign, bool accumulate)
{
    uint32_t blkStrideRep = mAlign / (BLOCK_SIZE / (uint32_t)sizeof(float)); // src repeat stride in 32B blocks
    for (uint32_t r0 = 0; r0 < rows; r0 += REDUCE_MAX_REPEAT) {
        uint32_t rep = (rows - r0 > (uint32_t)REDUCE_MAX_REPEAT) ? (uint32_t)REDUCE_MAX_REPEAT : (rows - r0);
        LocalTensor<float> accChunk = acc[r0];
        LocalTensor<float> srcChunk = src[r0 * mAlign];
        // M-chunking: sum partials over 64-wide mask windows into accChunk (dstRepStride=1 block).
        bool first = !accumulate;
        for (uint32_t m0 = 0; m0 < mLen; m0 += REDUCE_MAX_MASK) {
            uint32_t mask = (mLen - m0 > (uint32_t)REDUCE_MAX_MASK) ? (uint32_t)REDUCE_MAX_MASK : (mLen - m0);
            if (first) {
                WholeReduceSum<float>(accChunk, srcChunk[m0], (int32_t)mask, (int32_t)rep, 1, 1, (int32_t)blkStrideRep);
                first = false;
            } else {
                LocalTensor<float> part = partBuf_.Get<float>();
                WholeReduceSum<float>(part, srcChunk[m0], (int32_t)mask, (int32_t)rep, 1, 1, (int32_t)blkStrideRep);
                Add(accChunk, accChunk, part, rep);
            }
        }
    }
}

template <typename T>
__aicore__ inline void CdistReduceHL<T>::ReduceRowsMax(const LocalTensor<float>& acc, const LocalTensor<float>& src,
                                                       uint32_t rows, uint32_t mLen, uint32_t mAlign, bool accumulate)
{
    constexpr uint32_t BLK_FP32 = BLOCK_SIZE / (uint32_t)sizeof(float); // 8 fp32 per 32B block
    constexpr uint32_t REP_LANES = BLK_FP32 * BLK_FP32;                 // 64 lanes = 8 blocks per repeat
    uint32_t srcStrideBlk = mAlign / BLK_FP32;                          // source row stride in 32B blocks
    // Rows per chunk. Bounded by: (a) the Stage-1 Level-0 `Max` repeat limit — repeat==rep must be <=255
    // (uint8_t); (b) [rowsPad, 8] fitting x1ExpBuf_ (HL_PLANE_ELEMS). Use a multiple of 8 <=255 (=248) so
    // the BlockReduceMax padding is clean and no repeat overflows. x1ExpBuf_ is free here (post-Sub).
    uint32_t rowsCap = ((uint32_t)REDUCE_MAX_REPEAT / BLK_FP32) * BLK_FP32;     // 248 (mult of 8, <=255)
    uint32_t fitCap = ((uint32_t)HL_PLANE_ELEMS / BLK_FP32) & ~(BLK_FP32 - 1u); // by UB size
    if (rowsCap > fitCap)
        rowsCap = fitCap;
    if (rowsCap < BLK_FP32)
        rowsCap = BLK_FP32;
    LocalTensor<float> accPlane = x1ExpBuf_.Get<float>(); // packed [rowsPad, 8] scratch (1 block/row)

    for (uint32_t r0 = 0; r0 < rows; r0 += rowsCap) {
        uint32_t rep = (rows - r0 > rowsCap) ? rowsCap : (rows - r0);  // real rows this chunk
        uint32_t repPad = (rep + BLK_FP32 - 1u) / BLK_FP32 * BLK_FP32; // rows rounded up to *8
        LocalTensor<float> srcChunk = src[(uint64_t)r0 * mAlign];      // [rep, mLen] (row stride mAlign)

        // --- Stage 1: zero the [repPad, 8] accumulator, then Max-fold source 8-col chunks into it. Only
        // the `rep` real rows are folded; padding rows [rep, repPad) stay 0. All bases 32B-aligned. ---
        Duplicate<float>(accPlane, (float)0, repPad * BLK_FP32);
        for (uint32_t m0 = 0; m0 < mLen; m0 += BLK_FP32) {
            uint32_t w = (mLen - m0 > BLK_FP32) ? BLK_FP32 : (mLen - m0); // valid cols this chunk (<=8)
            BinaryRepeatParams rp;
            rp.dstBlkStride = 1;
            rp.src0BlkStride = 1;
            rp.src1BlkStride = 1;
            rp.dstRepStride = 1; // accPlane packed: 1 block (8 fp32) per row
            rp.src0RepStride = 1;
            rp.src1RepStride = (uint8_t)srcStrideBlk; // source: mAlign per row (whole 32B blocks)
            // accPlane[:, 0:w] = max(accPlane[:, 0:w], srcChunk[:, m0:m0+w]); m0 is a multiple of 8 (aligned)
            Max<float>(accPlane, accPlane, srcChunk[m0], (uint64_t)w, (uint8_t)rep, rp);
        }

        // --- Stage 2: BlockReduceMax packs one max per 32B block. Each repeat spans 8 blocks (8 rows)
        // and emits 8 packed results; drive with repeat=repPad/8, mask=64. Write into a scratch (partBuf_,
        // repPad results incl. padding), then copy exactly the `rep` real results contiguously into acc so
        // the up-to-7 padding outputs never touch neighbouring output rows. ---
        LocalTensor<float> redOut = partBuf_.Get<float>();
        BlockReduceMax<float>(redOut, accPlane, (int32_t)(repPad / BLK_FP32), (int32_t)REP_LANES,
                              /*dstRepStride=*/1, /*srcBlkStride=*/1, /*srcRepStride=*/BLK_FP32);
        // contiguous copy redOut[0:rep] -> acc[r0:r0+rep] (aligned base; Max with self == copy)
        Max(acc[r0], redOut, redOut, (int32_t)rep);
    }
}

// Fill dst (fp32 [bSize*pSize*RAlign], row-major with RAlign stride per p) with the p-norm partial over
// the current mSize M-columns. finalize=true → apply the 1/p normalization in place (single-tile M).
template <typename T>
__aicore__ inline void CdistReduceHL<T>::ComputeOneSizeHL(const LocalTensor<float>& x1Local,
                                                          const LocalTensor<float>& x2Local,
                                                          const LocalTensor<float>& dst, bool finalize)
{
    uint32_t M = (ubLoopNumM_ == 1) ? M_ : mSize_;
    uint32_t mA = MAlign_; // T-typed block-aligned element stride, shared by UB rows and compute plane
    LocalTensor<float> x1exp = x1ExpBuf_.Get<float>();
    LocalTensor<float> diff = diffBuf_.Get<float>();
    // R chunk: at most hlRTile rows per broadcast/reduce so the fp32 plane fits HL_PLANE_ELEMS.
    uint32_t hlRTile = (uint32_t)HL_PLANE_ELEMS / mA;
    if (hlRTile < 1)
        hlRTile = 1;

    for (uint32_t b = 0; b < bSize_; b++) {
        for (uint32_t p = 0; p < pSize_; p++) {
            LocalTensor<float> x1row = x1Local[b * pSize_ * mA + p * mA]; // [1, M]
            for (uint32_t r0 = 0; r0 < rSize_; r0 += hlRTile) {
                uint32_t rows = (rSize_ - r0 > hlRTile) ? hlRTile : (rSize_ - r0);
                LocalTensor<float> x2b = x2Local[b * rSize_ * mA + r0 * mA];              // [rows, M]
                LocalTensor<float> accRow = dst[b * pSize_ * RAlign_ + p * RAlign_ + r0]; // [rows]
                // --- broadcast x1row [1,M] -> x1exp [rows, M] (rank2, axis 0) ---
                {
                    uint32_t dShape[2] = {rows, mA};
                    uint32_t sShape[2] = {1u, mA};
                    BroadcastTiling bt;
                    GetBroadcastTilingInfo<float, 2>(2u, dShape, sShape, false, bt);
                    Broadcast<float, 2>(x1exp, x1row, dShape, sShape, &bt);
                }
                // --- diff = |x1exp - x2b| (^power) over [rows, MAlign] ---
                uint32_t planeCnt = rows * mA;
                if (p_ == 2.0f) {
                    Sub(diff, x1exp, x2b, planeCnt);
                    Mul(diff, diff, diff, planeCnt);
                    ReduceRowsSum(accRow, diff, rows, M, mA, false);
                } else if (p_ == 1.0f) {
                    AbsSub(diff, x1exp, x2b, planeCnt);
                    ReduceRowsSum(accRow, diff, rows, M, mA, false);
                } else if (p_ == 0.0f) {
                    AbsSub(diff, x1exp, x2b, planeCnt);
                    Ceil(diff, diff, planeCnt);
                    Mins(diff, diff, (float)1, planeCnt);
                    ReduceRowsSum(accRow, diff, rows, M, mA, false);
                } else if (p_ == static_cast<float>(INFINITY)) {
                    AbsSub(diff, x1exp, x2b, planeCnt);
                    ReduceRowsMax(accRow, diff, rows, M, mA, false);
                } else {
                    AbsSub(diff, x1exp, x2b, planeCnt);
                    Ln<float, lnConfig>(diff, diff, planeCnt);
                    Muls(diff, diff, (float)p_, planeCnt);
                    Exp<float, expConfig>(diff, diff, planeCnt);
                    ReduceRowsSum(accRow, diff, rows, M, mA, false);
                }
            }
        }
    }
}

// Single-tile (ubLoopNumM_==1) compute: reduce directly into yFp32_ then normalize.
// After this, yFp32_ is EnQue'd as float in yQueue_. CopyOut handles fp16→T cast.
// x1 is DeQue'd here and re-EnQue'd for reuse by the next R iteration (x1 persists across R loop).
// x2 is DeQue'd and freed (x2 changes each R iteration).
template <typename T>
__aicore__ inline void CdistReduceHL<T>::Compute()
{
    LocalTensor<float> x1Local = x1Queue_.DeQue<float>();
    LocalTensor<float> x2Local = x2Queue_.DeQue<float>();
    yFp32_ = yQueue_.template AllocTensor<float>();
    Duplicate<float>(yFp32_, (float)0, bSize_ * pSize_ * RAlign_);
    ComputeOneSizeHL(x1Local, x2Local, yFp32_, /*finalize=*/true);
    // normalize in place (whole tile).
    int32_t processNum = bSize_ * pSize_ * RAlign_;
    if (p_ == 2.0f) {
        Sqrt<float, sqrtConfig>(yFp32_, yFp32_, processNum);
    } else if (p_ != 1.0f && p_ != 0.0f && p_ != static_cast<float>(INFINITY)) {
        Ln<float, lnConfig>(yFp32_, yFp32_, processNum);
        Muls(yFp32_, yFp32_, (float)(1 / p_), processNum);
        Exp<float, expConfig>(yFp32_, yFp32_, processNum);
    }
    yQueue_.EnQue(yFp32_);
    // re-EnQue x1 for next R iteration (x1 is UB-resident, reused across R)
    x1Queue_.EnQue(x1Local);
    x2Queue_.FreeTensor(x2Local);
}

// M-split segment compute: reduce this M-segment into tmpLocal_, then accumulate into yFp32_.
// x1/x2 are re-copied per mIdx by the caller, so they are freed here (unlike Compute where x1 persists).
template <typename T>
__aicore__ inline void CdistReduceHL<T>::ComputeSplitM()
{
    int32_t processNum = bSize_ * pSize_ * RAlign_;
    LocalTensor<float> x1Local = x1Queue_.DeQue<float>();
    LocalTensor<float> x2Local = x2Queue_.DeQue<float>();
    yFp32_ = yQueue_.DeQue<float>();
    tmpLocal_ = tmpQueue_.DeQue<float>();
    ComputeOneSizeHL(x1Local, x2Local, tmpLocal_, /*finalize=*/false);
    if (p_ == static_cast<float>(INFINITY)) {
        Max(yFp32_, tmpLocal_, yFp32_, processNum);
    } else {
        Add(yFp32_, tmpLocal_, yFp32_, processNum);
    }
    yQueue_.EnQue(yFp32_);
    x1Queue_.FreeTensor(x1Local);
    x2Queue_.FreeTensor(x2Local);
    tmpQueue_.EnQue(tmpLocal_);
}

template <typename T>
__aicore__ inline void CdistReduceHL<T>::CalSplitMResult(int32_t processNum)
{
    yFp32_ = yQueue_.DeQue<float>();
    if (p_ == 2.0f) {
        Sqrt<float, sqrtConfig>(yFp32_, yFp32_, processNum);
    }
    if (p_ != 1.0f && p_ != 2.0f && p_ != static_cast<float>(INFINITY) && p_ != 0.0f) {
        Ln<float, lnConfig>(yFp32_, yFp32_, processNum);
        Muls(yFp32_, yFp32_, (float)(1 / p_), processNum);
        Exp<float, expConfig>(yFp32_, yFp32_, processNum);
    }
    yQueue_.EnQue(yFp32_);
}

template <typename T>
__aicore__ inline void CdistReduceHL<T>::ProcessNoSplitM(uint32_t bOffset, uint32_t pOffset, uint32_t rOffsetBlock,
                                                         uint32_t blockFactorR)
{
    uint64_t offsetX1 = 0;
    uint64_t offsetX2 = 0;
    uint64_t offsetY = 0;
    uint64_t rOffset = 0;
    offsetX1 = (uint64_t)bOffset * P_ * M_ + (uint64_t)pOffset * M_;
    // MAlign_ = T-typed block-aligned row stride (elements). Used for BOTH the T-typed compact CopyIn
    // and the fp32 compute plane: Cast(dst,src,count) preserves element layout, so after cast the fp32
    // rows share the same MAlign_ element stride. MAlign_*4 is 32B-aligned (fp16 MAlign multiple of 16,
    // fp32 multiple of 8) → valid WholeReduceSum srcRepStride. Keeps a single consistent stride.
    MAlign_ = ((M_ * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE) / sizeof(T);
    CopyInX1(offsetX1);
    for (uint32_t rIdx = 0; rIdx < ubLoopNumR_; rIdx++) {
        rOffset = rOffsetBlock + rIdx * ubFactorR_;
        rSize_ = (rIdx == ubLoopNumR_ - 1) ? (blockFactorR - ubFactorR_ * rIdx) : ubFactorR_;
        if (rSize_ == 0) {
            continue;
        }
        RAlign_ = ((rSize_ * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE) / sizeof(T);
        offsetX2 = (uint64_t)bOffset * R_ * M_ + (uint64_t)rOffset * M_;
        offsetY = (uint64_t)bOffset * P_ * R_ + (uint64_t)pOffset * R_ + rOffset;
        CopyInX2(offsetX2);
        Compute();
        CopyOut(offsetY);
    }
    // x1 was re-EnQue'd by the last Compute — DeQue and free it so the queue is clean for the
    // next (b,p) tile (otherwise the stale x1 buffer would be DeQue'd first on the next round).
    LocalTensor<float> x1Done = x1Queue_.DeQue<float>();
    x1Queue_.FreeTensor(x1Done);
}

template <typename T>
__aicore__ inline void CdistReduceHL<T>::ProcessSplitM(uint32_t bOffset, uint32_t pOffset, uint32_t rOffsetBlock,
                                                       uint32_t blockFactorR)
{
    uint64_t offsetX1 = 0;
    uint64_t offsetX2 = 0;
    uint64_t offsetY = 0;
    int32_t processNum = 0;
    uint64_t mOffset = 0;
    uint64_t rOffset = 0;
    for (uint32_t rIdx = 0; rIdx < ubLoopNumR_; rIdx++) {
        rOffset = rOffsetBlock + rIdx * ubFactorR_;
        rSize_ = (rIdx == ubLoopNumR_ - 1) ? (blockFactorR - ubFactorR_ * rIdx) : ubFactorR_;
        if (rSize_ == 0) {
            continue;
        }
        processNum = bSize_ * pSize_ * RAlign_;
        RAlign_ = ((rSize_ * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE) / sizeof(T);
        processNum = bSize_ * pSize_ * RAlign_;
        offsetY = (uint64_t)bOffset * P_ * R_ + (uint64_t)pOffset * R_ + rOffset;
        // reset yFp32_ accumulator for this (p, rSeg).
        yFp32_ = yQueue_.template AllocTensor<float>();
        Duplicate<float>(yFp32_, (float)0, processNum);
        yQueue_.EnQue(yFp32_);
        for (uint32_t mIdx = 0; mIdx < ubLoopNumM_; mIdx++) {
            mOffset = mIdx * ubFactorM_;
            mSize_ = (mIdx == ubLoopNumM_ - 1) ? ubTailFactorM_ : ubFactorM_;
            MAlign_ = ((mSize_ * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE) / sizeof(T);
            offsetX1 = (uint64_t)bOffset * P_ * M_ + (uint64_t)pOffset * M_ + mOffset;
            offsetX2 = (uint64_t)bOffset * R_ * M_ + (uint64_t)rOffset * M_ + mOffset;
            CopyInX1(offsetX1);
            CopyInX2(offsetX2);
            ComputeSplitM();
        }
        CalSplitMResult(processNum);
        PipeBarrier<PIPE_ALL>();
        CopyOut(offsetY);
    }
}

template <typename T>
__aicore__ inline void CdistReduceHL<T>::Process()
{
    uint32_t bOffset = 0;
    uint32_t pOffset = 0;
    uint32_t bOffsetBlock = 0;
    uint32_t pOffsetBlock = 0;
    uint32_t rOffsetBlock = 0;
    uint32_t blockFactorB = 0;
    uint32_t blockFactorP = 0;
    uint32_t blockFactorR = 0;
    uint32_t blockNumP = blockMainNumP_ + blockTailNumP_;
    uint32_t blockNumR = blockMainNumR_ + blockTailNumR_;
    uint32_t bBlockIdx = blockIdx_ / (blockNumP * blockNumR);
    uint32_t prBlockIdx = blockIdx_ % (blockNumP * blockNumR);
    uint32_t pBlockIdx = prBlockIdx / blockNumR;
    uint32_t rBlockIdx = prBlockIdx % blockNumR;
    blockFactorB = (bBlockIdx < blockMainNumB_) ? blockMainFactorB_ : blockTailFactorB_;
    bOffsetBlock = (bBlockIdx < blockMainNumB_) ?
                       blockMainFactorB_ * bBlockIdx :
                       blockMainFactorB_ * blockMainNumB_ + (bBlockIdx - blockMainNumB_) * blockTailFactorB_;
    blockFactorR = (rBlockIdx < blockMainNumR_) ? blockMainFactorR_ : blockTailFactorR_;
    rOffsetBlock = (rBlockIdx < blockMainNumR_) ?
                       blockMainFactorR_ * rBlockIdx :
                       blockMainFactorR_ * blockMainNumR_ + (rBlockIdx - blockMainNumR_) * blockTailFactorR_;
    blockFactorP = (pBlockIdx < blockMainNumP_) ? blockMainFactorP_ : blockTailFactorP_;
    pOffsetBlock = (pBlockIdx < blockMainNumP_) ?
                       blockMainFactorP_ * pBlockIdx :
                       blockMainFactorP_ * blockMainNumP_ + (pBlockIdx - blockMainNumP_) * blockTailFactorP_;
    for (uint32_t bIdx = 0; bIdx < ubLoopNumB_; bIdx++) {
        bOffset = bOffsetBlock + bIdx * ubFactorB_;
        bSize_ = (bIdx == ubLoopNumB_ - 1) ? (blockFactorB - ubFactorB_ * bIdx) : ubFactorB_;
        if (bSize_ == 0) {
            continue;
        }
        for (uint32_t pIdx = 0; pIdx < ubLoopNumP_; pIdx++) {
            pOffset = pOffsetBlock + pIdx * ubFactorP_;
            pSize_ = (pIdx == ubLoopNumP_ - 1) ? (blockFactorP - ubFactorP_ * pIdx) : ubFactorP_;
            if (pSize_ == 0) {
                continue;
            }
            if (ubLoopNumM_ == 1) {
                ProcessNoSplitM(bOffset, pOffset, rOffsetBlock, blockFactorR);
            } else {
                ProcessSplitM(bOffset, pOffset, rOffsetBlock, blockFactorR);
            }
        }
    }
    tmpQueue_.FreeTensor(tmpLocal_);
}

} // namespace NsCdist

#endif // !defined(__NPU_HOST__)

#endif // CDIST_REDUCE_HL_H
