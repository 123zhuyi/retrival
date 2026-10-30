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
 * \file as_strided_gather.h
 * \brief impl of AsStrided  gather
 */

#ifndef OP_KERNEL_AS_STRIDED_GATHER
#define OP_KERNEL_AS_STRIDED_GATHER

#include "op_kernel/platform_util.h"
#include "as_strided.h"

namespace AsStrided {
using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::UpdateMask;
constexpr uint16_t VF_LENGTH = Ops::Base::GetVRegSize();

template <typename T>
class KernelAsStridedGather {
public:
    __aicore__ inline KernelAsStridedGather(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR outShape, GM_ADDR outStride, GM_ADDR output,
                                const AsStridedWithGatherTilingData* tilingDataPtr);
    template <typename K>
    __aicore__ inline void CopyArray(const K* src, K* dst, int64_t size);
    __aicore__ inline uint16_t CeilDiv(uint32_t a, uint16_t b);
    __aicore__ inline int64_t GetMod(int64_t a, uint32_t b);
    __aicore__ inline void Process();
    __aicore__ inline void AsCopyGM2Ub();
    __aicore__ inline void AsComputeIdx();
    __aicore__ inline void GenDim1Index();
    __aicore__ inline void GenDim2Index();
    __aicore__ inline void GenDim3Index();
    __aicore__ inline void ComputeIdxOffset(int64_t idxInTotal, int64_t& idxOffset);
    __aicore__ inline void ComputeSameAxisOffset(int64_t idxInTotal, int64_t& idxOffset, int64_t& yGMOffset);
    __aicore__ inline void ComputeCoreOffset();
    __aicore__ inline void AsGather(int64_t idxOffset);
    __aicore__ inline void ComputeGMOffset(int64_t idxInTotal, int64_t& yGMOffset);
    __aicore__ inline void AsCopyOut(int64_t yGMOffset);

private:
    using RangeType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
    using IdxType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using CastType_ = std::conditional_t<sizeof(T) == 1,
                                         std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;

    TPipe pipe_;
    const AsStridedWithGatherTilingData* tdPtr_ = nullptr;
    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;
    int64_t blockIdx_ = 0;
    int64_t storageOffset_ = 0;
    int64_t blockNum_ = 0;
    int64_t bufferCnt_ = 2; // enable db

    uint32_t ubBatchArr_[TILING_ARRAY_LEN];
    uint32_t strideArr_[TILING_ARRAY_LEN];
    uint32_t sizeArr_[TILING_ARRAY_LEN];

    uint32_t realInUbSize_ = 0;
    uint32_t realInDataLen_ = 0;
    uint32_t dimNum_ = 10;
    uint32_t tilingAxisIdx_ = 9;
    uint32_t blockInnerAxisFactor_ = 0;
    uint32_t blockTailInnerAxisFactor_ = 0;
    uint32_t blockUbFactor_ = 0;
    uint32_t blockUbFactorTail_ = 0;
    uint32_t curUbFactor_ = 0;
    uint32_t blockTailUbFactor_ = 0;
    uint32_t blockTailUbFactorTail_ = 0;
    uint32_t blockFactor_ = 0;
    uint32_t blockMainCount_ = 0;
    uint32_t blockTailFactor_ = 0;

    uint32_t blockAxisIdx_ = 9;
    uint32_t blockOuterAxisFactor_ = 0;
    uint32_t blockTailOuterAxisFactor_ = 0;
    uint32_t coreOuterAxisFactor_ = 0;
    uint32_t coreInnerAxisFactor_ = 0;
    uint32_t coreInnerAxisTailFactor_ = 0;
    uint32_t preSize_ = 0;

    int64_t loopsUb_ = 0;
    int64_t coreStart_ = 0;

    LocalTensor<IdxType_> indexLocal_;
    LocalTensor<T> ubFactorLocal_;
    LocalTensor<T> yUbFactorLocal_;
};

template <typename T>
__aicore__ inline void KernelAsStridedGather<T>::Init(GM_ADDR input, GM_ADDR outShape, GM_ADDR outStride,
                                                      GM_ADDR output,
                                                      const AsStridedWithGatherTilingData* tilingDataPtr)
{
    blockIdx_ = GetBlockIdx();
    tdPtr_ = tilingDataPtr;
    storageOffset_ = tdPtr_->storageOffset;
    blockNum_ = tdPtr_->blockNum;

    CopyArray(tdPtr_->strideArr, strideArr_, TILING_ARRAY_LEN);
    CopyArray(tdPtr_->idxStrideArr, ubBatchArr_, TILING_ARRAY_LEN);
    CopyArray(tdPtr_->sizeArr, sizeArr_, TILING_ARRAY_LEN);
    realInUbSize_ = tdPtr_->inUbSize;
    realInDataLen_ = tdPtr_->inDataLen;
    dimNum_ = tdPtr_->outDimNum;
    tilingAxisIdx_ = tdPtr_->tilingAxisIdx;
    blockInnerAxisFactor_ = tdPtr_->mainBlockUbParam.innerAxisFactor;
    blockTailInnerAxisFactor_ = tdPtr_->tailBlockUbParam.innerAxisFactor;
    blockUbFactor_ = tdPtr_->mainBlockUbParam.ubFactor;
    blockUbFactorTail_ = tdPtr_->mainBlockUbParam.ubFactorTail;
    blockTailUbFactor_ = tdPtr_->tailBlockUbParam.ubFactor;
    blockTailUbFactorTail_ = tdPtr_->tailBlockUbParam.ubFactorTail;
    blockFactor_ = tdPtr_->mainBlockUbParam.loopsPerCore;
    blockTailFactor_ = tdPtr_->tailBlockUbParam.loopsPerCore;
    blockAxisIdx_ = tdPtr_->blockAxisIdx;
    blockOuterAxisFactor_ = tdPtr_->mainBlockUbParam.outerAxisFactor;
    blockTailOuterAxisFactor_ = tdPtr_->tailBlockUbParam.outerAxisFactor;
    coreOuterAxisFactor_ = tdPtr_->coreOuterAxisFactor;
    coreInnerAxisFactor_ = tdPtr_->coreInnerAxisFactor;
    coreInnerAxisTailFactor_ = tdPtr_->coreInnerAxisTailFactor;
    preSize_ = tdPtr_->preSize;

    blockMainCount_ = blockNum_ - 1;
    curUbFactor_ = blockUbFactor_;

    inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input) + storageOffset_);
    outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));

    pipe_.InitBuffer(inQue_, 1, realInUbSize_);
    pipe_.InitBuffer(outQue_, bufferCnt_, blockUbFactor_ * sizeof(T));
    pipe_.InitBuffer(indexBuf_, blockUbFactor_ * sizeof(IdxType_));

    indexLocal_ = indexBuf_.Get<IdxType_>();
    ubFactorLocal_ = inQue_.AllocTensor<T>();
}

template <typename T>
__aicore__ inline void KernelAsStridedGather<T>::ComputeCoreOffset()
{
    loopsUb_ = blockFactor_;
    if (blockAxisIdx_ == tilingAxisIdx_) { // 切同轴
        if (tilingAxisIdx_ != 0) {         // 不切在首轴
            uint32_t mulsFactor = blockIdx_ / coreOuterAxisFactor_;
            int64_t modFactor = 0;
            if (coreOuterAxisFactor_ != 1) { // 核外轴不为1
                modFactor = GetMod(blockIdx_, coreOuterAxisFactor_);
            }
            coreStart_ = mulsFactor * ubBatchArr_[tilingAxisIdx_] + modFactor * blockOuterAxisFactor_;
            int64_t isTailUb = GetMod(blockIdx_ + 1, coreOuterAxisFactor_);
            loopsUb_ = isTailUb == 0 ? blockTailOuterAxisFactor_ : blockOuterAxisFactor_;
        } else { // 切在首轴
            if (blockIdx_ > blockMainCount_ - 1) {
                coreStart_ = blockMainCount_ * blockFactor_;
                loopsUb_ = blockTailFactor_;
            } else {
                coreStart_ = blockIdx_ * blockFactor_;
            }
        }
    } else {
        if (blockIdx_ > blockMainCount_ - 1) {
            coreStart_ = blockMainCount_ * blockFactor_ + (blockIdx_ - blockMainCount_) * blockTailFactor_;
            loopsUb_ = blockTailFactor_;
        } else {
            coreStart_ = blockIdx_ * blockFactor_;
        }
    }
}

template <typename T>
__aicore__ inline void KernelAsStridedGather<T>::Process()
{
    ComputeCoreOffset();
    AsCopyGM2Ub();
    AsComputeIdx();
    ubFactorLocal_ = inQue_.DeQue<T>();
    for (int64_t idx_ub = 0; idx_ub < loopsUb_; ++idx_ub) {
        yUbFactorLocal_ = outQue_.AllocTensor<T>();
        int64_t idxInTotal = coreStart_ + idx_ub; // 全局ub索引
        int64_t yGMOffset = 0;
        int64_t idxOffset = 0;
        if ((blockAxisIdx_ == tilingAxisIdx_) && (tilingAxisIdx_ == 0)) {
            ComputeSameAxisOffset(idxInTotal, idxOffset, yGMOffset);
        } else {
            ComputeIdxOffset(idxInTotal, idxOffset);
            ComputeGMOffset(idxInTotal, yGMOffset);
        }
        AsGather(idxOffset);
        AsCopyOut(yGMOffset);
        outQue_.FreeTensor(yUbFactorLocal_);
    }
    inQue_.FreeTensor(ubFactorLocal_);
}

template <typename T>
__aicore__ inline void KernelAsStridedGather<T>::ComputeIdxOffset(int64_t idxInTotal, int64_t& idxOffset)
{
    // 从左向右，以首ub为基准，找到当前UB对“首UB在input上映射元素位置”的相对偏移量
    int64_t baseOffset = blockInnerAxisFactor_ * strideArr_[tilingAxisIdx_];
    for (int64_t idx_axis = 0; idx_axis < tilingAxisIdx_; ++idx_axis) {
        int64_t curMuls = idxInTotal / ubBatchArr_[idx_axis];
        int64_t nextMuls = idxInTotal / ubBatchArr_[idx_axis + 1];
        if (curMuls == 0 && nextMuls > 0) {
            idxOffset = idxOffset + nextMuls * strideArr_[idx_axis];
            idxInTotal = idxInTotal - nextMuls * ubBatchArr_[idx_axis + 1];
        }
    }
    if (ubBatchArr_[tilingAxisIdx_] != 1) {
        idxOffset = idxOffset + idxInTotal * baseOffset;
    }
    int64_t isTailUb = GetMod(idxInTotal + 1, ubBatchArr_[tilingAxisIdx_]);
    if (isTailUb == 0) {
        curUbFactor_ = blockTailUbFactorTail_;
    } else {
        curUbFactor_ = blockUbFactor_;
    }
}

template <typename T>
__aicore__ inline void KernelAsStridedGather<T>::ComputeSameAxisOffset(int64_t idxInTotal, int64_t& idxOffset,
                                                                       int64_t& yGMOffset)
{
    int64_t mulsFactor = 0;
    int64_t modFactor = 0;

    if (blockIdx_ > blockMainCount_ - 1) {
        idxInTotal = idxInTotal - blockMainCount_ * blockFactor_;
        modFactor = GetMod(idxInTotal, blockTailFactor_);
        idxOffset = blockMainCount_ * coreInnerAxisFactor_ + modFactor * blockTailInnerAxisFactor_;
        idxOffset = idxOffset * strideArr_[tilingAxisIdx_];

        int64_t isTailUb = GetMod(idxInTotal + 1, blockTailFactor_);
        if (isTailUb == 0) {
            curUbFactor_ = blockTailUbFactorTail_;
        } else {
            curUbFactor_ = blockTailUbFactor_;
        }

        yGMOffset = blockMainCount_ * coreInnerAxisFactor_ * preSize_ +
                    modFactor * blockTailInnerAxisFactor_ * preSize_;
    } else {
        mulsFactor = idxInTotal / blockFactor_;
        modFactor = GetMod(idxInTotal, blockFactor_);
        idxOffset = mulsFactor * coreInnerAxisFactor_ + modFactor * blockInnerAxisFactor_;
        idxOffset = idxOffset * strideArr_[tilingAxisIdx_];

        int64_t isTailUb = GetMod(idxInTotal + 1, blockFactor_);
        if (isTailUb == 0) {
            curUbFactor_ = blockUbFactorTail_;
        } else {
            curUbFactor_ = blockUbFactor_;
        }

        yGMOffset = mulsFactor * coreInnerAxisFactor_ * preSize_ + modFactor * blockInnerAxisFactor_ * preSize_;
    }
}

template <typename T>
__aicore__ inline void KernelAsStridedGather<T>::ComputeGMOffset(int64_t idxInTotal, int64_t& yGMOffset)
{
    int64_t tilingAxisEle = (ubBatchArr_[tilingAxisIdx_] - 1) * blockUbFactor_ + blockTailUbFactorTail_;
    int64_t curMuls = idxInTotal / ubBatchArr_[tilingAxisIdx_];
    int64_t curMods = GetMod(idxInTotal, ubBatchArr_[tilingAxisIdx_]);
    yGMOffset = curMuls * tilingAxisEle + curMods * blockUbFactor_;
}

template <typename T>
__aicore__ inline void KernelAsStridedGather<T>::AsCopyGM2Ub()
{
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = realInDataLen_ * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopyPadExtParams<T> copyInPadParams{false, 0, 0, 0};
    DataCopyPad(ubFactorLocal_, inputGM_, copyInParams, copyInPadParams);
    inQue_.EnQue<T>(ubFactorLocal_);
}

template <typename T>
__aicore__ inline void KernelAsStridedGather<T>::AsGather(int64_t idxOffset)
{
    // index
    __local_mem__ IdxType_* indexAddr = (__local_mem__ IdxType_*)indexLocal_.GetPhyAddr();
    // xUb
    __local_mem__ T* xUbAddr = (__local_mem__ T*)ubFactorLocal_.GetPhyAddr();
    // yUb
    __local_mem__ T* yUbAddr = (__local_mem__ T*)yUbFactorLocal_.GetPhyAddr();

    if constexpr (sizeof(T) == 1) {
        uint16_t vfLen = VF_LENGTH / sizeof(CastType_);
        uint16_t loopsCnt = CeilDiv(curUbFactor_, vfLen);
        uint32_t xUbMain = curUbFactor_ > vfLen ? vfLen : curUbFactor_;
        uint32_t xUbMain_copy = xUbMain;
        uint16_t mainLoopsCnt = (loopsCnt - 1) == 0 ? 1 : (loopsCnt - 1);
        uint16_t mainLoopsCnt2 = (loopsCnt - 1) == 0 ? 0 : (loopsCnt - 1);
        uint32_t xUbTail = curUbFactor_ > vfLen ? (curUbFactor_ - (loopsCnt - 1) * xUbMain) : 0;
        __VEC_SCOPE__
        {
            Reg::MaskReg p0 = AscendC::Reg::UpdateMask<IdxType_>(xUbMain);
            Reg::MaskReg p1 = AscendC::Reg::UpdateMask<IdxType_>(xUbTail);
            Reg::RegTensor<IdxType_> indexReg;
            Reg::RegTensor<IdxType_> vd0;
            Reg::RegTensor<T> dstReg;
            for (uint16_t idx_reg = 0; idx_reg < mainLoopsCnt; ++idx_reg) {
                Reg::DataCopy(indexReg, indexAddr + idx_reg * xUbMain_copy);
                Reg::Adds(vd0, indexReg, idxOffset, p0);
                Reg::DataCopyGather((Reg::RegTensor<CastType_>&)dstReg, xUbAddr, vd0, p0);
                __local_mem__ CastType_* yUbAddrB16 = reinterpret_cast<__local_mem__ CastType_*>(
                    yUbAddr + idx_reg * xUbMain_copy);
                Reg::DataCopy<CastType_, Reg::StoreDist::DIST_PACK_B16>(yUbAddrB16, (Reg::RegTensor<CastType_>&)dstReg,
                                                                        p0);
            }
            Reg::DataCopy(indexReg, indexAddr + mainLoopsCnt2 * xUbMain_copy);
            Reg::Adds(vd0, indexReg, idxOffset, p1);
            Reg::DataCopyGather((Reg::RegTensor<CastType_>&)dstReg, xUbAddr, vd0, p1);
            __local_mem__ CastType_* yUbAddrB16 = reinterpret_cast<__local_mem__ CastType_*>(
                yUbAddr + mainLoopsCnt2 * xUbMain_copy);
            Reg::DataCopy<CastType_, Reg::StoreDist::DIST_PACK_B16>(yUbAddrB16, (Reg::RegTensor<CastType_>&)dstReg, p1);
        }
    } else {
        uint16_t vfLen = VF_LENGTH / sizeof(CastType_);
        uint32_t vfLen_idxType = VF_LENGTH / sizeof(IdxType_);
        uint16_t loopsCnt = CeilDiv(curUbFactor_, vfLen);
        uint32_t xUbMain = curUbFactor_ > vfLen ? vfLen : curUbFactor_;
        uint32_t xUbMain_copy = xUbMain;
        uint16_t mainLoopsCnt = (loopsCnt - 1) == 0 ? 1 : (loopsCnt - 1);
        uint16_t mainLoopsCnt2 = (loopsCnt - 1) == 0 ? 0 : (loopsCnt - 1);
        uint32_t xUbTail = curUbFactor_ > vfLen ? (curUbFactor_ - (loopsCnt - 1) * xUbMain) : 0;
        __VEC_SCOPE__
        {
            Reg::MaskReg p0 = AscendC::Reg::UpdateMask<CastType_>(xUbMain);
            Reg::MaskReg p2 = AscendC::Reg::UpdateMask<IdxType_>(vfLen_idxType);
            Reg::MaskReg p1 = AscendC::Reg::UpdateMask<CastType_>(xUbTail);
            Reg::RegTensor<IdxType_> indexReg;
            Reg::RegTensor<IdxType_> vd0;
            Reg::RegTensor<T> dstReg;
            Reg::AddrReg aReg0;
            for (uint16_t idx_reg = 0; idx_reg < mainLoopsCnt; ++idx_reg) {
                Reg::DataCopy(indexReg, indexAddr + idx_reg * xUbMain_copy);
                Reg::Adds(vd0, indexReg, idxOffset, p2);
                Reg::DataCopyGather(dstReg, xUbAddr, vd0, p0);
                Reg::DataCopy(yUbAddr + idx_reg * xUbMain_copy, dstReg, p0);
            }
            Reg::DataCopy(indexReg, indexAddr + mainLoopsCnt2 * xUbMain_copy);
            Reg::Adds(vd0, indexReg, idxOffset, p2);
            Reg::DataCopyGather(dstReg, xUbAddr, vd0, p1);
            Reg::DataCopy(yUbAddr + mainLoopsCnt2 * xUbMain_copy, dstReg, p1);
        }
    }
    outQue_.EnQue<T>(yUbFactorLocal_);
}

template <typename T>
__aicore__ inline void KernelAsStridedGather<T>::AsCopyOut(int64_t yGMOffset)
{
    yUbFactorLocal_ = outQue_.DeQue<T>();
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = curUbFactor_ * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopyPad(outputGM_[yGMOffset], yUbFactorLocal_, copyInParams);
}

template <typename T>
__aicore__ inline void KernelAsStridedGather<T>::AsComputeIdx()
{
    // 计算index
    uint32_t dim = dimNum_ - tilingAxisIdx_;
    if (dim == 1) {
        GenDim1Index();
    } else if (dim == 2) {
        GenDim2Index();
    } else {
        GenDim3Index();
    }
}

template <typename T>
__aicore__ inline void KernelAsStridedGather<T>::GenDim3Index()
{
    __local_mem__ IdxType_* idxAddr = (__local_mem__ IdxType_*)indexLocal_.GetPhyAddr();
    uint16_t vfLen = VF_LENGTH / sizeof(IdxType_);

    uint32_t lastDimInOffset = strideArr_[dimNum_ - 1];
    uint32_t last2ndDimInOffset = strideArr_[dimNum_ - 2];
    uint32_t last3rdDimInOffset = strideArr_[dimNum_ - 3];
    uint32_t lastDimSize = sizeArr_[dimNum_ - 1];
    uint32_t last2ndDimSize = sizeArr_[dimNum_ - 2];
    uint32_t lastDimInc = vfLen % lastDimSize;
    uint32_t last2ndDimInc = vfLen / lastDimSize % last2ndDimSize;
    uint32_t last3rdDimInc = vfLen / (lastDimSize * last2ndDimSize);

    uint32_t leftDimSize = 0;
    uint16_t loopsCnt = 0;
    if (blockUbFactor_ > vfLen) {
        leftDimSize = blockUbFactor_ - vfLen;
        loopsCnt = CeilDiv(leftDimSize, vfLen);
    }

    __VEC_SCOPE__
    {
        RegTensor<RangeType_> tmp;
        RegTensor<IdxType_> idxReg;
        RegTensor<IdxType_> dim0Reg;
        RegTensor<IdxType_> tmpReg;
        RegTensor<IdxType_> dim1Reg;
        RegTensor<IdxType_> tmp1Reg;
        RegTensor<IdxType_> dim2Reg;
        RegTensor<IdxType_> dstReg;
        MaskReg mask = CreateMask<RangeType_, Reg::MaskPattern::ALL>();
        // vec_a: VL % a
        Reg::Arange(tmp, 0);
        idxReg = (RegTensor<IdxType_>&)tmp;
        Reg::Duplicate(dim0Reg, lastDimSize);
        Reg::Copy(dim2Reg, dim0Reg); // backup a
        Reg::Div(tmpReg, idxReg, dim0Reg, mask);
        Reg::Copy(dim1Reg, tmpReg); // backup VL / a
        Reg::Mul(tmpReg, tmpReg, dim0Reg, mask);
        Reg::Sub(dim0Reg, idxReg, tmpReg, mask);
        // vec_b: VL / a % b
        Reg::Duplicate(tmp1Reg, last2ndDimSize);
        Reg::Mul(dim2Reg, dim2Reg, tmp1Reg, mask); // backup b
        Reg::Div(tmpReg, dim1Reg, tmp1Reg, mask);
        Reg::Mul(tmpReg, tmpReg, tmp1Reg, mask);
        Reg::Sub(dim1Reg, dim1Reg, tmpReg, mask);
        // vec_c: VL / (a * b)
        Reg::Div(dim2Reg, idxReg, dim2Reg, mask);
        // index: vec_a * a_in_offset + vec_b * b_in_offset + vec_c * c_in_offset
        Reg::Muls(tmpReg, dim0Reg, lastDimInOffset, mask);
        Reg::Muls(tmp1Reg, dim1Reg, last2ndDimInOffset, mask);
        Reg::Muls(dstReg, dim2Reg, last3rdDimInOffset, mask);
        Reg::Add(dstReg, dstReg, tmpReg, mask);
        Reg::Add(dstReg, dstReg, tmp1Reg, mask);
        Reg::DataCopy(idxAddr, dstReg, mask);

        MaskReg lpMask;
        MaskReg selMask;
        RegTensor<IdxType_> zeroReg;
        RegTensor<IdxType_> oneReg;
        RegTensor<IdxType_> cmpReg;
        Reg::Duplicate(zeroReg, 0);
        Reg::Duplicate(oneReg, 1);
        for (uint16_t lpIdx = 0; lpIdx < loopsCnt; ++lpIdx) {
            lpMask = UpdateMask<IdxType_>(leftDimSize);
            /*   vec_a += a_inc
             *   cmp_a = vec_a >= a
             *   vec_a = vec_a - cmp_a * a
             */
            Reg::Adds(dim0Reg, dim0Reg, lastDimInc, lpMask);
            Reg::CompareScalar<IdxType_, CMPMODE::GE>(selMask, dim0Reg, lastDimSize, lpMask);
            Reg::Select(cmpReg, oneReg, zeroReg, selMask);
            Reg::Muls(tmpReg, cmpReg, lastDimSize, lpMask);
            Reg::Sub(dim0Reg, dim0Reg, tmpReg, lpMask);
            /*   vec_b += (b_inc + cmp_a)
             *   cmp_b = vec_b >= b
             *   vec_b = vec_b - cmp_b * b
             */
            Reg::Adds(cmpReg, cmpReg, last2ndDimInc, lpMask);
            Reg::Add(dim1Reg, dim1Reg, cmpReg, lpMask);
            Reg::CompareScalar<IdxType_, CMPMODE::GE>(selMask, dim1Reg, last2ndDimSize, lpMask);
            Reg::Select(cmpReg, oneReg, zeroReg, selMask);
            Reg::Muls(tmpReg, cmpReg, last2ndDimSize, lpMask);
            Reg::Sub(dim1Reg, dim1Reg, tmpReg, lpMask);
            // vec_c += (c_inc + cmp_b)
            Reg::Adds(dim2Reg, dim2Reg, last3rdDimInc, lpMask);
            Reg::Add(dim2Reg, dim2Reg, cmpReg, lpMask);
            // index: vec_a * a_in_offset + vec_b * b_in_offset + vec_c * c_in_offset
            Reg::Muls(tmpReg, dim0Reg, lastDimInOffset, lpMask);
            Reg::Muls(tmp1Reg, dim1Reg, last2ndDimInOffset, lpMask);
            Reg::Muls(dstReg, dim2Reg, last3rdDimInOffset, lpMask);
            Reg::Add(dstReg, dstReg, tmpReg, lpMask);
            Reg::Add(dstReg, dstReg, tmp1Reg, lpMask);
            Reg::DataCopy(idxAddr + (lpIdx + 1) * vfLen, dstReg, lpMask);
        }
    }
}

template <typename T>
__aicore__ inline void KernelAsStridedGather<T>::GenDim2Index()
{
    __local_mem__ IdxType_* idxAddr = (__local_mem__ IdxType_*)indexLocal_.GetPhyAddr();
    uint16_t vfLen = VF_LENGTH / sizeof(IdxType_);
    uint32_t lastDimInOffset = strideArr_[dimNum_ - 1];
    uint32_t last2ndDimInOffset = strideArr_[dimNum_ - 2];
    uint32_t lastDimSize = sizeArr_[dimNum_ - 1];

    uint32_t lastDimInc = vfLen % lastDimSize;
    uint32_t last2ndDimInc = vfLen / lastDimSize;

    uint32_t leftDimSize = 0;
    uint16_t loopsCnt = 0;
    if (blockUbFactor_ > vfLen) {
        leftDimSize = blockUbFactor_ - vfLen;
        loopsCnt = CeilDiv(leftDimSize, vfLen);
    }

    __VEC_SCOPE__
    {
        RegTensor<RangeType_> tmp;
        RegTensor<IdxType_> idxReg;
        RegTensor<IdxType_> dim0Reg;
        RegTensor<IdxType_> tmpReg;
        RegTensor<IdxType_> dim1Reg;
        RegTensor<IdxType_> dstReg;
        MaskReg mask = CreateMask<IdxType_, Reg::MaskPattern::ALL>();
        // vec_a: VL % a
        Reg::Arange(tmp, 0);
        idxReg = (RegTensor<IdxType_>&)tmp;
        Reg::Duplicate(dim0Reg, lastDimSize);
        Reg::Div(tmpReg, idxReg, dim0Reg, mask);
        Reg::Copy(dim1Reg, tmpReg); // vec_b: VL / a
        Reg::Mul(tmpReg, tmpReg, dim0Reg, mask);
        Reg::Sub(dim0Reg, idxReg, tmpReg, mask);
        // index: vec_a * a_in_offset + vec_b * b_in_offset
        Reg::Muls(tmpReg, dim0Reg, lastDimInOffset, mask);
        Reg::Muls(dstReg, dim1Reg, last2ndDimInOffset, mask);
        Reg::Add(dstReg, dstReg, tmpReg, mask);
        Reg::DataCopy(idxAddr, dstReg, mask);

        MaskReg lpMask;
        MaskReg selMask;
        RegTensor<IdxType_> zeroReg;
        RegTensor<IdxType_> oneReg;
        RegTensor<IdxType_> cmpReg;
        Reg::Duplicate(zeroReg, 0);
        Reg::Duplicate(oneReg, 1);
        for (uint16_t lpIdx = 0; lpIdx < loopsCnt; ++lpIdx) {
            lpMask = UpdateMask<IdxType_>(leftDimSize);
            /*   vec_a += a_inc
             *   cmp_a = vec_a >= a
             *   vec_a = vec_a - cmp_a * a
             */
            Reg::Adds(dim0Reg, dim0Reg, lastDimInc, lpMask);
            Reg::CompareScalar<IdxType_, CMPMODE::GE>(selMask, dim0Reg, lastDimSize, lpMask);
            Reg::Select(cmpReg, oneReg, zeroReg, selMask);
            Reg::Muls(tmpReg, cmpReg, lastDimSize, lpMask);
            Reg::Sub(dim0Reg, dim0Reg, tmpReg, lpMask);
            // vec_b += (b_inc + cmp_a)
            Reg::Adds(dim1Reg, dim1Reg, last2ndDimInc, lpMask);
            Reg::Add(dim1Reg, dim1Reg, cmpReg, lpMask);
            // index: vec_a * a_in_offset + vec_b * b_in_offset
            Reg::Muls(tmpReg, dim0Reg, lastDimInOffset, lpMask);
            Reg::Muls(dstReg, dim1Reg, last2ndDimInOffset, lpMask);
            Reg::Add(dstReg, dstReg, tmpReg, lpMask);
            Reg::DataCopy(idxAddr + (lpIdx + 1) * vfLen, dstReg, lpMask);
        }
    }
}

template <typename T>
__aicore__ inline void KernelAsStridedGather<T>::GenDim1Index()
{
    __local_mem__ IdxType_* idxAddr = (__local_mem__ IdxType_*)indexLocal_.GetPhyAddr();
    uint16_t vfLen = VF_LENGTH / sizeof(IdxType_);
    uint32_t lastDimInOffset = strideArr_[dimNum_ - 1];
    uint16_t loopsCnt = CeilDiv(blockUbFactor_, vfLen);
    uint32_t tempBUF = blockUbFactor_;
    __VEC_SCOPE__
    {
        RegTensor<RangeType_> tmp;
        RegTensor<IdxType_> srcReg;
        RegTensor<IdxType_> dstReg;
        MaskReg mask;
        Reg::Arange(tmp, 0);
        srcReg = (RegTensor<IdxType_>&)tmp;
        for (uint16_t lpIdx = 0; lpIdx < loopsCnt; ++lpIdx) {
            mask = UpdateMask<IdxType_>(tempBUF);
            Reg::Muls(dstReg, srcReg, lastDimInOffset, mask);
            Reg::DataCopy(idxAddr + lpIdx * vfLen, dstReg, mask);
            Reg::Adds(srcReg, srcReg, vfLen, mask);
        }
    }
}

template <typename T>
template <typename K>
__aicore__ inline void KernelAsStridedGather<T>::CopyArray(const K* src, K* dst, int64_t size)
{
    for (int64_t i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}

template <typename T>
__aicore__ inline uint16_t KernelAsStridedGather<T>::CeilDiv(uint32_t a, uint16_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
};

template <typename T>
__aicore__ inline int64_t KernelAsStridedGather<T>::GetMod(int64_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return a - b * (a / b);
};
} // namespace AsStrided
#endif // OP_KERNEL_AS_STRIDED_GATHER
