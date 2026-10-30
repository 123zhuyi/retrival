/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file split_medium_shape.h
 * \brief
 */
#ifndef SPLIT_MEDIUM_SHAPE
#define SPLIT_MEDIUM_SHAPE

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"

namespace Triangulator {
struct MediumBlockInfo {
    int64_t highActual;
    int64_t gmOffset;
};
using namespace AscendC;
using namespace Ops::Base;

constexpr uint32_t MEDIUM_BUFFER_NUM = 2;

template <typename T, const bool IS_LOWER>
class SplitMediumShape {
public:
    __aicore__ inline SplitMediumShape(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, __tiling_data_ptr__ TriangulatorMediumTilingData* tilingDataPtr,
                                TPipe* pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(const MediumBlockInfo& info);
    __aicore__ inline void CopyOut(const MediumBlockInfo& info);
    __aicore__ inline void DoVectorCopy(const MediumBlockInfo& info);
    __aicore__ inline void ComputeBlockInfo(int64_t offset, MediumBlockInfo& info);

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, MEDIUM_BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, MEDIUM_BUFFER_NUM> outQueue_;
    GlobalTensor<T> dstGm_;
    GlobalTensor<T> srcGm_;

    int64_t usedCoreNum_;
    int64_t normalCoreProcessNum_;
    int64_t tailCoreProcessNum_;
    int64_t highOuter_;
    int64_t highInner_;
    int64_t highTail_;
    int64_t col_;
    int64_t row_;
    int64_t planeArea_;
    int64_t copyCols_;
    int64_t headRows_;
    int64_t midRows_;
    int64_t tailRows_;
    uint32_t blockIdx_;
    uint32_t regCount_;
    DataCopyExtParams copyInParam_;
    DataCopyExtParams copyOutParam_;
};

template <typename T, const bool IS_LOWER>
__aicore__ inline void SplitMediumShape<T, IS_LOWER>::Init(
    GM_ADDR x, GM_ADDR y, __tiling_data_ptr__ TriangulatorMediumTilingData* tilingDataPtr, TPipe* pipeIn)
{
    pipe_ = pipeIn;

    // tiling_data
    blockIdx_ = GetBlockIdx();
    usedCoreNum_ = tilingDataPtr->usedCoreNum;
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    normalCoreProcessNum_ = tilingDataPtr->normalCoreProcessNum;
    tailCoreProcessNum_ = tilingDataPtr->tailCoreProcessNum;
    highInner_ = tilingDataPtr->highInner;
    highOuter_ = tilingDataPtr->highOuter;
    highTail_ = tilingDataPtr->highTail;
    col_ = tilingDataPtr->col;
    row_ = tilingDataPtr->row;
    planeArea_ = row_ * col_ * highInner_;
    copyCols_ = tilingDataPtr->copyCols;
    headRows_ = tilingDataPtr->headRows;
    midRows_ = tilingDataPtr->midRows;
    tailRows_ = tilingDataPtr->tailRows;
    constexpr uint32_t dataCount = GetVRegSize() / sizeof(T);
    constexpr uint32_t blockCount = GetUbBlockSize() / sizeof(T);
    uint32_t alignCount = col_ <= dataCount ? blockCount : dataCount;
    regCount_ = col_ <= dataCount ? CeilAlign(static_cast<uint32_t>(col_), blockCount) : dataCount;
    copyInParam_.dstStride = CeilAlign(static_cast<uint32_t>(col_), alignCount) * sizeof(T) / GetUbBlockSize() -
                             CeilDiv(static_cast<uint32_t>(col_ * sizeof(T)), static_cast<uint32_t>(GetUbBlockSize()));
    copyOutParam_.srcStride = copyInParam_.dstStride;
    copyInParam_.blockLen = col_ * sizeof(T);
    copyOutParam_.blockLen = col_ * sizeof(T);

    srcGm_.SetGlobalBuffer((__gm__ T*)x);
    dstGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(inQueue_, MEDIUM_BUFFER_NUM, tilingDataPtr->bufferSize);
    pipe_->InitBuffer(outQueue_, MEDIUM_BUFFER_NUM, tilingDataPtr->bufferSize);
}

template <typename T, const bool IS_LOWER>
__aicore__ inline void SplitMediumShape<T, IS_LOWER>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    int64_t startOffset = blockIdx_ < tailCoreProcessNum_ ? blockIdx_ * (normalCoreProcessNum_ + 1) :
                                                            tailCoreProcessNum_ + blockIdx_ * normalCoreProcessNum_;
    int64_t curBlocks = blockIdx_ < tailCoreProcessNum_ ? normalCoreProcessNum_ + 1 : normalCoreProcessNum_;
    int64_t endOffset = startOffset + curBlocks;
    MediumBlockInfo curBlockInfo;
    MediumBlockInfo nextBlockInfo;
    ComputeBlockInfo(startOffset, curBlockInfo);
    CopyIn(curBlockInfo);
    for (int64_t baseBlockIndex = startOffset + 1; baseBlockIndex < endOffset; baseBlockIndex++) {
        ComputeBlockInfo(baseBlockIndex, nextBlockInfo);
        CopyIn(nextBlockInfo);
        DoVectorCopy(curBlockInfo);
        CopyOut(curBlockInfo);
        curBlockInfo = nextBlockInfo;
    }
    DoVectorCopy(curBlockInfo);
    CopyOut(curBlockInfo);
}

template <typename T, const bool IS_LOWER>
__aicore__ inline void SplitMediumShape<T, IS_LOWER>::ComputeBlockInfo(int64_t offset, MediumBlockInfo& info)
{
    info.highActual = offset == highOuter_ - 1 ? highTail_ : highInner_;
    info.gmOffset = offset * planeArea_;
}

template <typename T, const bool IS_LOWER>
__aicore__ inline void SplitMediumShape<T, IS_LOWER>::CopyIn(const MediumBlockInfo& info)
{
    copyInParam_.blockCount = info.highActual * row_;
    DataCopyPadExtParams<T> copyInPadParam = {false, 0, 0, 0};

    LocalTensor<T> xLocal = inQueue_.AllocTensor<T>();
    AscendC::DataCopyPad(xLocal, srcGm_[info.gmOffset], copyInParam_, copyInPadParam);

    inQueue_.EnQue(xLocal);
}

template <typename T, const bool IS_LOWER>
__aicore__ inline void SplitMediumShape<T, IS_LOWER>::DoVectorCopy(const MediumBlockInfo& info)
{
    LocalTensor<T> xLocal = inQueue_.DeQue<T>();
    LocalTensor<T> yLocal = outQueue_.AllocTensor<T>();
    uint16_t vfInnerLoops = CeilDiv(static_cast<uint32_t>(col_), regCount_);
    uint16_t highActual = info.highActual;
    uint16_t headLoops = headRows_ * vfInnerLoops;
    uint16_t tailLoops = tailRows_ * vfInnerLoops;
    uint16_t midRows = midRows_;
    __local_mem__ T* xAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* yAddr = (__local_mem__ T*)yLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> vregZero;
        AscendC::Reg::RegTensor<T> vregSrc;
        AscendC::Reg::RegTensor<T> vregDst;
        AscendC::Reg::Duplicate(vregZero, 0);
        uint32_t count = regCount_;
        AscendC::Reg::MaskReg rowMask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::MaskReg selectMask;
        for (uint16_t m = 0; m < highActual; ++m) {
            for (uint16_t i = 0; i < headLoops; ++i) {
                if constexpr (IS_LOWER) {
                    AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(yAddr, vregZero, regCount_, rowMask);
                } else {
                    AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(vregSrc, xAddr, regCount_);
                    AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(yAddr, vregSrc, regCount_, rowMask);
                }
            }
            if constexpr (IS_LOWER) {
                xAddr += headLoops * regCount_;
            }
            for (uint16_t i = 0; i < midRows; ++i) {
                uint32_t copyEleNum = copyCols_ + i;
                for (uint16_t j = 0; j < vfInnerLoops; ++j) {
                    AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(vregSrc, xAddr, regCount_);
                    selectMask = AscendC::Reg::UpdateMask<T>(copyEleNum);
                    if constexpr (IS_LOWER) {
                        AscendC::Reg::Select(vregDst, vregSrc, vregZero, selectMask);
                        AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(yAddr, vregDst, regCount_,
                                                                                      rowMask);
                    } else {
                        AscendC::Reg::Select(vregDst, vregZero, vregSrc, selectMask);
                        AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(yAddr, vregDst, regCount_,
                                                                                      rowMask);
                    }
                }
            }
            for (uint16_t i = 0; i < tailLoops; ++i) {
                if constexpr (IS_LOWER) {
                    AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(vregSrc, xAddr, regCount_);
                    AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(yAddr, vregSrc, regCount_, rowMask);
                } else {
                    AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(yAddr, vregZero, regCount_, rowMask);
                }
            }
            if constexpr (!IS_LOWER) {
                xAddr += tailLoops * regCount_;
            }
        }
    }
    inQueue_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
}

template <typename T, const bool IS_LOWER>
__aicore__ inline void SplitMediumShape<T, IS_LOWER>::CopyOut(const MediumBlockInfo& info)
{
    copyOutParam_.blockCount = info.highActual * row_;
    LocalTensor<T> yLocal = outQueue_.DeQue<T>();
    AscendC::DataCopyPad(dstGm_[info.gmOffset], yLocal, copyOutParam_);
    outQueue_.FreeTensor(yLocal);
}
} // namespace Triangulator

#endif // SPLIT_MEDIUM_SHAPE
