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
 * \file split_tiny_shape.h
 * \brief
 */
#ifndef SPLIT_TINY_SHAPE
#define SPLIT_TINY_SHAPE

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"

namespace Triangulator {
struct TinyBlockInfo {
    int64_t highActual;
    int64_t gmOffset;
};
using namespace AscendC;
using namespace Ops::Base;

constexpr uint32_t TINY_BUFFER_NUM = 2;
constexpr uint8_t TINY_SHAPE_UPPER_RIGHT = 0;
constexpr uint8_t TINY_SHAPE_ON_LINE = 1;
constexpr uint8_t TINY_SHAPE_LOWER_LEFT = 2;
constexpr int64_t MASK_UB_SIZE = 32;
constexpr int64_t DOUBLE = 2;
constexpr int64_t QUADRUPLE = 4;
constexpr int64_t OCTA = 8;

template <typename T>
class SplitTinyShape {
public:
    __aicore__ inline SplitTinyShape(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, __tiling_data_ptr__ TriangulatorTinyTilingData* tilingDataPtr,
                                TPipe* pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(const TinyBlockInfo& info);
    __aicore__ inline void CopyOut(const TinyBlockInfo& info);
    __aicore__ inline void DoVectorCopy(const TinyBlockInfo& info);
    __aicore__ inline void ComputeBlockInfo(int64_t offset, TinyBlockInfo& info);

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, TINY_BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, TINY_BUFFER_NUM> outQueue_;
    GlobalTensor<T> dstGm_;
    GlobalTensor<T> srcGm_;
    LocalTensor<uint64_t> maskLocal_;

    int64_t usedCoreNum_;
    int64_t normalCoreProcessNum_;
    int64_t tailCoreProcessNum_;
    int64_t highOuter_;
    int64_t highInner_;
    int64_t highTail_;
    int64_t planeArea_;
    uint32_t blockIdx_;
    DataCopyExtParams copyInParam_;
    DataCopyExtParams copyOutParam_;
};

template <typename T>
__aicore__ inline void SplitTinyShape<T>::Init(GM_ADDR x, GM_ADDR y,
                                               __tiling_data_ptr__ TriangulatorTinyTilingData* tilingDataPtr,
                                               TPipe* pipeIn)
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
    highOuter_ = tilingDataPtr->highOuter;
    highInner_ = tilingDataPtr->highInner;
    highTail_ = tilingDataPtr->highTail;
    planeArea_ = tilingDataPtr->planeArea;
    copyInParam_.dstStride = GetVRegSize() / GetUbBlockSize() - CeilDiv(static_cast<uint32_t>(planeArea_ * sizeof(T)),
                                                                        static_cast<uint32_t>(GetUbBlockSize()));
    copyOutParam_.srcStride = copyInParam_.dstStride;
    copyInParam_.blockLen = planeArea_ * sizeof(T);
    copyOutParam_.blockLen = planeArea_ * sizeof(T);

    srcGm_.SetGlobalBuffer((__gm__ T*)x);
    dstGm_.SetGlobalBuffer((__gm__ T*)y);

    TBuf<TPosition::VECCALC> maskBuf;
    pipe_->InitBuffer(maskBuf, MASK_UB_SIZE);
    pipe_->InitBuffer(inQueue_, TINY_BUFFER_NUM, tilingDataPtr->bufferSize);
    pipe_->InitBuffer(outQueue_, TINY_BUFFER_NUM, tilingDataPtr->bufferSize);

    maskLocal_ = maskBuf.Get<uint64_t>();
    for (int64_t i = 0; i < MASK_UB_SIZE / sizeof(uint64_t); i++) {
        maskLocal_.SetValue(i, tilingDataPtr->highMask[i]);
    }
}

template <typename T>
__aicore__ inline void SplitTinyShape<T>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    int64_t startOffset = blockIdx_ < tailCoreProcessNum_ ? blockIdx_ * (normalCoreProcessNum_ + 1) :
                                                            tailCoreProcessNum_ + blockIdx_ * normalCoreProcessNum_;
    int64_t curBlocks = blockIdx_ < tailCoreProcessNum_ ? normalCoreProcessNum_ + 1 : normalCoreProcessNum_;
    int64_t endOffset = startOffset + curBlocks;
    TinyBlockInfo curBlockInfo;
    TinyBlockInfo nextBlockInfo;
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

template <typename T>
__aicore__ inline void SplitTinyShape<T>::ComputeBlockInfo(int64_t offset, TinyBlockInfo& info)
{
    info.highActual = offset == highOuter_ - 1 ? highTail_ : highInner_;
    info.gmOffset = offset * highInner_ * planeArea_;
}

template <typename T>
__aicore__ inline void SplitTinyShape<T>::CopyIn(const TinyBlockInfo& info)
{
    copyInParam_.blockCount = info.highActual;
    DataCopyPadExtParams<T> copyInPadParam = {false, 0, 0, 0};

    LocalTensor<T> xLocal = inQueue_.AllocTensor<T>();
    AscendC::DataCopyPad(xLocal, srcGm_[info.gmOffset], copyInParam_, copyInPadParam);

    inQueue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void SplitTinyShape<T>::DoVectorCopy(const TinyBlockInfo& info)
{
    LocalTensor<T> xLocal = inQueue_.DeQue<T>();
    LocalTensor<T> yLocal = outQueue_.AllocTensor<T>();

    auto* srcAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    auto* dstAddr = (__local_mem__ T*)yLocal.GetPhyAddr();
    auto* maskUbAddr = (__local_mem__ uint8_t*)maskLocal_.GetPhyAddr();

    constexpr int64_t dataCount = GetVRegSize() / sizeof(T);
    uint16_t repeatTimes = static_cast<uint16_t>(info.highActual);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> vreg;
        AscendC::Reg::RegTensor<T> dreg;
        AscendC::Reg::RegTensor<T> sreg;

        AscendC::Reg::MaskReg maskReg;
        AscendC::Reg::MaskReg maskRegD;
        AscendC::Reg::MaskReg maskRegQ;
        AscendC::Reg::MaskReg maskRegO;
        AscendC::Reg::MaskReg moveMask = AscendC::Reg::CreateMask<T>();

        AscendC::Reg::Duplicate(sreg, 0);

        if constexpr (sizeof(T) / sizeof(int8_t) == DOUBLE) {
            AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::MaskDist::DIST_US>(maskRegO, maskUbAddr);
        } else if constexpr (sizeof(T) / sizeof(int8_t) == QUADRUPLE) {
            AscendC::Reg::DataCopy(maskReg, maskUbAddr);
            AscendC::Reg::MaskUnPack(maskRegD, maskReg);
            AscendC::Reg::MaskUnPack(maskRegO, maskRegD);
        } else if constexpr (sizeof(T) / sizeof(int8_t) == OCTA) {
            AscendC::Reg::DataCopy(maskReg, maskUbAddr);
            AscendC::Reg::MaskUnPack(maskRegD, maskReg);
            AscendC::Reg::MaskUnPack(maskRegQ, maskRegD);
            AscendC::Reg::MaskUnPack(maskRegO, maskRegQ);
        } else {
            AscendC::Reg::DataCopy(maskRegO, maskUbAddr);
        }

        for (uint16_t i = 0; i < repeatTimes; i++) {
            AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(vreg, srcAddr, dataCount);
            AscendC::Reg::DataCopy(dstAddr, sreg, moveMask);
            AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(dstAddr, vreg, dataCount, maskRegO);
        }
    }
    inQueue_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void SplitTinyShape<T>::CopyOut(const TinyBlockInfo& info)
{
    copyOutParam_.blockCount = info.highActual;
    LocalTensor<T> yLocal = outQueue_.DeQue<T>();
    AscendC::DataCopyPad(dstGm_[info.gmOffset], yLocal, copyOutParam_);
    outQueue_.FreeTensor(yLocal);
}
} // namespace Triangulator

#endif // SPLIT_TINY_SHAPE
