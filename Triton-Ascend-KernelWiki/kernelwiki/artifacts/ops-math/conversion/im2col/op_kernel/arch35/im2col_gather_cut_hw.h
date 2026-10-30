/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IM2COL_GATHER_CUT_HW_H
#define IM2COL_GATHER_CUT_HW_H
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "im2col_tilingdata.h"

namespace Im2col {
using namespace AscendC;

/**
 * @tparam T 原始数据类型
 * @tparam U 做datacopygather的数据类型
 * @tparam Y 做vci的数据类型
 * @tparam isPadding 是否需要补pad
 */
template <typename T, typename U, typename Y, bool isPadding>
class Im2colGatherCutHw {
public:
    __aicore__ inline Im2colGatherCutHw(TPipe& pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const Im2ColNCHWTilingData* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessPad();
    __aicore__ inline void ProcessNoPad();
    __aicore__ inline void NoPadInWInH();
    __aicore__ inline void NoPadOutWInH();
    __aicore__ inline void NoPadInWOutH();
    __aicore__ inline void NoPadOutWOutH();
    __aicore__ inline void PadInWInH();
    __aicore__ inline void PadOutWInH();
    __aicore__ inline void PadInWOutH();
    __aicore__ inline void PadOutWOutH();
    __aicore__ inline void DataCopyIn(int64_t blockCount, int64_t blockLen, int64_t srcOffset, int64_t srcStride,
                                      int64_t dstStride, LocalTensor<T>& xUb);
    __aicore__ inline void DataCopyInPad(int64_t blockCount, int64_t blockLen, int64_t srcOffset, int64_t dstOffset,
                                         int64_t srcStride, int64_t dstStride, LocalTensor<T>& xUb);
    __aicore__ inline void DataCopyOut(int64_t localOffset, int64_t blockCount, int64_t blockLen, int64_t dstOffset,
                                       int64_t srcStride, int64_t dstStride, LocalTensor<T>& yUb);
    __aicore__ inline void DataCopyOutZero(int64_t blockCount, int64_t blockLen, int64_t dstOffset, int64_t srcStride,
                                           int64_t dstStride, int32_t procUbCount);
    __aicore__ inline void CalcIdxAndGather(int64_t wBlockCount, int64_t hBlockCount, int64_t blockLen, int64_t wFactor,
                                            int64_t hFactor, int64_t wUbFactor, LocalTensor<T>& xUb);
    __aicore__ inline void CalcOneUBIdxAndGather(int64_t wBlockCount, int64_t hBlockCount, int64_t blockLen,
                                                 int64_t wFactor, int64_t hFactor, int64_t wUbFactor,
                                                 LocalTensor<T>& xUb);
    __aicore__ inline void IsDataCopyPad(int64_t wBlockCount, int64_t hBlockCount, int64_t blockLen, int32_t inIdx,
                                         LocalTensor<T>& xUb);

private:
    __aicore__ inline __gm__ T* GetTensorAddr(int64_t index);

private:
    TPipe& pipe_;
    constexpr static int32_t BUFFER_NUM = 2;
    constexpr static int64_t BLOCK_ELENUM = Ops::Base::GetUbBlockSize() / sizeof(T);
    constexpr static int64_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
    constexpr static int64_t VL_LEN = Ops::Base::GetVRegSize();
    constexpr static int32_t DATA_COPY_GATHER = 128; // DataCopyGather并行度
    const Im2ColNCHWTilingData* tilingData_;
    const Im2ColInputInfo* inputInfo_;
    TBuf<> inXBuf_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueY_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    LocalTensor<T> yLocal_;
    LocalTensor<T> yUb_;
    LocalTensor<T> inTensorX_;

    int64_t inWOffset_ = 0;
    int64_t inHOffset_ = 0;
    int64_t inWOffsetPad_ = 0;
    int64_t inHOffsetPad_ = 0;
    int64_t outWOffset_ = 0;
    int64_t outHOffset_ = 0;
    int64_t batchOffset_ = 0;
    int32_t blockLen_ = 0;   // burstLen
    int64_t batchSize_ = 0;  // NC的大小
    int64_t ubCount_ = 0;    // 输出单个HW切的UB块数
    int64_t kernelOutW_ = 0; // 单kernel的W方向上有效elem个数
    int64_t kernelNumW_ = 0; // 输入W方向上卷积核个数
    int64_t kernelNumH_ = 0; // 输入H方向上卷积核个数
    int32_t ubLineW_ = 0;    // UB内处理的W行数
    int32_t ubLineH_ = 0;    // UB内处理的H行数
    int32_t ubFactorW_ = 0;  // W方向上ubFactor
    int32_t ubFactorH_ = 0;  // H方向上ubFactor
    int64_t outW_ = 0;
    int64_t outH_ = 0;
    int64_t inKernelW_ = 0;
    int64_t inKernelH_ = 0;
    int64_t xGmOffset_ = 0;
    int64_t xGmOffsetPad_ = 0;
    int64_t yGmOffset_ = 0;
    int64_t dtypeSize_ = sizeof(T);
    int32_t isPad_ = 0;
    int64_t wPadTop_ = 0; // 当前UB输入W方向上TOP pad大小
    int64_t hPadTop_ = 0; // 当前UB输入H方向上TOP pad大小
    int32_t maxGatherNum_ = DATA_COPY_GATHER / sizeof(T);
    int64_t blockFactor_ = 0;
    int64_t blockCount_ = 0;
    int64_t rectAnglesPerCore_ = 0;
    int32_t inputElemNum_ = 0;

    int32_t uVL_ = Ops::Base::GetVRegSize() / sizeof(U);
    int32_t blockIdx_ = 0;
    int32_t coreNum_ = 0;
    DataCopyExtParams copyInParam_{0, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParam_{false, 0, 0, 0};
    DataCopyExtParams copyOutParam_{0, 0, 0, 0, 0};
    LoopModeParams loopMode_{1, 1, 0, 0, 0, 0};
};

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::Init(GM_ADDR x, GM_ADDR y,
                                                                   const Im2ColNCHWTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    inputInfo_ = &tilingData_->input;
    ubCount_ = tilingData_->outHWrectAngles;
    kernelNumW_ = tilingData_->convKernelNumInWidth;
    kernelNumH_ = tilingData_->convKernelNumInHeight;
    blockLen_ = tilingData_->w4ubFactorW;
    batchSize_ = inputInfo_->N * inputInfo_->C;
    kernelOutW_ = Ops::Base::CeilDiv(inputInfo_->wKernelSize, inputInfo_->wDilation);
    ubLineW_ = tilingData_->lines4ubFactorW;
    ubLineH_ = tilingData_->lines4ubFactorH;
    outW_ = tilingData_->convKernelNumInWidth * tilingData_->convKernelNumInHeight;
    outH_ = inputInfo_->hKernelSize * inputInfo_->wKernelSize;
    ubFactorH_ = tilingData_->ubFactorH;
    ubFactorW_ = tilingData_->ubFactorW;

    int64_t inputSize = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_->inputBufferSize), BLOCK_SIZE);
    inputElemNum_ = inputSize / sizeof(T);
    int64_t outputSize = tilingData_->outputBufferSize;
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    if constexpr (isPadding) {
        pipe_.InitBuffer(inXBuf_, inputSize * BUFFER_NUM);
        inTensorX_ = inXBuf_.template Get<T>();
    } else {
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, inputSize);
    }
    pipe_.InitBuffer(outQueueY_, BUFFER_NUM, outputSize);
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::DataCopyIn(int64_t blockCount, int64_t blockLen,
                                                                         int64_t srcOffset, int64_t srcStride,
                                                                         int64_t dstStride, LocalTensor<T>& xUb)
{
    SetLoopModePara(loopMode_, DataCopyMVType::OUT_TO_UB);
    copyInParam_.blockCount = blockCount;
    copyInParam_.blockLen = blockLen * dtypeSize_;
    copyInParam_.srcStride = srcStride * dtypeSize_;
    copyInParam_.dstStride = dstStride / BLOCK_ELENUM;
    DataCopyPad(xUb, xGm_[srcOffset], copyInParam_, padParam_);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    inQueueX_.EnQue(xUb);
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::DataCopyInPad(int64_t blockCount, int64_t blockLen,
                                                                            int64_t srcOffset, int64_t dstOffset,
                                                                            int64_t srcStride, int64_t dstStride,
                                                                            LocalTensor<T>& xUb)
{
    copyInParam_.blockCount = blockCount;
    copyInParam_.blockLen = blockLen * dtypeSize_;
    copyInParam_.srcStride = srcStride * dtypeSize_;
    copyInParam_.dstStride = dstStride / BLOCK_ELENUM;
    DataCopyPad(xUb[dstOffset], xGm_[srcOffset], copyInParam_, padParam_);
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::IsDataCopyPad(int64_t wBlockCount, int64_t hBlockCount,
                                                                            int64_t blockLen, int32_t inIdx,
                                                                            LocalTensor<T>& xUb)
{
    int64_t srcWOffsetPad = inWOffsetPad_;
    int64_t srcHOffsetPad = inHOffsetPad_;
    int64_t srcHOffsetPadWStride = inHOffsetPad_;
    int64_t gmOffset = xGmOffset_;
    int64_t ubOffset = 0;
    int64_t leftPad = inputInfo_->wPaddingBefore - srcWOffsetPad > 0 ? inputInfo_->wPaddingBefore - srcWOffsetPad : 0;
    int64_t leftPadAlign = leftPad > 0 ? (leftPad + BLOCK_ELENUM - 1) / BLOCK_ELENUM * BLOCK_ELENUM : 0;
    int64_t rightPad = srcWOffsetPad + blockLen - inputInfo_->wPaddingBefore - inputInfo_->W > 0 ?
                           srcWOffsetPad + blockLen - inputInfo_->wPaddingBefore - inputInfo_->W :
                           0;
    int64_t blockLenPadAlign = (blockLen - leftPad + BLOCK_ELENUM - 1) / BLOCK_ELENUM * BLOCK_ELENUM + leftPadAlign;
    int32_t count = blockLenPadAlign * wBlockCount * hBlockCount;
    int32_t loop = 0;
    padParam_.isPad = true;
    if (inIdx >= 1) {
        int32_t nextIdx = inIdx + 1;
        LocalTensor<T> nextLocal = inTensorX_[(nextIdx & 1) * inputElemNum_];
        Duplicate(nextLocal, static_cast<T>(0), inputElemNum_);
        if ((nextIdx & 1) == 0) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        } else {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        }
    }

    if ((inIdx & 1) == 0) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    } else {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    }
    for (int64_t w = 0; w < wBlockCount; w++) {
        for (int64_t h = 0; h < hBlockCount; h++) {
            if (srcHOffsetPad < inputInfo_->hPaddingBefore ||
                srcHOffsetPad >= inputInfo_->hPaddingBefore + inputInfo_->H) {
                ubOffset += blockLenPadAlign;
                srcHOffsetPad = srcHOffsetPad + inputInfo_->hDilation;
                gmOffset = srcHOffsetPad > inputInfo_->hPaddingBefore ?
                               gmOffset + (srcHOffsetPad - inputInfo_->hPaddingBefore) * inputInfo_->W :
                               gmOffset;
            } else if (srcWOffsetPad < inputInfo_->wPaddingBefore) {
                DataCopyInPad(1, blockLen - leftPad - rightPad, gmOffset, ubOffset + leftPadAlign, 0, 0, xUb);
                ubOffset += blockLenPadAlign;
                gmOffset += inputInfo_->hDilation * inputInfo_->W;
                srcHOffsetPad = srcHOffsetPad + inputInfo_->hDilation;
            } else if (srcWOffsetPad >= inputInfo_->wPaddingBefore) {
                DataCopyInPad(1, blockLen - rightPad, gmOffset, ubOffset, 0, 0, xUb);
                ubOffset += blockLenPadAlign;
                gmOffset += inputInfo_->hDilation * inputInfo_->W;
                srcHOffsetPad = srcHOffsetPad + inputInfo_->hDilation;
            }
        }
        srcHOffsetPadWStride += inputInfo_->hStride;
        srcHOffsetPad = srcHOffsetPadWStride;
        loop = srcHOffsetPadWStride < inputInfo_->hPaddingBefore ?
                   0 :
                   srcHOffsetPadWStride - inHOffset_ - inputInfo_->hPaddingBefore;
        gmOffset = xGmOffset_ + loop * inputInfo_->W;
    }
    if ((inIdx & 1) == 0) {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    } else {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    }
    padParam_.isPad = false;
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::DataCopyOut(int64_t localOffset, int64_t blockCount,
                                                                          int64_t blockLen, int64_t dstOffset,
                                                                          int64_t srcStride, int64_t dstStride,
                                                                          LocalTensor<T>& yUb)
{
    copyOutParam_.blockCount = blockCount;
    copyOutParam_.blockLen = blockLen * dtypeSize_;
    copyOutParam_.srcStride = srcStride / BLOCK_ELENUM;
    copyOutParam_.dstStride = dstStride * dtypeSize_;
    DataCopyPad(yGm_[dstOffset], yUb[localOffset], copyOutParam_);
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::DataCopyOutZero(int64_t blockCount, int64_t blockLen,
                                                                              int64_t dstOffset, int64_t srcStride,
                                                                              int64_t dstStride, int32_t procUbCount)
{
    yLocal_ = outQueueY_.AllocTensor<T>();
    int64_t blockLenAlign = (blockLen + BLOCK_ELENUM - 1) / BLOCK_ELENUM * BLOCK_ELENUM;
    int32_t count = blockLenAlign * blockCount;
    Duplicate(yLocal_, static_cast<T>(0), count);
    if (procUbCount % 2 == 0) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    } else {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    }
    copyOutParam_.blockCount = blockCount;
    copyOutParam_.blockLen = blockLen * dtypeSize_;
    copyOutParam_.srcStride = srcStride / BLOCK_ELENUM;
    copyOutParam_.dstStride = dstStride * dtypeSize_;
    DataCopyPad(yGm_[dstOffset], yLocal_, copyOutParam_);
    outQueueY_.FreeTensor(yLocal_);
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::CalcOneUBIdxAndGather(int64_t wBlockCount,
                                                                                    int64_t hBlockCount,
                                                                                    int64_t blockLen, int64_t wFactor,
                                                                                    int64_t hFactor, int64_t wUbFactor,
                                                                                    LocalTensor<T>& xUb)
{
    int64_t srcWOffsetPad = inWOffsetPad_;
    int64_t leftPad = inputInfo_->wPaddingBefore - srcWOffsetPad > 0 ? inputInfo_->wPaddingBefore - srcWOffsetPad : 0;
    int64_t leftPadAlign = leftPad > 0 ? (leftPad + BLOCK_ELENUM - 1) / BLOCK_ELENUM * BLOCK_ELENUM : 0;
    int64_t blockLenAlign = (blockLen - leftPad + BLOCK_ELENUM - 1) / BLOCK_ELENUM * BLOCK_ELENUM + leftPadAlign;
    int64_t wStride = inputInfo_->wStride;
    uint16_t loop1 = uVL_ / wUbFactor;
    uint16_t loop2 = loop1 / hFactor;
    uint32_t maskSize = uVL_;
    uint32_t count = wUbFactor / wFactor;
    int64_t hwAlign = hBlockCount * blockLenAlign;
    int64_t dilation = inputInfo_->wDilation;
    int64_t stride = 0;
    yLocal_ = outQueueY_.AllocTensor<T>();
    __ubuf__ T* srcPtr = (__ubuf__ T*)xUb.GetPhyAddr() + leftPadAlign - leftPad;
    __ubuf__ U* dstPtrU = (__ubuf__ U*)yLocal_.GetPhyAddr();
    __ubuf__ T* dstPtrT = (__ubuf__ T*)yLocal_.GetPhyAddr();

    if (loop2 == 0) {
        uint16_t loop = hFactor / loop1;
        uint16_t tail = (hFactor - loop1 * loop) * wUbFactor;
        uint16_t loopH = hBlockCount;
        uint32_t maskTailSize = tail;
        uint32_t factor = loop1 * wUbFactor;
        uint32_t maskLoop = factor;
        uint32_t strideH = blockLenAlign;
        stride = dilation * loop1;
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<Y> indexReg;
            AscendC::Reg::RegTensor<U> tmp;
            AscendC::Reg::RegTensor<U> tmp1;
            AscendC::Reg::RegTensor<U> addReg;
            AscendC::Reg::RegTensor<U> addReg1;
            AscendC::Reg::RegTensor<U> subReg;
            AscendC::Reg::RegTensor<U> subReg1;
            AscendC::Reg::RegTensor<U> divReg;
            AscendC::Reg::RegTensor<U> divReg2;
            AscendC::Reg::RegTensor<U> mulsReg;
            AscendC::Reg::RegTensor<U> mulsReg0;
            AscendC::Reg::RegTensor<U> mulsReg1;
            AscendC::Reg::RegTensor<U> mulsReg2;
            AscendC::Reg::RegTensor<U> mulsReg3;
            AscendC::Reg::RegTensor<U> dstReg;
            AscendC::Reg::RegTensor<U> movReg;
            AscendC::Reg::RegTensor<T> dstRegT;
            AscendC::Reg::UnalignReg uDst;
            AscendC::Reg::MaskReg mask;
            AscendC::Reg::MaskReg maskOut;
            AscendC::Reg::MaskReg maskTail;

            mask = AscendC::Reg::UpdateMask<U>(maskSize);
            maskOut = AscendC::Reg::UpdateMask<U>(maskLoop);
            maskTail = AscendC::Reg::UpdateMask<U>(maskTailSize);
            Y startIdx = (Y)0;
            AscendC::Reg::Arange(indexReg, startIdx);
            AscendC::Reg::Duplicate(tmp, (U)wFactor, mask);
            AscendC::Reg::Duplicate(tmp1, (U)wUbFactor, mask);
            AscendC::Reg::Div(divReg, (AscendC::Reg::RegTensor<U>&)indexReg, tmp, mask);
            AscendC::Reg::Div(divReg2, (AscendC::Reg::RegTensor<U>&)indexReg, tmp1, mask);
            AscendC::Reg::Muls(mulsReg0, divReg2, (U)count, mask);
            AscendC::Reg::Muls(mulsReg, divReg, (U)wFactor, mask);
            AscendC::Reg::Muls(mulsReg1, divReg2, (U)dilation, mask);
            AscendC::Reg::Sub(subReg, (AscendC::Reg::RegTensor<U>&)indexReg, mulsReg, mask);
            AscendC::Reg::Sub(subReg1, divReg, mulsReg0, mask);
            AscendC::Reg::Muls(mulsReg2, subReg, (U)wStride, mask);
            AscendC::Reg::Muls(mulsReg3, subReg1, (U)hwAlign, mask);
            AscendC::Reg::Add(addReg, mulsReg3, mulsReg2, mask);
            AscendC::Reg::Add(addReg1, addReg, mulsReg1, mask);
            AscendC::Reg::Move(movReg, addReg1);
            for (uint16_t j = 0; j < loopH; j++) {
                for (uint16_t i = 0; i < loop; i++) {
                    AscendC::Reg::Gather(dstReg, srcPtr, addReg1, maskOut);
                    // Copy out
                    if constexpr (sizeof(T) == sizeof(int8_t)) {
                        // Convert B16 to B8
                        AscendC::Reg::Pack(dstRegT, dstReg);
                        AscendC::Reg::DataCopyUnAlign(dstPtrT, dstRegT, uDst, factor);
                        AscendC::Reg::DataCopyUnAlignPost(dstPtrT, uDst, 0);
                    } else {
                        AscendC::Reg::DataCopyUnAlign(dstPtrU, dstReg, uDst, factor);
                        AscendC::Reg::DataCopyUnAlignPost(dstPtrU, uDst, 0);
                    }
                    AscendC::Reg::Adds(addReg1, addReg1, (U)stride, mask);
                }
                AscendC::Reg::Gather(dstReg, srcPtr, addReg1, maskTail);
                if constexpr (sizeof(T) == sizeof(int8_t)) {
                    // Convert B16 to B8
                    AscendC::Reg::Pack(dstRegT, dstReg);
                    AscendC::Reg::DataCopyUnAlign(dstPtrT, dstRegT, uDst, tail);
                    AscendC::Reg::DataCopyUnAlignPost(dstPtrT, uDst, 0);
                } else {
                    AscendC::Reg::DataCopyUnAlign(dstPtrU, dstReg, uDst, tail);
                    AscendC::Reg::DataCopyUnAlignPost(dstPtrU, uDst, 0);
                }
                AscendC::Reg::Adds(movReg, movReg, (U)strideH, mask);
                AscendC::Reg::Move(addReg1, movReg);
            }
        }
    } else {
        uint16_t loopH = hBlockCount / loop2;
        uint16_t tail = hBlockCount - loopH * loop2;
        uint32_t maskTail = tail * hFactor * wUbFactor;
        uint32_t tailFactor = maskTail;
        uint32_t factor = loop2 * hFactor * wUbFactor;
        maskSize = factor;
        stride = blockLenAlign * loop2;
        uint32_t strideH = blockLenAlign;
        uint32_t hwFactor = hFactor * wUbFactor;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<Y> indexReg;
            AscendC::Reg::RegTensor<U> tmp;
            AscendC::Reg::RegTensor<U> tmp1;
            AscendC::Reg::RegTensor<U> addReg;
            AscendC::Reg::RegTensor<U> addReg1;
            AscendC::Reg::RegTensor<U> addReg2;
            AscendC::Reg::RegTensor<U> hwReg;
            AscendC::Reg::RegTensor<U> subReg;
            AscendC::Reg::RegTensor<U> subReg1;
            AscendC::Reg::RegTensor<U> subReg2;
            AscendC::Reg::RegTensor<U> divReg;
            AscendC::Reg::RegTensor<U> divReg1;
            AscendC::Reg::RegTensor<U> divReg2;
            AscendC::Reg::RegTensor<U> divReg3;
            AscendC::Reg::RegTensor<U> mulsReg;
            AscendC::Reg::RegTensor<U> mulsReg0;
            AscendC::Reg::RegTensor<U> mulsReg1;
            AscendC::Reg::RegTensor<U> mulsReg2;
            AscendC::Reg::RegTensor<U> mulsReg4;
            AscendC::Reg::RegTensor<U> mulsReg5;
            AscendC::Reg::RegTensor<U> mulsReg6;
            AscendC::Reg::RegTensor<U> dstReg;
            AscendC::Reg::RegTensor<T> dstRegT;
            AscendC::Reg::UnalignReg uDst;
            AscendC::Reg::MaskReg mask;
            AscendC::Reg::MaskReg maskOutTail;

            mask = AscendC::Reg::UpdateMask<U>(maskSize);
            maskOutTail = AscendC::Reg::UpdateMask<U>(maskTail);
            Y startIdx = (Y)0;
            AscendC::Reg::Arange(indexReg, startIdx);
            AscendC::Reg::Duplicate(hwReg, (U)hwFactor, mask);
            AscendC::Reg::Duplicate(tmp, (U)wFactor, mask);
            AscendC::Reg::Duplicate(tmp1, (U)wUbFactor, mask);
            AscendC::Reg::Div(divReg, (AscendC::Reg::RegTensor<U>&)indexReg, hwReg, mask);
            AscendC::Reg::Div(divReg1, (AscendC::Reg::RegTensor<U>&)indexReg, tmp1, mask);
            AscendC::Reg::Div(divReg2, (AscendC::Reg::RegTensor<U>&)indexReg, tmp, mask);
            AscendC::Reg::Muls(mulsReg, divReg2, (U)wFactor, mask);
            AscendC::Reg::Muls(mulsReg0, divReg1, (U)count, mask);
            AscendC::Reg::Sub(subReg, (AscendC::Reg::RegTensor<U>&)indexReg, mulsReg, mask);
            AscendC::Reg::Sub(subReg1, divReg2, mulsReg0, mask);
            AscendC::Reg::Muls(mulsReg2, subReg, (U)wStride, mask);
            AscendC::Reg::Muls(mulsReg4, divReg, (U)strideH, mask);
            AscendC::Reg::Muls(mulsReg5, subReg1, (U)hwAlign, mask);
            AscendC::Reg::Muls(mulsReg6, divReg, (U)hFactor, mask);
            AscendC::Reg::Sub(subReg2, divReg1, mulsReg6, mask);
            AscendC::Reg::Muls(mulsReg1, subReg2, (U)dilation, mask);
            AscendC::Reg::Add(addReg, mulsReg5, mulsReg2, mask);
            AscendC::Reg::Add(addReg1, addReg, mulsReg1, mask);
            AscendC::Reg::Add(addReg2, addReg1, mulsReg4, mask);

            for (uint16_t i = 0; i < loopH; i++) {
                AscendC::Reg::Gather(dstReg, srcPtr, addReg2, mask);
                // Copy out
                if constexpr (sizeof(T) == sizeof(int8_t)) {
                    // Convert B16 to B8
                    AscendC::Reg::Pack(dstRegT, dstReg);
                    AscendC::Reg::DataCopyUnAlign(dstPtrT, dstRegT, uDst, factor);
                    AscendC::Reg::DataCopyUnAlignPost(dstPtrT, uDst, 0);
                } else {
                    AscendC::Reg::DataCopyUnAlign(dstPtrU, dstReg, uDst, factor);
                    AscendC::Reg::DataCopyUnAlignPost(dstPtrU, uDst, 0);
                }
                AscendC::Reg::Adds(addReg2, addReg2, (U)stride, mask);
            }
            AscendC::Reg::Gather(dstReg, srcPtr, addReg2, maskOutTail);
            if constexpr (sizeof(T) == sizeof(int8_t)) {
                // Convert B16 to B8
                AscendC::Reg::Pack(dstRegT, dstReg);
                AscendC::Reg::DataCopyUnAlign(dstPtrT, dstRegT, uDst, tailFactor);
                AscendC::Reg::DataCopyUnAlignPost(dstPtrT, uDst, 0);
            } else {
                AscendC::Reg::DataCopyUnAlign(dstPtrU, dstReg, uDst, tailFactor);
                AscendC::Reg::DataCopyUnAlignPost(dstPtrU, uDst, 0);
            }
        }
    }
    outQueueY_.EnQue(yLocal_);
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::CalcIdxAndGather(int64_t wBlockCount, int64_t hBlockCount,
                                                                               int64_t blockLen, int64_t wFactor,
                                                                               int64_t hFactor, int64_t wUbFactor,
                                                                               LocalTensor<T>& xUb)
{
    int64_t srcWOffsetPad = inWOffsetPad_;
    int64_t leftPad = inputInfo_->wPaddingBefore - srcWOffsetPad > 0 ? inputInfo_->wPaddingBefore - srcWOffsetPad : 0;
    int64_t leftPadAlign = leftPad > 0 ? (leftPad + BLOCK_ELENUM - 1) / BLOCK_ELENUM * BLOCK_ELENUM : 0;
    int64_t blockLenAlign = (blockLen - leftPad + BLOCK_ELENUM - 1) / BLOCK_ELENUM * BLOCK_ELENUM + leftPadAlign;
    int64_t wStride = inputInfo_->wStride;
    uint32_t ubFactor = wUbFactor;
    uint32_t offset = (wUbFactor + BLOCK_ELENUM - 1) / BLOCK_ELENUM * BLOCK_ELENUM;
    uint16_t loop = hBlockCount;
    uint16_t loop1 = hFactor;
    uint32_t maskSize = wUbFactor;
    uint32_t maskB8 = wUbFactor;
    int64_t hwAlign = hBlockCount * blockLenAlign;
    int64_t dilation = inputInfo_->wDilation;
    int64_t stride = blockLenAlign - dilation * (loop1 - 1);
    yLocal_ = outQueueY_.AllocTensor<T>();
    __ubuf__ T* srcPtr = (__ubuf__ T*)xUb.GetPhyAddr() + leftPadAlign - leftPad;
    __ubuf__ U* dstPtrU = (__ubuf__ U*)yLocal_.GetPhyAddr();
    __ubuf__ T* dstPtrT = (__ubuf__ T*)yLocal_.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<Y> indexReg;
        AscendC::Reg::RegTensor<U> tmp;
        AscendC::Reg::RegTensor<U> tmp1;
        AscendC::Reg::RegTensor<U> addReg;
        AscendC::Reg::RegTensor<U> subReg;
        AscendC::Reg::RegTensor<U> divReg;
        AscendC::Reg::RegTensor<U> mulsReg;
        AscendC::Reg::RegTensor<U> mulsReg2;
        AscendC::Reg::RegTensor<U> mulsReg3;
        AscendC::Reg::RegTensor<U> dstReg;
        AscendC::Reg::RegTensor<T> dstRegT;
        AscendC::Reg::UnalignReg uDst;
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::MaskReg maskOut;
        AscendC::Reg::MaskReg maskT;

        mask = AscendC::Reg::UpdateMask<U>(maskSize);
        maskOut = AscendC::Reg::UpdateMask<U>(ubFactor);
        maskT = AscendC::Reg::UpdateMask<T>(maskB8);
        Y startIdx = (Y)0;
        AscendC::Reg::Arange(indexReg, startIdx);
        AscendC::Reg::Duplicate(tmp, (U)wFactor, mask);
        AscendC::Reg::Duplicate(tmp1, (U)dilation, mask);
        AscendC::Reg::Div(divReg, (AscendC::Reg::RegTensor<U>&)indexReg, tmp, mask);
        AscendC::Reg::Muls(mulsReg, divReg, (U)wFactor, mask);
        AscendC::Reg::Sub(subReg, (AscendC::Reg::RegTensor<U>&)indexReg, mulsReg, mask);
        AscendC::Reg::Muls(mulsReg2, subReg, (U)wStride, mask);
        AscendC::Reg::Muls(mulsReg3, divReg, (U)hwAlign, mask);
        AscendC::Reg::Add(addReg, mulsReg3, mulsReg2, mask);
        for (uint16_t i = 0; i < loop; i++) {
            for (uint16_t j = 0; j < loop1; j++) {
                AscendC::Reg::Gather(dstReg, srcPtr, addReg, maskOut);
                // Copy out
                if constexpr (sizeof(T) == sizeof(int8_t)) {
                    // Convert B16 to B8
                    AscendC::Reg::Pack(dstRegT, dstReg);
                    AscendC::Reg::StoreAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(dstPtrT, dstRegT, offset, maskT);
                } else {
                    AscendC::Reg::StoreAlign<U, Reg::PostLiteral::POST_MODE_UPDATE>(dstPtrU, dstReg, offset, maskOut);
                }
                AscendC::Reg::Adds(addReg, addReg, (U)dilation, mask);
            }
            AscendC::Reg::Sub(addReg, addReg, tmp1, mask);
            AscendC::Reg::Adds(addReg, addReg, (U)stride, mask);
        }
    }
    outQueueY_.EnQue(yLocal_);
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::NoPadInWInH()
{
    int32_t ubFactorH = tilingData_->ubFactorH;
    int32_t ubFactorW = tilingData_->ubFactorW;
    int32_t procUbCount = rectAnglesPerCore_;
    int64_t perMatrixWUbCnt = Ops::Base::CeilDiv(kernelNumW_, static_cast<int64_t>(ubFactorW));
    int64_t perMatrixHUbCnt = Ops::Base::CeilDiv(inputInfo_->wKernelSize, static_cast<int64_t>(ubFactorH));
    int64_t wUbCount = perMatrixWUbCnt * kernelNumH_;
    int64_t hUbCount = perMatrixHUbCnt * inputInfo_->hKernelSize;
    int64_t blockUbCount = tilingData_->rectAnglesPerCore * blockIdx_;
    int64_t ncIdx = blockUbCount / tilingData_->outHWrectAngles;
    int64_t hUbCountOffset = (blockUbCount - ncIdx * tilingData_->outHWrectAngles) / wUbCount;
    int64_t wUbCountOffset = blockUbCount - ncIdx * tilingData_->outHWrectAngles - hUbCountOffset * wUbCount;
    int64_t ubCountOffset = 0;      // 当前HW的ub块索引
    int64_t matrixWUbCntOffset = 0; // 当前HW的矩阵块W方向索引
    int64_t matrixHUbCntOffset = 0; // 当前HW的矩阵块H方向索引
    int64_t nowMatrixWUbOffset = 0; // 矩阵块内W方向UB索引
    int64_t nowMatrixHUbOffset = 0; // 矩阵块内H方向UB索引
    inKernelW_ = inputInfo_->W - (kernelNumW_ - 1) * inputInfo_->wStride;
    inKernelH_ = inputInfo_->H - (kernelNumH_ - 1) * inputInfo_->hStride;

    while (procUbCount > 0) {
        ubFactorW = tilingData_->ubFactorW;
        ubFactorH = tilingData_->ubFactorH;
        ubCountOffset = hUbCountOffset * wUbCount + wUbCountOffset;
        matrixWUbCntOffset = wUbCountOffset / perMatrixWUbCnt;
        matrixHUbCntOffset = hUbCountOffset / perMatrixHUbCnt;
        nowMatrixWUbOffset = (wUbCountOffset - matrixWUbCntOffset * perMatrixWUbCnt) * ubFactorW;
        nowMatrixHUbOffset = (hUbCountOffset - matrixHUbCntOffset * perMatrixHUbCnt) * ubFactorH;
        outWOffset_ = matrixWUbCntOffset * kernelNumW_ + nowMatrixWUbOffset;
        outHOffset_ = matrixHUbCntOffset * inputInfo_->wKernelSize + nowMatrixHUbOffset;
        inWOffset_ = (outWOffset_ - outWOffset_ / kernelNumW_ * kernelNumW_) * inputInfo_->wStride +
                     nowMatrixHUbOffset * inputInfo_->wDilation;
        inHOffset_ = outWOffset_ / kernelNumW_ * inputInfo_->hStride +
                     outHOffset_ / inputInfo_->wKernelSize * inputInfo_->hDilation;
        xGmOffset_ = ncIdx * inputInfo_->H * inputInfo_->W + inHOffset_ * inputInfo_->W + inWOffset_;
        yGmOffset_ = ncIdx * outW_ * outH_ + outHOffset_ * outW_ + outWOffset_;
        int64_t blockLen = (inputInfo_->W - inWOffset_) >= tilingData_->w4ubFactorW ? tilingData_->w4ubFactorW :
                                                                                      (inputInfo_->W - inWOffset_);
        ubFactorW = nowMatrixWUbOffset == (perMatrixWUbCnt - 1) * ubFactorW ? kernelNumW_ - nowMatrixWUbOffset :
                                                                              ubFactorW;
        ubFactorH = nowMatrixHUbOffset == (perMatrixHUbCnt - 1) * ubFactorH ?
                        inputInfo_->wKernelSize - nowMatrixHUbOffset :
                        ubFactorH;
        int64_t srcStride = 0;
        int64_t dstStride = 0;
        LocalTensor<T> xUb = inQueueX_.AllocTensor<T>();
        DataCopyIn(1, blockLen, xGmOffset_, srcStride, dstStride, xUb);
        LocalTensor<T> srcUb = inQueueX_.DeQue<T>();
        CalcIdxAndGather(1, 1, blockLen, ubFactorW, ubFactorH, ubFactorW, srcUb);
        LocalTensor<T> yUb = outQueueY_.DeQue<T>();
        DataCopyOut(0, ubFactorH, ubFactorW, yGmOffset_, 0, outW_ - ubFactorW, yUb);
        if (wUbCountOffset == wUbCount - 1) {
            wUbCountOffset = 0;
            hUbCountOffset = hUbCountOffset == hUbCount - 1 ? 0 : hUbCountOffset + 1;
            ncIdx = hUbCountOffset == 0 ? ncIdx + 1 : ncIdx;
        } else {
            wUbCountOffset++;
        }
        procUbCount--;
        outQueueY_.FreeTensor(yUb);
        inQueueX_.FreeTensor(srcUb);
    }
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::NoPadOutWInH()
{
    int32_t ubFactorH = tilingData_->ubFactorH;
    int32_t ubFactorW = tilingData_->ubFactorW;
    int32_t procUbCount = rectAnglesPerCore_;
    int64_t perMatrixHUbCnt = Ops::Base::CeilDiv(inputInfo_->wKernelSize, static_cast<int64_t>(ubFactorH));
    int64_t perWUbMatrixCnt = Ops::Base::CeilDiv(static_cast<int64_t>(ubFactorW), kernelNumW_);
    int64_t wUbCount = Ops::Base::CeilDiv(outW_, static_cast<int64_t>(ubFactorW));
    int64_t hUbCount = perMatrixHUbCnt * inputInfo_->hKernelSize;
    int64_t blockUbCount = tilingData_->rectAnglesPerCore * blockIdx_;
    int64_t ncIdx = blockUbCount / tilingData_->outHWrectAngles;
    int64_t hUbCountOffset = (blockUbCount - ncIdx * tilingData_->outHWrectAngles) / wUbCount; // H方向上UB块偏移
    int64_t wUbCountOffset = blockUbCount - ncIdx * tilingData_->outHWrectAngles -
                             hUbCountOffset * wUbCount; // W方向上UB块偏移
    int64_t ubCountOffset = 0;                          // 当前HW的ub块索引
    int64_t matrixWUbCntOffset = 0;                     // 当前HW的矩阵块W方向索引
    int64_t matrixHUbCntOffset = 0;                     // 当前HW的矩阵块H方向索引
    int64_t nowMatrixHUbOffset = 0;                     // 矩阵块内H方向UB索引
    int64_t inKernelHOffset = 0;                        // 输入H方向上卷积核所在行数
    int64_t perWUbMatrixCntReal = 0;
    inKernelW_ = inputInfo_->W - (kernelNumW_ - 1) * inputInfo_->wStride;
    inKernelH_ = inputInfo_->H - (kernelNumH_ - 1) * inputInfo_->hStride;
    while (procUbCount > 0) {
        ubFactorW = tilingData_->ubFactorW;
        ubFactorH = tilingData_->ubFactorH;
        ubCountOffset = hUbCountOffset * wUbCount + wUbCountOffset;
        matrixWUbCntOffset = wUbCountOffset * perWUbMatrixCnt;
        matrixHUbCntOffset = hUbCountOffset / perMatrixHUbCnt;
        perWUbMatrixCntReal = kernelNumH_ - matrixWUbCntOffset >= perWUbMatrixCnt ? perWUbMatrixCnt :
                                                                                    kernelNumH_ - matrixWUbCntOffset;
        nowMatrixHUbOffset = (hUbCountOffset - matrixHUbCntOffset * perMatrixHUbCnt) * ubFactorH;
        outWOffset_ = matrixWUbCntOffset * kernelNumW_;
        outHOffset_ = matrixHUbCntOffset * inputInfo_->wKernelSize + nowMatrixHUbOffset;
        inWOffset_ = (outWOffset_ - outWOffset_ / kernelNumW_ * kernelNumW_) * inputInfo_->wStride +
                     nowMatrixHUbOffset * inputInfo_->wDilation;
        inHOffset_ = outWOffset_ / kernelNumW_ * inputInfo_->hStride +
                     outHOffset_ / inputInfo_->wKernelSize * inputInfo_->hDilation;
        xGmOffset_ = ncIdx * inputInfo_->H * inputInfo_->W + inHOffset_ * inputInfo_->W + inWOffset_;
        yGmOffset_ = ncIdx * outW_ * outH_ + outHOffset_ * outW_ + outWOffset_;
        int64_t blockLen = (inputInfo_->W - inWOffset_) >= tilingData_->w4ubFactorW ? tilingData_->w4ubFactorW :
                                                                                      (inputInfo_->W - inWOffset_);
        ubFactorW = wUbCountOffset == wUbCount - 1 ? outW_ - outWOffset_ : ubFactorW;
        ubFactorH = nowMatrixHUbOffset == (perMatrixHUbCnt - 1) * ubFactorH ?
                        inputInfo_->wKernelSize - nowMatrixHUbOffset :
                        ubFactorH;
        int64_t srcStride = (inputInfo_->hStride - 1) * inputInfo_->W + inputInfo_->W - blockLen;
        int64_t dstStride = 0;
        int64_t blockCount = perWUbMatrixCntReal;
        LocalTensor<T> xUb = inQueueX_.AllocTensor<T>();
        DataCopyIn(blockCount, blockLen, xGmOffset_, srcStride, dstStride, xUb);
        LocalTensor<T> srcUb = inQueueX_.DeQue<T>();
        if (wUbCount == 1 && outW_ <= maxGatherNum_) {
            CalcOneUBIdxAndGather(1, 1, blockLen, kernelNumW_, ubFactorH, ubFactorW, srcUb);
            yUb_ = outQueueY_.DeQue<T>();
            DataCopyOut(0, 1, ubFactorW * ubFactorH, yGmOffset_, 0, 0, yUb_);
        } else {
            CalcIdxAndGather(blockCount, 1, blockLen, kernelNumW_, ubFactorH, ubFactorW, srcUb);
            yUb_ = outQueueY_.DeQue<T>();
            DataCopyOut(0, ubFactorH, ubFactorW, yGmOffset_, 0, outW_ - ubFactorW, yUb_);
        }
        if (wUbCountOffset == wUbCount - 1) {
            wUbCountOffset = 0;
            hUbCountOffset = hUbCountOffset == hUbCount - 1 ? 0 : hUbCountOffset + 1;
            ncIdx = hUbCountOffset == 0 ? ncIdx + 1 : ncIdx;
        } else {
            wUbCountOffset++;
        }
        procUbCount--;
        outQueueY_.FreeTensor(yUb_);
        inQueueX_.FreeTensor(srcUb);
    }
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::NoPadInWOutH()
{
    int32_t ubFactorH = tilingData_->ubFactorH;
    int32_t ubFactorW = tilingData_->ubFactorW;
    int32_t procUbCount = rectAnglesPerCore_;
    int64_t perHUbMatrixCnt = Ops::Base::CeilDiv(static_cast<int64_t>(ubFactorH), inputInfo_->wKernelSize);
    int64_t perMatrixWUbCnt = Ops::Base::CeilDiv(kernelNumW_, static_cast<int64_t>(ubFactorW));
    int64_t wUbCount = kernelNumH_ * perMatrixWUbCnt;
    int64_t hUbCount = Ops::Base::CeilDiv(inputInfo_->wKernelSize * inputInfo_->hKernelSize,
                                          static_cast<int64_t>(ubFactorH));
    int64_t blockUbCount = tilingData_->rectAnglesPerCore * blockIdx_;
    int64_t ncIdx = blockUbCount / tilingData_->outHWrectAngles;
    int64_t hUbCountOffset = (blockUbCount - ncIdx * tilingData_->outHWrectAngles) / wUbCount;
    int64_t wUbCountOffset = blockUbCount - ncIdx * tilingData_->outHWrectAngles - hUbCountOffset * wUbCount;
    int64_t ubCountOffset = 0;      // 当前HW的ub块索引
    int64_t matrixWUbCntOffset = 0; // 当前HW的矩阵块W方向索引
    int64_t matrixHUbCntOffset = 0; // 当前HW的矩阵块H方向索引
    int64_t nowMatrixWUbOffset = 0; // 矩阵块内W方向UB索引
    int64_t perHUbMatrixCntReal = 0;
    inKernelW_ = inputInfo_->W - (kernelNumW_ - 1) * inputInfo_->wStride;
    inKernelH_ = inputInfo_->H - (kernelNumH_ - 1) * inputInfo_->hStride;
    while (procUbCount > 0) {
        ubFactorW = tilingData_->ubFactorW;
        ubFactorH = tilingData_->ubFactorH;
        ubCountOffset = hUbCountOffset * wUbCount + wUbCountOffset;
        matrixWUbCntOffset = wUbCountOffset / perMatrixWUbCnt;
        matrixHUbCntOffset = hUbCountOffset * perHUbMatrixCnt;
        nowMatrixWUbOffset = (wUbCountOffset - matrixWUbCntOffset * perMatrixWUbCnt) * ubFactorW;
        perHUbMatrixCntReal = inputInfo_->hKernelSize - matrixHUbCntOffset >= perHUbMatrixCnt ?
                                  perHUbMatrixCnt :
                                  inputInfo_->hKernelSize - matrixHUbCntOffset;
        outWOffset_ = matrixWUbCntOffset * kernelNumW_ + nowMatrixWUbOffset;
        outHOffset_ = matrixHUbCntOffset * inputInfo_->wKernelSize;
        inWOffset_ = (outWOffset_ - outWOffset_ / kernelNumW_ * kernelNumW_) * inputInfo_->wStride;
        inHOffset_ = outWOffset_ / kernelNumW_ * inputInfo_->hStride +
                     outHOffset_ / inputInfo_->wKernelSize * inputInfo_->hDilation;
        xGmOffset_ = ncIdx * inputInfo_->H * inputInfo_->W + inHOffset_ * inputInfo_->W + inWOffset_;
        yGmOffset_ = ncIdx * outW_ * outH_ + outHOffset_ * outW_ + outWOffset_;
        int64_t blockLen = (inputInfo_->W - inWOffset_) >= tilingData_->w4ubFactorW ? tilingData_->w4ubFactorW :
                                                                                      (inputInfo_->W - inWOffset_);
        ubFactorW = nowMatrixWUbOffset == (perMatrixWUbCnt - 1) * ubFactorW ? kernelNumW_ - nowMatrixWUbOffset :
                                                                              ubFactorW;
        ubFactorH = hUbCountOffset == hUbCount - 1 ? outH_ - outHOffset_ : ubFactorH;
        int64_t srcStride = (inputInfo_->hDilation - 1) * inputInfo_->W + inputInfo_->W - blockLen;
        int64_t dstStride = 0;
        int64_t blockCount = perHUbMatrixCntReal;
        LocalTensor<T> xUb = inQueueX_.AllocTensor<T>();
        DataCopyIn(blockCount, blockLen, xGmOffset_, srcStride, dstStride, xUb);
        LocalTensor<T> srcUb = inQueueX_.DeQue<T>();
        CalcIdxAndGather(1, blockCount, blockLen, ubFactorW, inputInfo_->wKernelSize, ubFactorW, srcUb);
        LocalTensor<T> yUb = outQueueY_.DeQue<T>();
        DataCopyOut(0, ubFactorH, ubFactorW, yGmOffset_, 0, outW_ - ubFactorW, yUb);
        if (wUbCountOffset == wUbCount - 1) {
            wUbCountOffset = 0;
            hUbCountOffset = hUbCountOffset == hUbCount - 1 ? 0 : hUbCountOffset + 1;
            ncIdx = hUbCountOffset == 0 ? ncIdx + 1 : ncIdx;
        } else {
            wUbCountOffset++;
        }
        procUbCount--;
        outQueueY_.FreeTensor(yUb);
        inQueueX_.FreeTensor(srcUb);
    }
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::NoPadOutWOutH()
{
    int32_t ubFactorH = tilingData_->ubFactorH;
    int32_t ubFactorW = tilingData_->ubFactorW;
    int32_t procUbCount = rectAnglesPerCore_;
    int64_t perHUbMatrixCnt = Ops::Base::CeilDiv(static_cast<int64_t>(ubFactorH), inputInfo_->wKernelSize);
    int64_t perWUbMatrixCnt = Ops::Base::CeilDiv(static_cast<int64_t>(ubFactorW), kernelNumW_);
    int64_t wUbCount = Ops::Base::CeilDiv(outW_, static_cast<int64_t>(ubFactorW));
    int64_t hUbCount = Ops::Base::CeilDiv(inputInfo_->wKernelSize * inputInfo_->hKernelSize,
                                          static_cast<int64_t>(ubFactorH));
    int64_t blockUbCount = tilingData_->rectAnglesPerCore * blockIdx_;
    int64_t ncIdx = blockUbCount / tilingData_->outHWrectAngles;
    int64_t hUbCountOffset = (blockUbCount - ncIdx * tilingData_->outHWrectAngles) / wUbCount;
    int64_t wUbCountOffset = blockUbCount - ncIdx * tilingData_->outHWrectAngles - hUbCountOffset * wUbCount;
    int64_t ubCountOffset = 0;      // 当前HW的ub块索引
    int64_t matrixWUbCntOffset = 0; // 当前HW的矩阵块W方向索引
    int64_t matrixHUbCntOffset = 0; // 当前HW的矩阵块H方向索引
    int64_t perHUbMatrixCntReal = 0;
    int64_t perWUbMatrixCntReal = 0;
    while (procUbCount > 0) {
        ubFactorW = tilingData_->ubFactorW;
        ubFactorH = tilingData_->ubFactorH;
        ubCountOffset = hUbCountOffset * wUbCount + wUbCountOffset;
        matrixWUbCntOffset = wUbCountOffset * perWUbMatrixCnt;
        matrixHUbCntOffset = hUbCountOffset * perHUbMatrixCnt;
        perHUbMatrixCntReal = inputInfo_->hKernelSize - matrixHUbCntOffset >= perHUbMatrixCnt ?
                                  perHUbMatrixCnt :
                                  inputInfo_->hKernelSize - matrixHUbCntOffset;
        perWUbMatrixCntReal = kernelNumH_ - matrixWUbCntOffset >= perWUbMatrixCnt ? perWUbMatrixCnt :
                                                                                    kernelNumH_ - matrixWUbCntOffset;
        outWOffset_ = matrixWUbCntOffset * kernelNumW_;
        outHOffset_ = matrixHUbCntOffset * inputInfo_->wKernelSize;
        inWOffset_ = (outWOffset_ - outWOffset_ / kernelNumW_ * kernelNumW_) * inputInfo_->wStride;
        inHOffset_ = outWOffset_ / kernelNumW_ * inputInfo_->hStride +
                     outHOffset_ / inputInfo_->wKernelSize * inputInfo_->hDilation;
        xGmOffset_ = ncIdx * inputInfo_->H * inputInfo_->W + inHOffset_ * inputInfo_->W + inWOffset_;
        yGmOffset_ = ncIdx * outW_ * outH_ + outHOffset_ * outW_ + outWOffset_;
        int64_t blockLen = (inputInfo_->W - inWOffset_) >= tilingData_->w4ubFactorW ? tilingData_->w4ubFactorW :
                                                                                      (inputInfo_->W - inWOffset_);
        ubFactorW = wUbCountOffset == wUbCount - 1 ? outW_ - outWOffset_ : ubFactorW;
        ubFactorH = hUbCountOffset == hUbCount - 1 ? outH_ - outHOffset_ : ubFactorH;
        int64_t blockLenAlign = (blockLen + BLOCK_ELENUM - 1) / BLOCK_ELENUM * BLOCK_ELENUM;
        int64_t srcStride = (inputInfo_->hStride - 1) * inputInfo_->W + inputInfo_->W - blockLen;
        int64_t dstStride = blockLenAlign * (perHUbMatrixCntReal - 1);
        int64_t blockCount = perWUbMatrixCntReal;
        loopMode_.loop1Size = perHUbMatrixCntReal;
        loopMode_.loop1SrcStride = inputInfo_->W * inputInfo_->hDilation * dtypeSize_;
        loopMode_.loop1DstStride = blockLenAlign * dtypeSize_;
        LocalTensor<T> xUb = inQueueX_.AllocTensor<T>();
        DataCopyIn(blockCount, blockLen, xGmOffset_, srcStride, dstStride, xUb);
        LocalTensor<T> srcUb = inQueueX_.DeQue<T>();
        if (wUbCount == 1 && outW_ <= maxGatherNum_) {
            CalcOneUBIdxAndGather(1, perHUbMatrixCntReal, blockLen, kernelNumW_, inputInfo_->wKernelSize, ubFactorW,
                                  srcUb);
            yUb_ = outQueueY_.DeQue<T>();
            DataCopyOut(0, 1, ubFactorW * ubFactorH, yGmOffset_, 0, 0, yUb_);
        } else {
            CalcIdxAndGather(perWUbMatrixCntReal, perHUbMatrixCntReal, blockLen, kernelNumW_, inputInfo_->wKernelSize,
                             ubFactorW, srcUb);
            yUb_ = outQueueY_.DeQue<T>();
            DataCopyOut(0, ubFactorH, ubFactorW, yGmOffset_, 0, outW_ - ubFactorW, yUb_);
        }
        if (wUbCountOffset == wUbCount - 1) {
            wUbCountOffset = 0;
            hUbCountOffset = hUbCountOffset == hUbCount - 1 ? 0 : hUbCountOffset + 1;
            ncIdx = hUbCountOffset == 0 ? ncIdx + 1 : ncIdx;
        } else {
            wUbCountOffset++;
        }
        procUbCount--;
        outQueueY_.FreeTensor(yUb_);
        inQueueX_.FreeTensor(srcUb);
    }
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::PadInWInH()
{
    int32_t ubFactorH = tilingData_->ubFactorH;
    int32_t ubFactorW = tilingData_->ubFactorW;
    int32_t procUbCount = rectAnglesPerCore_;
    int64_t perMatrixWUbCnt = Ops::Base::CeilDiv(kernelNumW_, static_cast<int64_t>(ubFactorW));
    int64_t perMatrixHUbCnt = Ops::Base::CeilDiv(inputInfo_->wKernelSize, static_cast<int64_t>(ubFactorH));
    int64_t wUbCount = perMatrixWUbCnt * kernelNumH_;
    int64_t hUbCount = perMatrixHUbCnt * inputInfo_->hKernelSize;
    int64_t blockUbCount = tilingData_->rectAnglesPerCore * blockIdx_;
    int64_t ncIdx = blockUbCount / tilingData_->outHWrectAngles;
    int64_t hUbCountOffset = (blockUbCount - ncIdx * tilingData_->outHWrectAngles) / wUbCount;
    int64_t wUbCountOffset = blockUbCount - ncIdx * tilingData_->outHWrectAngles - hUbCountOffset * wUbCount;
    int64_t ubCountOffset = 0;      // 当前HW的ub块索引
    int64_t matrixWUbCntOffset = 0; // 当前HW的矩阵块W方向索引
    int64_t matrixHUbCntOffset = 0; // 当前HW的矩阵块H方向索引
    int64_t nowMatrixWUbOffset = 0; // 矩阵块内W方向UB索引
    int64_t nowMatrixHUbOffset = 0; // 矩阵块内H方向UB索引
    int64_t wPad = inputInfo_->W + inputInfo_->wPaddingBefore + inputInfo_->wPaddingAfter;
    int32_t inIdx = 0;
    Duplicate(inTensorX_, static_cast<T>(0), BUFFER_NUM * inputElemNum_);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);

    while (procUbCount > 0) {
        ubFactorW = tilingData_->ubFactorW;
        ubFactorH = tilingData_->ubFactorH;
        ubCountOffset = hUbCountOffset * wUbCount + wUbCountOffset;
        matrixWUbCntOffset = wUbCountOffset / perMatrixWUbCnt;
        matrixHUbCntOffset = hUbCountOffset / perMatrixHUbCnt;
        nowMatrixWUbOffset = (wUbCountOffset - matrixWUbCntOffset * perMatrixWUbCnt) * ubFactorW;
        nowMatrixHUbOffset = (hUbCountOffset - matrixHUbCntOffset * perMatrixHUbCnt) * ubFactorH;
        outWOffset_ = matrixWUbCntOffset * kernelNumW_ + nowMatrixWUbOffset;
        outHOffset_ = matrixHUbCntOffset * inputInfo_->wKernelSize + nowMatrixHUbOffset;
        inWOffsetPad_ = (outWOffset_ - outWOffset_ / kernelNumW_ * kernelNumW_) * inputInfo_->wStride +
                        nowMatrixHUbOffset * inputInfo_->wDilation;
        wPadTop_ = inputInfo_->wPaddingBefore - inWOffsetPad_ > 0 ? inputInfo_->wPaddingBefore - inWOffsetPad_ : 0;
        inHOffsetPad_ = outWOffset_ / kernelNumW_ * inputInfo_->hStride +
                        outHOffset_ / inputInfo_->wKernelSize * inputInfo_->hDilation;
        hPadTop_ = inputInfo_->hPaddingBefore - inHOffsetPad_ > 0 ? inputInfo_->hPaddingBefore - inHOffsetPad_ : 0;
        inWOffset_ = inWOffsetPad_ - inputInfo_->wPaddingBefore + wPadTop_;
        inHOffset_ = inHOffsetPad_ - inputInfo_->hPaddingBefore + hPadTop_;
        xGmOffset_ = ncIdx * inputInfo_->H * inputInfo_->W + inHOffset_ * inputInfo_->W + inWOffset_;
        xGmOffsetPad_ = inHOffsetPad_ * (inputInfo_->W + inputInfo_->wPaddingBefore + inputInfo_->wPaddingAfter) +
                        inWOffsetPad_;
        yGmOffset_ = ncIdx * outW_ * outH_ + outHOffset_ * outW_ + outWOffset_;
        int64_t blockLen = (wPad - inWOffsetPad_) >= tilingData_->w4ubFactorW ? tilingData_->w4ubFactorW :
                                                                                (wPad - inWOffsetPad_);
        ubFactorW = nowMatrixWUbOffset == (perMatrixWUbCnt - 1) * ubFactorW ? kernelNumW_ - nowMatrixWUbOffset :
                                                                              ubFactorW;
        ubFactorH = nowMatrixHUbOffset == (perMatrixHUbCnt - 1) * ubFactorH ?
                        inputInfo_->wKernelSize - nowMatrixHUbOffset :
                        ubFactorH;
        if (inHOffsetPad_ < inputInfo_->hPaddingBefore ||
            inHOffsetPad_ >= (inputInfo_->H + inputInfo_->hPaddingBefore) ||
            inWOffsetPad_ >= (inputInfo_->W + inputInfo_->wPaddingBefore) ||
            inWOffsetPad_ + blockLen <= inputInfo_->wPaddingBefore) {
            DataCopyOutZero(ubFactorH, ubFactorW, yGmOffset_, 0, outW_ - ubFactorW, procUbCount);
            if (wUbCountOffset == wUbCount - 1) {
                wUbCountOffset = 0;
                hUbCountOffset = hUbCountOffset == hUbCount - 1 ? 0 : hUbCountOffset + 1;
                ncIdx = hUbCountOffset == 0 ? ncIdx + 1 : ncIdx;
            } else {
                wUbCountOffset++;
            }
            procUbCount--;
            continue;
        }
        LocalTensor<T> srcUb = inTensorX_[(inIdx & 1) * inputElemNum_];
        IsDataCopyPad(1, 1, blockLen, inIdx, srcUb);
        CalcIdxAndGather(1, 1, blockLen, ubFactorW, ubFactorH, ubFactorW, srcUb);
        LocalTensor<T> yUb = outQueueY_.DeQue<T>();
        DataCopyOut(0, ubFactorH, ubFactorW, yGmOffset_, 0, outW_ - ubFactorW, yUb);
        if (wUbCountOffset == wUbCount - 1) {
            wUbCountOffset = 0;
            hUbCountOffset = hUbCountOffset == hUbCount - 1 ? 0 : hUbCountOffset + 1;
            ncIdx = hUbCountOffset == 0 ? ncIdx + 1 : ncIdx;
        } else {
            wUbCountOffset++;
        }
        procUbCount--;
        inIdx++;
        outQueueY_.FreeTensor(yUb);
    }
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::PadOutWInH()
{
    int32_t ubFactorH = tilingData_->ubFactorH;
    int32_t ubFactorW = tilingData_->ubFactorW;
    int32_t procUbCount = rectAnglesPerCore_;
    int64_t perMatrixHUbCnt = Ops::Base::CeilDiv(inputInfo_->wKernelSize, static_cast<int64_t>(ubFactorH));
    int64_t perWUbMatrixCnt = Ops::Base::CeilDiv(static_cast<int64_t>(ubFactorW), kernelNumW_);
    int64_t wUbCount = Ops::Base::CeilDiv(outW_, static_cast<int64_t>(ubFactorW));
    int64_t hUbCount = perMatrixHUbCnt * inputInfo_->hKernelSize;
    int64_t blockUbCount = tilingData_->rectAnglesPerCore * blockIdx_;
    int64_t ncIdx = blockUbCount / tilingData_->outHWrectAngles;
    int64_t hUbCountOffset = (blockUbCount - ncIdx * tilingData_->outHWrectAngles) / wUbCount; // H方向上UB块偏移
    int64_t wUbCountOffset = blockUbCount - ncIdx * tilingData_->outHWrectAngles -
                             hUbCountOffset * wUbCount; // W方向上UB块偏移
    int64_t ubCountOffset = 0;                          // 当前HW的ub块索引
    int64_t matrixWUbCntOffset = 0;                     // 当前HW的矩阵块W方向索引
    int64_t matrixHUbCntOffset = 0;                     // 当前HW的矩阵块H方向索引
    int64_t nowMatrixHUbOffset = 0;                     // 矩阵块内H方向UB索引
    int64_t inKernelHOffset = 0;                        // 输入H方向上卷积核所在行数
    int64_t perWUbMatrixCntReal = 0;
    int64_t wPad = inputInfo_->W + inputInfo_->wPaddingBefore + inputInfo_->wPaddingAfter;
    int32_t inIdx = 0;
    Duplicate(inTensorX_, static_cast<T>(0), BUFFER_NUM * inputElemNum_);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);

    while (procUbCount > 0) {
        ubFactorW = tilingData_->ubFactorW;
        ubFactorH = tilingData_->ubFactorH;
        ubCountOffset = hUbCountOffset * wUbCount + wUbCountOffset;
        matrixWUbCntOffset = wUbCountOffset * perWUbMatrixCnt;
        matrixHUbCntOffset = hUbCountOffset / perMatrixHUbCnt;
        nowMatrixHUbOffset = (hUbCountOffset - matrixHUbCntOffset * perMatrixHUbCnt) * ubFactorH;
        perWUbMatrixCntReal = kernelNumH_ - matrixWUbCntOffset >= perWUbMatrixCnt ? perWUbMatrixCnt :
                                                                                    kernelNumH_ - matrixWUbCntOffset;
        outWOffset_ = matrixWUbCntOffset * kernelNumW_;
        outHOffset_ = matrixHUbCntOffset * inputInfo_->wKernelSize + nowMatrixHUbOffset;
        inWOffsetPad_ = (outWOffset_ - outWOffset_ / kernelNumW_ * kernelNumW_) * inputInfo_->wStride +
                        nowMatrixHUbOffset * inputInfo_->wDilation;
        wPadTop_ = inputInfo_->wPaddingBefore - inWOffsetPad_ > 0 ? inputInfo_->wPaddingBefore - inWOffsetPad_ : 0;
        inHOffsetPad_ = outWOffset_ / kernelNumW_ * inputInfo_->hStride +
                        outHOffset_ / inputInfo_->wKernelSize * inputInfo_->hDilation;
        hPadTop_ = inputInfo_->hPaddingBefore - inHOffsetPad_ > 0 ? inputInfo_->hPaddingBefore - inHOffsetPad_ : 0;
        inWOffset_ = inWOffsetPad_ - inputInfo_->wPaddingBefore + wPadTop_;
        inHOffset_ = inHOffsetPad_ - inputInfo_->hPaddingBefore + hPadTop_;

        xGmOffset_ = ncIdx * inputInfo_->H * inputInfo_->W + inHOffset_ * inputInfo_->W + inWOffset_;
        yGmOffset_ = ncIdx * outW_ * outH_ + outHOffset_ * outW_ + outWOffset_;
        int64_t blockLen = (wPad - inWOffsetPad_) >= tilingData_->w4ubFactorW ? tilingData_->w4ubFactorW :
                                                                                (wPad - inWOffsetPad_);
        ubFactorW = wUbCountOffset == wUbCount - 1 ? outW_ - outWOffset_ : ubFactorW;
        ubFactorH = nowMatrixHUbOffset == (perMatrixHUbCnt - 1) * ubFactorH ?
                        inputInfo_->wKernelSize - nowMatrixHUbOffset :
                        ubFactorH;
        bool isHInPad = inHOffsetPad_ + inputInfo_->hStride * (perWUbMatrixCntReal - 1) < inputInfo_->hPaddingBefore;
        if (isHInPad || inHOffsetPad_ >= (inputInfo_->H + inputInfo_->hPaddingBefore) ||
            inWOffsetPad_ >= (inputInfo_->W + inputInfo_->wPaddingBefore) ||
            inWOffsetPad_ + blockLen <= inputInfo_->wPaddingBefore) {
            DataCopyOutZero(ubFactorH, ubFactorW, yGmOffset_, 0, outW_ - ubFactorW, procUbCount);
            if (wUbCountOffset == wUbCount - 1) {
                wUbCountOffset = 0;
                hUbCountOffset = hUbCountOffset == hUbCount - 1 ? 0 : hUbCountOffset + 1;
                ncIdx = hUbCountOffset == 0 ? ncIdx + 1 : ncIdx;
            } else {
                wUbCountOffset++;
            }
            procUbCount--;
            continue;
        }

        LocalTensor<T> srcUb = inTensorX_[(inIdx & 1) * inputElemNum_];
        IsDataCopyPad(perWUbMatrixCntReal, 1, blockLen, inIdx, srcUb);
        if (wUbCount == 1 && outW_ <= maxGatherNum_) {
            CalcOneUBIdxAndGather(1, 1, blockLen, kernelNumW_, ubFactorH, ubFactorW, srcUb);
            yUb_ = outQueueY_.DeQue<T>();
            DataCopyOut(0, 1, ubFactorW * ubFactorH, yGmOffset_, 0, 0, yUb_);
        } else {
            CalcIdxAndGather(perWUbMatrixCntReal, 1, blockLen, kernelNumW_, ubFactorH, ubFactorW, srcUb);
            yUb_ = outQueueY_.DeQue<T>();
            DataCopyOut(0, ubFactorH, ubFactorW, yGmOffset_, 0, outW_ - ubFactorW, yUb_);
        }
        if (wUbCountOffset == wUbCount - 1) {
            wUbCountOffset = 0;
            hUbCountOffset = hUbCountOffset == hUbCount - 1 ? 0 : hUbCountOffset + 1;
            ncIdx = hUbCountOffset == 0 ? ncIdx + 1 : ncIdx;
        } else {
            wUbCountOffset++;
        }
        procUbCount--;
        inIdx++;
        outQueueY_.FreeTensor(yUb_);
    }
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::PadInWOutH()
{
    int32_t ubFactorH = tilingData_->ubFactorH;
    int32_t ubFactorW = tilingData_->ubFactorW;
    int32_t procUbCount = rectAnglesPerCore_;
    int64_t perHUbMatrixCnt = Ops::Base::CeilDiv(static_cast<int64_t>(ubFactorH), inputInfo_->wKernelSize);
    int64_t perMatrixWUbCnt = Ops::Base::CeilDiv(kernelNumW_, static_cast<int64_t>(ubFactorW));
    int64_t wUbCount = kernelNumH_ * perMatrixWUbCnt;
    int64_t hUbCount = Ops::Base::CeilDiv(inputInfo_->wKernelSize * inputInfo_->hKernelSize,
                                          static_cast<int64_t>(ubFactorH));
    int64_t blockUbCount = tilingData_->rectAnglesPerCore * blockIdx_;
    int64_t ncIdx = blockUbCount / tilingData_->outHWrectAngles;
    int64_t hUbCountOffset = (blockUbCount - ncIdx * tilingData_->outHWrectAngles) / wUbCount;
    int64_t wUbCountOffset = blockUbCount - ncIdx * tilingData_->outHWrectAngles - hUbCountOffset * wUbCount;
    int64_t ubCountOffset = 0;      // 当前HW的ub块索引
    int64_t matrixWUbCntOffset = 0; // 当前HW的矩阵块W方向索引
    int64_t matrixHUbCntOffset = 0; // 当前HW的矩阵块H方向索引
    int64_t nowMatrixWUbOffset = 0; // 矩阵块内W方向UB索引
    int64_t perHUbMatrixCntReal = 0;
    int64_t wPad = inputInfo_->W + inputInfo_->wPaddingBefore + inputInfo_->wPaddingAfter;
    int32_t inIdx = 0;
    Duplicate(inTensorX_, static_cast<T>(0), BUFFER_NUM * inputElemNum_);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    while (procUbCount > 0) {
        ubFactorW = tilingData_->ubFactorW;
        ubFactorH = tilingData_->ubFactorH;
        ubCountOffset = hUbCountOffset * wUbCount + wUbCountOffset;
        matrixWUbCntOffset = wUbCountOffset / perMatrixWUbCnt;
        matrixHUbCntOffset = hUbCountOffset * perHUbMatrixCnt;
        nowMatrixWUbOffset = (wUbCountOffset - matrixWUbCntOffset * perMatrixWUbCnt) * ubFactorW;
        perHUbMatrixCntReal = inputInfo_->hKernelSize - matrixHUbCntOffset >= perHUbMatrixCnt ?
                                  perHUbMatrixCnt :
                                  inputInfo_->hKernelSize - matrixHUbCntOffset;
        outWOffset_ = matrixWUbCntOffset * kernelNumW_ + nowMatrixWUbOffset;
        outHOffset_ = matrixHUbCntOffset * inputInfo_->wKernelSize;
        inWOffsetPad_ = (outWOffset_ - outWOffset_ / kernelNumW_ * kernelNumW_) * inputInfo_->wStride;
        wPadTop_ = inputInfo_->wPaddingBefore - inWOffsetPad_ > 0 ? inputInfo_->wPaddingBefore - inWOffsetPad_ : 0;
        inHOffsetPad_ = outWOffset_ / kernelNumW_ * inputInfo_->hStride +
                        outHOffset_ / inputInfo_->wKernelSize * inputInfo_->hDilation;
        hPadTop_ = inputInfo_->hPaddingBefore - inHOffsetPad_ > 0 ? inputInfo_->hPaddingBefore - inHOffsetPad_ : 0;
        inWOffset_ = inWOffsetPad_ - inputInfo_->wPaddingBefore + wPadTop_;
        inHOffset_ = inHOffsetPad_ - inputInfo_->hPaddingBefore + hPadTop_;
        xGmOffset_ = ncIdx * inputInfo_->H * inputInfo_->W + inHOffset_ * inputInfo_->W + inWOffset_;
        yGmOffset_ = ncIdx * outW_ * outH_ + outHOffset_ * outW_ + outWOffset_;
        int64_t blockLen = (wPad - inWOffsetPad_) >= tilingData_->w4ubFactorW ? tilingData_->w4ubFactorW :
                                                                                (wPad - inWOffsetPad_);
        ubFactorW = nowMatrixWUbOffset == (perMatrixWUbCnt - 1) * ubFactorW ? kernelNumW_ - nowMatrixWUbOffset :
                                                                              ubFactorW;
        ubFactorH = hUbCountOffset == hUbCount - 1 ? outH_ - outHOffset_ : ubFactorH;
        bool isHInPad = inHOffsetPad_ + inputInfo_->hDilation * (perHUbMatrixCntReal - 1) < inputInfo_->hPaddingBefore;
        if (isHInPad || inHOffsetPad_ >= (inputInfo_->H + inputInfo_->hPaddingBefore) ||
            inWOffsetPad_ >= (inputInfo_->W + inputInfo_->wPaddingBefore) ||
            inWOffsetPad_ + blockLen <= inputInfo_->wPaddingBefore) {
            DataCopyOutZero(ubFactorH, ubFactorW, yGmOffset_, 0, outW_ - ubFactorW, procUbCount);
            if (wUbCountOffset == wUbCount - 1) {
                wUbCountOffset = 0;
                hUbCountOffset = hUbCountOffset == hUbCount - 1 ? 0 : hUbCountOffset + 1;
                ncIdx = hUbCountOffset == 0 ? ncIdx + 1 : ncIdx;
            } else {
                wUbCountOffset++;
            }
            procUbCount--;
            continue;
        }

        LocalTensor<T> srcUb = inTensorX_[(inIdx & 1) * inputElemNum_];
        IsDataCopyPad(1, perHUbMatrixCntReal, blockLen, inIdx, srcUb);
        CalcIdxAndGather(1, perHUbMatrixCntReal, blockLen, ubFactorW, inputInfo_->wKernelSize, ubFactorW, srcUb);
        LocalTensor<T> yUb = outQueueY_.DeQue<T>();
        DataCopyOut(0, ubFactorH, ubFactorW, yGmOffset_, 0, outW_ - ubFactorW, yUb);
        if (wUbCountOffset == wUbCount - 1) {
            wUbCountOffset = 0;
            hUbCountOffset = hUbCountOffset == hUbCount - 1 ? 0 : hUbCountOffset + 1;
            ncIdx = hUbCountOffset == 0 ? ncIdx + 1 : ncIdx;
        } else {
            wUbCountOffset++;
        }
        procUbCount--;
        inIdx++;
        outQueueY_.FreeTensor(yUb);
    }
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::PadOutWOutH()
{
    int32_t ubFactorH = tilingData_->ubFactorH;
    int32_t ubFactorW = tilingData_->ubFactorW;
    int32_t procUbCount = rectAnglesPerCore_;
    int64_t perHUbMatrixCnt = Ops::Base::CeilDiv(static_cast<int64_t>(ubFactorH), inputInfo_->wKernelSize);
    int64_t perWUbMatrixCnt = Ops::Base::CeilDiv(static_cast<int64_t>(ubFactorW), kernelNumW_);
    int64_t wUbCount = Ops::Base::CeilDiv(outW_, static_cast<int64_t>(ubFactorW));
    int64_t hUbCount = Ops::Base::CeilDiv(inputInfo_->wKernelSize * inputInfo_->hKernelSize,
                                          static_cast<int64_t>(ubFactorH));
    int64_t blockUbCount = tilingData_->rectAnglesPerCore * blockIdx_;
    int64_t ncIdx = blockUbCount / tilingData_->outHWrectAngles;
    int64_t hUbCountOffset = (blockUbCount - ncIdx * tilingData_->outHWrectAngles) / wUbCount;
    int64_t wUbCountOffset = blockUbCount - ncIdx * tilingData_->outHWrectAngles - hUbCountOffset * wUbCount;
    int64_t ubCountOffset = 0;      // 当前HW的ub块索引
    int64_t matrixWUbCntOffset = 0; // 当前HW的矩阵块W方向索引
    int64_t matrixHUbCntOffset = 0; // 当前HW的矩阵块H方向索引
    int64_t perHUbMatrixCntReal = 0;
    int64_t perWUbMatrixCntReal = 0;
    int64_t wPad = inputInfo_->W + inputInfo_->wPaddingBefore + inputInfo_->wPaddingAfter;
    int32_t inIdx = 0;
    Duplicate(inTensorX_, static_cast<T>(0), BUFFER_NUM * inputElemNum_);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    while (procUbCount > 0) {
        ubFactorW = tilingData_->ubFactorW;
        ubFactorH = tilingData_->ubFactorH;
        ubCountOffset = hUbCountOffset * wUbCount + wUbCountOffset;
        matrixWUbCntOffset = wUbCountOffset * perWUbMatrixCnt;
        matrixHUbCntOffset = hUbCountOffset * perHUbMatrixCnt;
        perHUbMatrixCntReal = inputInfo_->hKernelSize - matrixHUbCntOffset >= perHUbMatrixCnt ?
                                  perHUbMatrixCnt :
                                  inputInfo_->hKernelSize - matrixHUbCntOffset;
        perWUbMatrixCntReal = kernelNumH_ - matrixWUbCntOffset >= perWUbMatrixCnt ? perWUbMatrixCnt :
                                                                                    kernelNumH_ - matrixWUbCntOffset;
        outWOffset_ = matrixWUbCntOffset * kernelNumW_;
        outHOffset_ = matrixHUbCntOffset * inputInfo_->wKernelSize;
        inWOffsetPad_ = (outWOffset_ - outWOffset_ / kernelNumW_ * kernelNumW_) * inputInfo_->wStride;
        wPadTop_ = inputInfo_->wPaddingBefore - inWOffsetPad_ > 0 ? inputInfo_->wPaddingBefore - inWOffsetPad_ : 0;
        inHOffsetPad_ = outWOffset_ / kernelNumW_ * inputInfo_->hStride +
                        outHOffset_ / inputInfo_->wKernelSize * inputInfo_->hDilation;
        hPadTop_ = inputInfo_->hPaddingBefore - inHOffsetPad_ > 0 ? inputInfo_->hPaddingBefore - inHOffsetPad_ : 0;
        inWOffset_ = inWOffsetPad_ - inputInfo_->wPaddingBefore + wPadTop_;
        inHOffset_ = inHOffsetPad_ - inputInfo_->hPaddingBefore + hPadTop_;
        xGmOffset_ = ncIdx * inputInfo_->H * inputInfo_->W + inHOffset_ * inputInfo_->W + inWOffset_;
        yGmOffset_ = ncIdx * outW_ * outH_ + outHOffset_ * outW_ + outWOffset_;
        int64_t blockLen = (wPad - inWOffsetPad_) >= tilingData_->w4ubFactorW ? tilingData_->w4ubFactorW :
                                                                                (wPad - inWOffsetPad_);
        ubFactorW = wUbCountOffset == wUbCount - 1 ? outW_ - outWOffset_ : ubFactorW;
        ubFactorH = hUbCountOffset == hUbCount - 1 ? outH_ - outHOffset_ : ubFactorH;

        int64_t hMaxOffsetPad = inHOffsetPad_ + inputInfo_->hDilation * (perHUbMatrixCntReal - 1) +
                                inputInfo_->hStride * (perWUbMatrixCntReal - 1);
        bool isHInPad = hMaxOffsetPad < inputInfo_->hPaddingBefore;
        if (isHInPad || inHOffsetPad_ >= (inputInfo_->H + inputInfo_->hPaddingBefore) ||
            inWOffsetPad_ >= (inputInfo_->W + inputInfo_->wPaddingBefore) ||
            inWOffsetPad_ + blockLen <= inputInfo_->wPaddingBefore) {
            DataCopyOutZero(ubFactorH, ubFactorW, yGmOffset_, 0, outW_ - ubFactorW, procUbCount);
            if (wUbCountOffset == wUbCount - 1) {
                wUbCountOffset = 0;
                hUbCountOffset = hUbCountOffset == hUbCount - 1 ? 0 : hUbCountOffset + 1;
                ncIdx = hUbCountOffset == 0 ? ncIdx + 1 : ncIdx;
            } else {
                wUbCountOffset++;
            }
            procUbCount--;
            continue;
        }

        LocalTensor<T> srcUb = inTensorX_[(inIdx & 1) * inputElemNum_];
        IsDataCopyPad(perWUbMatrixCntReal, perHUbMatrixCntReal, blockLen, inIdx, srcUb);
        if (wUbCount == 1 && outW_ <= maxGatherNum_) {
            CalcOneUBIdxAndGather(1, perHUbMatrixCntReal, blockLen, kernelNumW_, inputInfo_->wKernelSize, ubFactorW,
                                  srcUb);
            yUb_ = outQueueY_.DeQue<T>();
            DataCopyOut(0, 1, ubFactorW * ubFactorH, yGmOffset_, 0, 0, yUb_);
        } else {
            CalcIdxAndGather(perWUbMatrixCntReal, perHUbMatrixCntReal, blockLen, kernelNumW_, inputInfo_->wKernelSize,
                             ubFactorW, srcUb);
            yUb_ = outQueueY_.DeQue<T>();
            DataCopyOut(0, ubFactorH, ubFactorW, yGmOffset_, 0, outW_ - ubFactorW, yUb_);
        }
        if (wUbCountOffset == wUbCount - 1) {
            wUbCountOffset = 0;
            hUbCountOffset = hUbCountOffset == hUbCount - 1 ? 0 : hUbCountOffset + 1;
            ncIdx = hUbCountOffset == 0 ? ncIdx + 1 : ncIdx;
        } else {
            wUbCountOffset++;
        }
        procUbCount--;
        inIdx++;
        outQueueY_.FreeTensor(yUb_);
    }
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::ProcessPad()
{
    if (ubFactorW_ < tilingData_->convKernelNumInWidth && ubFactorH_ < inputInfo_->wKernelSize) {
        // W和H都截断
        PadInWInH();
    } else if (ubFactorW_ >= tilingData_->convKernelNumInWidth && ubFactorH_ < inputInfo_->wKernelSize) {
        // W不截断，H截断
        PadOutWInH();
    } else if (ubFactorW_ < tilingData_->convKernelNumInWidth && ubFactorH_ >= inputInfo_->wKernelSize) {
        // W截断，H不截断
        PadInWOutH();
    } else {
        // W和H都不截断
        PadOutWOutH();
    }
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::ProcessNoPad()
{
    if (ubFactorW_ < tilingData_->convKernelNumInWidth && ubFactorH_ < inputInfo_->wKernelSize) {
        // W和H都截断
        NoPadInWInH();
    } else if (ubFactorW_ >= tilingData_->convKernelNumInWidth && ubFactorH_ < inputInfo_->wKernelSize) {
        // W不截断，H截断
        NoPadOutWInH();
    } else if (ubFactorW_ < tilingData_->convKernelNumInWidth && ubFactorH_ >= inputInfo_->wKernelSize) {
        // W截断，H不截断
        NoPadInWOutH();
    } else {
        // W和H都不截断
        NoPadOutWOutH();
    }
}

template <typename T, typename U, typename Y, bool isPadding>
__aicore__ inline void Im2colGatherCutHw<T, U, Y, isPadding>::Process()
{
    if (blockIdx_ * tilingData_->rectAnglesPerCore > tilingData_->totalRectAngles) {
        return;
    }
    rectAnglesPerCore_ = tilingData_->totalRectAngles - blockIdx_ * tilingData_->rectAnglesPerCore <
                                 tilingData_->rectAnglesPerCore ?
                             tilingData_->totalRectAngles - blockIdx_ * tilingData_->rectAnglesPerCore :
                             tilingData_->rectAnglesPerCore;
    if constexpr (isPadding) {
        ProcessPad();
    } else {
        ProcessNoPad();
    }
}
} // namespace Im2col
#endif // namespace Im2col
