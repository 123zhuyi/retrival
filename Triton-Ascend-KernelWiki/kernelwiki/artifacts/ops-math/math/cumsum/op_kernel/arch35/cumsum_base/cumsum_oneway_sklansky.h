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
 * \file cumsum_oneway_sklansky.h
 * \brief calculate the oneway sklansky cumsum
 */

#ifndef CANN_OPS_BUILT_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_BASE_CUMSUM_ONEWAY_SKLANSKY_H
#define CANN_OPS_BUILT_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_BASE_CUMSUM_ONEWAY_SKLANSKY_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "cumsum_base.h"

namespace Cumsum {
using namespace AscendC;

struct OnewaySklanskyInitData {
    int64_t mLen = 0;
    int64_t rLen = 0;
    int64_t nLen = 0;
    int32_t ubBufferSize = 0;
    int32_t mFold = 1;
    bool reverse = false;
};

struct OnewaySklanskyProcessData {
    int64_t mOffset = 0;
    int64_t rOffset = 0;
    int64_t nOffset = 0;
    int32_t mFactor = 0;
    int32_t rFactor = 0;
    int32_t nFactor = 0;
    bool exclusive = false;
    bool isLenChange = false;
};

template <typename DataType, typename PromoteDataType>
class CumsumOnewaySklansky : public CumsumBase<DataType> {
public:
    __aicore__ inline CumsumOnewaySklansky(TPipe& pipe) : CumsumBase<DataType>(pipe) {}
    __aicore__ inline void BaseInit(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const OnewaySklanskyInitData& param);
    __aicore__ inline void BaseProcessPre(const OnewaySklanskyProcessData& param);
    __aicore__ inline void BaseProcess();
    __aicore__ inline void BaseCopyOut();
    __aicore__ inline LocalTensor<PromoteDataType>& GetBaseResTensor();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyInExclusive();
    __aicore__ inline void CopyInDataCast();
    __aicore__ inline void Compute();
    __aicore__ inline void ComputeWithFlag(int32_t addOffset, uint32_t realDupSize, __local_mem__ PromoteDataType* xBuf,
                                           int32_t flag);
    __aicore__ inline void ComputeInner(int32_t addLoop, uint32_t realDupSize);

private:
    constexpr static int32_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
    constexpr static uint32_t ADD_SIZE = Ops::Base::GetVRegSize();
    constexpr static uint32_t ADD_ELEM = ADD_SIZE / sizeof(PromoteDataType);
    int64_t lenM_ = 0;
    int64_t lenR_ = 0;
    int64_t lenN_ = 0;
    int64_t mOffset_ = 0;
    int64_t rOffset_ = 0;
    int64_t nOffset_ = 0;
    int32_t mFactor_ = 0;
    int32_t nFactor_ = 0;
    int32_t rFactor_ = 0;
    int32_t dupSize_ = 0;
    uint16_t addCount_ = 0;
    uint16_t groupMainCount_ = 0;
    uint16_t groupTailCount_ = 0;
    uint16_t addTailCount_ = 0;
    uint32_t nAddFactor_ = 0;
    int32_t mFold_ = 1;
    bool reverse_ = false;
    bool exclusive_ = false;
    bool isLenChange_ = true;
    TBuf<> inBufX_;
    TBuf<> inCastBufX_;
    LocalTensor<DataType> inTensorX_;
    LocalTensor<PromoteDataType> inCastTensorX_;
    LocalTensor<PromoteDataType> computerBuffer_;
    bool isFirstCopyIn_ = true;
};

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumOnewaySklansky<DataType, PromoteDataType>::BaseInit(GM_ADDR x, GM_ADDR y,
                                                                                 GM_ADDR workspace,
                                                                                 const OnewaySklanskyInitData& param)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    lenM_ = param.mLen;
    lenR_ = param.rLen;
    lenN_ = param.nLen;
    mFold_ = param.mFold;
    reverse_ = param.reverse;
    this->pipe_.InitBuffer(inBufX_, param.ubBufferSize);
    if constexpr (!IsSameType<DataType, PromoteDataType>::value) {
        this->pipe_.InitBuffer(inCastBufX_, param.ubBufferSize * sizeof(PromoteDataType) / sizeof(DataType));
        inCastTensorX_ = inCastBufX_.template Get<PromoteDataType>();
    }
    inTensorX_ = inBufX_.template Get<DataType>();
    this->Init(x, y);
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumOnewaySklansky<DataType, PromoteDataType>::BaseProcessPre(
    const OnewaySklanskyProcessData& param)
{
    mOffset_ = param.mOffset;
    rOffset_ = param.rOffset;
    nOffset_ = param.nOffset;
    mFactor_ = param.mFactor;
    rFactor_ = param.rFactor;
    nFactor_ = param.nFactor;
    exclusive_ = param.exclusive;
    isLenChange_ = param.isLenChange;
    dupSize_ = Ops::Base::CeilAlign(static_cast<int32_t>(nFactor_ * sizeof(DataType)), BLOCK_SIZE);
    nAddFactor_ = dupSize_ / sizeof(DataType);
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumOnewaySklansky<DataType, PromoteDataType>::CopyIn()
{
    DataCopyPadExtParams<DataType> dataCopyPadExtParams = {false, 0, 0, 0};
    DataCopyExtParams copyParams{0, 0, 0, 0, 0};
    copyParams.blockCount = rFactor_;
    copyParams.blockLen = nFactor_ * sizeof(DataType);
    copyParams.srcStride = (lenN_ - nFactor_) * sizeof(DataType);
    copyParams.dstStride = (mFold_ - 1) * dupSize_ / BLOCK_SIZE;
    int64_t xGmOffset = 0;
    int32_t copyRowOffset = 0;
    int32_t copyColOffset = 0;
    for (int32_t i = 0; i < mFactor_; i++) {
        xGmOffset = (mOffset_ + i) * lenR_ * lenN_ + rOffset_ * lenN_ + nOffset_;
        copyRowOffset = Ops::Base::FloorDiv(i, mFold_);
        copyColOffset = i % mFold_;
        DataCopyPad(inTensorX_[copyRowOffset * rFactor_ * nAddFactor_ * mFold_ + copyColOffset * nAddFactor_],
                    this->xGm_[xGmOffset], copyParams, dataCopyPadExtParams);
    }
    CopyInDataCast();
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumOnewaySklansky<DataType, PromoteDataType>::CopyInExclusive()
{
    DataCopyPadExtParams<DataType> dataCopyPadExtParams = {false, 0, 0, 0};
    DataCopyExtParams copyParams{0, 0, 0, 0, 0};
    copyParams.blockCount = isLenChange_ ? (rFactor_ - 1) : rFactor_;
    copyParams.blockLen = nFactor_ * sizeof(DataType);
    copyParams.srcStride = (lenN_ - nFactor_) * sizeof(DataType);
    copyParams.dstStride = (mFold_ - 1) * dupSize_ / BLOCK_SIZE;
    int64_t xGmOffset = 0;
    int32_t inXOffset = 0;
    int32_t copyRowOffset = 0;
    int32_t copyColOffset = 0;
    int32_t dupOffset = 0;
    for (int32_t i = 0; i < mFactor_; i++) {
        copyRowOffset = Ops::Base::FloorDiv(i, mFold_);
        copyColOffset = i % mFold_;
        inXOffset = copyRowOffset * rFactor_ * nAddFactor_ * mFold_ + copyColOffset * nAddFactor_;
        if (isLenChange_) {
            if (reverse_) {
                if (copyParams.blockCount > 0) {
                    xGmOffset = (mOffset_ + i) * lenR_ * lenN_ + (rOffset_ + 1) * lenN_ + nOffset_;
                    DataCopyPad(inTensorX_[inXOffset], this->xGm_[xGmOffset], copyParams, dataCopyPadExtParams);
                    SetFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    WaitFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                }
                dupOffset = copyRowOffset * rFactor_ * nAddFactor_ * mFold_ + (rFactor_ - 1) * nAddFactor_ * mFold_;
                Duplicate(inTensorX_[dupOffset], static_cast<DataType>(0), nAddFactor_ * mFold_);
            } else {
                if (copyParams.blockCount > 0) {
                    xGmOffset = (mOffset_ + i) * lenR_ * lenN_ + rOffset_ * lenN_ + nOffset_;
                    inXOffset += nAddFactor_ * mFold_;
                    DataCopyPad(inTensorX_[inXOffset], this->xGm_[xGmOffset], copyParams, dataCopyPadExtParams);
                    SetFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    WaitFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                }
                dupOffset = copyRowOffset * rFactor_ * nAddFactor_ * mFold_;
                Duplicate(inTensorX_[dupOffset], static_cast<DataType>(0), nAddFactor_ * mFold_);
            }
        } else {
            if (reverse_) {
                xGmOffset = (mOffset_ + i) * lenR_ * lenN_ + (rOffset_ + 1) * lenN_ + nOffset_;
            } else {
                xGmOffset = (mOffset_ + i) * lenR_ * lenN_ + (rOffset_ - 1) * lenN_ + nOffset_;
            }
            DataCopyPad(inTensorX_[inXOffset], this->xGm_[xGmOffset], copyParams, dataCopyPadExtParams);
        }
    }
    CopyInDataCast();
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumOnewaySklansky<DataType, PromoteDataType>::CopyInDataCast()
{
    SetFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    WaitFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

    if constexpr (!IsSameType<DataType, PromoteDataType>::value) {
        Cast(inCastTensorX_, inTensorX_, RoundMode::CAST_NONE,
             Ops::Base::CeilDiv(mFactor_, mFold_) * mFold_ * dupSize_ * rFactor_ / sizeof(DataType));
        computerBuffer_ = inCastTensorX_;
    } else {
        computerBuffer_ = inTensorX_;
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumOnewaySklansky<DataType, PromoteDataType>::ComputeWithFlag(
    int32_t addOffset, uint32_t realDupSize, __local_mem__ PromoteDataType* xBuf, int32_t flag)
{
    uint16_t groupMainCount = groupMainCount_;
    uint16_t groupTailCount = groupTailCount_;
    uint16_t addCount = addCount_;
    uint16_t addTailCount = addTailCount_;
    int32_t computeLen = nAddFactor_ * mFold_;
    uint16_t mLoop = Ops::Base::CeilDiv(mFactor_, mFold_);
    int32_t mFactor = rFactor_ * computeLen;
    uint16_t nLoop = static_cast<uint16_t>(Ops::Base::CeilDiv(realDupSize, ADD_SIZE));
    int32_t realDupElem = realDupSize / sizeof(PromoteDataType);

    __VEC_SCOPE__
    {
        Reg::RegTensor<PromoteDataType> x1RegTensor;
        Reg::RegTensor<PromoteDataType> x2RegTensor;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < mLoop; i++) {
            uint32_t addTotal = realDupElem;
            for (uint16_t j = 0; j < nLoop; j++) {
                mask = Reg::UpdateMask<uint32_t>(addTotal);
                for (uint16_t m = 0; m < groupMainCount; m++) {
                    DataCopy(x1RegTensor, xBuf + j * ADD_ELEM + flag * m * addOffset + i * mFactor);
                    for (uint16_t n = 1; n <= addCount; n++) {
                        DataCopy(x2RegTensor,
                                 xBuf + j * ADD_ELEM + flag * (m * addOffset + n * computeLen) + i * mFactor);
                        Add(x2RegTensor, x1RegTensor, x2RegTensor, mask);
                        DataCopy(xBuf + j * ADD_ELEM + flag * (m * addOffset + n * computeLen) + i * mFactor,
                                 x2RegTensor, mask);
                    }
                }

                for (uint16_t m = 0; m < groupTailCount; m++) {
                    DataCopy(x1RegTensor, xBuf + j * ADD_ELEM + flag * groupMainCount * addOffset + i * mFactor);
                    for (uint16_t n = 1; n <= addTailCount; n++) {
                        DataCopy(x2RegTensor, xBuf + j * ADD_ELEM +
                                                  flag * (groupMainCount * addOffset + n * computeLen) + i * mFactor);
                        Add(x2RegTensor, x1RegTensor, x2RegTensor, mask);
                        DataCopy(
                            xBuf + j * ADD_ELEM + flag * (groupMainCount * addOffset + n * computeLen) + i * mFactor,
                            x2RegTensor, mask);
                    }
                }
            }
        }
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumOnewaySklansky<DataType, PromoteDataType>::ComputeInner(int32_t addLoop,
                                                                                     uint32_t realDupSize)
{
    int32_t startAddOffset = 0;
    int32_t addLoopPow = __CumsumUtil::CalPow2(addLoop);
    int32_t nextAddLoopPow = __CumsumUtil::CalPow2(addLoop + 1);
    int32_t startOffset = reverse_ ? (rFactor_ - addLoopPow) * nAddFactor_ : (addLoopPow - 1) * nAddFactor_;
    startOffset *= mFold_;
    addCount_ = addLoopPow;
    int32_t addOffset = nextAddLoopPow * nAddFactor_ * mFold_;
    groupMainCount_ = Ops::Base::FloorDiv(rFactor_, nextAddLoopPow);
    groupTailCount_ = (rFactor_ % nextAddLoopPow > addCount_) ? 1 : 0;
    addTailCount_ = rFactor_ - groupMainCount_ * nextAddLoopPow - addLoopPow;
    __local_mem__ PromoteDataType* xBuf = (__local_mem__ PromoteDataType*)computerBuffer_[startOffset + startAddOffset]
                                              .GetPhyAddr();

    int32_t flag = reverse_ ? -1 : 1;
    ComputeWithFlag(addOffset, realDupSize, xBuf, flag);
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumOnewaySklansky<DataType, PromoteDataType>::Compute()
{
    uint32_t realDupSize = nAddFactor_ * sizeof(PromoteDataType) * mFold_;
    int32_t rLoop = __CumsumUtil::CalLog2(rFactor_);
    int32_t nLoop = Ops::Base::CeilDiv(realDupSize, ADD_SIZE);

    for (int32_t k = 0; k < rLoop; k++) {
        ComputeInner(k, realDupSize);
    }

    SetFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    WaitFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumOnewaySklansky<DataType, PromoteDataType>::BaseProcess()
{
    if constexpr (!IsSameType<DataType, PromoteDataType>::value) {
        if (!isFirstCopyIn_) {
            WaitFlag<HardEvent::V_MTE2>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        }
    }
    if (exclusive_) {
        CopyInExclusive();
    } else {
        CopyIn();
    }
    if constexpr (!IsSameType<DataType, PromoteDataType>::value) {
        SetFlag<HardEvent::V_MTE2>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        isFirstCopyIn_ = false;
    }

    Compute();
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline LocalTensor<PromoteDataType>& CumsumOnewaySklansky<DataType, PromoteDataType>::GetBaseResTensor()
{
    return computerBuffer_;
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumOnewaySklansky<DataType, PromoteDataType>::BaseCopyOut()
{
    LocalTensor<DataType> outputTensor = inTensorX_;
    if constexpr (!IsSameType<DataType, PromoteDataType>::value) {
        LocalTensor<DataType> castTmp = inCastTensorX_.template ReinterpretCast<DataType>();
        Cast(castTmp, inCastTensorX_, RoundMode::CAST_RINT,
             Ops::Base::CeilDiv(mFactor_, mFold_) * mFold_ * dupSize_ * rFactor_ / sizeof(DataType));
        outputTensor = castTmp;
    }
    SetFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    WaitFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    DataCopyExtParams copyParams{0, 0, 0, 0, 0};
    copyParams.blockCount = rFactor_;
    copyParams.blockLen = nFactor_ * sizeof(DataType);
    copyParams.srcStride = (mFold_ - 1) * dupSize_ / BLOCK_SIZE;
    copyParams.dstStride = (lenN_ - nFactor_) * sizeof(DataType);
    int64_t yGmOffset = 0;
    int32_t copyRowOffset = 0;
    int32_t copyColOffset = 0;

    for (int32_t i = 0; i < mFactor_; i++) {
        yGmOffset = (mOffset_ + i) * lenR_ * lenN_ + rOffset_ * lenN_ + nOffset_;
        copyRowOffset = Ops::Base::FloorDiv(i, mFold_);
        copyColOffset = i % mFold_;
        DataCopyPad(this->yGm_[yGmOffset],
                    outputTensor[copyRowOffset * rFactor_ * nAddFactor_ * mFold_ + copyColOffset * nAddFactor_],
                    copyParams);
    }

    if constexpr (!IsSameType<DataType, PromoteDataType>::value) {
        SetFlag<HardEvent::MTE3_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        WaitFlag<HardEvent::MTE3_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    } else {
        SetFlag<HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        WaitFlag<HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    }
}

} // namespace Cumsum

#endif
