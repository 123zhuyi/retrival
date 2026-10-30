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
 * \file cumsum_twoway_sklansky.h
 * \brief functional class: two way sklansky
 */
#ifndef CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUMBASE_CUMSUM_TWO_WAY_SKLANSKY_H
#define CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUMBASE_CUMSUM_TWO_WAY_SKLANSKY_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "cumsum_base.h"

namespace Cumsum {
using namespace AscendC;

struct TwowaySklanskyInitData {
    int64_t lenM = 0;              // shape M
    int64_t lenR = 0;              // shape R
    int64_t lenN = 0;              // shape N
    int64_t xBufferSize = 0;       // 搬入ub所占空间
    int64_t xUnfoldBufferSize = 0; // 还原折叠后所需空间
};

struct TwowaySklanskyProcessData {
    int64_t offsetM = 0;         // gm上M轴的偏移量, 单位: R * N
    int64_t offsetR = 0;         // gm上M轴的偏移量, 单位: N
    int64_t offsetN = 0;         // gm上M轴的偏移量, 单位：element
    int32_t ubFactorR = 0;       // 单次ub可处理的R长度, 单位: N
    int32_t ubFactorN = 0;       // 单次ub可处理可处理的N, 单位：element
    int32_t rFoldAfterSize = 0;  // r折叠后长度，单位：N
    int32_t rFoldAfterCount = 0; // r折叠后后首次sklansky的次数
    int32_t rFoldCount = 0;      // r折叠次数
    bool isExclusive = false;    // attr
    bool isLenChange = false;    // 配合exclusive属性
    bool isReverse = false;      // attr
};

template <typename DataType, typename PromoteDataType>
class CumsumTwowaySklansky : public CumsumBase<DataType> {
public:
    __aicore__ inline CumsumTwowaySklansky(TPipe& pipe) : CumsumBase<DataType>(pipe) {}
    constexpr static uint32_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
    constexpr static uint32_t MAX_N = 32; // 双向折叠场景下，N最大尾32
    constexpr static uint32_t VL_SIZE = Ops::Base::GetVRegSize();
    constexpr static uint32_t VL_ELEM = VL_SIZE / sizeof(PromoteDataType);
    constexpr static uint32_t MAX_REPEAT_TIME = 255;
    constexpr static uint32_t MAX_REPEAT_TIME_ELEM = MAX_REPEAT_TIME * VL_ELEM;
    constexpr static uint32_t MAX_FOLD = 8; // 双向折叠场景下，最大折叠次数为8
    constexpr static uint32_t kMinMultiFoldCount = 1;

public:
    __aicore__ inline void BaseInit(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const TwowaySklanskyInitData& initData);
    __aicore__ inline void BaseProcessPre(const TwowaySklanskyProcessData& processData);
    __aicore__ inline void BaseProcess();
    __aicore__ inline void BaseCopyOut();
    __aicore__ inline LocalTensor<PromoteDataType>& GetBaseResTensor();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void CalcAddFactorIndex();
    __aicore__ inline void SetCopyParam(int32_t iter, uint32_t rCountReal, uint32_t& ubOffset, int64_t& gmOffset,
                                        DataCopyExtParams& dataCopyParams);
    __aicore__ inline void VfNoAlignCopyOut(uint32_t dirtyDataNum, uint32_t actualNum);
    __aicore__ inline void Compute();
    __aicore__ inline void FirstSkalansky();
    __aicore__ inline void VfFirstAdd(uint16_t startOffset, uint16_t addCircle, uint16_t addCount, uint16_t addOffset);
    __aicore__ inline void VfGatherBeforeSecondSkalansky();
    __aicore__ inline void SecondSkalansky();
    __aicore__ inline void VfSecondAdd(int32_t k, uint16_t startOffset, uint16_t addCircle, uint16_t addOffset);
    __aicore__ inline void VfSecondVLBuffer(__ubuf__ PromoteDataType* secondfoldPtr,
                                            __ubuf__ PromoteDataType* addFactorDataPtr, uint16_t startOffset,
                                            uint16_t addCircle, uint16_t addOffset);

private:
    TwowaySklanskyInitData initData_;
    TwowaySklanskyProcessData processData_;

    TBuf<> inBufX_;
    TBuf<> inCastBufX_;
    TBuf<> computeBuf_;
    TBuf<> outNoAlignBuf_;
    TBuf<> secondSkalanskyAddFactorIndexBuf_;
    TBuf<> secondSkalanskyAddFactorDataBuf_;
    LocalTensor<DataType> oriFoldBuffer_;
    LocalTensor<PromoteDataType> foldBuffer_;
    LocalTensor<PromoteDataType> secondfoldBuffer_;
    LocalTensor<DataType> outNoAlignBuffer_;
    LocalTensor<DataType> outputLocal_;
    LocalTensor<int32_t> secondSkalanskyAddFactorIndexBuffer_;
    LocalTensor<PromoteDataType> secondSkalanskyAddFactorDataBuffer_;

    int32_t reverseFlag_ = 1;
    int64_t inputOffset_ = 0;
    int64_t outputOffset_ = 0;
    uint16_t nAlignBlockElem_ = 0;
    uint16_t indexLoopCount_ = 0;
    uint32_t tailR_ = 0;       // 尾块R, 单位：N
    uint32_t foldOneElem_ = 0; // 折叠后一块大小, 单位: element
    uint32_t numAlignBlock_ = 0;
};

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySklansky<DataType, PromoteDataType>::BaseInit(GM_ADDR x, GM_ADDR y,
                                                                                 GM_ADDR workspace,
                                                                                 const TwowaySklanskyInitData& initData)
{
    CumsumBase<DataType>::Init(x, y);
    initData_ = initData;
    this->pipe_.InitBuffer(inBufX_, initData_.xBufferSize);           // R和N block对齐之后的空间大小
    this->pipe_.InitBuffer(computeBuf_, initData_.xUnfoldBufferSize); // 还原折叠后所需空间
    this->pipe_.InitBuffer(outNoAlignBuf_, VL_SIZE); // reverse=true, 搬出时额外申请一个256B大小的空间存放非对齐数据
    // secondSkalansky时AddFactor的index储存空间，max: 8KB
    this->pipe_.InitBuffer(secondSkalanskyAddFactorIndexBuf_, VL_SIZE * MAX_N);
    // secondSkalansky时AddFactor的data储存空间，256B * 2 * 4, max: 2KB
    this->pipe_.InitBuffer(secondSkalanskyAddFactorDataBuf_, VL_SIZE * MAX_FOLD);
    oriFoldBuffer_ = inBufX_.template Get<DataType>();
    secondfoldBuffer_ = computeBuf_.template Get<PromoteDataType>();
    outNoAlignBuffer_ = outNoAlignBuf_.template Get<DataType>();
    secondSkalanskyAddFactorIndexBuffer_ = secondSkalanskyAddFactorIndexBuf_.template Get<int32_t>();
    secondSkalanskyAddFactorDataBuffer_ = secondSkalanskyAddFactorDataBuf_.template Get<PromoteDataType>();

    if constexpr (!std::is_same<DataType, PromoteDataType>::value) {
        this->pipe_.InitBuffer(inCastBufX_, sizeof(PromoteDataType) / sizeof(DataType) * initData_.xBufferSize);
        foldBuffer_ = inCastBufX_.template Get<PromoteDataType>();
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySklansky<DataType, PromoteDataType>::BaseProcessPre(
    const TwowaySklanskyProcessData& processData)
{
    processData_ = processData;
    reverseFlag_ = processData_.isReverse ? -1 : 1;
    // gm offset 计算公式：offsetM * lenR * lenN + offsetR * lenN + offsetN
    nAlignBlockElem_ = Ops::Base::CeilAlign(static_cast<uint32_t>(processData_.ubFactorN * sizeof(DataType)),
                                            BLOCK_SIZE) /
                       sizeof(DataType);
    outputOffset_ = processData_.offsetM * initData_.lenR * initData_.lenN + processData_.offsetR * initData_.lenN +
                    processData_.offsetN;
    // 根据isOffsetChange和isReverse来调整offsetR
    processData_.offsetR = processData_.isExclusive ?
                               (processData_.isReverse ? processData_.offsetR + 1 : processData_.offsetR - 1) :
                               processData_.offsetR;
    inputOffset_ = processData_.offsetM * initData_.lenR * initData_.lenN + processData_.offsetR * initData_.lenN +
                   processData_.offsetN;
    // 尾块R的长度
    tailR_ = processData_.ubFactorR -
             Ops::Base::FloorDiv(processData_.ubFactorR, processData_.rFoldAfterSize) * processData_.rFoldAfterSize;
    tailR_ = (tailR_ == 0) ? processData_.rFoldAfterSize : tailR_;
    foldOneElem_ = processData_.rFoldAfterSize * processData_.ubFactorN;
    numAlignBlock_ = Ops::Base::CeilAlign(static_cast<uint32_t>(foldOneElem_ * sizeof(PromoteDataType)), BLOCK_SIZE) /
                     sizeof(PromoteDataType);
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySklansky<DataType, PromoteDataType>::BaseProcess()
{
    CopyIn();
    Compute();
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySklansky<DataType, PromoteDataType>::CalcAddFactorIndex()
{
    for (uint16_t i = 0; i < processData_.ubFactorN; i++) {
        int32_t startIdx = i * VL_ELEM % processData_.ubFactorN;
        CreateVecIndex(secondSkalanskyAddFactorIndexBuffer_[i * VL_ELEM], startIdx, VL_ELEM);
        if (startIdx == 0 && indexLoopCount_ != 0) {
            break;
        }
        indexLoopCount_++;
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySklansky<DataType, PromoteDataType>::SetCopyParam(int32_t iter, uint32_t rCountReal,
                                                                                     uint32_t& ubOffset,
                                                                                     int64_t& gmOffset,
                                                                                     DataCopyExtParams& dataCopyParams)
{
    ubOffset = iter * nAlignBlockElem_;
    dataCopyParams.blockCount = processData_.rFoldAfterSize; // 每次搬运rSize行数据
    if (iter == rCountReal - 1) {
        dataCopyParams.blockCount = tailR_;
    }
    if (processData_.isReverse) {
        gmOffset = inputOffset_ + (processData_.ubFactorR - (iter + 1) * processData_.rFoldAfterSize) *
                                      processData_.ubFactorN; // 倒搬数据
        if (iter == rCountReal - 1) {
            ubOffset = (tailR_ == 0) ? ubOffset : ubOffset + (processData_.rFoldAfterSize - tailR_) * VL_ELEM;
            gmOffset = inputOffset_; // 倒搬数据，保证尾块不越界
        }
    } else {
        gmOffset = inputOffset_ + iter * foldOneElem_;
        if (processData_.isLenChange && iter == 0) {
            ubOffset = VL_SIZE / sizeof(PromoteDataType) + ubOffset; // ub内buffer首行需要置零且越过第一行进行搬运数据
        }
    }
    if (processData_.offsetR < 0 && iter == 0) {
        // 保证首块搬入偏移不小于0
        gmOffset = processData_.offsetM * initData_.lenR * initData_.lenN + processData_.offsetN;
    }
    if (processData_.isLenChange && iter == 0) {
        dataCopyParams.blockCount = processData_.rFoldAfterSize - 1; // isLenChange为true，首块需要少搬运一行数据
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySklansky<DataType, PromoteDataType>::CopyIn()
{
    // 搬入时要补齐剩下的空间为0，不能再从gm上搬运
    Duplicate<DataType>(oriFoldBuffer_, 0, processData_.rFoldAfterSize * (VL_SIZE / sizeof(PromoteDataType)));
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventID);
    WaitFlag<HardEvent::V_MTE2>(eventID);

    CalcAddFactorIndex(); // 计算每轮addFactor的初始偏移量

    uint32_t rCountReal = CeilDivision(processData_.ubFactorR, processData_.rFoldAfterSize);
    uint32_t ubOffset = 0;
    int64_t gmOffset = 0;
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = processData_.rFoldAfterSize;
    dataCopyParams.blockLen = processData_.ubFactorN * sizeof(DataType);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (VL_SIZE - nAlignBlockElem_ * sizeof(PromoteDataType)) /
                               (sizeof(PromoteDataType) / sizeof(DataType)) / BLOCK_SIZE;
    DataCopyPadExtParams<DataType> padParams{false, 0, 0, 0};
    for (int32_t i = 0; i < rCountReal; i++) { // for循环可以用loopsize替换，凡事会多搬数据，需要考虑性能问题
        SetCopyParam(i, rCountReal, ubOffset, gmOffset, dataCopyParams);
        DataCopyPad(oriFoldBuffer_[ubOffset], this->xGm_[gmOffset], dataCopyParams, padParams);
    }

    eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventID);
    WaitFlag<HardEvent::MTE2_V>(eventID);

    if constexpr (std::is_same<DataType, PromoteDataType>::value) {
        foldBuffer_ = oriFoldBuffer_;
    } else {
        uint64_t mask[2] = {UINT64_MAX, 0};
        int32_t fullReapeatCount = processData_.rFoldAfterSize / MAX_REPEAT_TIME;
        for (int32_t i = 0; i < fullReapeatCount; i++) {
            int32_t offset = i * MAX_REPEAT_TIME_ELEM;
            Cast(foldBuffer_[offset], oriFoldBuffer_[offset], RoundMode::CAST_NONE, mask, MAX_REPEAT_TIME,
                 {1, 1, 8, 4});
        }
        int32_t tailReapeatTime = processData_.rFoldAfterSize - fullReapeatCount * MAX_REPEAT_TIME;
        if (tailReapeatTime > 0) {
            int32_t offset = fullReapeatCount * MAX_REPEAT_TIME_ELEM;
            Cast(foldBuffer_[offset], oriFoldBuffer_[offset], RoundMode::CAST_NONE, mask, tailReapeatTime,
                 {1, 1, 8, 4});
        }
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySklansky<DataType, PromoteDataType>::Compute()
{
    FirstSkalansky();
    VfGatherBeforeSecondSkalansky();
    SecondSkalansky();
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySklansky<DataType, PromoteDataType>::VfNoAlignCopyOut(uint32_t dirtyDataNum,
                                                                                         uint32_t actualNum)
{
    __ubuf__ DataType* outputPtr = (__ubuf__ DataType*)outputLocal_.GetPhyAddr();
    __ubuf__ DataType* outNoAlignPtr = (__ubuf__ DataType*)outNoAlignBuffer_.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<DataType, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::UnalignReg u0;
        AscendC::Reg::RegTensor<DataType> outNoAlignReg;
        auto srcUbT = outputPtr + dirtyDataNum; // 非对齐的有效数据
        AscendC::Reg::DataCopyUnAlignPre(u0, srcUbT);
        AscendC::Reg::DataCopyUnAlign(outNoAlignReg, u0, srcUbT);
        AscendC::Reg::DataCopy(outNoAlignPtr, outNoAlignReg, p0);
    }
    DataCopyExtParams dataCopyNoAlignParams;
    dataCopyNoAlignParams.blockCount = 1;
    dataCopyNoAlignParams.blockLen = actualNum * sizeof(DataType);
    dataCopyNoAlignParams.srcStride = 0;
    dataCopyNoAlignParams.dstStride = 0;

    event_t eventNoAlignID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventNoAlignID);
    WaitFlag<HardEvent::V_MTE3>(eventNoAlignID);
    DataCopyPad(this->yGm_[outputOffset_], outNoAlignBuffer_, dataCopyNoAlignParams); // 搬运头部非对齐数据
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySklansky<DataType, PromoteDataType>::BaseCopyOut()
{
    if constexpr (std::is_same<DataType, PromoteDataType>::value) {
        outputLocal_ = secondfoldBuffer_;
    } else {
        outputLocal_ = secondfoldBuffer_.template ReinterpretCast<DataType>();
        Cast(outputLocal_, secondfoldBuffer_, RoundMode::CAST_RINT, processData_.rFoldCount * foldOneElem_);
    }
    uint32_t ubOffset = 0;
    uint32_t dirtyDataNum = 0;
    uint32_t mixDataLen = BLOCK_SIZE / sizeof(DataType);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = processData_.ubFactorN * processData_.ubFactorR * sizeof(DataType);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    if (processData_.isReverse) {
        if (processData_.rFoldCount > kMinMultiFoldCount) {
            dirtyDataNum = processData_.ubFactorN *
                           (processData_.rFoldCount * processData_.rFoldAfterSize - processData_.ubFactorR);
        }
        if (dirtyDataNum > 0) {
            uint32_t remainder = dirtyDataNum % mixDataLen;
            VfNoAlignCopyOut(dirtyDataNum, mixDataLen - remainder);

            uint32_t effectiveDataHead = dirtyDataNum + (mixDataLen - remainder);
            outputOffset_ = outputOffset_ + (mixDataLen - remainder); // 调整输出起始位置到对齐
            ubOffset = effectiveDataHead;
            dataCopyParams.blockLen = (processData_.ubFactorN * processData_.ubFactorR - (mixDataLen - remainder)) *
                                      sizeof(DataType);
        }
    }

    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID);
    WaitFlag<HardEvent::V_MTE3>(eventID);
    DataCopyPad(this->yGm_[outputOffset_], outputLocal_[ubOffset], dataCopyParams);
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline LocalTensor<PromoteDataType>& CumsumTwowaySklansky<DataType, PromoteDataType>::GetBaseResTensor()
{
    return secondfoldBuffer_;
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySklansky<DataType, PromoteDataType>::FirstSkalansky()
{
    // ub内第一次skalansky --- VADD
    for (int32_t k = 0; k < processData_.rFoldAfterCount; k++) {
        uint16_t startOffset = (processData_.isReverse) ? processData_.rFoldAfterSize - __CumsumUtil::CalPow2(k) :
                                                          __CumsumUtil::CalPow2(k) - 1;  // 31, 30, 28 or 0, 1, 3
        uint16_t addCircle = processData_.rFoldAfterSize / __CumsumUtil::CalPow2(k + 1); // 16 8 4 2 1
        uint16_t addCount = __CumsumUtil::CalPow2(k);                                    // 1 2 4 8 16
        uint16_t addOffset = __CumsumUtil::CalPow2(k + 1); // 每一轮add之间的偏移量 2 4 8 16 32
        VfFirstAdd(startOffset, addCircle, addCount, addOffset);
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySklansky<DataType, PromoteDataType>::VfFirstAdd(uint16_t startOffset,
                                                                                   uint16_t addCircle,
                                                                                   uint16_t addCount,
                                                                                   uint16_t addOffset)
{
    int32_t reverseFlag = reverseFlag_;
    uint32_t addReg1Offset = 0;
    uint32_t addReg2Offset = 0;
    __ubuf__ PromoteDataType* foldPtr = (__ubuf__ PromoteDataType*)foldBuffer_.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<PromoteDataType, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<PromoteDataType> sumReg;
        AscendC::Reg::RegTensor<PromoteDataType> addReg1;
        AscendC::Reg::RegTensor<PromoteDataType> addReg2;

        for (uint16_t m = 0; m < addCircle; m++) {
            addReg1Offset = (startOffset + reverseFlag * m * addOffset) * VL_ELEM;
            for (uint16_t n = 0; n < addCount; n++) {
                addReg2Offset = (startOffset + reverseFlag * (m * addOffset + n + 1)) * VL_ELEM;
                AscendC::Reg::DataCopy(addReg1, foldPtr + addReg1Offset);
                AscendC::Reg::DataCopy(addReg2, foldPtr + addReg2Offset);
                AscendC::Reg::Add(sumReg, addReg1, addReg2, p0);
                AscendC::Reg::DataCopy(foldPtr + addReg2Offset, sumReg, p0);
            }
        }
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySklansky<DataType, PromoteDataType>::VfGatherBeforeSecondSkalansky()
{
    int32_t nSize = processData_.ubFactorN;
    int32_t reverseFlag = reverseFlag_;
    uint32_t dstOffset = 0;
    uint32_t baseOffset = processData_.rFoldCount * foldOneElem_ < numAlignBlock_ ?
                              0 :
                              processData_.rFoldCount * foldOneElem_ - numAlignBlock_;
    uint32_t dstBaseOffset = processData_.isReverse ? baseOffset : 0;
    uint32_t num = foldOneElem_;
    uint32_t numAlignBlock = numAlignBlock_;
    uint16_t times = CeilDivision(num, VL_ELEM);
    uint16_t foldCount = processData_.rFoldCount;

    __ubuf__ PromoteDataType* srcPtr = (__ubuf__ PromoteDataType*)foldBuffer_.GetPhyAddr();
    __ubuf__ PromoteDataType* dstPtr = (__ubuf__ PromoteDataType*)secondfoldBuffer_.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg p0;
        AscendC::Reg::MaskReg p1;
        AscendC::Reg::RegTensor<int32_t> indexReg;
        AscendC::Reg::RegTensor<int32_t> tmp;
        AscendC::Reg::RegTensor<int32_t> tmp1;
        AscendC::Reg::RegTensor<int32_t> tmp2;
        AscendC::Reg::RegTensor<int32_t> subReg;
        AscendC::Reg::RegTensor<int32_t> addReg;
        AscendC::Reg::RegTensor<int32_t> loopAddReg;
        AscendC::Reg::RegTensor<int32_t> nReg;
        AscendC::Reg::RegTensor<PromoteDataType> dstReg;

        p0 = AscendC::Reg::UpdateMask<int32_t>(num);
        uint32_t num1 = foldOneElem_;             // num被当成引用传给了p0，每次update都会减256B
        AscendC::Reg::Duplicate(nReg, nSize, p0); // [9 9 9 9 9 ...]
        for (uint16_t i = 0; i < times; i++) {
            AscendC::Reg::Arange(indexReg, i * VL_ELEM);         // [0 1 2 3 4 5 ...]
            AscendC::Reg::Div(tmp, indexReg, nReg, p0);          // [0000...1111...66667]
            AscendC::Reg::Muls(tmp1, tmp, (int32_t)VL_ELEM, p0); // [000..64...448]
            AscendC::Reg::Mul(subReg, tmp, nReg, p0);            // [000...99999.....63]
            AscendC::Reg::Sub(tmp2, indexReg, subReg, p0);       // [0123...0123...0123..]
            AscendC::Reg::Add(addReg, tmp1, tmp2, p0);           // [0123...64,65....]
            p1 = AscendC::Reg::UpdateMask<int32_t>(num1);
            for (uint16_t j = 0; j < foldCount; j++) {
                AscendC::Reg::Adds(loopAddReg, addReg, (int32_t)nAlignBlockElem_ * j, p0);
                AscendC::Reg::DataCopyGather(dstReg, srcPtr, (AscendC::Reg::RegTensor<uint32_t>&)loopAddReg, p1);
                dstOffset = dstBaseOffset + i * VL_ELEM + reverseFlag * j * numAlignBlock;
                AscendC::Reg::DataCopy(dstPtr + dstOffset, dstReg, p1);
            }
        }
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySklansky<DataType, PromoteDataType>::SecondSkalansky()
{
    // ub内第二次Skakansky
    int32_t rFoldCountLog = __CumsumUtil::CalLog2(processData_.rFoldCount);
    uint16_t startOffset = 0;
    uint16_t addCircle = 0;
    uint16_t addOffset = 0;

    for (int32_t k = 0; k < rFoldCountLog; k++) {
        startOffset = (processData_.isReverse) ? processData_.rFoldCount - __CumsumUtil::CalPow2(k) :
                                                 __CumsumUtil::CalPow2(k) - 1;
        addCircle = processData_.rFoldCount / __CumsumUtil::CalPow2(k + 1); // 4 2 1 每一列的add轮次数
        addOffset = __CumsumUtil::CalPow2(k + 1);
        VfSecondAdd(k, startOffset, addCircle, addOffset);
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySklansky<DataType, PromoteDataType>::VfSecondAdd(int32_t k, uint16_t startOffset,
                                                                                    uint16_t addCircle,
                                                                                    uint16_t addOffset)
{
    uint32_t num = foldOneElem_;
    uint32_t baseNum = num * __CumsumUtil::CalPow2(k);
    uint16_t times = CeilDivision(baseNum, VL_ELEM);
    uint16_t loopMain = CeilDivision(times, indexLoopCount_) - 1;     // 主块循环次数
    uint16_t indexLoopCountTail = times - indexLoopCount_ * loopMain; // 尾块add次数
    uint16_t loopTail = 1;
    int32_t reverseFlag = reverseFlag_;
    int32_t ReserveOffset = processData_.isReverse ? -1 * baseNum : numAlignBlock_;
    int32_t singleRemainder = VL_ELEM % processData_.ubFactorN;
    int32_t mainNum = loopMain * indexLoopCount_ * VL_ELEM;
    uint32_t baseNumOver = baseNum - mainNum;

    __ubuf__ PromoteDataType* secondfoldPtr = (__ubuf__ PromoteDataType*)secondfoldBuffer_.GetPhyAddr();
    __ubuf__ int32_t* addFactorIndexPtr = (__ubuf__ int32_t*)secondSkalanskyAddFactorIndexBuffer_.GetPhyAddr();
    __ubuf__ PromoteDataType* addFactorDataPtr = (__ubuf__ PromoteDataType*)
                                                     secondSkalanskyAddFactorDataBuffer_.GetPhyAddr();

    VfSecondVLBuffer(secondfoldPtr, addFactorDataPtr, startOffset, addCircle, addOffset);

    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<PromoteDataType, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg p1;
        AscendC::Reg::MaskReg p2;
        AscendC::Reg::RegTensor<PromoteDataType> addDataReg;
        AscendC::Reg::RegTensor<int32_t> addFactorIndexReg;
        AscendC::Reg::RegTensor<PromoteDataType> addFactorDataReg;

        for (uint16_t m = 0; m < addCircle; m++) {
            uint32_t baseNum1 = baseNum;
            uint32_t baseNum2 = baseNumOver;
            int32_t addDataStart = (startOffset + reverseFlag * m * addOffset) * num;
            int32_t addDataMainReserveOffset = addDataStart + ReserveOffset;
            int32_t addDataTailReserveOffset = addDataStart + ReserveOffset + mainNum;
            // 主块
            for (uint16_t i = 0; i < loopMain; i++) {
                for (uint16_t j = 0; j < indexLoopCount_; j++) {
                    p1 = AscendC::Reg::UpdateMask<int32_t>(baseNum1);
                    // 连续搬入AddFactor的index和data
                    AscendC::Reg::AddrReg addrOffset = AscendC::Reg::CreateAddrReg<PromoteDataType>(j, VL_ELEM);
                    AscendC::Reg::DataCopy(addFactorIndexReg, addFactorIndexPtr, addrOffset);
                    AscendC::Reg::Adds(addFactorIndexReg, addFactorIndexReg, m * VL_ELEM * 2, p0);
                    AscendC::Reg::DataCopyGather(addFactorDataReg, addFactorDataPtr,
                                                 (AscendC::Reg::RegTensor<uint32_t>&)addFactorIndexReg, p0);
                    int32_t addDataOffset = addDataMainReserveOffset + (i * indexLoopCount_ + j) * VL_ELEM;
                    AscendC::Reg::DataCopy(addDataReg, secondfoldPtr + addDataOffset);     // 连续对齐搬入addData
                    AscendC::Reg::Add(addDataReg, addDataReg, addFactorDataReg, p1);       // Add
                    AscendC::Reg::DataCopy(secondfoldPtr + addDataOffset, addDataReg, p1); // 搬出
                }
            }
            // 尾块，只需执行一次即可
            for (uint16_t i = 0; i < loopTail; i++) {
                for (uint16_t j = 0; j < indexLoopCountTail; j++) {
                    p2 = AscendC::Reg::UpdateMask<int32_t>(baseNum2);
                    AscendC::Reg::AddrReg addrOffset = AscendC::Reg::CreateAddrReg<PromoteDataType>(j, VL_ELEM);
                    AscendC::Reg::DataCopy(addFactorIndexReg, addFactorIndexPtr, addrOffset);
                    AscendC::Reg::Adds(addFactorIndexReg, addFactorIndexReg, m * VL_ELEM * 2, p0);
                    AscendC::Reg::DataCopyGather(addFactorDataReg, addFactorDataPtr,
                                                 (AscendC::Reg::RegTensor<uint32_t>&)addFactorIndexReg, p0);
                    int32_t addDataOffset = addDataTailReserveOffset + (i * indexLoopCount_ + j) * VL_ELEM;
                    AscendC::Reg::DataCopy(addDataReg, secondfoldPtr + addDataOffset);
                    AscendC::Reg::Add(addDataReg, addDataReg, addFactorDataReg, p2);
                    AscendC::Reg::DataCopy(secondfoldPtr + addDataOffset, addDataReg, p2);
                }
            }
        }
    }
}

template <typename DataType, typename PromoteDataType>
__aicore__ inline void CumsumTwowaySklansky<DataType, PromoteDataType>::VfSecondVLBuffer(
    __ubuf__ PromoteDataType* secondfoldPtr, __ubuf__ PromoteDataType* addFactorDataPtr, uint16_t startOffset,
    uint16_t addCircle, uint16_t addOffset)
{
    int32_t reverseFlag = reverseFlag_;
    uint32_t num = foldOneElem_;
    int32_t nSize = processData_.ubFactorN;
    int32_t singleRemainder = VL_ELEM % processData_.ubFactorN;
    uint16_t innerLoop = 2;
    int32_t ReserveStartOffset = processData_.isReverse ? 0 :
                                                          processData_.ubFactorN * (processData_.rFoldAfterSize - 1);

    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<PromoteDataType, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<int32_t> indexReg;
        AscendC::Reg::RegTensor<int32_t> tmp;
        AscendC::Reg::RegTensor<int32_t> tmp1;
        AscendC::Reg::RegTensor<int32_t> tmp2;
        AscendC::Reg::RegTensor<int32_t> tmp3;
        AscendC::Reg::RegTensor<int32_t> nReg;
        AscendC::Reg::RegTensor<int32_t> secondFoldReg;
        AscendC::Reg::RegTensor<PromoteDataType> addFactorDataReg;

        for (uint16_t m = 0; m < addCircle; m++) {
            int32_t startIdx = 0;
            int32_t addFactorDataStart = (startOffset + reverseFlag * m * addOffset) * num;
            AscendC::Reg::Duplicate(nReg, nSize, p0);  // [9 9 9 9 9 ...]
            for (uint16_t i = 0; i < innerLoop; i++) { // buffer 长度 256B * 2
                AscendC::Reg::Arange(indexReg, startIdx);
                AscendC::Reg::Div(tmp, indexReg, nReg, p0);          // [0000...1111...66667]
                AscendC::Reg::Muls(tmp1, tmp, (int32_t)VL_ELEM, p0); // [000..64...448]
                AscendC::Reg::Mul(tmp3, tmp, nReg, p0);              // [000...99999.....63]
                AscendC::Reg::Sub(tmp2, indexReg, tmp3, p0);         // [0123...0123...0123..]
                AscendC::Reg::Adds(secondFoldReg, tmp2, addFactorDataStart + ReserveStartOffset, p0);
                AscendC::Reg::DataCopyGather(addFactorDataReg, secondfoldPtr,
                                             (AscendC::Reg::RegTensor<uint32_t>&)secondFoldReg, p0);
                AscendC::Reg::DataCopy(addFactorDataPtr + (int32_t)VL_ELEM * 2 * m + (int32_t)VL_ELEM * i,
                                       addFactorDataReg, p0);
                startIdx = startIdx + singleRemainder;
            }
        }
    }
}

} // namespace Cumsum

#endif // CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUMBASE_CUMSUM_TWO_WAY_SKLANSKY_H
