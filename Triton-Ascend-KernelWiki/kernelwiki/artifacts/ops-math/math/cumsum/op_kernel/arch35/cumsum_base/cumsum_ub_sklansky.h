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
 * \file cumsum_ub_sklansky.h
 * \brief functional class: template class: ub sklansky + (two way sklansky or one way sklansky)
 */
#ifndef CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUMBASE_CUMSUM_UB_SKLANSKY_H
#define CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUMBASE_CUMSUM_UB_SKLANSKY_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "cumsum_base.h"
#include "cumsum_twoway_sklansky.h"
#include "cumsum_oneway_sklansky.h"

namespace Cumsum {
using namespace AscendC;

struct UbSklanskyInitData {
    int64_t lenM = 0; // shape M
    int64_t lenR = 0; // shape R
    int64_t lenN = 0; // shape N
    int32_t mFold = 1;
    int32_t xBufferSize = 0;

    // 双向参数，计算buffersize需要，后续考虑移到processPre处理
    int32_t rFoldAfterSize = 0;      // r折叠后长度(主块)，单位：N
    int32_t rFoldAfterCount = 0;     // r折叠后后首次sklansky的次数(主块)
    int32_t rFoldCount = 0;          // r折叠次数(主块)
    int32_t ubFactorN = 0;           // 单次ub可处理可处理的N, 单位：element
    int32_t ubFactorR = 0;           // 单次ub可处理的R长度（主块）, 单位: N
    int32_t ubTailFactorR = 0;       // 单次ub可处理的R长度（尾块）, 单位: N
    int32_t ubCountR = 0;            // ub间处理数据，块数
    int32_t rTailFoldAfterSize = 0;  // r折叠后长度(尾块)，单位：N
    int32_t rTailFoldAfterCount = 0; // r折叠后后首次sklansky的次数(尾块)
    int32_t rTailFoldCount = 0;      // r折叠次数(尾块)
    int32_t xUnfoldBufferSize = 0;

    // 单向参数
    bool reverse = false;

    // ub间参数
    int32_t memoLen = 0;     // ub间缓存的块数
    int32_t perMemoSize = 0; // 每一级缓存的大小，单位：element，block对齐
    int32_t ubSklanskyBufSize = 0;
};

struct UbSklanskyProcessData {
    int64_t offsetM = 0;      // gm上M轴的偏移量, 单位: R * N
    int64_t offsetR = 0;      // gm上M轴的偏移量, 单位: N
    int64_t offsetN = 0;      // gm上M轴的偏移量, 单位：element
    bool isExclusive = false; // attr
    bool isLenChange = false; // 配合isExclusive属性对ub间进行控制
    int32_t ubFactorN = 0;    // 单向会变
    int32_t ubFactorM = 1;    // 单向会变
};

template <typename DataType, typename PromoteDataType, typename UbInner>
class CumsumUbSklansky : public CumsumBase<DataType> {
public:
    __aicore__ inline CumsumUbSklansky(TPipe& pipe) : CumsumBase<DataType>(pipe), ubInner_(pipe) {}
    constexpr static int32_t VL_SIZE = Ops::Base::GetVRegSize();
    constexpr static int32_t VL_ELEM = VL_SIZE / sizeof(PromoteDataType);
    constexpr static int32_t BLK_SIZE = Ops::Base::GetUbBlockSize();
    constexpr static int32_t BLK_ELEM = BLK_SIZE / sizeof(PromoteDataType);
    constexpr static int32_t CONST1 = 1;
    constexpr static int32_t MAX_MEMO_ROW = 10;
    constexpr static int32_t MAX_REPEAT_TIME = 255;
    constexpr static int32_t TWOWAY_MAX_N = 32;
    constexpr static int32_t DATA_FACTOR_SIZE = 2;

public:
    __aicore__ inline void BaseInit(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const UbSklanskyInitData& initData);
    __aicore__ inline void BaseProcessPre(const UbSklanskyProcessData& processData);
    __aicore__ inline void BaseProcess();

private:
    __aicore__ inline void SkalanskyBetweenUb(int32_t ubIndex);
    __aicore__ inline void SetTwowaySklanskyParam(int32_t ubIndex,
                                                  TwowaySklanskyProcessData& twowaySklanskyProcessData);
    __aicore__ inline void SetOnewaySklanskyParam(int32_t ubIndex,
                                                  OnewaySklanskyProcessData& onewaySklanskyProcessData);
    __aicore__ inline bool CheckOffsetAndLen(int32_t ubIndex);
    __aicore__ inline void VfBetweenUbAddTwoway(__local_mem__ PromoteDataType* dstPtr,
                                                __local_mem__ PromoteDataType* srcPtr, uint32_t cacheStartInRes,
                                                uint16_t memCacheId);
    __aicore__ inline void VfBetweenUbAddTwowayBak(__local_mem__ PromoteDataType* dstPtr,
                                                   __local_mem__ PromoteDataType* srcPtr, uint32_t cacheStartInRes,
                                                   uint16_t memCacheId);
    __aicore__ inline void VfBetweenUbWriteMemTwoway(__local_mem__ PromoteDataType* dstPtr,
                                                     __local_mem__ PromoteDataType* srcPtr, uint32_t cacheStartInRes,
                                                     uint16_t memCacheId);
    __aicore__ inline void VfBetweenUbAddOneway(__local_mem__ PromoteDataType* dstPtr,
                                                __local_mem__ PromoteDataType* srcPtr, uint32_t cacheStartInRes,
                                                uint16_t memCacheId);
    __aicore__ inline void VfBetweenUbWriteMemOneway(__local_mem__ PromoteDataType* dstPtr,
                                                     __local_mem__ PromoteDataType* srcPtr, uint32_t cacheStartInRes,
                                                     uint16_t memCacheId);
    __aicore__ inline uint16_t CalAddFactorIndex();
    __aicore__ inline void CalAddFactorValue(__local_mem__ PromoteDataType* dstPtr,
                                             __local_mem__ PromoteDataType* addFactorDataPtr, int32_t singleRemainder,
                                             uint16_t memCacheId);

private:
    UbInner ubInner_;
    TBuf<> memoBuf_;
    LocalTensor<PromoteDataType> memoBuffer_; // ub间结果缓存
    int32_t memoFlags_[MAX_MEMO_ROW] = {0};
    UbSklanskyInitData initData_;
    UbSklanskyProcessData processData_;
    int64_t inputOffset_ = 0;
    int32_t ubBetweenFactorR_ = 0; // ub间需要处理的r的长度
    int32_t ubBetweenAddR_ = 0;
    int32_t addLenN_ = 0;
    TBuf<> addFactorIndexBuf_;
    TBuf<> addFactorDataBuf_;
    LocalTensor<uint32_t> addFactorIndexBuffer_;
    LocalTensor<PromoteDataType> addFactorDataBuffer_;
};

template <typename DataType, typename PromoteDataType, typename UbInner>
__aicore__ inline void CumsumUbSklansky<DataType, PromoteDataType, UbInner>::BaseInit(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const UbSklanskyInitData& initData)
{
    CumsumBase<DataType>::Init(x, y);
    initData_ = initData;
    if constexpr (IsSameType<UbInner, CumsumTwowaySklansky<DataType, PromoteDataType>>::value) {
        TwowaySklanskyInitData twowaySklanskyInitData;
        twowaySklanskyInitData.lenM = initData_.lenM;
        twowaySklanskyInitData.lenR = initData_.lenR;
        twowaySklanskyInitData.lenN = initData_.lenN;
        twowaySklanskyInitData.xBufferSize = initData_.xBufferSize;
        twowaySklanskyInitData.xUnfoldBufferSize = initData_.xUnfoldBufferSize;
        this->pipe_.InitBuffer(addFactorIndexBuf_, VL_SIZE * TWOWAY_MAX_N); // AddFactor的index储存空间，max: 8KB
        this->pipe_.InitBuffer(addFactorDataBuf_, VL_SIZE * DATA_FACTOR_SIZE); // AddFactor的data储存空间，max: 512B
        addFactorIndexBuffer_ = addFactorIndexBuf_.template Get<uint32_t>();
        addFactorDataBuffer_ = addFactorDataBuf_.template Get<PromoteDataType>();
        ubInner_.BaseInit(x, y, workspace, twowaySklanskyInitData);
    } else {
        OnewaySklanskyInitData onewaySklanskyInitData;
        onewaySklanskyInitData.mLen = initData_.lenM;
        onewaySklanskyInitData.rLen = initData_.lenR;
        onewaySklanskyInitData.nLen = initData_.lenN;
        onewaySklanskyInitData.mFold = initData_.mFold;
        onewaySklanskyInitData.ubBufferSize = initData_.xBufferSize;
        onewaySklanskyInitData.reverse = initData_.reverse;
        ubInner_.BaseInit(x, y, workspace, onewaySklanskyInitData);
    }
    this->pipe_.InitBuffer(memoBuf_, initData_.memoLen * initData_.perMemoSize * sizeof(PromoteDataType));
    memoBuffer_ = memoBuf_.template Get<PromoteDataType>();
}

template <typename DataType, typename PromoteDataType, typename UbInner>
__aicore__ inline void CumsumUbSklansky<DataType, PromoteDataType, UbInner>::BaseProcessPre(
    const UbSklanskyProcessData& processData)
{
    processData_ = processData;
    // gm Offset计算公式：OffsetM * lenR * lenN + offsetR * lenN + offsetN
    inputOffset_ = processData_.offsetM * initData_.lenR * initData_.lenN + processData_.offsetR * initData_.lenN +
                   processData_.offsetN;
    ubBetweenFactorR_ = initData_.ubFactorR * (initData_.ubCountR - 1) +
                        initData_.ubTailFactorR; // UB间处理的R的数量，revers倒序的时候使用
}

template <typename DataType, typename PromoteDataType, typename UbInner>
__aicore__ inline bool CumsumUbSklansky<DataType, PromoteDataType, UbInner>::CheckOffsetAndLen(int32_t ubIndex)
{
    if (processData_.isLenChange) { // 核间进行控制
        return false;
    } else { // 首核不需要进行控制，或没有上层核间控制
        return (ubIndex == 0) ? true : false;
    }
}

template <typename DataType, typename PromoteDataType, typename UbInner>
__aicore__ inline void CumsumUbSklansky<DataType, PromoteDataType, UbInner>::SetTwowaySklanskyParam(
    int32_t ubIndex, TwowaySklanskyProcessData& twowaySklanskyProcessData)
{
    twowaySklanskyProcessData.isReverse = initData_.reverse;
    twowaySklanskyProcessData.offsetM = processData_.offsetM; // M分核
    twowaySklanskyProcessData.offsetN = processData_.offsetN; // N全载
    twowaySklanskyProcessData.ubFactorN = initData_.ubFactorN;
    twowaySklanskyProcessData.rFoldAfterSize = initData_.rFoldAfterSize;
    twowaySklanskyProcessData.rFoldAfterCount = initData_.rFoldAfterCount;
    twowaySklanskyProcessData.rFoldCount = initData_.rFoldCount;
    twowaySklanskyProcessData.ubFactorR = initData_.ubFactorR;
    if (ubIndex == initData_.ubCountR - 1) { // 尾块ubFactorR需要rSize/rFoldAfterCount/rFoldCount重新计算
        twowaySklanskyProcessData.rFoldAfterSize = initData_.rTailFoldAfterSize;
        twowaySklanskyProcessData.rFoldAfterCount = initData_.rTailFoldAfterCount;
        twowaySklanskyProcessData.rFoldCount = initData_.rTailFoldCount;
        twowaySklanskyProcessData.ubFactorR = initData_.ubTailFactorR;
    }
    if (initData_.reverse) {
        twowaySklanskyProcessData.offsetR = processData_.offsetR + ubBetweenFactorR_ -
                                            (ubIndex + 1) * initData_.ubFactorR; // revers倒序
        if (ubIndex == initData_.ubCountR - 1) {
            twowaySklanskyProcessData.offsetR = processData_.offsetR; // 尾块OffsetR特殊处理为起始offsetR
        }
    } else {
        twowaySklanskyProcessData.offsetR = processData_.offsetR + ubIndex * initData_.ubFactorR;
    }
    twowaySklanskyProcessData.isExclusive = processData_.isExclusive;
    if (processData_.isExclusive) {
        twowaySklanskyProcessData.isLenChange = CheckOffsetAndLen(ubIndex);
    }
}

template <typename DataType, typename PromoteDataType, typename UbInner>
__aicore__ inline void CumsumUbSklansky<DataType, PromoteDataType, UbInner>::SetOnewaySklanskyParam(
    int32_t ubIndex, OnewaySklanskyProcessData& onewaySklanskyProcessData)
{
    onewaySklanskyProcessData.mOffset = processData_.offsetM;
    onewaySklanskyProcessData.mFactor = processData_.ubFactorM;
    onewaySklanskyProcessData.nOffset = processData_.offsetN;
    onewaySklanskyProcessData.nFactor = processData_.ubFactorN;
    if (initData_.reverse) {
        if (ubIndex == initData_.ubCountR - 1) {
            onewaySklanskyProcessData.rFactor = initData_.ubTailFactorR;
            ubBetweenAddR_ = initData_.ubTailFactorR;
            onewaySklanskyProcessData.rOffset = processData_.offsetR;
        } else {
            onewaySklanskyProcessData.rFactor = initData_.ubFactorR;
            ubBetweenAddR_ = initData_.ubFactorR;
            onewaySklanskyProcessData.rOffset = processData_.offsetR + ubBetweenFactorR_ -
                                                (ubIndex + 1) * initData_.ubFactorR; // revers倒序
        }
    } else {
        if (ubIndex == initData_.ubCountR - 1) {
            onewaySklanskyProcessData.rFactor = initData_.ubTailFactorR;
            ubBetweenAddR_ = initData_.ubTailFactorR;
            onewaySklanskyProcessData.rOffset = processData_.offsetR + initData_.ubFactorR * ubIndex;
        } else {
            onewaySklanskyProcessData.rFactor = initData_.ubFactorR;
            ubBetweenAddR_ = initData_.ubFactorR;
            onewaySklanskyProcessData.rOffset = processData_.offsetR + initData_.ubFactorR * ubIndex;
        }
    }

    onewaySklanskyProcessData.exclusive = processData_.isExclusive;
    if (processData_.isExclusive) {
        onewaySklanskyProcessData.isLenChange = CheckOffsetAndLen(ubIndex);
    }
    addLenN_ = Ops::Base::CeilAlign(static_cast<int32_t>(processData_.ubFactorN * sizeof(DataType)), BLK_SIZE) /
               sizeof(DataType);
}

template <typename DataType, typename PromoteDataType, typename UbInner>
__aicore__ inline void CumsumUbSklansky<DataType, PromoteDataType, UbInner>::BaseProcess()
{
    Duplicate<PromoteDataType>(memoBuffer_, 0, initData_.memoLen * initData_.perMemoSize);
    for (int32_t i = 0; i < initData_.ubCountR; i++) {
        if constexpr (IsSameType<UbInner, CumsumTwowaySklansky<DataType, PromoteDataType>>::value) {
            TwowaySklanskyProcessData twowaySklanskyProcessData;
            SetTwowaySklanskyParam(i, twowaySklanskyProcessData);
            ubInner_.BaseProcessPre(twowaySklanskyProcessData);
        } else {
            OnewaySklanskyProcessData onewaySklanskyProcessData;
            SetOnewaySklanskyParam(i, onewaySklanskyProcessData);
            ubInner_.BaseProcessPre(onewaySklanskyProcessData);
        }
        ubInner_.BaseProcess();
        SkalanskyBetweenUb(i);
        ubInner_.BaseCopyOut();
    }
}

template <typename DataType, typename PromoteDataType, typename UbInner>
__aicore__ inline uint16_t CumsumUbSklansky<DataType, PromoteDataType, UbInner>::CalAddFactorIndex()
{
    uint16_t loopCount = 0;
    int32_t startIdx = 0;
    // 计算每轮addFactor的初始偏移量
    for (uint16_t i = 0; i < initData_.ubFactorN; i++) {
        startIdx = i * VL_ELEM % initData_.ubFactorN;
        for (uint16_t j = 0; j < VL_ELEM; j++) {
            addFactorIndexBuffer_.SetValue(i * VL_ELEM + j, startIdx + j);
        }
        if (startIdx == 0 && loopCount != 0) {
            break;
        }
        loopCount++;
    }
    return loopCount;
}

template <typename DataType, typename PromoteDataType, typename UbInner>
__aicore__ inline void CumsumUbSklansky<DataType, PromoteDataType, UbInner>::CalAddFactorValue(
    __local_mem__ PromoteDataType* dstPtr, __local_mem__ PromoteDataType* addFactorDataPtr, int32_t singleRemainder,
    uint16_t memCacheId)
{
    uint16_t dataFactorSize = DATA_FACTOR_SIZE;
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<PromoteDataType, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<int32_t> indexReg;
        AscendC::Reg::RegTensor<int32_t> tmp0;
        AscendC::Reg::RegTensor<int32_t> tmp1;
        AscendC::Reg::RegTensor<int32_t> tmp2;
        AscendC::Reg::RegTensor<int32_t> tmp3;
        AscendC::Reg::RegTensor<int32_t> nReg;
        AscendC::Reg::RegTensor<PromoteDataType> addFactorDataReg;

        int32_t startIdx = 0;
        AscendC::Reg::Duplicate(nReg, initData_.ubFactorN, p0); // [9 9 9 9 9 ...]
        for (uint16_t ii = 0; ii < dataFactorSize; ii++) {      // buffer 长度 256B * 2
            AscendC::Reg::Arange(indexReg, startIdx);
            AscendC::Reg::Div(tmp0, indexReg, nReg, p0);          // [0000...1111...66667]
            AscendC::Reg::Muls(tmp1, tmp0, (int32_t)VL_ELEM, p0); // [000..64...448]
            AscendC::Reg::Mul(tmp3, tmp0, nReg, p0);              // [000...99999.....63]
            AscendC::Reg::Sub(tmp2, indexReg, tmp3, p0);          // [0123...0123...0123..]
            AscendC::Reg::DataCopyGather(addFactorDataReg, dstPtr + memCacheId * initData_.perMemoSize,
                                         (AscendC::Reg::RegTensor<uint32_t>&)tmp2, p0);
            AscendC::Reg::DataCopy(addFactorDataPtr + (int32_t)VL_ELEM * ii, addFactorDataReg, p0);
            startIdx = startIdx + singleRemainder;
        }
    }
}

template <typename DataType, typename PromoteDataType, typename UbInner>
__aicore__ inline void CumsumUbSklansky<DataType, PromoteDataType, UbInner>::VfBetweenUbAddTwoway(
    __local_mem__ PromoteDataType* dstPtr, __local_mem__ PromoteDataType* srcPtr, uint32_t cacheStartInRes,
    uint16_t memCacheId)
{
    uint16_t indexLoopCount = CalAddFactorIndex();
    int32_t singleRemainder = VL_ELEM % initData_.ubFactorN;
    __local_mem__ PromoteDataType* addFactorDataPtr = (__local_mem__ PromoteDataType*)addFactorDataBuffer_.GetPhyAddr();
    __local_mem__ uint32_t* addFactorIndexPtr = (__local_mem__ uint32_t*)addFactorIndexBuffer_.GetPhyAddr();

    CalAddFactorValue(dstPtr, addFactorDataPtr, singleRemainder, memCacheId);

    uint32_t baseNumOri = initData_.ubFactorR * initData_.ubFactorN;
    uint16_t times = CeilDivision(baseNumOri, VL_ELEM);
    uint16_t loopMain = CeilDivision(times, indexLoopCount) - 1; // 主块循环次数
    uint16_t indexLoopCountTail = times - indexLoopCount * loopMain;
    uint32_t mainCount = loopMain * indexLoopCount;
    uint32_t mainNum = mainCount * VL_ELEM;
    uint32_t baseNumOver = baseNumOri - mainNum;

    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<PromoteDataType, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg p3;
        AscendC::Reg::RegTensor<uint32_t> addFactorIndexReg;
        AscendC::Reg::RegTensor<PromoteDataType> addFactorDataReg;
        AscendC::Reg::RegTensor<PromoteDataType> curReg;
        // dstTensor[memCacheId] 与 srcTensor相加
        for (uint16_t i = 0; i < loopMain; i++) {
            for (uint16_t j = 0; j < indexLoopCount; j++) {
                // 连续搬入AddFactor的index和data
                p3 = AscendC::Reg::UpdateMask<int32_t>(mainNum);
                AscendC::Reg::AddrReg addrOffset = AscendC::Reg::CreateAddrReg<PromoteDataType>(j, VL_ELEM);
                AscendC::Reg::DataCopy(addFactorIndexReg, addFactorIndexPtr, addrOffset);
                AscendC::Reg::DataCopyGather(addFactorDataReg, addFactorDataPtr, addFactorIndexReg, p0);
                // 搬入第二块数据
                int32_t addDataOffset = (i * indexLoopCount + j) * VL_ELEM;
                AscendC::Reg::DataCopy(curReg, srcPtr + addDataOffset);
                AscendC::Reg::Add(curReg, curReg, addFactorDataReg, p3);
                AscendC::Reg::DataCopy(srcPtr + addDataOffset, curReg, p3);
            }
        }

        for (uint16_t j = 0; j < indexLoopCountTail; j++) {
            // 连续搬入AddFactor的index和data
            p3 = AscendC::Reg::UpdateMask<int32_t>(baseNumOver);
            AscendC::Reg::AddrReg addrOffset = AscendC::Reg::CreateAddrReg<PromoteDataType>(j, VL_ELEM);
            AscendC::Reg::DataCopy(addFactorIndexReg, addFactorIndexPtr, addrOffset);
            AscendC::Reg::DataCopyGather(addFactorDataReg, addFactorDataPtr, addFactorIndexReg, p0);
            // 搬入第二块数据
            int32_t addDataOffset = (mainCount + j) * VL_ELEM;
            AscendC::Reg::DataCopy(curReg, srcPtr + addDataOffset);
            AscendC::Reg::Add(curReg, curReg, addFactorDataReg, p3);
            AscendC::Reg::DataCopy(srcPtr + addDataOffset, curReg, p3);
        }
    }
}

template <typename DataType, typename PromoteDataType, typename UbInner>
__aicore__ inline void CumsumUbSklansky<DataType, PromoteDataType, UbInner>::VfBetweenUbAddOneway(
    __local_mem__ PromoteDataType* dstPtr, __local_mem__ PromoteDataType* srcPtr, uint32_t cacheStartInRes,
    uint16_t memCacheId)
{
    uint16_t rCount = ubBetweenAddR_;
    uint32_t addLen = addLenN_ * initData_.mFold;
    uint32_t addLenAlign = addLenN_ * initData_.mFold; // block对齐
    uint16_t mLoop = Ops::Base::CeilDiv(processData_.ubFactorM, initData_.mFold);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<PromoteDataType> memoReg;
        AscendC::Reg::RegTensor<PromoteDataType> curReg;
        AscendC::Reg::MaskReg p0;
        p0 = AscendC::Reg::UpdateMask<int32_t>(addLen);
        AscendC::Reg::DataCopy(memoReg, dstPtr + memCacheId * initData_.perMemoSize);
        for (uint16_t i = 0; i < mLoop; i++) {
            for (uint16_t ii = 0; ii < rCount; ii++) {
                uint16_t j = i * rCount + ii;
                AscendC::Reg::AddrReg addrOffset = AscendC::Reg::CreateAddrReg<PromoteDataType>(j, addLenAlign);
                AscendC::Reg::DataCopy(curReg, srcPtr, addrOffset);
                AscendC::Reg::Add(curReg, curReg, memoReg, p0);
                AscendC::Reg::DataCopy(srcPtr, curReg, addrOffset, p0);
            }
        }
    }
}

template <typename DataType, typename PromoteDataType, typename UbInner>
__aicore__ inline void CumsumUbSklansky<DataType, PromoteDataType, UbInner>::VfBetweenUbWriteMemTwoway(
    __local_mem__ PromoteDataType* dstPtr, __local_mem__ PromoteDataType* srcPtr, uint32_t cacheStartInRes,
    uint16_t memCacheId)
{
    __VEC_SCOPE__
    {
        // 把resub最后一行写入dstTensor[cacheID] -> 使用datacopygather
        AscendC::Reg::MaskReg p0;
        AscendC::Reg::RegTensor<PromoteDataType> dstCaheReg;
        AscendC::Reg::RegTensor<int32_t> indexReg2;
        AscendC::Reg::Arange(indexReg2, cacheStartInRes);
        uint32_t nLen = initData_.ubFactorN;
        p0 = AscendC::Reg::UpdateMask<int32_t>(nLen);
        AscendC::Reg::DataCopyGather(dstCaheReg, srcPtr, (AscendC::Reg::RegTensor<uint32_t>&)indexReg2, p0);
        AscendC::Reg::DataCopy(dstPtr + memCacheId * initData_.perMemoSize, dstCaheReg, p0);
    }
}

template <typename DataType, typename PromoteDataType, typename UbInner>
__aicore__ inline void CumsumUbSklansky<DataType, PromoteDataType, UbInner>::VfBetweenUbWriteMemOneway(
    __local_mem__ PromoteDataType* dstPtr, __local_mem__ PromoteDataType* srcPtr, uint32_t cacheStartInRes,
    uint16_t memCacheId)
{
    uint32_t addLen = addLenN_ * initData_.mFold;
    uint32_t addLenAlign = addLenN_ * initData_.mFold; // block对齐
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg p0;
        AscendC::Reg::RegTensor<PromoteDataType> dstCaheReg;
        p0 = AscendC::Reg::UpdateMask<int32_t>(addLen);
        AscendC::Reg::DataCopy(dstCaheReg, srcPtr + cacheStartInRes * addLenAlign);
        AscendC::Reg::DataCopy(dstPtr + memCacheId * initData_.perMemoSize, dstCaheReg, p0);
    }
}

template <typename DataType, typename PromoteDataType, typename UbInner>
__aicore__ inline void CumsumUbSklansky<DataType, PromoteDataType, UbInner>::SkalanskyBetweenUb(int32_t ubIndex)
{
    int32_t cacheID = __CumsumUtil::GetCacheID(ubIndex);
    __local_mem__ PromoteDataType* dstPtr = (__local_mem__ PromoteDataType*)memoBuffer_.GetPhyAddr();
    LocalTensor<PromoteDataType> computeBuffer = ubInner_.GetBaseResTensor();
    __local_mem__ PromoteDataType* srcPtr = (__local_mem__ PromoteDataType*)computeBuffer.GetPhyAddr();

    uint32_t cacheStartInRes = 0;
    if constexpr (IsSameType<UbInner, CumsumTwowaySklansky<DataType, PromoteDataType>>::value) {
        uint32_t nLen = initData_.ubFactorN;
        cacheStartInRes = (initData_.ubFactorR - 1) * nLen; // compact之后的最后一行第一个元素的index
        if (initData_.reverse) {
            cacheStartInRes = 0;
        }
    } else {
        cacheStartInRes = initData_.ubFactorR - 1; // r轴的行index
        if (initData_.reverse) {
            cacheStartInRes = 0;
        }
    }
    for (uint16_t j = 0; j < initData_.memoLen; ++j) {
        if constexpr (IsSameType<UbInner, CumsumTwowaySklansky<DataType, PromoteDataType>>::value) {
            VfBetweenUbAddTwoway(dstPtr, srcPtr, cacheStartInRes, j);
        } else {
            VfBetweenUbAddOneway(dstPtr, srcPtr, cacheStartInRes, j);
        }
        if (cacheID > j) {
            // 等于将dstTensor[j]清0
            Duplicate<PromoteDataType>(memoBuffer_[j * initData_.perMemoSize], 0, initData_.perMemoSize);
        }

        if (j == cacheID) {
            if constexpr (IsSameType<UbInner, CumsumTwowaySklansky<DataType, PromoteDataType>>::value) {
                VfBetweenUbWriteMemTwoway(dstPtr, srcPtr, cacheStartInRes, j);
            } else {
                VfBetweenUbWriteMemOneway(dstPtr, srcPtr, cacheStartInRes, j);
            }
        }
    }
}

} // namespace Cumsum

#endif // CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUMBASE_CUMSUM_UB_SKLANSKY_H
