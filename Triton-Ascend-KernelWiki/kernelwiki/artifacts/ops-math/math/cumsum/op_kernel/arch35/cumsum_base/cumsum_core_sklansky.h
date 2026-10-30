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
 * \file cumsum_core_sklansky.h
 * \brief cumsum sklansky between cores.
 */

#ifndef CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_BASE_CUMSUM_CORE_SKLANSKY_H
#define CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_BASE_CUMSUM_CORE_SKLANSKY_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "cumsum_base.h"

namespace Cumsum {

using namespace AscendC;

constexpr int32_t SS_ITER_MAX = 6;
constexpr int32_t ADD_DATA_CASTED_MULTI = 2;
constexpr int8_t ADD_FACTOR_NDDMA_DIM = 2;

template <typename T, typename PromtT>
class CumsumCoreSklansky : public CumsumBase<T> {
public:
    struct CoreSklanskyPara {
        bool isReverse = false;                // 属性
        int32_t nBorrowLen = 0;                // N借轴开多核的数量
        int64_t rLen = 0;                      // R轴总长
        int64_t nLen = 0;                      // N轴总长
        int64_t rFactor = 0;                   // R的blockFactor
        bool isTailN = false;                  // 处理的是否是尾块N
        int32_t nFactor = 0;                   // N的blockFactor
        int32_t addFactorRepeatTimes = 0;      // addFactor重复几次
        int32_t mCoreOffset = 0;               // 当前核在M轴上的偏移量，单位1
        int32_t nCoreOffset = 0;               // 当前核在N轴上的偏移量，单位nFactor
        int64_t mOffset = 0;                   // 当前核在M轴上的偏移量的起始位置，单位元素个数
        int32_t nOffset = 0;                   // 当前核在N轴上的偏移量的起始位置，单位元素个数
        int32_t coreGroupCount = 0;            // 按M*nBorrowLen，分组后的数量
        int32_t coreGroupCoreNum = 0;          // 按M*nBorrowLen，分组后每一组使用的核数
        int32_t sklanskyItersCount = 0;        // sklansky迭代行数
        int32_t addFactorBufferSize = 0;       // addFactor，UB缓存的大小
        int32_t addDataBufferSize = 0;         // addData，UB缓存的大小
        int32_t addFactorOffsetBufferSize = 0; // addFactor偏移量，UB缓存的大小

        CumsumSklanskyItersTiling sklanskyItersTiling[SS_ITER_MAX]; // sklansky迭代每一行的tiling
    };

public:
    __aicore__ inline CumsumCoreSklansky(TPipe& pipe) : CumsumBase<T>(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* td);

    __aicore__ inline void Process();

private:
    __aicore__ inline void PreProcess();

    __aicore__ inline bool CalcIterGroupOffset(int32_t ssi);

    __aicore__ inline void CopyInAddFactor(int32_t ssi);

    __aicore__ inline void DoDtypeCastAddFactorIfNeed();

    __aicore__ inline void LoopAddData(int32_t ssi);

    __aicore__ inline CumsumIterGroupTiling& JudgeIterGroupTiling(CumsumSklanskyItersTiling& ssiTiling);

    __aicore__ inline int64_t CalcIterGroupCoreStart(CumsumSklanskyItersTiling& ssiTiling,
                                                     CumsumIterGroupTiling& iterGroupTiling);

    __aicore__ inline void CalcUbLoopInfo(CumsumIterGroupTiling& iterGroupTiling, int32_t& ubLoop, int32_t& ubFactor,
                                          int32_t& ubTailFactor);

    __aicore__ inline void SetCopyParam(bool isUbTail, int32_t ubFactor, int32_t ubTailFactor,
                                        DataCopyExtParams& copyParam);

    __aicore__ inline void Mte2AddData(int64_t addDataStart, DataCopyExtParams& copyParam);

    __aicore__ inline void DoDtypeCastAddDataIfNeed(DataCopyExtParams& copyParam);

    __aicore__ inline void DoAdd(DataCopyExtParams& copyParam);

    __aicore__ inline void DoAddTailN(DataCopyExtParams& copyParam);

    __aicore__ inline void CalcAddFactorCircleCount();

    __aicore__ inline void DoAddMainN(DataCopyExtParams& copyParam);

    __aicore__ inline void DoDtypeCastBackAddDataIfNeed(DataCopyExtParams& copyParam);

    __aicore__ inline void Mte3AddData(int64_t addDataStart, DataCopyExtParams& copyParam);

private:
    /* 一次性初始化的类成员 */
    CoreSklanskyPara coreSsPara_; // 外部初始化的参数，主要来源于TilingData
    uint32_t vlElementCount_ = 0; // VL长度容纳的元素个数

    /* 伴随sklansky迭代行数会更新的类成员 */
    int32_t iterGroupOffset_ = 0;       // 当前核在每一行迭代中的组偏移
    int32_t iterGroupInnerOffset_ = 0;  // 当前核在每一行迭代中的某一组中组内的核偏移
    uint16_t addFactorCircleCount_ = 0; // addFactor每一轮重复的次数

    /* UB buffer 信息 */
    TBuf<> addFactor_;       // 前一R分组中的最大值的一行
    TBuf<> addData_;         // 与前一R分组中的最大值的一行，进行行相加的每一行数据
    TBuf<> addFactorCasted_; // 前一R分组中的最大值的一行，精度提升后
    TBuf<> addDataCasted_; // 与前一R分组中的最大值的一行，进行行相加的每一行数据，精度提升后
    TBuf<> addFactorOffset_; // 每次copy addFactor数据的偏移量

    static constexpr MultiCopyConfig nddmaConfig_ = {false};
}; // class CumsumCoreSklansky

template <typename T, typename PromtT>
__aicore__ inline void CumsumCoreSklansky<T, PromtT>::Init(GM_ADDR x, GM_ADDR y, const CumsumSklanskyTilingData* td)
{
    CumsumBase<T>::Init(x, y);

    coreSsPara_.isReverse = td->reverse == 1 ? true : false;
    coreSsPara_.nBorrowLen = td->nBorrow;
    coreSsPara_.rLen = td->lenR;
    coreSsPara_.nLen = td->lenN;
    coreSsPara_.rFactor = td->rBlockPara.blockFactor;
    coreSsPara_.coreGroupCount = td->coreGroupCount;
    coreSsPara_.coreGroupCoreNum = td->coreGroupCoreNum;
    coreSsPara_.sklanskyItersCount = td->sklanskyItersCount;
    coreSsPara_.mCoreOffset = this->blockIdx_ / (coreSsPara_.nBorrowLen * coreSsPara_.coreGroupCoreNum);
    coreSsPara_.nCoreOffset = (this->blockIdx_ -
                               coreSsPara_.nBorrowLen * coreSsPara_.coreGroupCoreNum * coreSsPara_.mCoreOffset) /
                              coreSsPara_.coreGroupCoreNum;
    coreSsPara_.mOffset = coreSsPara_.mCoreOffset * coreSsPara_.rLen * coreSsPara_.nLen;
    coreSsPara_.nOffset = coreSsPara_.nCoreOffset * td->nBlockPara.blockFactor;
    coreSsPara_.isTailN = (coreSsPara_.nCoreOffset == coreSsPara_.nBorrowLen - 1);
    coreSsPara_.nFactor = coreSsPara_.isTailN ? td->nBlockPara.blockTailFactor : td->nBlockPara.blockFactor;
    coreSsPara_.addFactorRepeatTimes = coreSsPara_.isTailN ? td->addFactorRepeatTimesTail :
                                                             td->addFactorRepeatTimesMain;
    coreSsPara_.sklanskyItersTiling[0] = td->sklanskyItersTiling0;
    coreSsPara_.sklanskyItersTiling[1] = td->sklanskyItersTiling1;
    coreSsPara_.sklanskyItersTiling[2] = td->sklanskyItersTiling2;
    coreSsPara_.sklanskyItersTiling[3] = td->sklanskyItersTiling3;
    coreSsPara_.sklanskyItersTiling[4] = td->sklanskyItersTiling4;
    coreSsPara_.sklanskyItersTiling[5] = td->sklanskyItersTiling5;
    coreSsPara_.addFactorBufferSize = td->addFactorBufferSize;
    coreSsPara_.addDataBufferSize = td->addDataBufferSize;
    coreSsPara_.addFactorOffsetBufferSize = td->addFactorOffsetBufferSize;
    vlElementCount_ = Ops::Base::GetVRegSize() / sizeof(PromtT);
}

template <typename T, typename PromtT>
__aicore__ inline void CumsumCoreSklansky<T, PromtT>::Process()
{
    PreProcess();

    for (int32_t i = 0; i < coreSsPara_.sklanskyItersCount; i++) {
        // 核间全同步，必须保证blockdim的所有核都能走到这里，下边有continue，所以放在这里
        SyncAll();

        // 计算addFactor的轮次
        if (i == 0) {
            CalcAddFactorCircleCount();
        }

        // 不工作的核跳过
        if (this->blockIdx_ >= coreSsPara_.coreGroupCount * coreSsPara_.coreGroupCoreNum) {
            continue;
        }

        // 计算当前核在行迭代中，所属组的偏移量，如果返回false，说明当前核在行迭代中轮空
        if (!CalcIterGroupOffset(i)) {
            continue;
        }

        // 搬入addFactor
        CopyInAddFactor(i);

        // 如果需要提升精度，做Cast
        DoDtypeCastAddFactorIfNeed();

        // UB上轮询处理Add
        LoopAddData(i);
    }
}

template <typename T, typename PromtT>
__aicore__ inline void CumsumCoreSklansky<T, PromtT>::PreProcess()
{
    this->pipe_.Reset();
    this->pipe_.InitBuffer(addFactor_, coreSsPara_.addFactorBufferSize);
    this->pipe_.InitBuffer(addData_, coreSsPara_.addDataBufferSize);
    this->pipe_.InitBuffer(addFactorOffset_, coreSsPara_.addFactorOffsetBufferSize);
    if constexpr (!IsSameType<T, PromtT>::value) {
        this->pipe_.InitBuffer(addFactorCasted_, coreSsPara_.addFactorBufferSize);
        this->pipe_.InitBuffer(addDataCasted_, coreSsPara_.addDataBufferSize * ADD_DATA_CASTED_MULTI);
    }
}

template <typename T, typename PromtT>
__aicore__ inline bool CumsumCoreSklansky<T, PromtT>::CalcIterGroupOffset(int32_t ssi)
{
    iterGroupOffset_ = (this->blockIdx_ -
                        coreSsPara_.nBorrowLen * coreSsPara_.coreGroupCoreNum * coreSsPara_.mCoreOffset -
                        coreSsPara_.coreGroupCoreNum * coreSsPara_.nCoreOffset) /
                       coreSsPara_.sklanskyItersTiling[ssi].iterGroupCoreNum;

    // 每一组分到的核可能用不完，用不完的核轮空跳过。当前核的组偏移大于组数，则跳过
    if (iterGroupOffset_ > coreSsPara_.sklanskyItersTiling[ssi].iterGroupCount - 1) {
        return false;
    }

    return true;
}

template <typename T, typename PromtT>
__aicore__ inline void CumsumCoreSklansky<T, PromtT>::CopyInAddFactor(int32_t ssi)
{
    CumsumSklanskyItersTiling& ssiTiling = coreSsPara_.sklanskyItersTiling[ssi];

    /* 计算addFactor搬入的起始位置，M方向偏移+R方向偏移+N方向偏移 */
    // R方向偏移
    int64_t rOffset = 0;
    if (coreSsPara_.isReverse == false) {
        rOffset = ((ssiTiling.iterGroupStartOffset + iterGroupOffset_ * ssiTiling.iterGroupAddOffset + 1) *
                       coreSsPara_.rFactor -
                   1) *
                  coreSsPara_.nLen;
    } else {
        rOffset = (coreSsPara_.rLen -
                   (ssiTiling.iterGroupStartOffset + iterGroupOffset_ * ssiTiling.iterGroupAddOffset + 1) *
                       coreSsPara_.rFactor) *
                  coreSsPara_.nLen;
    }

    int64_t addFactorOffset = coreSsPara_.mOffset + rOffset + coreSsPara_.nOffset;

    // 搬入UB，使用nddma broadcast
    NdDmaDci();
    MultiCopyParams<T, ADD_FACTOR_NDDMA_DIM> copyParam;
    copyParam.constantValue = 0;
    copyParam.loopInfo = {
        {1, 0},
        {1, static_cast<uint32_t>(coreSsPara_.nFactor)},
        {static_cast<uint32_t>(coreSsPara_.nFactor), static_cast<uint32_t>(coreSsPara_.addFactorRepeatTimes)},
        {0, 0},
        {0, 0}};
    DataCopy<T, ADD_FACTOR_NDDMA_DIM, nddmaConfig_>(addFactor_.Get<T>(), this->yGm_[addFactorOffset], copyParam);

    SetFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    WaitFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
}

template <typename T, typename PromtT>
__aicore__ inline void CumsumCoreSklansky<T, PromtT>::DoDtypeCastAddFactorIfNeed()
{
    // 不需要提升精度直接返回
    if constexpr (IsSameType<T, PromtT>::value) {
        return;
    }

    Cast(addFactorCasted_.Get<PromtT>(), addFactor_.Get<T>(), RoundMode::CAST_NONE,
         coreSsPara_.nFactor * coreSsPara_.addFactorRepeatTimes);
}

/* UB上轮询处理add数据，分为5步
1、搬入数据
2、如果需要提升精度，Cast
3、Add
4、如果需要提升精度，还原Cast
5、搬出数据 */
template <typename T, typename PromtT>
__aicore__ inline void CumsumCoreSklansky<T, PromtT>::LoopAddData(int32_t ssi)
{
    CumsumSklanskyItersTiling& ssiTiling = coreSsPara_.sklanskyItersTiling[ssi];
    CumsumIterGroupTiling& iterGroupTiling = JudgeIterGroupTiling(ssiTiling);

    // 计算当前核在每一行迭代中的某一组中组内的核偏移
    iterGroupInnerOffset_ = this->blockIdx_ -
                            coreSsPara_.nBorrowLen * coreSsPara_.coreGroupCoreNum * coreSsPara_.mCoreOffset -
                            coreSsPara_.coreGroupCoreNum * coreSsPara_.nCoreOffset -
                            ssiTiling.iterGroupCoreNum * iterGroupOffset_;

    // 如果当前核不需要处理数据，轮空，直接返回。这种场景不存在，这里只是做异常保护
    if (iterGroupInnerOffset_ >= iterGroupTiling.iterGroupMainBlockCount + iterGroupTiling.iterGroupTailBlockCount) {
        return;
    }

    // 计算当前核处理数据的起始位置
    int64_t iterGroupCoreStart = CalcIterGroupCoreStart(ssiTiling, iterGroupTiling);

    /* 计算UB轮询的信息 */
    int32_t ubLoop = 0;
    int32_t ubFactor = 0;
    int32_t ubTailFactor = 0;
    CalcUbLoopInfo(iterGroupTiling, ubLoop, ubFactor, ubTailFactor);

    /* UB上的轮询 */
    for (int32_t i = 0; i < ubLoop; i++) {
        // 计算addData搬入的起始偏移量
        int64_t addDataStart = iterGroupCoreStart + i * ubFactor * coreSsPara_.nLen;

        // 是否是最后一次UB轮询
        bool isTailUbLoop = (i == ubLoop - 1);

        // 计算copy参数
        DataCopyExtParams copyParam;
        SetCopyParam(isTailUbLoop, ubFactor, ubTailFactor, copyParam);

        // 搬入addData，GM->UB
        Mte2AddData(addDataStart, copyParam);

        // 如果需要提升精度，Cast
        DoDtypeCastAddDataIfNeed(copyParam);

        // 执行Add
        DoAdd(copyParam);

        // 如果需要提升精度，还原Cast
        DoDtypeCastBackAddDataIfNeed(copyParam);

        // 搬出addData，UB->GM
        Mte3AddData(addDataStart, copyParam);
    }
}

template <typename T, typename PromtT>
__aicore__ inline CumsumIterGroupTiling& CumsumCoreSklansky<T, PromtT>::JudgeIterGroupTiling(
    CumsumSklanskyItersTiling& ssiTiling)
{
    if (coreSsPara_.isTailN) {
        if (iterGroupOffset_ < ssiTiling.iterGroupCount - 1) {
            return ssiTiling.mainIterGroupTailNTiling; // 主块Group，尾块N
        } else {
            return ssiTiling.tailIterGroupTailNTiling; // 尾块Group，尾块N
        }
    } else {
        if (iterGroupOffset_ < ssiTiling.iterGroupCount - 1) {
            return ssiTiling.mainIterGroupMainNTiling; // 主块Group，主块N
        } else {
            return ssiTiling.tailIterGroupMainNTiling; // 主块Group，主块N
        }
    }
}

template <typename T, typename PromtT>
__aicore__ inline int64_t CumsumCoreSklansky<T, PromtT>::CalcIterGroupCoreStart(CumsumSklanskyItersTiling& ssiTiling,
                                                                                CumsumIterGroupTiling& iterGroupTiling)
{
    /* 计算当前核在每一行迭代中的某一组中组内的核偏移，单位是元素个数 */
    int64_t iterGroupCoreOffset = 0;
    if (iterGroupInnerOffset_ < iterGroupTiling.iterGroupMainBlockCount) { // 主核
        iterGroupCoreOffset = iterGroupInnerOffset_ * iterGroupTiling.iterGroupBlockFactor * coreSsPara_.nLen;
    } else { // 尾核
        iterGroupCoreOffset = (iterGroupTiling.iterGroupMainBlockCount * iterGroupTiling.iterGroupBlockFactor +
                               (iterGroupInnerOffset_ - iterGroupTiling.iterGroupMainBlockCount) *
                                   (iterGroupTiling.iterGroupBlockFactor - 1)) *
                              coreSsPara_.nLen;
    }

    /* 计算当前核处理的组的开始偏移量，单位元素个数 */
    int64_t iterGroupStart = 0;
    if (coreSsPara_.isReverse == false) {
        iterGroupStart = (iterGroupOffset_ * ssiTiling.iterGroupAddOffset + ssiTiling.iterGroupAddCount) *
                         coreSsPara_.rFactor * coreSsPara_.nLen;
    } else {
        int64_t rOffset = coreSsPara_.rLen -
                          (iterGroupOffset_ + 1) * ssiTiling.iterGroupAddOffset * coreSsPara_.rFactor;
        if (rOffset < 0) {
            iterGroupStart = 0;
        } else {
            iterGroupStart = rOffset * coreSsPara_.nLen;
        }
    }

    // M偏移量 + N偏移量 + 组起始位置 + 组内的偏移量
    return coreSsPara_.mOffset + coreSsPara_.nOffset + iterGroupStart + iterGroupCoreOffset;
}

template <typename T, typename PromtT>
__aicore__ inline void CumsumCoreSklansky<T, PromtT>::CalcUbLoopInfo(CumsumIterGroupTiling& iterGroupTiling,
                                                                     int32_t& ubLoop, int32_t& ubFactor,
                                                                     int32_t& ubTailFactor)
{
    if (iterGroupInnerOffset_ < iterGroupTiling.iterGroupMainBlockCount) { // 主核
        ubLoop = iterGroupTiling.iterGroupMainCoreUbFactorCount;
        ubFactor = iterGroupTiling.iterGroupMainCoreUbFactor;
        ubTailFactor = iterGroupTiling.iterGroupMainCoreUbTailFactor;
    } else { // 尾核
        ubLoop = iterGroupTiling.iterGroupTailCoreUbFactorCount;
        ubFactor = iterGroupTiling.iterGroupTailCoreUbFactor;
        ubTailFactor = iterGroupTiling.iterGroupTailCoreUbTailFactor;
    }
}

template <typename T, typename PromtT>
__aicore__ inline void CumsumCoreSklansky<T, PromtT>::SetCopyParam(bool isUbTail, int32_t ubFactor,
                                                                   int32_t ubTailFactor, DataCopyExtParams& copyParam)
{
    copyParam.blockCount = ubFactor;
    if (isUbTail) { // UB尾块
        copyParam.blockCount = ubTailFactor;
    }
    copyParam.blockLen = coreSsPara_.nFactor * sizeof(T);
    copyParam.srcStride = (coreSsPara_.nLen - coreSsPara_.nFactor) * sizeof(T);
    copyParam.dstStride = 0;
}

template <typename T, typename PromtT>
__aicore__ inline void CumsumCoreSklansky<T, PromtT>::Mte2AddData(int64_t addDataStart, DataCopyExtParams& copyParam)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams = {false, 0, 0, 0};
    DataCopyPad<T, PaddingMode::Compact>(addData_.Get<T>(), this->yGm_[addDataStart], copyParam, dataCopyPadExtParams);

    SetFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    WaitFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
}

template <typename T, typename PromtT>
__aicore__ inline void CumsumCoreSklansky<T, PromtT>::DoDtypeCastAddDataIfNeed(DataCopyExtParams& copyParam)
{
    // 如果不需要提升精度，直接返回
    if constexpr (IsSameType<T, PromtT>::value) {
        return;
    }

    Cast(addDataCasted_.Get<PromtT>(), addData_.Get<T>(), RoundMode::CAST_NONE,
         copyParam.blockCount * copyParam.blockLen / sizeof(T));
}

template <typename T, typename PromtT>
__aicore__ inline void CumsumCoreSklansky<T, PromtT>::DoAdd(DataCopyExtParams& copyParam)
{
    if (coreSsPara_.isTailN) { // 尾块N
        DoAddTailN(copyParam);
    } else { // 主块N
        DoAddMainN(copyParam);
    }
}

template <typename T, typename PromtT>
__aicore__ inline void CumsumCoreSklansky<T, PromtT>::DoAddTailN(DataCopyExtParams& copyParam)
{
    /* 初始化Add动作的参数 */
    __ubuf__ PromtT* addFactor = nullptr;
    __ubuf__ PromtT* addData = nullptr;
    uint16_t addLoop = 0;  // 以VL为单位，需要执行Add的次数
    uint32_t addTotal = 0; // 一共需要Add的数据量
    if constexpr (IsSameType<T, PromtT>::value) {
        addFactor = (__ubuf__ T*)addFactor_.Get<T>().GetPhyAddr();
        addData = (__ubuf__ T*)addData_.Get<T>().GetPhyAddr();
        addLoop = Ops::Base::CeilDiv(copyParam.blockCount * copyParam.blockLen, Ops::Base::GetVRegSize());
        addTotal = copyParam.blockCount * copyParam.blockLen / sizeof(PromtT);
    } else {
        addFactor = (__ubuf__ PromtT*)addFactorCasted_.Get<PromtT>().GetPhyAddr();
        addData = (__ubuf__ PromtT*)addDataCasted_.Get<PromtT>().GetPhyAddr();
        addLoop = Ops::Base::CeilDiv(copyParam.blockCount * copyParam.blockLen / sizeof(T) * sizeof(PromtT),
                                     static_cast<uint64_t>(Ops::Base::GetVRegSize()));
        addTotal = copyParam.blockCount * copyParam.blockLen / sizeof(T) * sizeof(PromtT) / sizeof(PromtT);
    }

    uint16_t addFactorCircleLoop = Ops::Base::CeilDiv(addLoop, addFactorCircleCount_);
    __ubuf__ uint32_t* addFactorOffset = (__ubuf__ uint32_t*)addFactorOffset_.Get<int32_t>().GetPhyAddr();
    uint32_t vlElementCount = vlElementCount_;
    uint16_t addFactorCircleCountTail = addLoop - (addFactorCircleLoop - 1) * addFactorCircleCount_;
    uint32_t addTotalTail = addTotal - ((addFactorCircleLoop - 1) * addFactorCircleCount_) * vlElementCount;

    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<PromtT, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::RegTensor<uint32_t> addFactorIndexRegTensor;
        AscendC::Reg::RegTensor<PromtT> addFactorRegTensor;
        AscendC::Reg::RegTensor<PromtT> addDataRegTensor;
        for (uint16_t i = 0; i < addFactorCircleLoop - 1; i++) {
            for (uint16_t j = 0; j < addFactorCircleCount_; j++) {
                // 搬入addFactor
                AscendC::Reg::AddrReg addrReg = AscendC::Reg::CreateAddrReg<uint32_t>(j, vlElementCount);
                AscendC::Reg::DataCopy(addFactorIndexRegTensor, addFactorOffset, addrReg);
                AscendC::Reg::DataCopyGather(addFactorRegTensor, addFactor, addFactorIndexRegTensor, p0);

                // 对齐连续搬入addData
                DataCopy(addDataRegTensor, addData + (i * addFactorCircleCount_ + j) * vlElementCount);

                // Add
                mask = AscendC::Reg::UpdateMask<uint32_t>(addTotal);
                Add(addDataRegTensor, addFactorRegTensor, addDataRegTensor, mask);

                // 对齐连续搬出addData
                DataCopy(addData + (i * addFactorCircleCount_ + j) * vlElementCount, addDataRegTensor, mask);
            }
        }

        // 尾块，只需执行一次
        for (uint16_t j = 0; j < addFactorCircleCountTail; j++) {
            // 搬入addFactor
            AscendC::Reg::AddrReg addrReg = AscendC::Reg::CreateAddrReg<uint32_t>(j, vlElementCount);
            AscendC::Reg::DataCopy(addFactorIndexRegTensor, addFactorOffset, addrReg);
            AscendC::Reg::DataCopyGather(addFactorRegTensor, addFactor, addFactorIndexRegTensor, p0);

            // 对齐连续搬入addData
            DataCopy(addDataRegTensor,
                     addData + ((addFactorCircleLoop - 1) * addFactorCircleCount_ + j) * vlElementCount);

            // Add
            mask = AscendC::Reg::UpdateMask<uint32_t>(addTotalTail);
            Add(addDataRegTensor, addFactorRegTensor, addDataRegTensor, mask);

            // 对齐连续搬出addData
            DataCopy(addData + ((addFactorCircleLoop - 1) * addFactorCircleCount_ + j) * vlElementCount,
                     addDataRegTensor, mask);
        }
    } // __VEC_SCOPE__
}

template <typename T, typename PromtT>
__aicore__ inline void CumsumCoreSklansky<T, PromtT>::CalcAddFactorCircleCount()
{
    for (uint16_t i = 0; i < coreSsPara_.nFactor; i++) {
        int32_t startIdx = i * vlElementCount_ % coreSsPara_.nFactor;
        if (startIdx == 0 && i != 0) {
            break;
        }
        addFactorCircleCount_++;
        LocalTensor<int32_t> addFactorTensor = addFactorOffset_.Get<int32_t>();
        CreateVecIndex(addFactorTensor[i * vlElementCount_], startIdx, vlElementCount_);
    }
}

template <typename T, typename PromtT>
__aicore__ inline void CumsumCoreSklansky<T, PromtT>::DoAddMainN(DataCopyExtParams& copyParam)
{
    LocalTensor<PromtT> addFactor;
    LocalTensor<PromtT> addData;
    if constexpr (IsSameType<T, PromtT>::value) {
        addFactor = addFactor_.Get<T>();
        addData = addData_.Get<T>();
    } else {
        addFactor = addFactorCasted_.Get<PromtT>();
        addData = addDataCasted_.Get<PromtT>();
    }

    for (int32_t i = 0; i < copyParam.blockCount; i++) {
        Add(addData[i * vlElementCount_], addFactor, addData[i * vlElementCount_], vlElementCount_);
    }
}

template <typename T, typename PromtT>
__aicore__ inline void CumsumCoreSklansky<T, PromtT>::DoDtypeCastBackAddDataIfNeed(DataCopyExtParams& copyParam)
{
    // 如果不需要提升精度，直接返回
    if constexpr (IsSameType<T, PromtT>::value) {
        return;
    }

    Cast(addData_.Get<T>(), addDataCasted_.Get<PromtT>(), RoundMode::CAST_RINT,
         copyParam.blockCount * copyParam.blockLen / sizeof(T));
}

template <typename T, typename PromtT>
__aicore__ inline void CumsumCoreSklansky<T, PromtT>::Mte3AddData(int64_t addDataStart, DataCopyExtParams& copyParam)
{
    SetFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    WaitFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    copyParam.dstStride = copyParam.srcStride;
    copyParam.srcStride = 0;
    DataCopyPad<T, PaddingMode::Compact>(this->yGm_[addDataStart], addData_.Get<T>(), copyParam);

    SetFlag<HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    WaitFlag<HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
}

} // namespace Cumsum

#endif // CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_BASE_CUMSUM_CORE_SKLANSKY_H
