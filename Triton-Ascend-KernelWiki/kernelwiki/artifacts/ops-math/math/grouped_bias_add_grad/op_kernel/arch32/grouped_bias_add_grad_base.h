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
 * \file grouped_bias_add_grad_base.h
 * \brief
 */

#ifndef GROUPED_BIAS_ADD_GRAD_BASE_H
#define GROUPED_BIAS_ADD_GRAD_BASE_H

#include "kernel_operator.h"
namespace GroupedBiasAddGradAll {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t BLOCK = 32;
constexpr int64_t B64_BLOCK_NUM = BLOCK / sizeof(int64_t);
constexpr int64_t B32_BLOCK_NUM = BLOCK / sizeof(float);
constexpr int64_t B16_BLOCK_NUM = BLOCK / sizeof(half);
constexpr int32_t UB_GROUP_SUM_NUM = 8;
constexpr uint32_t USE_UB = 1;
constexpr uint32_t USE_WS = 0;
constexpr int32_t TWO_NUM = 2;

template <typename T>
class GroupedBiasAddGradBase
{
public:
    __aicore__ inline GroupedBiasAddGradBase(){};

    __aicore__ inline void InitBaseParams(
        GM_ADDR grad_y, GM_ADDR grad_bias, GM_ADDR workspace, const GroupedBiasAddGradTilingData& tilingData);
    __aicore__ inline void CustomTensorReduce(
        LocalTensor<float>& dst, LocalTensor<float>& src, int64_t rows, int64_t cols, bool append = false);
    __aicore__ inline void ComputePerG(const int64_t cPreValue, const int64_t tailC);
    __aicore__ inline void ComputePerGUb(const int64_t cPreValue, const int64_t tailC);
    __aicore__ inline void ComputePerLoop(const int64_t loop, const int64_t cPreValue);
    __aicore__ inline void ComputeBasePara();
    __aicore__ inline void CopyInGradY(const int64_t gradYGmAddr);
    __aicore__ inline void CalIntraGroupSum();
    __aicore__ inline void CopyOutWorkspace(const int64_t groupSumGmAddr);
    __aicore__ inline void CalBetweenGroupSum();
    __aicore__ inline void CopyInGroupSum(const int64_t sumGmAddr, const int64_t cNum);
    __aicore__ inline void CastAndCopyOut(LocalTensor<float>& sumOutLocal);
    __aicore__ inline void CopyOut(const int64_t gradBiasGmAddr);
    __aicore__ inline void ComputePerLoopUb(
        const int64_t loop, const int64_t cPreValue, LocalTensor<float>& groupSumLocal);
    __aicore__ inline void CalBetweenGroupSumUb(LocalTensor<float>& groupSumLocal);
    __aicore__ inline void CalcGroupInterval(
        LocalTensor<int32_t>& interval, LocalTensor<int32_t>& groupIdx, GlobalTensor<int32_t>& groupIdxGm,
        int32_t rightPad);
    __aicore__ inline void CalcGroupInterval(
        LocalTensor<int64_t>& interval, LocalTensor<int64_t>& groupIdx, GlobalTensor<int32_t>& groupIdxGm,
        int32_t rightPad);

    // 变量区
    /* ascendc variable */
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue_; // 存放输入、所有组内结果 basec * baseh(组内累加结果搬到ws情况使用)
    TQue<QuePosition::VECOUT, BUFFER_NUM> gradBiasQue_; // 存放组内累加结果、cast后最终组间累加结果 1 * baseh

    TBuf<QuePosition::VECCALC> castBuf_; // 存放cast32结果basec * baseh、cast16前组间累加结果 1 * baseh
    TBuf<QuePosition::VECCALC> sumBuf_;  // 存放每次loop累加结果

    GlobalTensor<T> gradYGm_;
    GlobalTensor<T> gradBiasGm_;
    GlobalTensor<float> groupSumWorkspaceGm_;

    int64_t usedCoreNum_{0};
    int64_t normalCoreNum_{0};
    int64_t normalCoreProcessNum_{0};
    int64_t tailCoreProcessNum_{0};
    int64_t wsUnitNum_{0};
    int64_t dimG_{0};
    int64_t dimC_{0};
    int64_t dimH_{0};

    int64_t baseH_{0};
    int64_t baseC_{0};
    int64_t loopCNum_{0};
    int64_t hNum_{0};
    int64_t processC_{0};      // 当前核当前循环处理的C方向大小，尾块时修改值
    int64_t processH_{0};      // 当前核当前循环处理的H方向大小，尾块时修改值
    int64_t processHAlign_{0}; // processH_ block对齐

    int64_t blockIdx_{0};
    int64_t processGHByCore_{0}; // 每个核处理的块数（按照G、H切的块）
    bool isFp32_{false};
    int32_t groupIdxType_{0};
    int64_t gIdx_{0}; // 当前核处理的G方向索引
    int64_t hIdx_{0}; // 当前核处理的H方向索引
};

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::InitBaseParams(
    GM_ADDR grad_y, GM_ADDR grad_bias, GM_ADDR workspace, const GroupedBiasAddGradTilingData& tilingData)
{
    usedCoreNum_ = tilingData.usedCoreNum;
    normalCoreNum_ = tilingData.normalCoreNum;
    normalCoreProcessNum_ = tilingData.normalCoreProcessNum;
    tailCoreProcessNum_ = tilingData.tailCoreProcessNum;
    wsUnitNum_ = tilingData.wsUnitNum;
    dimG_ = tilingData.dimG;
    dimH_ = tilingData.dimH;
    baseH_ = tilingData.baseH;
    baseC_ = tilingData.baseC;
    loopCNum_ = tilingData.loopCNum;
    groupIdxType_ = tilingData.groupIdxType;

    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    // init normal core and tail core process GHnum
    if (blockIdx_ < normalCoreNum_) {
        processGHByCore_ = normalCoreProcessNum_;
    } else {
        processGHByCore_ = tailCoreProcessNum_;
    }
    // init params
    gradYGm_.SetGlobalBuffer((__gm__ T*)grad_y);
    gradBiasGm_.SetGlobalBuffer((__gm__ T*)grad_bias);
    groupSumWorkspaceGm_.SetGlobalBuffer((__gm__ float*)workspace);

    isFp32_ = sizeof(T) == sizeof(float);
    pipe_.InitBuffer(inQue_, BUFFER_NUM, baseC_ * baseH_ * sizeof(float));
    pipe_.InitBuffer(gradBiasQue_, BUFFER_NUM, baseH_ * sizeof(float));
    pipe_.InitBuffer(castBuf_, baseC_ * baseH_ * sizeof(float));
    pipe_.InitBuffer(sumBuf_, UB_GROUP_SUM_NUM * baseH_ * sizeof(float));
    hNum_ = (dimH_ + baseH_ - 1) / baseH_;
}

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::CalcGroupInterval(
    LocalTensor<int32_t>& interval, LocalTensor<int32_t>& groupIdx, GlobalTensor<int32_t>& groupIdxGm, int32_t rightPad)
{
    if (this->groupIdxType_ == 0) {
        DataCopyExtParams copyParamsTemp{
            static_cast<uint16_t>(1), static_cast<uint32_t>((this->dimG_ - 1) * sizeof(int32_t)),
            static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

        DataCopyPadExtParams<int32_t> padParamsTemp{
            true, static_cast<uint8_t>(1), static_cast<uint8_t>(rightPad), static_cast<int32_t>(0)};
        DataCopyPad(groupIdx, groupIdxGm[0], copyParamsTemp, padParamsTemp);

        event_t eventMte2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2toV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2toV);

        Sub(interval, interval, groupIdx, this->dimG_);
    } else {
        event_t eventMte2toS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventMte2toS);
        WaitFlag<HardEvent::MTE2_S>(eventMte2toS);

        groupIdx.SetValue(0, 0);
        for (int64_t i = 1; i < this->dimG_; i++) {
            groupIdx.SetValue(i, interval(i - 1) + groupIdx(i - 1));
        }
    }
}

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::CalcGroupInterval(
    LocalTensor<int64_t>& interval, LocalTensor<int64_t>& groupIdx, GlobalTensor<int32_t>& groupIdxGm, int32_t rightPad)
{
    LocalTensor<int32_t> intervalInt32 = interval.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> groupIdxInt32 = groupIdx.template ReinterpretCast<int32_t>();

    if (this->groupIdxType_ == 0) {
        DataCopyExtParams copyParamsTemp{
            static_cast<uint16_t>(1), static_cast<uint32_t>((this->dimG_ - 1) * sizeof(int64_t)),
            static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

        DataCopyPadExtParams<int32_t> padParamsTemp{
            true, static_cast<uint8_t>(2), static_cast<uint8_t>(rightPad * 2), static_cast<int32_t>(0)};
        DataCopyPad(groupIdxInt32, groupIdxGm[0], copyParamsTemp, padParamsTemp);

        event_t eventMte2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2toV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2toV);

        Cast(intervalInt32, interval, RoundMode::CAST_NONE, this->dimG_);
        Cast(groupIdxInt32, groupIdx, RoundMode::CAST_NONE, this->dimG_);

        PipeBarrier<PIPE_V>();
        Sub(intervalInt32, intervalInt32, groupIdxInt32, this->dimG_);
    } else {
        event_t eventMte2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2toV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2toV);

        Cast(intervalInt32, interval, RoundMode::CAST_NONE, this->dimG_);
        Cast(groupIdxInt32, groupIdx, RoundMode::CAST_NONE, this->dimG_);
        PipeBarrier<PIPE_V>();

        groupIdxInt32.SetValue(0, 0);
        for (int64_t i = 1; i < this->dimG_; i++) {
            groupIdxInt32.SetValue(i, intervalInt32(i - 1) + groupIdxInt32(i - 1));
        }
    }
}

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::CustomTensorReduce(
    LocalTensor<float>& dst, LocalTensor<float>& src, int64_t rows, int64_t cols, bool append)
{
    if (!append) {
        Duplicate(dst, static_cast<float>(0), cols);
    }
    int64_t curRows = 0;
    int64_t halfRows = 0;
    curRows = rows;
    while (curRows > 1) {
        halfRows = (curRows + 1) / TWO_NUM;
        Add(src[0], src[0], src[halfRows * cols], (curRows - halfRows) * cols);
        PipeBarrier<PIPE_V>();
        curRows = halfRows;
    }
    PipeBarrier<PIPE_V>();
    Add(dst[0], dst[0], src[0], cols);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::ComputePerG(const int64_t cPreValue, const int64_t tailC)
{
    ComputeBasePara();
    for (int64_t loop = 0; loop < loopCNum_; loop++) {
        bool isLastC = loop == (loopCNum_ - 1);
        if (unlikely(isLastC)) {
            processC_ = tailC;
        }
        ComputePerLoop(loop, cPreValue);
    }
    event_t eventMte3toMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
    CalBetweenGroupSum();
}

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::ComputePerGUb(const int64_t cPreValue, const int64_t tailC)
{
    ComputeBasePara();
    LocalTensor<float> groupSumLocal = sumBuf_.Get<float>();
    for (int64_t loop = 0; loop < loopCNum_; loop++) {
        bool isLastC = loop == (loopCNum_ - 1);
        if (unlikely(isLastC)) {
            processC_ = tailC;
        }
        ComputePerLoopUb(loop, cPreValue, groupSumLocal);
    }
    CalBetweenGroupSumUb(groupSumLocal);
}

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::ComputeBasePara()
{
    int64_t tailH = dimH_ % baseH_ == 0 ? baseH_ : dimH_ % baseH_;
    // 每个核每次进来初始化processC_, processH_
    processC_ = baseC_;
    processH_ = baseH_;
    bool isLastH = hIdx_ == (hNum_ - 1);
    if (unlikely(isLastH)) {
        processH_ = tailH;
    }
    if constexpr (IsSameType<T, float>::value) {
        processHAlign_ = (processH_ + B32_BLOCK_NUM - 1) / B32_BLOCK_NUM * B32_BLOCK_NUM;
    } else {
        processHAlign_ = (processH_ + B16_BLOCK_NUM - 1) / B16_BLOCK_NUM * B16_BLOCK_NUM;
    }
}

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::ComputePerLoop(const int64_t loop, const int64_t cPreValue)
{
    int64_t gradYGmAddr = cPreValue * dimH_ + hIdx_ * baseH_ + loop * baseC_ * dimH_;
    int64_t groupSumGmAddr = blockIdx_ * wsUnitNum_ * baseH_ + loop * processHAlign_;
    // grad_y搬进ub
    CopyInGradY(gradYGmAddr);
    // 组内二分累加
    CalIntraGroupSum();
    // 组内二分累加结果搬到workspace
    CopyOutWorkspace(groupSumGmAddr);
}

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::CopyInGradY(const int64_t gradYGmAddr)
{
    LocalTensor<T> gradYLocal = inQue_.AllocTensor<T>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(processC_), static_cast<uint32_t>(processH_ * sizeof(T)),
        static_cast<uint32_t>((dimH_ - processH_) * sizeof(T)), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> padParams{
        true, static_cast<uint8_t>(0), static_cast<uint8_t>(processHAlign_ - processH_), static_cast<T>(0)};
    DataCopyPad(gradYLocal, gradYGm_[gradYGmAddr], copyParams, padParams);
    inQue_.EnQue(gradYLocal);
}

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::CalIntraGroupSum()
{
    LocalTensor<T> gradYLocal = inQue_.DeQue<T>();
    LocalTensor<float> castFp32 = castBuf_.Get<float>();
    if constexpr (IsSameType<T, float>::value) {
        Muls(castFp32, gradYLocal, 1.0f, processHAlign_ * processC_);
    } else {
        Cast(castFp32, gradYLocal, RoundMode::CAST_NONE, processHAlign_ * processC_);
    }
    PipeBarrier<PIPE_V>();
    inQue_.FreeTensor(gradYLocal);
    LocalTensor<float> groupSumLocal = gradBiasQue_.AllocTensor<float>();
    CustomTensorReduce(groupSumLocal, castFp32, processC_, processHAlign_);
    gradBiasQue_.EnQue<float>(groupSumLocal);
}

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::CopyOutWorkspace(const int64_t groupSumGmAddr)
{
    LocalTensor<float> groupSumLocal = gradBiasQue_.DeQue<float>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(processHAlign_ * sizeof(float)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(groupSumWorkspaceGm_[groupSumGmAddr], groupSumLocal, copyParams);
    gradBiasQue_.FreeTensor(groupSumLocal);
}

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::CalBetweenGroupSum()
{
    int64_t cNum = loopCNum_ < baseC_ ? loopCNum_ : baseC_;
    int64_t sumGmAddr = blockIdx_ * wsUnitNum_ * baseH_;
    LocalTensor<float> sumOutLocal = castBuf_.Get<float>();
    // 所有组内累加结果搬进ub
    CopyInGroupSum(sumGmAddr, cNum);
    // 组内累加结果再进行分组累加
    LocalTensor<float> sumLocal = inQue_.DeQue<float>();
    CustomTensorReduce(sumOutLocal, sumLocal, cNum, processHAlign_);
    inQue_.FreeTensor(sumLocal);
    auto loopNum = (loopCNum_ + baseC_ - 1) / baseC_;
    for (int64_t loop = 1; loop < loopNum; loop++) {
        int64_t sumGmAddr = blockIdx_ * wsUnitNum_ * baseH_ + loop * baseC_ * processHAlign_;
        bool isLastLoop = loop == (loopNum - 1);
        if (unlikely(isLastLoop)) {
            cNum = loopCNum_ % baseC_ == 0 ? baseC_ : loopCNum_ % baseC_;
        }
        LocalTensor<float> sumOutLocal = castBuf_.Get<float>();
        // 所有组内累加结果搬进ub
        CopyInGroupSum(sumGmAddr, cNum);
        // 组内累加结果再进行分组累加
        LocalTensor<float> sumLocal = inQue_.DeQue<float>();
        CustomTensorReduce(sumOutLocal, sumLocal, cNum, processHAlign_, true);
        inQue_.FreeTensor(sumLocal);
    }
    CastAndCopyOut(sumOutLocal);
}

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::CopyInGroupSum(const int64_t sumGmAddr, const int64_t cNum)
{
    LocalTensor<float> sumLocal = inQue_.AllocTensor<float>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(cNum * processHAlign_ * sizeof(float)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(cNum * processHAlign_ - processHAlign_),
        static_cast<uint32_t>(0)};

    DataCopyPadExtParams<float> padParams{
        false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<float>(0)};
    DataCopyPad(sumLocal, groupSumWorkspaceGm_[sumGmAddr], copyParams, padParams);
    inQue_.EnQue(sumLocal);
}

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::CastAndCopyOut(LocalTensor<float>& sumOutLocal)
{
    LocalTensor<T> gradBiasLocal = gradBiasQue_.AllocTensor<T>();
    if constexpr (IsSameType<T, float>::value) {
        Muls(gradBiasLocal, sumOutLocal, 1.0f, processH_);
    } else {
        Cast(gradBiasLocal, sumOutLocal, RoundMode::CAST_ROUND, processH_);
    }
    gradBiasQue_.EnQue<T>(gradBiasLocal);
    int64_t gradBiasGmAddr = gIdx_ * dimH_ + hIdx_ * baseH_;
    CopyOut(gradBiasGmAddr);
}

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::CopyOut(const int64_t gradBiasGmAddr)
{
    LocalTensor<T> gradBiasLocal = gradBiasQue_.DeQue<T>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(processH_ * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(gradBiasGm_[gradBiasGmAddr], gradBiasLocal, copyParams);
    gradBiasQue_.FreeTensor(gradBiasLocal);
}

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::ComputePerLoopUb(
    const int64_t loop, const int64_t cPreValue, LocalTensor<float>& groupSumLocal)
{
    int64_t gradYGmAddr = cPreValue * dimH_ + hIdx_ * baseH_ + loop * baseC_ * dimH_;
    // grad_y搬进ub
    CopyInGradY(gradYGmAddr);
    // 组内二分累加
    LocalTensor<T> gradYLocal = inQue_.DeQue<T>();
    LocalTensor<float> castFp32 = castBuf_.Get<float>();
    if constexpr (IsSameType<T, float>::value) {
        Muls(castFp32, gradYLocal, 1.0f, processHAlign_ * processC_);
    } else {
        Cast(castFp32, gradYLocal, RoundMode::CAST_NONE, processHAlign_ * processC_);
    }
    PipeBarrier<PIPE_V>();
    inQue_.FreeTensor(gradYLocal);
    auto tmpLocal = groupSumLocal[loop * baseH_];
    CustomTensorReduce(tmpLocal, castFp32, processC_, processHAlign_);
}

template <typename T>
__aicore__ inline void GroupedBiasAddGradBase<T>::CalBetweenGroupSumUb(LocalTensor<float>& groupSumLocal)
{
    LocalTensor<float> sumOutLocal = castBuf_.Get<float>();
    // 组内累加结果再进行分组累加
    CustomTensorReduce(sumOutLocal, groupSumLocal, loopCNum_, baseH_);
    CastAndCopyOut(sumOutLocal);
}
} // namespace GroupedBiasAddGradAll

#endif // GROUPED_BIAS_ADD_GRAD_BASE_H
