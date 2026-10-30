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
 * \file grouped_bias_add_grad_unequal_c_perf.h
 * \brief
 */

#ifndef GROUPED_BIAS_ADD_GRAD_UNEQUAL_C_PERF_H
#define GROUPED_BIAS_ADD_GRAD_UNEQUAL_C_PERF_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "grouped_bias_add_grad_base.h"

namespace GroupedBiasAddGradAll {
using namespace AscendC;

template <typename T, typename G, const uint32_t USE_TYPE>
class GroupedBiasAddGradUnequalCPerf : public GroupedBiasAddGradBase<T>
{
public:
    __aicore__ inline GroupedBiasAddGradUnequalCPerf(){};
    __aicore__ inline void Init(
        GM_ADDR grad_y, GM_ADDR group_idx, GM_ADDR grad_bias, GM_ADDR workspace,
        const GroupedBiasAddGradTilingData& tilingData);
    __aicore__ inline void CopyInGroupIdAndCalcInterval(LocalTensor<int32_t>& interval, LocalTensor<int32_t>& groupIdx);
    __aicore__ inline void CopyInGroupIdAndCalcInterval(LocalTensor<int64_t>& interval, LocalTensor<int64_t>& groupIdx);
    __aicore__ inline void SortGroupIndex(LocalTensor<int32_t>& interval);
    __aicore__ inline void Process();

private:
    GlobalTensor<int32_t> groupIdxGm_;
    TQue<QuePosition::VECIN, 1> groupIntervalInQue_;
    TQue<QuePosition::VECIN, 1> groupIdxInQue_;
    TBuf<QuePosition::VECCALC> sortedTemp_;
    TBuf<QuePosition::VECCALC> groupIndex_;

    uint32_t sortedElemAlign_{0};
};

template <typename T, typename G, const uint32_t USE_TYPE>
__aicore__ inline void GroupedBiasAddGradUnequalCPerf<T, G, USE_TYPE>::Init(
    GM_ADDR grad_y, GM_ADDR group_idx, GM_ADDR grad_bias, GM_ADDR workspace,
    const GroupedBiasAddGradTilingData& tilingData)
{
    // Init tiling data
    this->InitBaseParams(grad_y, grad_bias, workspace, tilingData);

    groupIdxGm_.SetGlobalBuffer((__gm__ int32_t*)group_idx);

    // 个数按照32B对齐
    int64_t groupIdxAlign = (this->dimG_ + B32_BLOCK_NUM - 1) / B32_BLOCK_NUM * B32_BLOCK_NUM;
    if constexpr (IsSameType<T, int64_t>::value) {
        groupIdxAlign = (this->dimG_ + B64_BLOCK_NUM - 1) / B64_BLOCK_NUM * B64_BLOCK_NUM;
    }

    this->pipe_.InitBuffer(groupIntervalInQue_, 1, groupIdxAlign * sizeof(G));
    this->pipe_.InitBuffer(groupIdxInQue_, 1, groupIdxAlign * sizeof(G));
    // sort 需要按照32个数对齐
    sortedElemAlign_ = (this->dimG_ + BLOCK - 1) / BLOCK * BLOCK;
    this->pipe_.InitBuffer(sortedTemp_, sortedElemAlign_ * sizeof(int32_t) * TWO_NUM);
    this->pipe_.InitBuffer(groupIndex_, sortedElemAlign_ * sizeof(int32_t) * TWO_NUM);
}

template <typename T, typename G, const uint32_t USE_TYPE>
__aicore__ inline void GroupedBiasAddGradUnequalCPerf<T, G, USE_TYPE>::CopyInGroupIdAndCalcInterval(
    LocalTensor<int32_t>& interval, LocalTensor<int32_t>& groupIdx)
{
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(this->dimG_ * sizeof(int32_t)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    int32_t dimGMod = this->dimG_ % B32_BLOCK_NUM;
    int32_t processRightPad = dimGMod ? (B32_BLOCK_NUM - dimGMod) : 0;

    DataCopyPadExtParams<int32_t> padParams{
        true, static_cast<uint8_t>(0), static_cast<uint8_t>(processRightPad), static_cast<int32_t>(0)};
    DataCopyPad(interval, groupIdxGm_[0], copyParams, padParams);

    GroupedBiasAddGradBase<T>::CalcGroupInterval(interval, groupIdx, groupIdxGm_, processRightPad);

    groupIntervalInQue_.EnQue(interval);
    groupIdxInQue_.EnQue(groupIdx);
}

template <typename T, typename G, const uint32_t USE_TYPE>
__aicore__ inline void GroupedBiasAddGradUnequalCPerf<T, G, USE_TYPE>::CopyInGroupIdAndCalcInterval(
    LocalTensor<int64_t>& interval, LocalTensor<int64_t>& groupIdx)
{
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(this->dimG_ * sizeof(int64_t)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    int32_t dimGMod = this->dimG_ % B64_BLOCK_NUM;
    int32_t processRightPad = dimGMod ? (B64_BLOCK_NUM - dimGMod) : 0;

    DataCopyPadExtParams<int32_t> padParams{
        true, static_cast<uint8_t>(0), static_cast<uint8_t>(processRightPad * 2), static_cast<int32_t>(0)};

    LocalTensor<int32_t> intervalInt32 = interval.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> groupIdxInt32 = groupIdx.template ReinterpretCast<int32_t>();
    DataCopyPad(intervalInt32, groupIdxGm_[0], copyParams, padParams);

    GroupedBiasAddGradBase<T>::CalcGroupInterval(interval, groupIdx, groupIdxGm_, processRightPad);

    groupIntervalInQue_.EnQue(intervalInt32);
    groupIdxInQue_.EnQue(groupIdxInt32);
}

template <typename T, typename G, const uint32_t USE_TYPE>
__aicore__ inline void GroupedBiasAddGradUnequalCPerf<T, G, USE_TYPE>::SortGroupIndex(LocalTensor<int32_t>& interval)
{
    int64_t sortRepeatTimes = sortedElemAlign_ / BLOCK;
    LocalTensor<int32_t> indexLocal = groupIndex_.Get<int32_t>();
    LocalTensor<float> groupIdxCast = this->inQue_.template AllocTensor<float>();
    Duplicate<float>(groupIdxCast, -1, this->baseC_ * this->baseH_);

    CreateVecIndex(indexLocal, 0, this->dimG_);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint32_t> indexCast = indexLocal.ReinterpretCast<uint32_t>();

    Cast(groupIdxCast, interval, RoundMode::CAST_NONE, this->dimG_);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> sortedGroupIdxLocal = this->castBuf_.template Get<float>();
    LocalTensor<float> sortTmpLocal = sortedTemp_.Get<float>();

    groupIdxCast.SetSize(sortedElemAlign_);
    sortTmpLocal.SetSize(sortedElemAlign_ * TWO_NUM);
    sortedGroupIdxLocal.SetSize(sortedElemAlign_ * TWO_NUM);
    Sort<float, true>(sortedGroupIdxLocal, groupIdxCast, indexCast, sortTmpLocal, sortRepeatTimes);

    PipeBarrier<PIPE_V>();
    Extract(groupIdxCast, indexCast, sortedGroupIdxLocal, sortRepeatTimes);

    this->inQue_.template FreeTensor(groupIdxCast);
}

template <typename T, typename G, const uint32_t USE_TYPE>
__aicore__ inline void GroupedBiasAddGradUnequalCPerf<T, G, USE_TYPE>::Process()
{
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }

    LocalTensor<G> cValueTensor = groupIntervalInQue_.AllocTensor<G>();
    LocalTensor<G> groupIdxTensor = groupIdxInQue_.AllocTensor<G>();
    CopyInGroupIdAndCalcInterval(cValueTensor, groupIdxTensor);

    LocalTensor<int32_t> cValueLocal = groupIntervalInQue_.DeQue<int32_t>();
    LocalTensor<int32_t> groupIdxLocal = groupIdxInQue_.DeQue<int32_t>();
    SortGroupIndex(cValueLocal);
    LocalTensor<int32_t> indexLocal = groupIndex_.Get<int32_t>();
    int32_t coreNumber = this->usedCoreNum_ / TWO_NUM;
    int32_t offset = this->blockIdx_ / coreNumber;

    event_t eventVtoS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVtoS);
    WaitFlag<HardEvent::V_S>(eventVtoS);

    for (int64_t i = 0; i < this->processGHByCore_; i++) {
        this->gIdx_ = indexLocal(TWO_NUM * i + offset);

        int32_t cValue = cValueLocal(this->gIdx_);
        this->loopCNum_ = (cValue + this->baseC_ - 1) / this->baseC_;
        int64_t tailC = cValue % this->baseC_ == 0 ? this->baseC_ : cValue % this->baseC_;
        int64_t cPreValue = groupIdxLocal(this->gIdx_);
        for (int64_t h = 0; h < (this->hNum_ + coreNumber - 1) / coreNumber; h++) {
            this->hIdx_ = (this->blockIdx_ % coreNumber + coreNumber * h) % this->hNum_;
            if (unlikely(cValue == 0)) {
                int64_t tailH = this->dimH_ % this->baseH_ == 0 ? this->baseH_ : this->dimH_ % this->baseH_;
                this->processH_ = this->baseH_;
                bool isLastH = this->hIdx_ == (this->hNum_ - 1);
                if (unlikely(isLastH)) {
                    this->processH_ = tailH;
                }
                InitOutput<T>(
                    this->gradBiasGm_[this->gIdx_ * this->dimH_ + this->hIdx_ * this->baseH_], this->processH_, 0);
            } else if constexpr (USE_TYPE == USE_UB) {
                this->ComputePerGUb(cPreValue, tailC);
            } else if (this->loopCNum_ <= UB_GROUP_SUM_NUM) {
                this->ComputePerGUb(cPreValue, tailC);
            } else {
                this->ComputePerG(cPreValue, tailC);
            }
        }
    }
    groupIdxInQue_.FreeTensor(groupIdxLocal);
    groupIntervalInQue_.FreeTensor(cValueLocal);
}
} // namespace GroupedBiasAddGradAll
#endif