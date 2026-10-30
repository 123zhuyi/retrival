/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file amp_update_scale.h
 * \brief
 */
#ifndef AMP_UPDATE_SCALE_H
#define AMP_UPDATE_SCALE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
constexpr uint32_t BlockBytes = 32;
// IEEE 754 float32 位字段常量
constexpr uint32_t FLOAT32_ABS_MASK = 0x7FFFFFFF;
constexpr int FLOAT32_EXP_SHIFT = 23;
constexpr uint32_t FLOAT32_EXP_MASK = 0xFF;

template<typename T>
class AmpUpdateScale {
    public:
        __aicore__ inline AmpUpdateScale() {
        }

        __aicore__ inline void Init(GM_ADDR current_scale, GM_ADDR growth_tracker, GM_ADDR found_inf, GM_ADDR updated_scale,
                                    GM_ADDR updated_growth_tracker, float growthFactor, float backoffFactor, int32_t growthInterval, TPipe *tmpPipe) {
            pipe_ = tmpPipe;

            growthFactor_ = growthFactor;
            backoffFactor_ = backoffFactor;
            growthInterval_ = growthInterval;

            currentScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(current_scale), 1);
            growthTrackerGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(growth_tracker), 1);
            foundInfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(found_inf), 1);

            updatedScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(updated_scale), 1);
            updatedGrowthTrackerGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(updated_growth_tracker), 1);

            pipe_->InitBuffer(currentScaleBuf_, BlockBytes);
            pipe_->InitBuffer(growthTrackerBuf_, BlockBytes);
            pipe_->InitBuffer(foundInfBuf_, BlockBytes);

            pipe_->InitBuffer(updatedScaleBuf_, BlockBytes);
            pipe_->InitBuffer(updatedGrowthTrackerBuf_, BlockBytes);

            if constexpr (IsSameType<T, bfloat16_t>::value) {
                pipe_->InitBuffer(castBuf_, BlockBytes);
            }
        }

        __aicore__ inline void Process() {
            Compute();
        }

    private:
        __aicore__ inline void SyncM2toV()
        {
            event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
        }

        __aicore__ inline void SyncVtoM3()
        {
            event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventId);
            WaitFlag<HardEvent::V_MTE3>(eventId);
        }

        __aicore__ inline void SyncM3toM2()
        {
            event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        }

        __aicore__ inline bool IsFinite(float value)
        {
            uint32_t tmpValue = *((uint32_t*)&value);
            if ((tmpValue & FLOAT32_ABS_MASK) >> FLOAT32_EXP_SHIFT == FLOAT32_EXP_MASK) {
                return false;
            } else {
                return true;
            }
        }

        __aicore__ inline void LoadInputData() {
            LocalTensor<T> currentScaleLocalT = currentScaleBuf_.Get<T>();
            LocalTensor<int32_t> growthTrackerLocalT = growthTrackerBuf_.Get<int32_t>();
            LocalTensor<T> foundInfLocalT = foundInfBuf_.Get<T>();

            DataCopyExtParams copyParamsCurrentScale{1, 1 * sizeof(T), 0, 0, 0};
            DataCopyExtParams copyParamsGrowthTracker{1, 1 * sizeof(int32_t), 0, 0, 0};
            DataCopyPadExtParams<T> padParamsCurrentScale{false, 0, 0, 0};
            DataCopyPadExtParams<int32_t> padParamsGrowthTracker{false, 0, 0, 0};

            DataCopyPad(currentScaleLocalT, currentScaleGm_, copyParamsCurrentScale, padParamsCurrentScale);
            DataCopyPad(growthTrackerLocalT, growthTrackerGm_, copyParamsGrowthTracker, padParamsGrowthTracker);
            DataCopyPad(foundInfLocalT, foundInfGm_, copyParamsCurrentScale, padParamsCurrentScale);
            SyncM2toV();

            if constexpr (IsSameType<T, float>::value) {
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                currentScale_ = currentScaleLocalT.GetValue(0);
                foundInf_ = foundInfLocalT.GetValue(0);
            } else if constexpr (IsSameType<T, half>::value) {
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                currentScale_ = static_cast<float>(currentScaleLocalT.GetValue(0));
                foundInf_ = static_cast<float>(foundInfLocalT.GetValue(0));
            } else {
                LocalTensor<float> castTmp = castBuf_.Get<float>();
                Cast(castTmp, currentScaleLocalT, RoundMode::CAST_NONE, 1);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                currentScale_ = castTmp.GetValue(0);
                Cast(castTmp, foundInfLocalT, RoundMode::CAST_NONE, 1);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                foundInf_ = castTmp.GetValue(0);
            }
            growthTracker_ = growthTrackerLocalT.GetValue(0);
        }

        __aicore__ inline void ComputeScaleUpdate() {
            if (foundInf_) {
                currentScale_ *= backoffFactor_;
                growthTracker_ = 0;
            } else {
                successful_ = growthTracker_ + 1;
                if (successful_ == growthInterval_) {
                    newScale_ = currentScale_ * growthFactor_;
                    if (IsFinite(newScale_)) {
                        currentScale_ = newScale_;
                    }
                    growthTracker_ = 0;
                } else {
                    growthTracker_ = successful_;
                }
            }
        }

        __aicore__ inline void StoreOutputData() {
            LocalTensor<T> updatedScaleLocalT = updatedScaleBuf_.Get<T>();
            LocalTensor<int32_t> updatedGrowthTrackerLocalT = updatedGrowthTrackerBuf_.Get<int32_t>();

            if constexpr (IsSameType<T, float>::value) {
                updatedScaleLocalT.SetValue(0, currentScale_);
            } else if constexpr (IsSameType<T, half>::value) {
                updatedScaleLocalT.SetValue(0, static_cast<T>(currentScale_));
            } else {
                LocalTensor<float> castTmp = castBuf_.Get<float>();
                castTmp.SetValue(0, currentScale_);
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                Cast(updatedScaleLocalT, castTmp, RoundMode::CAST_RINT, 1);
            }
            updatedGrowthTrackerLocalT.SetValue(0, growthTracker_);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            SyncVtoM3();

            DataCopyExtParams copyParamsCurrentScale{1, 1 * sizeof(T), 0, 0, 0};
            DataCopyExtParams copyParamsGrowthTracker{1, 1 * sizeof(int32_t), 0, 0, 0};
            DataCopyPad(updatedScaleGm_, updatedScaleLocalT, copyParamsCurrentScale);
            DataCopyPad(updatedGrowthTrackerGm_, updatedGrowthTrackerLocalT, copyParamsGrowthTracker);
            SyncM3toM2();
        }

        __aicore__ inline void Compute() {
            LoadInputData();
            ComputeScaleUpdate();
            StoreOutputData();
        }

        TPipe *pipe_;

        float growthFactor_;
        float backoffFactor_;
        float currentScale_;
        float newScale_;
        float foundInf_;
        int32_t growthInterval_;
        int32_t growthTracker_;
        int32_t successful_;

        GlobalTensor<T> currentScaleGm_, foundInfGm_, updatedScaleGm_;
        GlobalTensor<int32_t> growthTrackerGm_, updatedGrowthTrackerGm_;

        TBuf<> growthTrackerBuf_, foundInfBuf_, currentScaleBuf_;
        TBuf<> updatedGrowthTrackerBuf_, updatedScaleBuf_;
        TBuf<> castBuf_;
};
#endif // AMP_UPDATE_SCALE_H