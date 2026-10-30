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
 * \file grouped_bias_add_grad_equal_c.h
 * \brief
 */

#ifndef GROUPED_BIAS_ADD_GRAD_EQUAL_C_H
#define GROUPED_BIAS_ADD_GRAD_EQUAL_C_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "grouped_bias_add_grad_base.h"

namespace GroupedBiasAddGradAll {
using namespace AscendC;

template <typename T, const uint32_t USE_TYPE>
class GroupedBiasAddGradEqualC : public GroupedBiasAddGradBase<T>
{
public:
    __aicore__ inline GroupedBiasAddGradEqualC(){};
    __aicore__ inline void Init(
        GM_ADDR grad_y, GM_ADDR grad_bias, GM_ADDR workspace, const GroupedBiasAddGradTilingData& tilingData);
    __aicore__ inline void Process();

private:
    int64_t dimC_{0};
};

template <typename T, const uint32_t USE_TYPE>
__aicore__ inline void GroupedBiasAddGradEqualC<T, USE_TYPE>::Init(
    GM_ADDR grad_y, GM_ADDR grad_bias, GM_ADDR workspace, const GroupedBiasAddGradTilingData& tilingData)
{
    // Init tiling data
    this->InitBaseParams(grad_y, grad_bias, workspace, tilingData);
    dimC_ = tilingData.dimC;
}

template <typename T, const uint32_t USE_TYPE>
__aicore__ inline void GroupedBiasAddGradEqualC<T, USE_TYPE>::Process()
{
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }

    int64_t tailC = dimC_ % this->baseC_ == 0 ? this->baseC_ : dimC_ % this->baseC_;
    for (int64_t i = 0; i < this->processGHByCore_; i++) {
        this->gIdx_ = (this->blockIdx_ + this->usedCoreNum_ * i) / this->hNum_;
        this->hIdx_ = (this->blockIdx_ + this->usedCoreNum_ * i) % this->hNum_;
        int64_t cPreValue = this->gIdx_ * dimC_;
        if (unlikely(dimC_ == 0)) {
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
        } else if constexpr (USE_TYPE == USE_WS) {
            this->ComputePerG(cPreValue, tailC);
        }
    }
}
} // namespace GroupedBiasAddGradAll
#endif