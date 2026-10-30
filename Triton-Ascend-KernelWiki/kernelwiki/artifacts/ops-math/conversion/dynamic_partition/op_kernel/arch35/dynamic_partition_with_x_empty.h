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
 * \file dynamic_partition_with_x_empty.h
 * \brief impl of DynamicPartition which x is empty
 */

#ifndef OP_KERNEL_DYNAMIC_PARTITION_WITH_X_EMPTY_H_
#define OP_KERNEL_DYNAMIC_PARTITION_WITH_X_EMPTY_H_

#include "dynamic_partition_with_h_mc.h"

namespace DynPart
{
using namespace AscendC;

template <typename T>
class DynPartWithXEmpty : public DynPartWithHMC<T>
{
public:
    __aicore__ inline DynPartWithXEmpty(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR partitions, GM_ADDR y, GM_ADDR yshape, GM_ADDR workspace,
                                const DynPartTilingData* tilingDataPtr, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcPartBase();

private:
    const DynPartTilingData* tdPtr_ = nullptr;
    int32_t begPart_ = 0;
    int32_t endPart_ = 0;
};

template <typename T>
__aicore__ inline void DynPartWithXEmpty<T>::Init(GM_ADDR x, GM_ADDR partitions, GM_ADDR y, GM_ADDR yshape,
                                                  GM_ADDR workspace, const DynPartTilingData* tilingDataPtr,
                                                  TPipe* pipeIn)
{
    tdPtr_ = tilingDataPtr;
    DynPartWithHMC<T>::Init(x, partitions, y, yshape, workspace, tilingDataPtr, pipeIn);
}

template <typename T>
__aicore__ inline void DynPartWithXEmpty<T>::Process()
{
    int32_t partLpCnt = tdPtr_->numPartitions / NUM_PARTITION_UNIT;
    int32_t partLeft = tdPtr_->numPartitions % NUM_PARTITION_UNIT;
    for (int32_t pLpIdx = 0; pLpIdx < partLpCnt; ++pLpIdx) {
        begPart_ = pLpIdx * NUM_PARTITION_UNIT;
        endPart_ = (pLpIdx + 1) * NUM_PARTITION_UNIT;
        CalcPartBase();
        auto ubPartBase = this->pBaseQue_.template DeQue<uint64_t>();
        this->RefreshOutputShapes(ubPartBase, NUM_PARTITION_UNIT,
                                  static_cast<int64_t>(pLpIdx * NUM_PARTITION_UNIT * SHAPE_GAP));
        this->pBaseQue_.FreeTensor(ubPartBase);
    }
    if (partLeft > 0) {
        begPart_ = partLpCnt * NUM_PARTITION_UNIT;
        endPart_ = tdPtr_->numPartitions;
        CalcPartBase();
        auto ubPartBase = this->pBaseQue_.template DeQue<uint64_t>();
        this->RefreshOutputShapes(ubPartBase, partLeft,
                                  static_cast<int64_t>(partLpCnt * NUM_PARTITION_UNIT * SHAPE_GAP));
        this->pBaseQue_.FreeTensor(ubPartBase);
    }
}

template <typename T>
__aicore__ inline void DynPartWithXEmpty<T>::CalcPartBase()
{
    this->CalcPartCnt(begPart_, endPart_);
    SyncAll();

    if (this->blockIdx_ == tdPtr_->usedCoreCnt - 1) {
        auto ubPartBase = this->pBaseQue_.template AllocTensor<uint64_t>();
        Duplicate(ubPartBase, 0UL, this->coreWSAlign_);
        this->pBaseQue_.EnQue(ubPartBase);
        ubPartBase = this->pBaseQue_.template DeQue<uint64_t>();
        auto ubPartCnt = this->pCntQue_.template AllocTensor<uint64_t>();
        for (int64_t i = 0; i <= this->blockIdx_; ++i) {
            // make sure ub is enough
            DataCopyPadExtParams<uint64_t> copyPadParams{false, 0, 0, 0};
            DataCopyExtParams copyParams1{1, static_cast<uint32_t>(this->coreWS_ * sizeof(uint64_t)), 0, 0, 0};
            DataCopyPad(ubPartCnt, this->ws_[i * this->coreWS_], copyParams1, copyPadParams);
            this->InsertSync(HardEvent::MTE2_V);
            Add(ubPartBase, ubPartBase, ubPartCnt, this->coreWS_);
            this->InsertSync(HardEvent::V_MTE2);
        }
        this->pCntQue_.FreeTensor(ubPartCnt);
        this->pBaseQue_.EnQue(ubPartBase);
    }
}

}  // namespace DynPart

#endif  // OP_KERNEL_DYNAMIC_PARTITION_WITH_X_EMPTY_H_