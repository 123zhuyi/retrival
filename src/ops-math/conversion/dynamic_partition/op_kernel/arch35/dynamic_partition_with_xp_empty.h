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
 * \file dynamic_partition_with_xp_empty.h
 * \brief impl of DynamicPartition which x and paritions are all empty
 */

#ifndef OP_KERNEL_DYNAMIC_PARTITION_WITH_XP_EMPTY_H_
#define OP_KERNEL_DYNAMIC_PARTITION_WITH_XP_EMPTY_H_

#include "dynamic_partition_base.h"

namespace DynPart
{
using namespace AscendC;

template <typename T>
class DynPartWithXPEmpty : public DynPartBase<T>
{
public:
    __aicore__ inline DynPartWithXPEmpty(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR partitions, GM_ADDR y, GM_ADDR yshape,
                                const DynPartTilingData* tilingDataPtr, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    const DynPartTilingData* tdPtr_ = nullptr;
};

template <typename T>
__aicore__ inline void DynPartWithXPEmpty<T>::Init(GM_ADDR x, GM_ADDR partitions, GM_ADDR y, GM_ADDR yshape,
                                                   const DynPartTilingData* tilingDataPtr, TPipe* pipeIn)
{
    tdPtr_ = tilingDataPtr;
    this->BaseInit(x, partitions, y, yshape, tilingDataPtr, pipeIn);
}

template <typename T>
__aicore__ inline void DynPartWithXPEmpty<T>::Process()
{
    int32_t partLpCnt = tdPtr_->numPartitions / NUM_PARTITION_UNIT;
    int32_t partLeft = tdPtr_->numPartitions % NUM_PARTITION_UNIT;
    for (int32_t pLpIdx = 0; pLpIdx < partLpCnt; ++pLpIdx) {
        int32_t begPart = pLpIdx * NUM_PARTITION_UNIT;
        int32_t endPart = (pLpIdx + 1) * NUM_PARTITION_UNIT;
        this->SetPartBaseZero(this->pBaseQue_);
        auto ubPartBase = this->pBaseQue_.template DeQue<uint64_t>();
        this->RefreshOutputShapes(ubPartBase, NUM_PARTITION_UNIT, pLpIdx * NUM_PARTITION_UNIT * SHAPE_GAP);
        this->pBaseQue_.FreeTensor(ubPartBase);
    }
    if (partLeft > 0) {
        int32_t begPart = partLpCnt * NUM_PARTITION_UNIT;
        int32_t endPart = tdPtr_->numPartitions;
        this->SetPartBaseZero(this->pBaseQue_);
        auto ubPartBase = this->pBaseQue_.template DeQue<uint64_t>();
        this->RefreshOutputShapes(ubPartBase, partLeft, partLpCnt * NUM_PARTITION_UNIT * SHAPE_GAP);
        this->pBaseQue_.FreeTensor(ubPartBase);
    }
}

}  // namespace DynPart

#endif  // OP_KERNEL_DYNAMIC_PARTITION_WITH_XP_EMPTY_H_