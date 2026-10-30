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
 * \file dynamic_partition_with_h_mc.h
 * \brief impl of DynamicPartition which multiple cores at axis H
 */

#ifndef OP_KERNEL_DYNAMIC_PARTITION_WITH_H_MC_H_
#define OP_KERNEL_DYNAMIC_PARTITION_WITH_H_MC_H_

#include "dynamic_partition_base.h"

namespace DynPart {
using namespace AscendC;

template <typename T, bool isSingleW = false>
class DynPartWithHMC : public DynPartBase<T> {
public:
    __aicore__ inline DynPartWithHMC(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR partitions, GM_ADDR y, GM_ADDR yshape, GM_ADDR workspace,
                                const DynPartTilingData* tilingDataPtr, TPipe* pipeIn);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CalcPartCnt(int32_t begPart, int32_t endPart);
    __aicore__ inline void CalcPartBase(int32_t begPart, int32_t endPart);

private:
    __aicore__ inline void InnerProcess4CalcPartCnt(uint32_t ubPartLen, int32_t begPart, int32_t endPart);

protected:
    GlobalTensor<uint64_t> ws_;
    GlobalTensor<int32_t> pInGM_;

private:
    const DynPartTilingData* tdPtr_ = nullptr;
    TQue<QuePosition::VECIN, 1> pR1InQue_; // calc partition base value for each core
    uint32_t ubPartLen_ = 0;
};

template <typename T, bool isSingleW>
__aicore__ inline void DynPartWithHMC<T, isSingleW>::Init(GM_ADDR x, GM_ADDR partitions, GM_ADDR y, GM_ADDR yshape,
                                                          GM_ADDR workspace, const DynPartTilingData* tilingDataPtr,
                                                          TPipe* pipeIn)
{
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace); // 获取用户workspace指针
    if (usrWorkspace == nullptr) {
        return;
    }
    tdPtr_ = tilingDataPtr;
    this->BaseInit(x, partitions, y, yshape, tilingDataPtr, pipeIn);

    // ping pong
    uint32_t leftUBSize = (tdPtr_->totalUBSize - this->coreWSAlign_ * sizeof(uint64_t) * NUM_TWO) / NUM_TWO /
                          BLOCK_SIZE * BLOCK_SIZE;
    ubPartLen_ = static_cast<uint32_t>(leftUBSize / sizeof(int32_t));
    this->bufPoolSync_.InitBuffer(pR1InQue_, NUM_TWO, leftUBSize);
    pInGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(partitions));
    ws_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(usrWorkspace));
}

template <typename T, bool isSingleW>
__aicore__ inline void DynPartWithHMC<T, isSingleW>::Process()
{
    int32_t partLpCnt = tdPtr_->numPartitions / NUM_PARTITION_UNIT;
    int32_t partLeft = tdPtr_->numPartitions % NUM_PARTITION_UNIT;

    for (int32_t pLpIdx = 0; pLpIdx < partLpCnt; ++pLpIdx) {
        int32_t begPart = pLpIdx * NUM_PARTITION_UNIT;
        int32_t endPart = (pLpIdx + 1) * NUM_PARTITION_UNIT;
        CalcPartBase(begPart, endPart);
        if constexpr (!isSingleW) {
            this->MultiplePartProcess(this->pBaseQue_, begPart, endPart);
        } else {
            this->SinglePartProcess(this->pBaseQue_, begPart, endPart);
        }
        auto ubPartBase = this->pBaseQue_.template DeQue<uint64_t>();
        this->RefreshOutputShapes(ubPartBase, NUM_PARTITION_UNIT,
                                  static_cast<int64_t>(pLpIdx * NUM_PARTITION_UNIT * SHAPE_GAP));
        this->pBaseQue_.FreeTensor(ubPartBase);
    }
    if (partLeft > 0) {
        int32_t begPart = partLpCnt * NUM_PARTITION_UNIT;
        int32_t endPart = tdPtr_->numPartitions;
        CalcPartBase(begPart, endPart);
        if constexpr (!isSingleW) {
            this->MultiplePartProcess(this->pBaseQue_, begPart, endPart);
        } else {
            this->SinglePartProcess(this->pBaseQue_, begPart, endPart);
        }
        auto ubPartBase = this->pBaseQue_.template DeQue<uint64_t>();
        this->RefreshOutputShapes(ubPartBase, partLeft,
                                  static_cast<int64_t>(partLpCnt * NUM_PARTITION_UNIT * SHAPE_GAP));
        this->pBaseQue_.FreeTensor(ubPartBase);
    }
}

template <typename T, bool isSingleW>
__aicore__ inline void DynPartWithHMC<T, isSingleW>::InnerProcess4CalcPartCnt(uint32_t ubPartLen, int32_t begPart,
                                                                              int32_t endPart)
{
    auto ubPart = pR1InQue_.template DeQue<int32_t>();
    auto ubPartCnt = this->pCntQue_.template DeQue<uint64_t>();
    auto ubPartBase = this->pBaseQue_.template DeQue<uint64_t>();
    __local_mem__ int32_t* ptrPart = (__local_mem__ int32_t*)ubPart.GetPhyAddr();
    __local_mem__ uint64_t* ptrPartBase = (__local_mem__ uint64_t*)ubPartBase.GetPhyAddr();
    __local_mem__ int32_t* ptrPartMid = reinterpret_cast<__local_mem__ int32_t*>(ptrPartBase);
    constexpr auto dFactor = sizeof(uint64_t) / sizeof(int32_t);
    uint16_t pLpCntVF = static_cast<uint16_t>(Ops::Base::CeilDiv(ubPartLen, this->b32VLSize_));

    for (int32_t partIdx = begPart; partIdx < endPart; ++partIdx) {
        uint32_t partSize = ubPartLen;
        uint32_t modPartIdx = partIdx % NUM_PARTITION_UNIT;
        __VEC_SCOPE__
        {
            RegTensor<int32_t> parts;
            RegTensor<int32_t> midPartCnt;
            RegTensor<int32_t> partCnt;
            Reg::Duplicate(partCnt, int32_t(0));
            RegTensor<int32_t> allTensor;
            Reg::Duplicate(allTensor, int32_t(1));
            MaskReg mask;
            MaskReg cmpMask;
            MaskReg vl1Mask = CreateMask<int32_t, Reg::MaskPattern::VL1>();

            for (uint16_t i = 0; i < pLpCntVF; ++i) {
                mask = UpdateMask<int32_t>(partSize);
                Reg::DataCopy(parts, ptrPart + i * this->b32VLSize_);
                Reg::CompareScalar(cmpMask, parts, partIdx, mask);
                Reg::ReduceSum(midPartCnt, allTensor, cmpMask);
                Reg::Add(partCnt, partCnt, midPartCnt, vl1Mask);
            }
            // partCnt must be less than max int32_t
            Reg::DataCopy<int32_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(ptrPartMid + modPartIdx * dFactor, partCnt,
                                                                           vl1Mask);
        }
    }
    Add(ubPartCnt, ubPartCnt, ubPartBase, this->coreWS_);
    pR1InQue_.FreeTensor(ubPart);
    this->pBaseQue_.EnQue(ubPartBase);
    this->pCntQue_.EnQue(ubPartCnt);
}

template <typename T, bool isSingleW>
__aicore__ inline void DynPartWithHMC<T, isSingleW>::CalcPartCnt(int32_t begPart, int32_t endPart)
{
    auto ubPartCnt = this->pCntQue_.template AllocTensor<uint64_t>();
    auto ubPartBase = this->pBaseQue_.template AllocTensor<uint64_t>();
    Duplicate(ubPartCnt, 0UL, this->coreWSAlign_);
    Duplicate(ubPartBase, 0UL, this->coreWSAlign_);
    this->pCntQue_.EnQue(ubPartCnt);
    this->pBaseQue_.EnQue(ubPartBase);

    DataCopyPadExtParams<int32_t> copyPadParams{false, 0, 0, 0};
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(ubPartLen_ * sizeof(int32_t)), 0, 0, 0};
    int64_t partLen = (this->blockIdx_ != tdPtr_->usedCoreCnt - 1) ? tdPtr_->hMSize : tdPtr_->hTSize;
    int64_t partLpCnt = partLen / ubPartLen_;
    int64_t partLeft = partLen % ubPartLen_;
    for (int64_t lpIdx = 0; lpIdx < partLpCnt; ++lpIdx) {
        auto ubPart = pR1InQue_.template AllocTensor<int32_t>();
        DataCopyPad(ubPart, pInGM_[this->blockIdx_ * tdPtr_->hMSize + lpIdx * ubPartLen_], copyParams, copyPadParams);
        pR1InQue_.EnQue(ubPart);
        InnerProcess4CalcPartCnt(ubPartLen_, begPart, endPart);
    }
    if (partLeft > 0) {
        auto ubPart = pR1InQue_.template AllocTensor<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(partLeft * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(ubPart, pInGM_[this->blockIdx_ * tdPtr_->hMSize + partLpCnt * ubPartLen_], copyParams,
                    copyPadParams);
        pR1InQue_.EnQue(ubPart);
        InnerProcess4CalcPartCnt(partLeft, begPart, endPart);
    }
    ubPartBase = this->pBaseQue_.template DeQue<uint64_t>();
    this->pBaseQue_.FreeTensor(ubPartBase);
    ubPartCnt = this->pCntQue_.template DeQue<uint64_t>();
    this->InsertSync(HardEvent::V_MTE3);
    DataCopyExtParams copyParams1{1, static_cast<uint32_t>(this->coreWS_ * sizeof(uint64_t)), 0, 0, 0};
    DataCopyPad(ws_[this->blockIdx_ * this->coreWS_], ubPartCnt, copyParams1);
    this->pCntQue_.FreeTensor(ubPartCnt);
}

template <typename T, bool isSingleW>
__aicore__ inline void DynPartWithHMC<T, isSingleW>::CalcPartBase(int32_t begPart, int32_t endPart)
{
    // size of ubPartBase is equal to ubPartCnt
    CalcPartCnt(begPart, endPart);
    SyncAll();

    auto ubPartBase = this->pBaseQue_.template AllocTensor<uint64_t>();
    Duplicate(ubPartBase, 0UL, this->coreWSAlign_);
    this->pBaseQue_.EnQue(ubPartBase);

    if (this->blockIdx_ != 0) {
        ubPartBase = this->pBaseQue_.template DeQue<uint64_t>();
        auto ubPartCnt = this->pCntQue_.template AllocTensor<uint64_t>();
        for (int64_t i = 0; i < this->blockIdx_; ++i) {
            // make sure ub is enough
            DataCopyPadExtParams<uint64_t> copyPadParams{false, 0, 0, 0};
            DataCopyExtParams copyParams1{1, static_cast<uint32_t>(this->coreWS_ * sizeof(uint64_t)), 0, 0, 0};
            DataCopyPad(ubPartCnt, ws_[i * this->coreWS_], copyParams1, copyPadParams);
            this->InsertSync(HardEvent::MTE2_V);
            Add(ubPartBase, ubPartBase, ubPartCnt, this->coreWS_);
            this->InsertSync(HardEvent::V_MTE2);
        }
        this->pCntQue_.FreeTensor(ubPartCnt);
        this->pBaseQue_.EnQue(ubPartBase);
    }
    this->bufPoolSync_.Reset();
}

} // namespace DynPart

#endif // OP_KERNEL_DYNAMIC_PARTITION_WITH_H_MC_H_
