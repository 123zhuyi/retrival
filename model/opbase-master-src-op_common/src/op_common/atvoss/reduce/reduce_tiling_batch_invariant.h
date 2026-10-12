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
 * \file reduce_batch_invariant_tiling.h
 * \brief batchInvariant=1 分支的 tiling 模板实现，被 reduce_tiling.cpp include
 *        注意：本文件非独立编译单元，不可单独编译
 */

#ifndef _REDUCE_BATCH_INVARIANT_TILING_H_
#define _REDUCE_BATCH_INVARIANT_TILING_H_

#include "op_common/atvoss/reduce/reduce_tiling.h"
#include "op_common/op_host/util/math_util.h"

namespace Ops {
namespace Base {

constexpr static int32_t AXES_STEP = 2; // A轴和R轴循环遍历的步长
constexpr static double THRES_HOLD = 0.95;
constexpr static uint64_t TAIL_R_THRES = 64;

inline void AssembleUnit(ReduceTilingUnit& unit, int32_t idx, uint64_t inner, uint64_t outer, uint64_t step)
{
    unit.idx = idx;
    unit.inner = inner;
    unit.outer = outer;
    unit.step = step;
}

template <class Pattern>
void ReduceOpTiling::ComputeRFirst(const uint64_t* shape)
{
    uint64_t dSize = ge::GetSizeByDataType(opInput_.inputDtype);
    uint64_t ubBlockSize = compileInfo_->ubBlockSize / dSize;
    uint64_t cacheSize = compileInfo_->cacheLineSize / dSize;
    uint64_t bBlockNum = basicBlock_ * Ratio() / opDag_.maxInputBytes;
    // AR和尾轴为R且尾轴R大于64，ub内不需要预留cacheline大小的空间；其余场景预留cacheline大小空间
    needReserveA_ = !(Pattern::ID == PATTERN_AR || (!Pattern::TailA && shape[Pattern::Dim - 1] > TAIL_R_THRES));
    OP_LOGD(context_, "needReserveA is %d", needReserveA_);
    // 尾轴为A时预留cachesize+ubBlockSize大小的空间，因为kernel侧尾轴A可能按照cacheSize + ubBlockSize对齐
    uint64_t maxRInUB = needReserveA_ ? bBlockNum / cacheSize : bBlockNum;
    uint64_t innerR = 1;
    int32_t startR = Pattern::TailA ? Pattern::Dim - AXES_STEP : Pattern::Dim - CONST1;
    int32_t endR = Pattern::FirstA ? CONST1 : CONST0;
    int32_t iRCut = -1;

    for (int32_t iR = startR; iR >= endR; iR -= AXES_STEP) {
        uint64_t axisLen = shape[iR];
        uint64_t alignedLen = (iR == Pattern::Dim - 1) ? CeilAlign(axisLen, ubBlockSize) : axisLen;
        if (innerR * alignedLen <= maxRInUB) {
            innerR *= alignedLen;
            continue;
        }
        iRCut = iR;
        break;
    }

    if (iRCut == -1) {
        AssembleUnit(unitR_, -1, innerR, 1, 1);
        tilingData_->factorRCntPerCore = 1;
        tilingData_->groupR = 1;
        return;
    }

    uint64_t ubFactorR = maxRInUB / innerR;
    // 如果R能被均匀切分且大于1/2ub内最大的R，则均匀切分，减少补pad
    if (!isPrime(shape[iRCut])) {
        uint64_t maxStep = maxRInUB / 2 / innerR;
        for (uint64_t i = ubFactorR; i >= maxStep; i--) {
            if (shape[iRCut] % i == 0) {
                ubFactorR = i;
                break;
            }
        }
    }
    if (iRCut == Pattern::Dim - 1) {
        ubFactorR = FloorAlign(ubFactorR, ubBlockSize);
    }
    innerR = innerR * ubFactorR;

    uint64_t outerR = CeilDiv(shape[iRCut], ubFactorR);
    for (int32_t iR = iRCut - AXES_STEP; iR >= endR; iR -= AXES_STEP) {
        outerR *= shape[iR];
    }

    AssembleUnit(unitR_, iRCut, innerR, outerR, ubFactorR);

    if (outerR > static_cast<uint64_t>(compileInfo_->vectorCoreNum)) {
        tilingData_->groupR = static_cast<uint64_t>(compileInfo_->vectorCoreNum);
        tilingData_->factorRCntPerCore = CeilDiv(outerR, static_cast<uint64_t>(compileInfo_->vectorCoreNum));
        tilingData_->groupR = CeilDiv(outerR, tilingData_->factorRCntPerCore);
    } else {
        tilingData_->factorRCntPerCore = 1;
        tilingData_->groupR = outerR;
    }
}

template <class Pattern>
void ReduceOpTiling::ComputeAWithRCannotFullLoad(const uint64_t* shape)
{
    uint64_t ubBlockSize = compileInfo_->ubBlockSize / opDag_.minInputBytes;
    uint64_t bBlockNum = basicBlock_ * Ratio() / opDag_.maxInputBytes;
    uint64_t innerR = unitR_.inner;
    uint64_t maxInnerA = resultBlock_ / opDag_.maxInputBytes;

    uint64_t innerA = 1;
    int32_t startA = Pattern::TailA ? Pattern::Dim - CONST1 : Pattern::Dim - AXES_STEP;
    int32_t endA = Pattern::FirstA ? CONST0 : CONST1;
    int32_t iACut = -1;
    for (int32_t iA = startA; iA >= endA; iA -= AXES_STEP) {
        uint64_t axisLen = shape[iA];
        uint64_t alignedLen = (iA == Pattern::Dim - CONST1) ? CeilAlign(axisLen, ubBlockSize) : axisLen;
        if (innerA * alignedLen * innerR <= bBlockNum && innerA * alignedLen <= maxInnerA) {
            innerA *= alignedLen;
            continue;
        }
        iACut = iA;
        break;
    }

    uint64_t outerA = 1;
    for (int32_t iA = iACut - AXES_STEP; iA >= endA; iA -= AXES_STEP) {
        outerA *= shape[iA];
    }
    uint64_t ubFactorA = bBlockNum / (innerR * innerA);
    if (iACut == -1) {
        outerA = 1;
        ubFactorA = shape[endA];
    } else {
        ubFactorA = iACut == Pattern::Dim - CONST1 ? FloorAlign(ubFactorA, ubBlockSize) : ubFactorA;
        if (innerA * ubFactorA > maxInnerA) {
            ubFactorA = FloorDiv(maxInnerA, innerA);
        }
        outerA *= CeilDiv(shape[iACut], ubFactorA);
        innerA = innerA * ubFactorA;
    }
    AssembleUnit(unitA_, iACut, innerA, outerA, ubFactorA);
}

template <class Pattern>
void ReduceOpTiling::ComputeAWithRFullLoad(const uint64_t* shape)
{
    uint64_t dSize = opDag_.minInputBytes;
    uint64_t cacheSize = compileInfo_->cacheLineSize / dSize;
    uint64_t ubBlockSize = compileInfo_->ubBlockSize / opDag_.minInputBytes;
    uint64_t bBlockNum = basicBlock_ * Ratio() / opDag_.maxInputBytes;
    uint64_t innerR = unitR_.inner;
    uint64_t maxInnerA = resultBlock_ / opDag_.maxInputBytes;

    uint64_t innerA = 1;
    int32_t startA = Pattern::TailA ? Pattern::Dim - CONST1 : Pattern::Dim - AXES_STEP;
    int32_t endA = Pattern::FirstA ? CONST0 : CONST1;
    uint64_t outerA = 1;
    uint64_t step = 1;
    for (int32_t iA = startA; iA >= endA; iA -= AXES_STEP) {
        outerA *= shape[iA];
    }
    int32_t iA;
    for (iA = startA; iA >= endA; iA -= AXES_STEP) {
        uint64_t axisLen = shape[iA];
        uint64_t maxStep = 1UL;
        bool splitHere = false;
        for (uint64_t s = 1; s <= axisLen; s++) {
            uint64_t tmpS = (iA == Pattern::Dim - 1) ? CeilAlign(s, ubBlockSize) : s;
            uint64_t tmpInnerA = innerA * tmpS;
            uint64_t tmpOuterA = outerA / axisLen * CeilDiv(axisLen, tmpS);
            double rate = static_cast<double>(tmpOuterA) /
                          static_cast<double>(CeilAlign(tmpOuterA, compileInfo_->vectorCoreNum));
            bool isContinue = (tmpInnerA * innerR <= bBlockNum && tmpInnerA <= maxInnerA);
            if (isContinue) {
                if (tmpInnerA <= cacheSize) {
                    maxStep = tmpS;
                } else {
                    maxStep = rate >= THRES_HOLD ? tmpS : maxStep;
                }
                continue;
            } else {
                splitHere = true;
                break;
            }
        }
        if (splitHere || iA - AXES_STEP < 0) {
            step = maxStep;
            innerA *= step;
            outerA = outerA / axisLen * CeilDiv(axisLen, step);
            break;
        }
        innerA *= (iA == Pattern::Dim - 1 ? CeilAlign(axisLen, ubBlockSize) : axisLen);
        outerA /= axisLen;
    }
    AssembleUnit(unitA_, iA, innerA, outerA, step);
}

template <class Pattern>
void ReduceOpTiling::SetTilingDataBatchInvariant(const uint64_t* shape)
{
    uint64_t groupR = tilingData_->groupR;
    tilingData_->ubFactorA = unitA_.step;
    tilingData_->ubFactorR = unitR_.step;
    tilingData_->factorATotalCnt = unitA_.outer;
    tilingData_->factorRTotalCnt = unitR_.outer;
    tilingData_->basicBlock = basicBlock_;
    tilingData_->resultBlock = resultBlock_;
    tilingData_->coreNum = static_cast<int32_t>(compileInfo_->vectorCoreNum);

    if (tilingData_->groupR > 1) {
        OP_CHECK_IF(context_->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Failed to set ScheduleMode!"), return);
    }

    for (int32_t i = 0; i < MAX_DIM; i++) {
        tilingData_->shape[i] = shape[i];
        tilingData_->stride[i] = viewStride_[i];
        tilingData_->sliceNum[i] = sliceNum_[i];
        tilingData_->sliceShape[i] = sliceShape_[i];
        tilingData_->sliceStride[i] = sliceStride_[i];
    }

    tilingData_->useNddma = IsUseNddma<Pattern>(shape);
    ComputeStride<Pattern>(shape);
    if (unitA_.outer * groupR <= compileInfo_->vectorCoreNum) {
        tilingData_->factorACntPerCore = 1;
    } else {
        tilingData_->factorACntPerCore = CeilDiv(unitA_.outer, FloorDiv(compileInfo_->vectorCoreNum, groupR));
    }
    uint32_t realCore = CeilDiv(unitA_.outer, tilingData_->factorACntPerCore) *
                        CeilDiv(unitR_.outer, tilingData_->factorRCntPerCore);
    tilingData_->realCoreNum = realCore;
    context_->SetBlockDim(realCore);
}

} // namespace Base
} // namespace Ops

#endif // _REDUCE_BATCH_INVARIANT_TILING_H_
