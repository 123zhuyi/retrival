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
 * \file as_strided_tiling_arch35.cpp
 * \brief as_strided_tiling_arch35
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <set>
#include "util/const_util.h"
#include "register/op_def_registry.h"
#include "as_strided_dualcut_tiling_arch35.h"
#include "as_strided_tiling_arch35.h"
#include "as_strided_merge_axis_tiling_arch35.h"
#include "util/math_util.h"
#include "op_host/tiling_base_util.h"
#include "log/log.h"
#include "util/platform_util.h"

using namespace std;
using namespace ge;

namespace optiling {
constexpr size_t CONST_TWO = 2;
constexpr size_t CONST_FOUR = 4;
constexpr size_t UB_BUFFER_B8 = 126976;
constexpr size_t UB_BUFFER_B16 = 63488;
constexpr size_t UB_BUFFER_B32 = 31744;
constexpr size_t UB_BUFFER_B64 = 15872;
constexpr size_t INPUT_DTYPE_B64 = 8;
constexpr size_t INPUT_DTYPE_B32 = 4;
constexpr size_t INPUT_DTYPE_B16 = 2;
constexpr size_t INPUT_DTYPE_B8 = 1;
constexpr uint32_t UB_ALIGN_SIZE = 32;
constexpr size_t MOVEALIGN_DIM5 = 5;
constexpr size_t MOVEALIGN_DIM4 = 4;
constexpr size_t MOVEALIGN_DIM3 = 3;
constexpr size_t MOVEALIGN_DIM2 = 2;
constexpr size_t MOVEALIGN_KEY = 100;
constexpr int64_t MOVEALIGN_FLAG = 128;
constexpr int64_t MOVEALIGN_STRIDE_CON = 32;
constexpr size_t DUAL_CUT_KEY = 200;
constexpr size_t ALL_STRIDES_ZERO_KEY = 300;
constexpr size_t SIMT_KEY = 400;
constexpr size_t WITH_GATHER_KEY = 500;
constexpr size_t EMPTY_TENSOR_KEY = 1000;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t DUAL_CUT_CONDITION1 = 64;
constexpr int32_t DUAL_CUT_CONDITION2 = 128;
constexpr int32_t DUAL_CUT_CONDITION3 = 32 * 1024;
constexpr int32_t VALID_DIM = 8;
constexpr int64_t SMALL_SHAPE_BYTES_THRESHOLD = 1024 * 1024;
constexpr int64_t LAST_STRIDE_THRESHOLD = 64;
constexpr uint16_t MAX_UINT16 = 65535;
constexpr uint16_t GATHER_AXES_LIMIT = 3;
constexpr uint16_t GATHER_UB_SIZE_LOWER_LIMIT = 2048;
constexpr size_t WORKSPACE_SIZE_8MB = 8 * 1024 * 1024;

std::map<ge::DataType, uint32_t> tilingTypeKeyMap = {
    {ge::DT_INT64, INPUT_DTYPE_B64},      {ge::DT_UINT64, INPUT_DTYPE_B64},       {ge::DT_COMPLEX64, INPUT_DTYPE_B64},
    {ge::DT_FLOAT, INPUT_DTYPE_B32},      {ge::DT_INT32, INPUT_DTYPE_B32},        {ge::DT_UINT32, INPUT_DTYPE_B32},
    {ge::DT_COMPLEX32, INPUT_DTYPE_B32},  {ge::DT_FLOAT16, INPUT_DTYPE_B16},      {ge::DT_BF16, INPUT_DTYPE_B16},
    {ge::DT_INT16, INPUT_DTYPE_B16},      {ge::DT_UINT16, INPUT_DTYPE_B16},       {ge::DT_UINT8, INPUT_DTYPE_B8},
    {ge::DT_INT8, INPUT_DTYPE_B8},        {ge::DT_BOOL, INPUT_DTYPE_B8},          {ge::DT_HIFLOAT8, INPUT_DTYPE_B8},
    {ge::DT_FLOAT8_E5M2, INPUT_DTYPE_B8}, {ge::DT_FLOAT8_E4M3FN, INPUT_DTYPE_B8},
};

ge::graphStatus AsStridedTilingClass::AsStridedSetTilingData(AsStridedTilingData& tilingData)
{
    tilingData_ = context_->GetTilingData<AsStridedTilingData>();
    tilingData_->axisOutTotalFactor = tilingData.axisOutTotalFactor;
    tilingData_->outerAxisFactor = tilingData.outerAxisFactor;
    tilingData_->innerAxisFactor = tilingData.innerAxisFactor;
    tilingData_->tilingAxisIdx = tilingData.tilingAxisIdx;
    tilingData_->outerAxisNum = tilingData.outerAxisNum;
    tilingData_->ubSize = tilingData.ubSize;
    tilingData_->innerAxisNum = tilingData.innerAxisNum;
    tilingData_->storageOffset = tilingData.storageOffset;
    tilingData_->blockNum = tilingData.blockNum;
    tilingData_->loopsTailCore = tilingData.loopsTailCore;
    tilingData_->innerAxisFactorTail = tilingData.innerAxisFactorTail;
    tilingData_->ubFactor = tilingData.ubFactor;
    tilingData_->ubFactorTail = tilingData.ubFactorTail;
    tilingData_->loopsPerCore = tilingData.loopsPerCore;
    tilingData_->en32BAligned = tilingData.en32BAligned;

    for(int64_t i = 0; i < TILING_ARRAY_LEN; i++) {
        tilingData_->outStrideArr[i] = tilingData.outStrideArr[i];
        tilingData_->innerAxis[i] = tilingData.innerAxis[i];
        tilingData_->outLoopArr[i] = tilingData.outLoopArr[i];
        tilingData_->gmInStride[i] = tilingData.gmInStride[i];
        tilingData_->gmShape[i] = tilingData.gmShape[i];
        tilingData_->gmOutStride[i] = tilingData.gmOutStride[i];
    }

    for(int64_t j = 0; j < TILING_NDDMA_LEN; j++) {
        tilingData_->nddmaLoop[j] = tilingData.nddmaLoop[j];
        tilingData_->nddmaDstStride[j] = tilingData.nddmaDstStride[j];
        tilingData_->nddmaTailLoop[j] = tilingData.nddmaTailLoop[j];
        tilingData_->nddmaSrcStride[j] = tilingData.nddmaSrcStride[j];
    }

    return ge::GRAPH_SUCCESS;
}

template<typename T, typename U>
void copyArrForTiling(T& tiling, const U& tilingParam, int64_t arrLen, int64_t nddmaLen)
{
    for(int64_t i = 0; i < arrLen; i++) {
        tiling.outStrideArr[i] = tilingParam.outStrideArr[i];
        tiling.innerAxis[i] = tilingParam.innerAxis[i];
        tiling.outLoopArr[i] = tilingParam.outLoopArr[i];
        tiling.gmInStride[i] = tilingParam.gmInStride[i];
        tiling.gmShape[i] = tilingParam.gmShape[i];
    }

    for(int64_t j = 0; j < nddmaLen; j++) {
        tiling.nddmaLoop[j] = tilingParam.nddmaLoop[j];
        tiling.nddmaDstStride[j] = tilingParam.nddmaDstStride[j];
        tiling.nddmaTailLoop[j] = tilingParam.nddmaTailLoop[j];
        tiling.nddmaSrcStride[j] = tilingParam.nddmaSrcStride[j];
    }
}

ge::graphStatus AsStridedTilingClass::SetTilingData(AsStridedTilingData& tiling,
                                     AsStridedTilingParam& tilingParam)
{
    if (tilingParam.tilingKey != ALL_STRIDES_ZERO_KEY && tilingParam.tilingKey != SIMT_KEY &&
        tilingParam.tilingKey != WITH_GATHER_KEY) {
        tiling.axisOutTotalFactor = tilingParam.axisOutTotalFactor;
        tiling.outerAxisFactor = tilingParam.outerAxisFactor;
        tiling.innerAxisFactor = tilingParam.innerAxisFactor;
        tiling.tilingAxisIdx = tilingParam.tilingAxisIdx;
        tiling.outerAxisNum = tilingParam.outerAxisNum;
        tiling.ubSize = tilingParam.ubSize;
        tiling.innerAxisNum = tilingParam.innerAxisNum;
        tiling.storageOffset = tilingParam.storageOffset;
        tiling.blockNum = tilingParam.blockNum;
        tiling.innerAxisFactorTail = tilingParam.innerAxisFactorTail;
        tiling.ubFactorTail = tilingParam.ubFactorTail;
        tiling.loopsPerCore = tilingParam.loopsPerCore;
        tiling.en32BAligned = tilingParam.en32BAligned;
        copyArrForTiling<AsStridedTilingData, AsStridedTilingParam>(tiling, tilingParam, TILING_ARRAY_LEN, TILING_NDDMA_LEN);
        this->AsStridedSetTilingData(tiling);
    }
    return ge::GRAPH_SUCCESS;
}

void AsStridedTilingClass::SetZeroStrideTilingData(AsStridedTilingParam& tilingParam)
{
    zeroStrideTilingData_ = context_->GetTilingData<AsStridedZeroStrideTilingData>();
    zeroStrideTilingData_->blockNum = tilingParam.blockNum;
    zeroStrideTilingData_->ubSizePlatForm = tilingParam.ubSizePlatForm;
    zeroStrideTilingData_->storageOffset = tilingParam.storageOffset;
    zeroStrideTilingData_->mainBlockFactor = tilingParam.mainBlockFactor;
    zeroStrideTilingData_->tailBlockFactor = tilingParam.tailBlockFactor;

    OP_LOGI(context_, "[SetZeroStrideTilingData]blockNum:%u, ubSizePlatForm:%lu, storageOffset:%ld, \
        mainBlockFactor:%ld, tailBlockFactor:%ld.",
        tilingParam.blockNum, tilingParam.ubSizePlatForm, tilingParam.storageOffset,
        tilingParam.mainBlockFactor, tilingParam.tailBlockFactor);
}

void AsStridedTilingClass::SetSimtTilingData(AsStridedTilingParam& tilingParam)
{
    simtTilingData_ = context_->GetTilingData<AsStridedSimtTilingData>();
    simtTilingData_->outDimNum = tilingParam.outDimNum;
    simtTilingData_->blockNum = tilingParam.blockNum;
    simtTilingData_->storageOffset = tilingParam.storageOffset;
    simtTilingData_->mainBlockFactor = tilingParam.mainBlockFactor;
    simtTilingData_->tailBlockFactor = tilingParam.tailBlockFactor;

    for(int64_t i = 0; i < TILING_ARRAY_LEN; i++) {
        simtTilingData_->sizeArr[i] = tilingParam.sizeArr[i];
        simtTilingData_->strideArr[i] = tilingParam.strideArr[i];
        simtTilingData_->outSizeStride[i] = tilingParam.outSizeStride[i];
    }

    OP_LOGI(context_, "[SetSimtTilingData]outDimNum:%u, blockNum:%u, storageOffset:%ld, \
        mainBlockFactor:%ld, tailBlockFactor:%ld.",
        tilingParam.outDimNum, tilingParam.blockNum, tilingParam.storageOffset,
        tilingParam.mainBlockFactor, tilingParam.tailBlockFactor);
}

static void SetWithGatherUbParam(UbParam& tilingDataUbParam, UbParam& ubParam)
{
    tilingDataUbParam.innerAxisFactor = ubParam.innerAxisFactor;
    tilingDataUbParam.innerAxisFactorTail = ubParam.innerAxisFactorTail;
    tilingDataUbParam.outerAxisFactor = ubParam.outerAxisFactor;
    tilingDataUbParam.ubFactor = ubParam.ubFactor;
    tilingDataUbParam.ubFactorTail = ubParam.ubFactorTail;
    tilingDataUbParam.loopsPerCore = ubParam.loopsPerCore;
}

void AsStridedTilingClass::SetWithGatherTilingData(AsStridedUbGatherParam& ubGatherParam)
{
    gatherTilingData_ = context_->GetTilingData<AsStridedWithGatherTilingData>();
    gatherTilingData_->storageOffset = ubGatherParam.storageOffset;
    gatherTilingData_->ubSizePlatForm = ubGatherParam.ubSizePlatForm;
    gatherTilingData_->tilingAxisIdx = ubGatherParam.tilingAxisIdx;
    gatherTilingData_->preSize = ubGatherParam.preSize;
    gatherTilingData_->blockNum = ubGatherParam.blockNum;
    gatherTilingData_->mainBlockCnt = ubGatherParam.mainBlockCnt;
    gatherTilingData_->outDimNum = ubGatherParam.outDimNum;
    gatherTilingData_->inUbSize = ubGatherParam.inUbSize;
    gatherTilingData_->inDataLen = ubGatherParam.inDataLen;
    gatherTilingData_->blockAxisIdx = ubGatherParam.blockAxisIdx;
    gatherTilingData_->coreCurAxisFactor = ubGatherParam.coreCurAxisFactor;
    gatherTilingData_->coreInnerAxisFactor = ubGatherParam.coreInnerAxisFactor;
    gatherTilingData_->coreInnerAxisTailFactor = ubGatherParam.coreInnerAxisTailFactor;
    gatherTilingData_->coreOuterAxisFactor = ubGatherParam.coreOuterAxisFactor;
    SetWithGatherUbParam(gatherTilingData_->mainBlockUbParam, ubGatherParam.mainBlockUbParam);
    SetWithGatherUbParam(gatherTilingData_->tailBlockUbParam, ubGatherParam.tailBlockUbParam);
    for(int64_t i = 0; i < TILING_ARRAY_LEN; i++) {
        gatherTilingData_->sizeArr[i] = ubGatherParam.sizeArr[i];
        gatherTilingData_->strideArr[i] = ubGatherParam.strideArr[i];
        gatherTilingData_->idxStrideArr[i] = ubGatherParam.idxStrideArr[i];
    }

    OP_LOGI(context_, "[SetWithGatherTilingData]outDimNum:%u, blockNum:%u, mainBlockCnt:%u, storageOffset:%ld, \
        ubSizePlatForm:%lu, inUbSize:%u.",
        ubGatherParam.outDimNum, ubGatherParam.blockNum, ubGatherParam.mainBlockCnt, ubGatherParam.storageOffset,
        ubGatherParam.ubSizePlatForm, ubGatherParam.inUbSize);
}

void AsStridedTilingClass::NoTilingMergeAxis(AsStridedTilingData& tiling, AsStridedTilingParam& tilingParam, gert::Shape outSize)
{
    OP_LOGD(context_, "NoTilingMergeAxis");
    tilingParam.ubUseFactor = 1;
    tilingParam.loopsPerCore = 1;

    tilingParam.loopsPerCore = (tilingParam.axisOutTotalFactor + tilingParam.blockNum - 1) / tilingParam.blockNum;

    for (size_t i = 0; i < tilingParam.outerAxisNum; i++) {
        tilingParam.outLoopArr[i + TILING_ARRAY_LEN - tilingParam.outerAxisNum] = outSize[i];
    }

    for (int32_t i = TILING_NDDMA_LEN - 2; i >= 0; i--) {
        tilingParam.nddmaDstStride[i] = outSize[i + tilingParam.outerAxisNum + 1] * tilingParam.nddmaDstStride[i + 1];
    }

    for (uint32_t i = 0; i < TILING_NDDMA_LEN; i++) {
        tilingParam.nddmaLoop[i] = outSize[i + tilingParam.outerAxisNum];
        tilingParam.ubUseFactor *= outSize[i + tilingParam.outerAxisNum];
    }

    tiling.ubFactor = tilingParam.ubUseFactor;
}

void AsStridedTilingClass::MergeAxisAfterTiling(
    [[maybe_unused]] const AsStridedTilingData& tiling, AsStridedTilingParam& tilingParam, gert::Shape outSize)
{
    OP_LOGD(context_, "Start fusing");
    tilingParam.axisOutTotalFactor = tilingParam.outerAxisFactor;
    for (uint32_t i = 0; i < tilingParam.tilingAxisIdx; i++) {
        tilingParam.axisOutTotalFactor *= outSize[i];
    }
    tilingParam.loopsPerCore = 1;
    tilingParam.loopsPerCore = (tilingParam.axisOutTotalFactor + tilingParam.blockNum - 1) / tilingParam.blockNum;

    if (tilingParam.tilingFlag == 1) { // Axis tiling is complete.
        for (uint32_t i = 0; i < tilingParam.outerAxisNum - 1; i++) {
            tilingParam.outLoopArr[i + TILING_ARRAY_LEN - tilingParam.outerAxisNum] = outSize[i];
        }
        tilingParam.outLoopArr[TILING_ARRAY_LEN - 1] = tilingParam.outerAxisFactor;
    }

    for (int32_t i = tilingParam.innerAxisNum - 1; i > 0; i--) {
        tilingParam.innerAxis[i] = outSize[tilingParam.tilingAxisIdx + i];
    }
    tilingParam.innerAxis[0] = tilingParam.innerAxisFactor;

    if (tilingParam.innerAxisNum >= TILING_NDDMA_LEN) {
        for (int32_t i = TILING_NDDMA_LEN - 2; i >= 0; i--) {
            tilingParam.nddmaDstStride[i] = tilingParam.innerAxis[i + 1] * tilingParam.nddmaDstStride[i + 1];
        }
        for (uint32_t i = 0; i < TILING_NDDMA_LEN; i++) {
            tilingParam.nddmaLoop[i] = tilingParam.innerAxis[i + tilingParam.innerAxisNum - TILING_NDDMA_LEN];
            tilingParam.nddmaTailLoop[i] = tilingParam.innerAxis[i + tilingParam.innerAxisNum - TILING_NDDMA_LEN];
            tilingParam.nddmaTailLoop[0] = tilingParam.innerAxisFactorTail;
        }
    } else {
        for (uint32_t i = 1; i < tilingParam.innerAxisNum; i++) {
            tilingParam.nddmaDstStride[TILING_NDDMA_LEN - 1 - i] =
                tilingParam.innerAxis[tilingParam.innerAxisNum - i] *
                tilingParam.nddmaDstStride[TILING_NDDMA_LEN - 1 - i + 1];
        }
        for (uint32_t i = 0; i < tilingParam.innerAxisNum; i++) {
            tilingParam.nddmaLoop[i + TILING_NDDMA_LEN - tilingParam.innerAxisNum] = tilingParam.innerAxis[i];
            tilingParam.nddmaTailLoop[i + TILING_NDDMA_LEN - tilingParam.innerAxisNum] = tilingParam.innerAxis[i];
            tilingParam.nddmaTailLoop[TILING_NDDMA_LEN - tilingParam.innerAxisNum] = tilingParam.innerAxisFactorTail;
        }
    }
}

static void MergeAxis4MoveAlign(gert::TilingContext* context, 
    AsStridedTilingParam& tilingParam, gert::Shape outSize, gert::Shape outStride, AsStridedTilingData& tiling)
{
    OP_LOGD(context, "MergeAxis4MoveAlign");
    tilingParam.axisOutTotalFactor = 1;
    tilingParam.ubUseFactor = 1;
    if (tilingParam.innerAxisNum > MOVEALIGN_DIM5) {
        tilingParam.outerAxisNum += 1;
    }
    for (uint32_t i = 0; i < tilingParam.outerAxisNum; i++) {
        tilingParam.axisOutTotalFactor *= outSize[i];
        tilingParam.outStrideArr[TILING_ARRAY_LEN - tilingParam.outerAxisNum + i] = outStride[i];
    }
    tilingParam.blockNum =
        tilingParam.axisOutTotalFactor > tilingParam.numCore ? tilingParam.numCore : tilingParam.axisOutTotalFactor;
    OP_LOGD(context, "BlockNum: %u", tilingParam.blockNum);

    tilingParam.loopsPerCore = (tilingParam.axisOutTotalFactor + tilingParam.blockNum - 1) / tilingParam.blockNum;

    for (size_t i = 0; i < tilingParam.outerAxisNum; i++) {
        tilingParam.outLoopArr[i + TILING_ARRAY_LEN - tilingParam.outerAxisNum] = outSize[i];
    }

    for (uint32_t i = 1; i < TILING_NDDMA_LEN; i++) {
        tilingParam.ubUseFactor *= outSize[i + tilingParam.outerAxisNum - 1];
    }
    tiling.ubFactor = tilingParam.ubUseFactor;
    tilingParam.ubFactorTail = 0;
    tilingParam.innerAxisFactorTail = 0;
    if ((tilingParam.nddmaDstStride[MOVEALIGN_DIM2] * tilingParam.sizeofDtype) % UB_ALIGN_SIZE != 0) {
        tilingParam.nddmaDstStride[MOVEALIGN_DIM2] =
            Ops::Base::CeilDiv(tilingParam.nddmaDstStride[MOVEALIGN_DIM2] * tilingParam.sizeofDtype, UB_ALIGN_SIZE) *
            UB_ALIGN_SIZE;
        tilingParam.nddmaDstStride[1] =
            tilingParam.nddmaDstStride[MOVEALIGN_DIM2] * tilingParam.nddmaLoop[MOVEALIGN_DIM2];
        tilingParam.en32BAligned = 1;
    } else if ((tilingParam.nddmaDstStride[1] * tilingParam.sizeofDtype) % UB_ALIGN_SIZE != 0) {
        tilingParam.nddmaDstStride[1] =
            Ops::Base::CeilDiv(tilingParam.nddmaDstStride[1] * tilingParam.sizeofDtype, UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        tilingParam.en32BAligned = 1;
    }
}

inline bool HasDuplicate(gert::TilingContext* context, gert::Shape outStride)
{
    // MoveAlign Condition 3
    OP_CHECK_IF(
        (outStride.GetDimNum() > VALID_DIM),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "outStride",
            std::to_string(outStride.GetDimNum()).c_str(),
            ("The shape dim of outStride must be within the range [0, " + std::to_string(VALID_DIM) + "].").c_str()),
        return false);
    int32_t numStride[VALID_DIM] = {0};
    for (uint32_t i = 0; i < outStride.GetDimNum(); i++) {
        numStride[i] = outStride[i];
    }
    std::sort(numStride, numStride + std::min(outStride.GetDimNum(), (size_t)VALID_DIM));
    for (uint32_t i = 1; i < outStride.GetDimNum(); i++) {
        if (numStride[i] == numStride[i - 1]) {
            return true;
        }
    }
    return false;
}

inline static bool CalStrideRange(gert::Shape outStride, const AsStridedTilingParam& tilingParam)
{
    // MoveAlign Condition 2
    int64_t minVal = outStride[0];
    int64_t maxVal = outStride[0];
    for (uint32_t i = 0; i < outStride.GetDimNum(); i++) {
        if (outStride[i] < minVal) {
            minVal = outStride[i];
        }
        if (outStride[i] > maxVal) {
            maxVal = outStride[i];
        }
    }
    if ((maxVal - minVal) * tilingParam.sizeofDtype < MOVEALIGN_STRIDE_CON) {
        return false;
    }
    return true;
}

inline static bool CheckLastDim(gert::Shape outSize, gert::Shape outStride, const AsStridedTilingParam& tilingParam)
{
    // MoveAlign Condition 1
    auto outShapeSize = outSize.GetDimNum();
    if (((outSize[outShapeSize - 1] * tilingParam.sizeofDtype) >= MOVEALIGN_FLAG) &&
        (outStride[outShapeSize - 1] == 1)) {
        return true;
    }
    return false;
}

/*
 * MoveAlign Condition:
 * 1. The size of last dim is larger than 128 and its stride is 1.
 * 2. The outStride range is larger than 32B.
 * 3. The outStride doesn' t have duplicate value.
 */
inline static bool IsMoveAlign(gert::TilingContext* context, gert::Shape outSize, gert::Shape outStride, AsStridedTilingParam& tilingParam)
{
    if (CheckLastDim(outSize, outStride, tilingParam) && CalStrideRange(outStride, tilingParam) &&
        (!HasDuplicate(context, outStride))) {
        OP_LOGD(context, "Need MoveAlign");
        return true;
    }
    return false;
}

inline static bool IsAllStridesZero(gert::TilingContext* context, gert::Shape outStride)
{
    for (uint32_t i = 0; i < outStride.GetDimNum(); i++) {
        if (outStride[i] != 0) {
            return false;
        }
    }
    OP_LOGD(context, "All Strides Are Zero.");
    return true;
}

inline static bool IsSmallShape(gert::TilingContext* context, gert::Shape outSize, uint32_t sizeofDtype)
{
    uint64_t totalOutElement = 1;
    for (uint32_t i = 0; i < outSize.GetDimNum(); i++) {
        totalOutElement *= outSize[i];
    }
    if (totalOutElement * sizeofDtype <= SMALL_SHAPE_BYTES_THRESHOLD) {
        return true;
        OP_LOGD(context, "Need SIMT.");
    }
    return false;
}

inline static bool CheckBndryForUint16GatherIdx(gert::Shape outSize, gert::Shape outStride,
    const AsStridedTilingParam& tilingParam, const AsStridedUbGatherParam& ubGatherParam)
{
    uint32_t dimNum = ubGatherParam.outDimNum - ubGatherParam.tilingAxisIdx;
    if (tilingParam.sizeofDtype > CONST_TWO) {
        return true;
    }

    if (dimNum == 1) {
        return outStride[ubGatherParam.outDimNum - 1] * (ubGatherParam.mainBlockUbParam.ubFactor - 1) <= MAX_UINT16;
    } else if (dimNum == CONST_TWO) {
        return outSize[ubGatherParam.outDimNum - 1] <= MAX_UINT16;
    } else if (dimNum > CONST_TWO) {
        uint32_t lastDim = outSize[ubGatherParam.outDimNum - 1];
        uint32_t prevDim = outSize[ubGatherParam.outDimNum - 2];
        return (lastDim * prevDim <= MAX_UINT16);
    }

    return true;
}

inline static bool IsUbGather(gert::TilingContext* context, const AsStridedTilingParam& tilingParam)
{
    if ((static_cast<uint64_t>(tilingParam.inputSize) * static_cast<uint64_t>(tilingParam.sizeofDtype) < tilingParam.ubSizePlatForm / CONST_FOUR)) {
        OP_LOGD(context, "Need UbGather.");
        return true;
    }
    return false;
}

inline static uint32_t CalcblockNum(const gert::TilingContext* context, uint64_t totalOutElement,
                                    const AsStridedTilingParam& tilingParam)
{
    uint32_t cacheLineSize = Ops::Base::GetCacheLineSize(context);
    OP_LOGD(context, "[CalcblockNum]cacheLineSize:%u.", cacheLineSize);
    return std::min(tilingParam.numCore,
                    static_cast<uint32_t>(Ops::Base::CeilDiv(totalOutElement * tilingParam.sizeofDtype,
                    static_cast<uint64_t>(cacheLineSize))));
}

inline static void SetAllStridesZeroTilingParam(const gert::TilingContext* context, gert::Shape outSize,
                                                AsStridedTilingParam& tilingParam)
{
    int64_t shapeSize = outSize.GetShapeSize();
    OP_CHECK_IF(
        (shapeSize <= 0),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "size",
            std::to_string(shapeSize).c_str(),
            "The shape size of this parameter must be greater than 0."),
        return);
    uint64_t totalOutElement = static_cast<uint64_t>(shapeSize);
    tilingParam.blockNum = CalcblockNum(context, totalOutElement, tilingParam);
    tilingParam.mainBlockFactor = static_cast<int64_t>(Ops::Base::CeilDiv(totalOutElement,
                                                                    static_cast<uint64_t>(tilingParam.blockNum)));
    tilingParam.blockNum = static_cast<uint32_t>(Ops::Base::CeilDiv(totalOutElement,
                                                                    static_cast<uint64_t>(tilingParam.mainBlockFactor)));
    tilingParam.tailBlockFactor = static_cast<int64_t>(totalOutElement) -
        static_cast<int64_t>(tilingParam.blockNum - 1) * tilingParam.mainBlockFactor;
}

inline static void SetSimtTilingParam(const gert::TilingContext* context, gert::Shape outSize, gert::Shape outStride,
                                      AsStridedTilingParam& tilingParam)
{
    uint64_t totalOutElement = outSize.GetShapeSize();
    tilingParam.blockNum = CalcblockNum(context, totalOutElement, tilingParam);

    tilingParam.outDimNum = outSize.GetDimNum();
    tilingParam.mainBlockFactor = static_cast<int64_t>(Ops::Base::CeilDiv(totalOutElement,
                                                                    static_cast<uint64_t>(tilingParam.blockNum)));
    tilingParam.blockNum = static_cast<uint32_t>(Ops::Base::CeilDiv(totalOutElement,
                                static_cast<uint64_t>(tilingParam.mainBlockFactor)));
    tilingParam.tailBlockFactor = static_cast<int64_t>(totalOutElement) -
        static_cast<int64_t>(tilingParam.blockNum - 1) * tilingParam.mainBlockFactor;
    for (uint32_t i = 0; i < tilingParam.outDimNum; i++) {
        tilingParam.sizeArr[i] = outSize[i];
        tilingParam.strideArr[i] = outStride[i];
    }
    for (int32_t i = tilingParam.outDimNum - 2; i >= 0; i--) {
        tilingParam.outSizeStride[i] = tilingParam.outSizeStride[i + 1] * outSize[i + 1];
    }
}

inline static void CalcTilingCore(const gert::TilingContext* context, gert::Shape outSize, AsStridedUbGatherParam& ubGatherParam)
{
    OP_CHECK_IF(
        (outSize.GetDimNum() <= 0),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "size",
            std::to_string(outSize.GetDimNum()).c_str(),
            "The shape dim of size must be greater than 0."),
        return);
    uint32_t preSize = 1;
    if (ubGatherParam.blockNum == 1) {
        ubGatherParam.blockAxisIdx = 0;
        ubGatherParam.coreInnerAxisFactor = outSize[0];
        ubGatherParam.coreInnerAxisTailFactor = outSize[0];
    } else {
        for (uint32_t i = 0; i < outSize.GetDimNum(); i++) {
            ubGatherParam.coreCurAxisFactor = outSize[i] * preSize;
            ubGatherParam.blockAxisIdx = i;
            if (ubGatherParam.coreCurAxisFactor >= ubGatherParam.blockNum) {
                break;
            } else {
                preSize *= outSize[i];
            }
        }
        ubGatherParam.coreInnerAxisFactor = Ops::Base::CeilDiv(ubGatherParam.coreCurAxisFactor, ubGatherParam.blockNum);
        ubGatherParam.blockNum = Ops::Base::CeilDiv(ubGatherParam.coreCurAxisFactor, ubGatherParam.coreInnerAxisFactor);
        ubGatherParam.coreInnerAxisTailFactor = ubGatherParam.coreCurAxisFactor -
                                                (ubGatherParam.blockNum - 1) * ubGatherParam.coreInnerAxisFactor;
        ubGatherParam.coreOuterAxisFactor = static_cast<uint32_t>(Ops::Base::CeilDiv(outSize[ubGatherParam.blockAxisIdx],
                                                        static_cast<int64_t>(ubGatherParam.coreInnerAxisFactor)));
    }

    OP_LOGD(context, "[CalcTilingCore]blockAxisIdx:%u, blockNum:%u.",
        ubGatherParam.blockAxisIdx, ubGatherParam.blockNum);
    OP_LOGD(context, "[CalcTilingCore]coreInnerAxisFactor:%u, coreInnerAxisTailFactor:%u, coreOuterAxisFactor:%u",
        ubGatherParam.coreInnerAxisFactor, ubGatherParam.coreInnerAxisTailFactor, ubGatherParam.coreOuterAxisFactor);
}

inline static void CalcMaxUbFactor(AsStridedTilingParam& tilingParam, AsStridedUbGatherParam& ubGatherParam,
                                   uint32_t& maxUbFactor, gert::Shape outSize, gert::Shape outStride)
{
    uint32_t requiredStorageSize = 0;
    for (size_t i = 0; i < outSize.GetDimNum(); i++) {
        requiredStorageSize += (outSize[i] - 1) * outStride[i];
    }
    ubGatherParam.inDataLen = requiredStorageSize + 1;
    ubGatherParam.inUbSize = Ops::Base::CeilDiv(ubGatherParam.inDataLen * tilingParam.sizeofDtype, UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
    uint32_t outUbAlign = ((tilingParam.ubSizePlatForm - ubGatherParam.inUbSize) / UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
    if (tilingParam.sizeofDtype == INPUT_DTYPE_B8) {
        // b8对应的索引类型是uint16
        maxUbFactor = outUbAlign / (sizeof(uint16_t) + tilingParam.sizeofDtype * BUFFER_NUM);
    } else if (tilingParam.sizeofDtype == INPUT_DTYPE_B16 or tilingParam.sizeofDtype == INPUT_DTYPE_B32) {
        maxUbFactor = outUbAlign / (tilingParam.sizeofDtype + tilingParam.sizeofDtype * BUFFER_NUM);
    } else {
        // b64对应的索引类型是uint32
        maxUbFactor = outUbAlign / (sizeof(uint32_t) + tilingParam.sizeofDtype * BUFFER_NUM);
    }
}

inline static void SetubParamWhenOutAxesLimit(gert::Shape outSize, uint32_t coreInnerAxisFactor,
                                              AsStridedUbGatherParam& ubGatherParam, UbParam& ubParam)
{
    // 切分思路：ub全载后三根轴，三根轴外到核切分内轴为ub循环次数
    OP_CHECK_IF(
        (ubGatherParam.tilingAxisIdx >= TILING_ARRAY_LEN),
        OP_LOGE("as_trided", "the axis idx is more than range"), return);
    OP_CHECK_IF(
        (outSize.GetDimNum() == 0),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("as_strided", "size",
            std::to_string(outSize.GetDimNum()).c_str(),
            "The shape dim of size must be greater than 0."),
        return);
    ubParam.innerAxisFactor = outSize[ubGatherParam.tilingAxisIdx];
    ubParam.outerAxisFactor = 1;
    ubParam.innerAxisFactorTail = ubParam.innerAxisFactor;
    ubParam.ubFactor = ubGatherParam.preSize;
    ubParam.ubFactorTail = ubGatherParam.preSize;
    ubParam.loopsPerCore = coreInnerAxisFactor;
    for (int32_t j = static_cast<int32_t>(ubGatherParam.tilingAxisIdx) - 1; j > static_cast<int32_t>(ubGatherParam.blockAxisIdx); j--) {
        ubParam.loopsPerCore *= outSize[j];
    }
}

inline static void SetubParamWhenAxesSame(uint32_t coreInnerAxisFactor, uint32_t maxUbFactor,
                                          AsStridedUbGatherParam& ubGatherParam, UbParam& ubParam)
{
    // 切在同一根轴，要保证ub不能切到核外
    ubParam.innerAxisFactor = maxUbFactor / ubGatherParam.preSize;
    ubParam.innerAxisFactor = std::min(coreInnerAxisFactor, ubParam.innerAxisFactor);
    ubParam.outerAxisFactor = Ops::Base::CeilDiv(coreInnerAxisFactor, ubParam.innerAxisFactor);
    ubParam.innerAxisFactorTail = coreInnerAxisFactor -
                                  ubParam.innerAxisFactor * (ubParam.outerAxisFactor - 1);
    ubParam.ubFactor = ubGatherParam.preSize * ubParam.innerAxisFactor;
    ubParam.ubFactorTail = ubGatherParam.preSize * ubParam.innerAxisFactorTail;
    ubParam.loopsPerCore = ubParam.outerAxisFactor;
}

inline static void CalctilingCoreWithInAxis(const gert::TilingContext* context, gert::Shape outSize, uint32_t maxUbFactor,
                                            AsStridedUbGatherParam& ubGatherParam)
{
    if (ubGatherParam.tilingAxisIdx == 0) {
        SetubParamWhenAxesSame(ubGatherParam.coreInnerAxisFactor, maxUbFactor, ubGatherParam,
                               ubGatherParam.mainBlockUbParam);
        SetubParamWhenAxesSame(ubGatherParam.coreInnerAxisTailFactor, maxUbFactor, ubGatherParam,
                               ubGatherParam.tailBlockUbParam);
    } else {
        uint32_t outerCoreNum = 1;
        for (uint32_t i = 0; i < ubGatherParam.blockAxisIdx; i++) {
            outerCoreNum *= outSize[i];
        }
        uint32_t innerCoreNum = ubGatherParam.blockNumMin / outerCoreNum;
        ubGatherParam.coreInnerAxisFactor = static_cast<uint32_t>(Ops::Base::CeilDiv(outSize[ubGatherParam.blockAxisIdx],
                                                                               static_cast<int64_t>(innerCoreNum)));
        innerCoreNum = static_cast<uint32_t>(Ops::Base::CeilDiv(outSize[ubGatherParam.blockAxisIdx],
                                                          static_cast<int64_t>(ubGatherParam.coreInnerAxisFactor)));

        UbParam& ubParam = ubGatherParam.mainBlockUbParam;
        ubParam.innerAxisFactor = maxUbFactor / ubGatherParam.preSize;
        ubParam.innerAxisFactor = std::min(ubGatherParam.coreInnerAxisFactor, ubParam.innerAxisFactor);
        ubParam.outerAxisFactor = Ops::Base::CeilDiv(ubGatherParam.coreInnerAxisFactor, ubParam.innerAxisFactor);

        ubGatherParam.coreInnerAxisFactor =
            ubParam.innerAxisFactor * ubParam.outerAxisFactor > outSize[ubGatherParam.blockAxisIdx] ?
                ubGatherParam.coreInnerAxisFactor :
                ubParam.innerAxisFactor * ubParam.outerAxisFactor;  // 按主核均分UB修正
        SetubParamWhenAxesSame(ubGatherParam.coreInnerAxisFactor, maxUbFactor, ubGatherParam,
                               ubGatherParam.mainBlockUbParam);
        innerCoreNum = std::min(innerCoreNum,
                                static_cast<uint32_t>(Ops::Base::CeilDiv(outSize[ubGatherParam.blockAxisIdx],
                                                      static_cast<int64_t>(ubGatherParam.coreInnerAxisFactor))));
        ubGatherParam.coreOuterAxisFactor = innerCoreNum;
        ubGatherParam.blockNum = innerCoreNum * outerCoreNum;   // 再次修正核数
        ubGatherParam.mainBlockCnt = (innerCoreNum - 1) * outerCoreNum;
        ubGatherParam.coreInnerAxisTailFactor = outSize[ubGatherParam.blockAxisIdx] -
                                                (innerCoreNum - 1) * ubGatherParam.coreInnerAxisFactor;
        if (ubGatherParam.coreInnerAxisTailFactor > ubGatherParam.coreInnerAxisFactor) {
            OP_LOGE(context,
                "[CalctilingCoreWithInAxis]coreInnerAxisTailFactor is larger than coreInnerAxisFactor, unexpected!!");
        }
        SetubParamWhenAxesSame(ubGatherParam.coreInnerAxisTailFactor, maxUbFactor, ubGatherParam,
                               ubGatherParam.tailBlockUbParam);
    }
}

inline static void SetubParamWhenAxesDiff(gert::Shape outSize, uint32_t coreInnerAxisFactor, uint32_t maxUbFactor,
                                          AsStridedUbGatherParam& ubGatherParam, UbParam& ubParam)
{
    ubParam.innerAxisFactor = maxUbFactor / ubGatherParam.preSize;
    ubParam.outerAxisFactor = static_cast<uint32_t>(Ops::Base::CeilDiv(outSize[ubGatherParam.tilingAxisIdx],
                                                                 static_cast<int64_t>(ubParam.innerAxisFactor)));
    ubParam.innerAxisFactorTail = outSize[ubGatherParam.tilingAxisIdx] -
                                  ubParam.innerAxisFactor * (ubParam.outerAxisFactor - 1);
    ubParam.ubFactor = ubGatherParam.preSize * ubParam.innerAxisFactor;
    ubParam.ubFactorTail = ubGatherParam.preSize * ubParam.innerAxisFactorTail;
    // 核内部轴到ub外部轴的累乘
    ubParam.loopsPerCore = coreInnerAxisFactor * ubParam.outerAxisFactor;
    for (int32_t j = static_cast<int32_t>(ubGatherParam.tilingAxisIdx) - 1; j > static_cast<int32_t>(ubGatherParam.blockAxisIdx); j--) {
        ubParam.loopsPerCore *= outSize[j];
    }
}

inline static void PrintUbGatherParam(const gert::TilingContext* context, AsStridedUbGatherParam& ubGatherParam)
{
    OP_LOGI(context, "[CalcTilingUb]tilingAxisIdx:%u, blockAxisIdx:%u, preSize:%u.",
        ubGatherParam.tilingAxisIdx, ubGatherParam.blockAxisIdx, ubGatherParam.preSize);
    OP_LOGI(context, "[CalcTilingUb]mainBlockUbParam, innerAxisFactor:%u, innerAxisFactorTail:%u, \
        outerAxisFactor:%u, ubFactor:%u, ubFactorTail:%u, loopsPerCore:%u.",
        ubGatherParam.mainBlockUbParam.innerAxisFactor, ubGatherParam.mainBlockUbParam.innerAxisFactorTail,
        ubGatherParam.mainBlockUbParam.outerAxisFactor, ubGatherParam.mainBlockUbParam.ubFactor,
        ubGatherParam.mainBlockUbParam.ubFactorTail, ubGatherParam.mainBlockUbParam.loopsPerCore);
    OP_LOGI(context, "[CalcTilingUb]tailBlockUbParam, innerAxisFactor:%u, innerAxisFactorTail:%u, \
        outerAxisFactor:%u, ubFactor:%u, ubFactorTail:%u, loopsPerCore:%u.",
        ubGatherParam.tailBlockUbParam.innerAxisFactor, ubGatherParam.tailBlockUbParam.innerAxisFactorTail,
        ubGatherParam.tailBlockUbParam.outerAxisFactor, ubGatherParam.tailBlockUbParam.ubFactor,
        ubGatherParam.tailBlockUbParam.ubFactorTail, ubGatherParam.tailBlockUbParam.loopsPerCore);
}

inline static void CalcTilingUb(const gert::TilingContext* context, gert::Shape outSize, gert::Shape outStride, AsStridedTilingParam& tilingParam,
                                AsStridedUbGatherParam& ubGatherParam)
{
    uint32_t maxUbFactor = 0;
    CalcMaxUbFactor(tilingParam, ubGatherParam, maxUbFactor, outSize, outStride);
    OP_LOGD(context, "[CalcTilingUb]maxUbFactor:%u.", maxUbFactor);

    ubGatherParam.preSize = 1;
    uint32_t curAxisFactor = 0;
    for (int32_t i = outSize.GetDimNum() - 1; i >= static_cast<int32_t>(ubGatherParam.blockAxisIdx); i--) {
        // UB内最多三根轴
        if (outSize.GetDimNum() - i > GATHER_AXES_LIMIT) {
            SetubParamWhenOutAxesLimit(outSize, ubGatherParam.coreInnerAxisFactor, ubGatherParam,
                                       ubGatherParam.mainBlockUbParam);
            SetubParamWhenOutAxesLimit(outSize, ubGatherParam.coreInnerAxisTailFactor, ubGatherParam,
                                       ubGatherParam.tailBlockUbParam);
            ubGatherParam.tilingFlag = 1;
            break;
        }
        ubGatherParam.tilingAxisIdx = i;
        curAxisFactor = outSize[i] * ubGatherParam.preSize;
        if (curAxisFactor >= maxUbFactor || i == static_cast<int32_t>(ubGatherParam.blockAxisIdx)) {
            break;
        }
        ubGatherParam.preSize *= outSize[i];
    }

    if (ubGatherParam.tilingFlag == 0) {
        if (ubGatherParam.tilingAxisIdx == ubGatherParam.blockAxisIdx) {
            // 切ub与切核在同一根轴，ub存在跨轴处理风险，需重新分核
            CalctilingCoreWithInAxis(context, outSize, maxUbFactor, ubGatherParam);
        } else {
            SetubParamWhenAxesDiff(outSize, ubGatherParam.coreInnerAxisFactor, maxUbFactor, ubGatherParam,
                                   ubGatherParam.mainBlockUbParam);
            SetubParamWhenAxesDiff(outSize, ubGatherParam.coreInnerAxisTailFactor, maxUbFactor, ubGatherParam,
                                   ubGatherParam.tailBlockUbParam);
        }
        ubGatherParam.tilingFlag = 1;
    }

    PrintUbGatherParam(context, ubGatherParam);
}

inline static void ComputeUbGatherParam(const gert::TilingContext* context, gert::Shape outSize, gert::Shape outStride,
                                        AsStridedTilingParam& tilingParam, AsStridedUbGatherParam& ubGatherParam)
{
    int64_t shapeSize = outSize.GetShapeSize();
    OP_CHECK_IF(
        (shapeSize <= 0),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "size",
            std::to_string(shapeSize).c_str(),
            "The shape size of this parameter must be greater than 0."),
        return);
    uint64_t totalOutElement = static_cast<uint64_t>(shapeSize);
    ubGatherParam.blockNum = CalcblockNum(context, totalOutElement, tilingParam);
    ubGatherParam.blockNumMin = ubGatherParam.blockNum;
    ubGatherParam.outDimNum = outSize.GetDimNum();

    CalcTilingCore(context, outSize, ubGatherParam);
    CalcTilingUb(context, outSize, outStride, tilingParam, ubGatherParam);
}

inline static void SetUbGatherTilingParam(gert::Shape outSize, gert::Shape outStride,
    const AsStridedTilingParam& tilingParam, AsStridedUbGatherParam& ubGatherParam)
{
    ubGatherParam.storageOffset = tilingParam.storageOffset;
    ubGatherParam.ubSizePlatForm = tilingParam.ubSizePlatForm;
    for (uint32_t i = 0; i < ubGatherParam.outDimNum; i++) {
        ubGatherParam.sizeArr[i] = outSize[i];
        ubGatherParam.strideArr[i] = outStride[i];
    }

    // 当coreInnerAxisTailFactor大于coreInnerAxisFactor时，此结果非预期
    ubGatherParam.idxStrideArr[ubGatherParam.tilingAxisIdx] =
        static_cast<uint32_t>(Ops::Base::CeilDiv(outSize[ubGatherParam.tilingAxisIdx],
        static_cast<int64_t>(ubGatherParam.mainBlockUbParam.innerAxisFactor)));  
    for (int32_t i = ubGatherParam.tilingAxisIdx - 1; i >= 0; i--) {
        ubGatherParam.idxStrideArr[i] = ubGatherParam.idxStrideArr[i + 1] * outSize[i];
    }
}

inline static void MoveAlignForAsStrided(gert::TilingContext* context, 
    AsStridedTilingParam& tilingParam, gert::Shape outSize, gert::Shape outStride, AsStridedTilingData& tiling)
{
    if (tilingParam.innerAxisNum == MOVEALIGN_DIM3) {
        if ((tilingParam.nddmaDstStride[MOVEALIGN_DIM2] * tilingParam.sizeofDtype) % UB_ALIGN_SIZE != 0) {
            tilingParam.nddmaDstStride[MOVEALIGN_DIM2] =
                Ops::Base::CeilDiv(
                    tilingParam.nddmaDstStride[MOVEALIGN_DIM2] * tilingParam.sizeofDtype, UB_ALIGN_SIZE) *
                UB_ALIGN_SIZE;
            tilingParam.nddmaDstStride[1] = 0;
            tilingParam.en32BAligned = 1;
        }
    } else if (tilingParam.innerAxisNum == MOVEALIGN_DIM4) {
        if ((tilingParam.nddmaDstStride[MOVEALIGN_DIM2] * tilingParam.sizeofDtype) % UB_ALIGN_SIZE != 0) {
            tilingParam.nddmaDstStride[MOVEALIGN_DIM2] =
                Ops::Base::CeilDiv(
                    tilingParam.nddmaDstStride[MOVEALIGN_DIM2] * tilingParam.sizeofDtype, UB_ALIGN_SIZE) *
                UB_ALIGN_SIZE;
            tilingParam.nddmaDstStride[1] =
                tilingParam.nddmaDstStride[MOVEALIGN_DIM2] * tilingParam.nddmaLoop[MOVEALIGN_DIM2];
            tilingParam.en32BAligned = 1;
        } else if ((tilingParam.nddmaDstStride[1] * tilingParam.sizeofDtype) % UB_ALIGN_SIZE != 0) {
            tilingParam.nddmaDstStride[1] =
                Ops::Base::CeilDiv(tilingParam.nddmaDstStride[1] * tilingParam.sizeofDtype, UB_ALIGN_SIZE) *
                UB_ALIGN_SIZE;
            tilingParam.en32BAligned = 1;
        }
    } else if (tilingParam.innerAxisNum >= MOVEALIGN_DIM5) {
        MergeAxis4MoveAlign(context, tilingParam, outSize, outStride, tiling);
    }
}

inline static bool CheckDualCut(gert::Shape& outStride, const AsStridedTilingParam& tilingParam)
{
    // 条件1：UB切分轴的Strided最小且小于等于128Byte
    // 条件2：UB切分轴左侧包含stride小于等于64B的轴
    // 以上条件为或的关系

    int dimNums = outStride.GetDimNum();
    uint32_t minimumStrideAxisIdx = -1;
    int64_t minimumStridedByte = 0xFFFFFFFF;
    int64_t byteStrides[dimNums];
    for (int i = 0; i < dimNums; i++) {
        byteStrides[i] = outStride.GetDim(i) * tilingParam.sizeofDtype;
        if (minimumStridedByte > byteStrides[i]) {
            minimumStridedByte = byteStrides[i];
            minimumStrideAxisIdx = i;
        }
    }

    OP_CHECK_IF(
        ((minimumStrideAxisIdx == tilingParam.tilingAxisIdx) && (minimumStridedByte <= DUAL_CUT_CONDITION2)),
        OP_LOGW("CheckDualCut", "Case#1: Minimum Stride Axis cutted by Sole cut and smaller than 128B, do dual cut"),
        return true);
    
    for (uint32_t i = 0; i < tilingParam.tilingAxisIdx; i++) {
        OP_CHECK_IF((byteStrides[i] <= DUAL_CUT_CONDITION1),
        OP_LOGW("CheckDualCut", "Case#2: Sole cut outer axis have axis stride smaller than 64 Byte, do dual cut"),
        return true);
    }

    return false;
}

inline static void SetTilingDataForDualCutting(AsStridedTilingParam& tilingParam, DualCutAxisSeeker& seeker)
{
    for (int i = 0; i < TILING_ARRAY_LEN; i++) {
        tilingParam.gmShape[i] = seeker.gmShape[i];
    }
    for (int j = 0; j < TILING_ARRAY_LEN; j++) {
        tilingParam.gmInStride[j] = seeker.gmInStride[j];
    }
    for (int k = 0; k < TILING_ARRAY_LEN; k++) {
        tilingParam.gmOutStride[k] = seeker.gmOutStride[k];
    }

    for (int i = 0; i < TILING_NDDMA_LEN; i++) {
        tilingParam.nddmaLoop[i] = seeker.ubShape[i];
        tilingParam.nddmaSrcStride[i] = seeker.ubInStride[i];
        tilingParam.nddmaDstStride[i] = seeker.ubOutStride[i];
    }

    tilingParam.axisOutTotalFactor = seeker.cutAxisNums;
    tilingParam.outerAxisNum = seeker.outerAxisNums;
    tilingParam.innerAxisNum = seeker.innerAxisNums;
    tilingParam.blockNum = seeker.useCoreNum;
    tilingParam.loopsPerCore = seeker.dimsPerCore;
    tilingParam.innerAxisFactorTail = seeker.dimsPerCoreTail;

    tilingParam.nddmaTailLoop[0] = seeker.cutAxisIdx01;
    tilingParam.nddmaTailLoop[1] = seeker.cutAxisIdx02;
    // 2 means the second nature number
    tilingParam.nddmaTailLoop[2] = seeker.cutAxisTail01;
    // 3 means the third nature number
    tilingParam.nddmaTailLoop[3] = seeker.cutAxisTail02;
}

void ProcessB64Data(gert::TilingContext* context, gert::Shape& outSize, gert::Shape& outStride, AsStridedTilingParam& tilingParam) 
{
    auto outShapeSize = outSize.GetDimNum();
    auto outStrideSize = outStride.GetDimNum();

    // 情况一：不需要进行处理：
    OP_CHECK_IF(
        (outShapeSize < TILING_NDDMA_LEN),
        OP_LOGD(context, "the outShape size is less 5, do not process"), return);
    
    OP_CHECK_IF(
        (outShapeSize >= TILING_ARRAY_LEN - 2),
        OP_LOGD(context, "the outShape size is more than or equal 8, do not process"), return);

    // 情况二：最后一维stride不为1，并且shape大于等于5维，补充一维size=1, stride=1
    if(outStrideSize > 0 && outStride[outStrideSize - 1] != 1) {
        OP_LOGD(context, "#case2: the last outStride is not 1, first add one dim and process");
        outSize.SetDimNum(outShapeSize + 1);
        outSize.SetDim(outShapeSize, 1);
        outStride.SetDimNum(outStrideSize + 1);
        outStride.SetDim(outStrideSize, 1);
    }

    OP_LOGD(context, "#case2 or case3: the last outStride is 1 now, to process");
    tilingParam.sizeofDtype = INPUT_DTYPE_B32;
    tilingParam.tilingKey = INPUT_DTYPE_B32;
    tilingParam.ubSize = (tilingParam.ubSizePlatForm / BUFFER_NUM) / tilingParam.sizeofDtype;
    outShapeSize = outSize.GetDimNum();
    outStrideSize = outStride.GetDimNum();
    for(size_t i = 0; i < outStrideSize - 1; i++) {
        outStride[i] *= 2;
    }
    outSize[outShapeSize - 1] *= 2;
    tilingParam.storageOffset *= 2;

    return;
}

static bool IsStrideAffect(gert::TilingContext* context, const AsStridedTilingParam& tilingParam, gert::Shape outStride, const DualCutAxisSeeker& seeker) 
{
    // 条件一，双切分后的stride小于64B的多于单切分
    // 条件二，单切分轴的右侧大的stride更靠近尾轴，假设尾轴之前存在stride小于尾轴，排序的影响

    int32_t singleStrideMore64 = 0;
    int32_t dualStrideMore64 = 0;
    bool singleConditionTailMore64 = false;
    bool singleConditionNotTailLess64 = false;

    OP_CHECK_IF(
        (outStride.GetDimNum() == 0),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "stride",
            std::to_string(outStride.GetDimNum()).c_str(),
            "The shape dim of stride must be greater than 0."),
        return false);

    if((outStride[outStride.GetDimNum() - 1] * tilingParam.sizeofDtype) > DUAL_CUT_CONDITION1) {
        singleConditionTailMore64 = true;
    }

    for(size_t i = tilingParam.tilingAxisIdx; i < outStride.GetDimNum(); i++) {
        if((outStride[i] * tilingParam.sizeofDtype) > DUAL_CUT_CONDITION1) {
            singleStrideMore64++;
        } else if(i != outStride.GetDimNum() - 1) {
            singleConditionNotTailLess64 =  true;
        }
    }

    for(size_t i = 0; i < seeker.ubAxis.size(); i++) {
        if((seeker.ubInStride[SHAPE_NDDMA_LEN - seeker.ubAxis.size() + i] * tilingParam.sizeofDtype) > DUAL_CUT_CONDITION1) {
            dualStrideMore64++;
        }
    }
    
    OP_LOGD(
        context, "dualStrideMore64: %d, singleStrideMore64 : %d, singleConditionTailMore64 : %d, singleConditionNotTailLess64 : %d", 
            dualStrideMore64, singleStrideMore64, singleConditionTailMore64, singleConditionNotTailLess64);
    
    return ( (dualStrideMore64 < singleStrideMore64) || (singleConditionTailMore64 && singleConditionNotTailLess64) );
}

ge::graphStatus AsStridedTilingClass::SingleCutOfNDDMAForAsStrided(AsStridedTilingParam& tilingParam, gert::Shape outSize, gert::Shape outStride,
    AsStridedTilingData& tiling)
{
    auto outShapeSize = outSize.GetDimNum();
    auto outStrideSize = outStride.GetDimNum();
    if (outStrideSize > TILING_NDDMA_LEN) {
        for (uint32_t i = 0; i < TILING_NDDMA_LEN; i++) {
            tilingParam.nddmaSrcStride[i] = outStride[i + outStrideSize - TILING_NDDMA_LEN];
        }
    } else {
        for (uint32_t i = 0; i < outStrideSize; i++) {
            tilingParam.nddmaSrcStride[i + TILING_NDDMA_LEN - outStrideSize] = outStride[i];
        }
    }

    // to find tiling axis
    uint32_t curProd = 1;
    for (int32_t i = outShapeSize - 1; i >= 0; i--) {
        tilingParam.curAxisFactor = outSize[i] * tilingParam.preSize;
        curProd = outSize[i] * tilingParam.preSize;
        if (outSize[outShapeSize - 1] % (UB_ALIGN_SIZE / tilingParam.sizeofDtype)) {
            curProd = tilingParam.curAxisFactor / outSize[outShapeSize - 1] * Ops::Base::CeilDiv(static_cast<uint32_t>(outSize[outShapeSize - 1]), (UB_ALIGN_SIZE / tilingParam.sizeofDtype)) * (UB_ALIGN_SIZE / tilingParam.sizeofDtype);
        }
        if (curProd >= tilingParam.ubSize) {
            tilingParam.tilingAxisIdx = i;
            tilingParam.outerAxisNum = i + 1;
            tilingParam.innerAxisNum = outShapeSize - tilingParam.outerAxisNum + 1;
            if (tilingParam.innerAxisNum > TILING_NDDMA_LEN) {
                OP_LOGD(context_, "NDDMA max axis is 5! So there is no need to tile!");
                tilingParam.outerAxisNum = outShapeSize - TILING_NDDMA_LEN;
                tilingParam.axisOutTotalFactor = 1;
                for (uint32_t j = 0; j < tilingParam.outerAxisNum; j++) {
                    tilingParam.axisOutTotalFactor *= outSize[j];
                    tilingParam.outStrideArr[TILING_ARRAY_LEN - tilingParam.outerAxisNum + j] = outStride[j];
                }
                tilingParam.outerAxisFactor = 1;
                tilingParam.blockNum = tilingParam.axisOutTotalFactor > tilingParam.numCore ?
                                           tilingParam.numCore :
                                           tilingParam.axisOutTotalFactor;
                NoTilingMergeAxis(tiling, tilingParam, outSize);
                tilingParam.tilingFlag = 1;
                break;
            }
            // 0.9: Priority schemes with average UB usage > 90%
            for (uint32_t j = tilingParam.ubSize; j >= (tilingParam.ubSize * 0.9); j--) {
                if ((tilingParam.curAxisFactor % j) == 0) {
                    tilingParam.outerAxisFactor = tilingParam.curAxisFactor / j;
                    if (outSize[i] % tilingParam.outerAxisFactor == 0) {

                        // 考虑对齐后的size可能会超过可用ub
                        uint32_t tempSize = 0;
                        if (i == static_cast<int32_t>(outShapeSize - 1)) {
                            tempSize = Ops::Base::CeilDiv(j, (UB_ALIGN_SIZE / tilingParam.sizeofDtype)) * (UB_ALIGN_SIZE / tilingParam.sizeofDtype);
                        } else {
                            tempSize = j / outSize[outShapeSize - 1];
                            tempSize *= Ops::Base::CeilDiv(static_cast<uint32_t>(outSize[outShapeSize - 1]), (UB_ALIGN_SIZE / tilingParam.sizeofDtype)) * (UB_ALIGN_SIZE / tilingParam.sizeofDtype);
                        }
                        if (tilingParam.ubSize < tempSize) {
                            continue;
                        }
                        
                        OP_LOGD(context_, "UB can use %u", j);
                        OP_LOGD(context_, "Can be total tiling");
                        tilingParam.innerAxisFactor = outSize[i] / tilingParam.outerAxisFactor;
                        tilingParam.tilingFlag = 1;
                        tilingParam.ubFactor = j;
                        tiling.ubFactor = tilingParam.ubFactor;
                        break;
                    }
                }
            }

            if (tilingParam.tilingFlag == 0) {
                OP_LOGD(context_, "There is no total tiling!");
                tilingParam.innerAxisFactor = tilingParam.ubSize / tilingParam.preSize;
                if (i != static_cast<int32_t>(outShapeSize - 1)) {
                    tilingParam.innerAxisFactor = tilingParam.ubSize / (curProd / outSize[i]);
                }
                tilingParam.outerAxisFactor =
                    (outSize[i] + tilingParam.innerAxisFactor - 1) / tilingParam.innerAxisFactor;
                tilingParam.innerAxisFactorTail =
                    (tilingParam.innerAxisFactor * tilingParam.outerAxisFactor == outSize[i]) ?
                        0 :
                        outSize[i] - tilingParam.innerAxisFactor * (tilingParam.outerAxisFactor - 1);
                tilingParam.ubFactor = tilingParam.preSize * tilingParam.innerAxisFactor;
                tilingParam.ubFactorTail = tilingParam.preSize * tilingParam.innerAxisFactorTail;
                tilingParam.tilingFlag = 1;
                tiling.ubFactor = tilingParam.ubFactor;
            }
            tilingParam.axisOutTotalFactor = tilingParam.outerAxisFactor;
            for (uint32_t j = 0; j < tilingParam.tilingAxisIdx; j++) {
                tilingParam.axisOutTotalFactor *= outSize[j];
            }
            tilingParam.blockNum = tilingParam.axisOutTotalFactor > tilingParam.numCore ?
                                       tilingParam.numCore :
                                       tilingParam.axisOutTotalFactor;
            MergeAxisAfterTiling(tiling, tilingParam, outSize);

            for (uint32_t j = 0; j <= tilingParam.tilingAxisIdx; j++) {
                tilingParam.outStrideArr[TILING_ARRAY_LEN - tilingParam.tilingAxisIdx - 1 + j] = outStride[j];
            }
            if (tilingParam.innerAxisNum <= TILING_NDDMA_LEN) {
                tilingParam.outStrideArr[TILING_ARRAY_LEN - 1] *= tilingParam.innerAxisFactor;
            }
            break;
        } else {
            tilingParam.preSize *= outSize[i];
        }
    }

    if (tilingParam.tilingFlag == 0) {
        if (outShapeSize > TILING_NDDMA_LEN) {
            OP_LOGD(context_, "NDDMA max axis is 5! So all is no need to tile!");
            tilingParam.outerAxisNum = outShapeSize - TILING_NDDMA_LEN;
            tilingParam.axisOutTotalFactor = 1;
            for (uint32_t i = 0; i <= tilingParam.outerAxisNum - 1; i++) {
                tilingParam.axisOutTotalFactor *= outSize[i];
                tilingParam.outStrideArr[TILING_ARRAY_LEN - tilingParam.outerAxisNum + i] = outStride[i];
            }
            tilingParam.outerAxisFactor = 1;
            tilingParam.innerAxisNum = outShapeSize;
            tilingParam.blockNum = tilingParam.axisOutTotalFactor > tilingParam.numCore ?
                                       tilingParam.numCore :
                                       tilingParam.axisOutTotalFactor;
            NoTilingMergeAxis(tiling, tilingParam, outSize);
        } else {
            OP_LOGD(context_, "No need to tile!");
            tilingParam.axisOutTotalFactor = 1;
            tilingParam.outerAxisFactor = 1;
            tilingParam.outerAxisNum = 1;
            tilingParam.innerAxisFactor = outSize[0];
            tilingParam.tilingAxisIdx = 0;
            tilingParam.innerAxisNum = outShapeSize;
            tilingParam.blockNum = 1;
            MergeAxisAfterTiling(tiling, tilingParam, outSize);
            tiling.ubFactor = tilingParam.curAxisFactor;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AsStridedTilingClass::NDDMAForAsStrided(
    AsStridedTilingParam& tilingParam, gert::Shape outSize, gert::Shape outStride,
    AsStridedTilingData& tiling)
{
    OP_LOGD(context_, "Enter SingleTilingForAsStrided");

    AsStridedTilingParam tempTilingParam = tilingParam; //先保存

    // stride all zero
    if (IsAllStridesZero(context_, outStride)) {
        SetAllStridesZeroTilingParam(context_, outSize, tilingParam);
        SetZeroStrideTilingData(tilingParam);
        tilingParam.tilingKey = ALL_STRIDES_ZERO_KEY;
        return ge::GRAPH_SUCCESS; 
    }

    SingleCutOfNDDMAForAsStrided(tilingParam, outSize,  outStride, tiling);

    // move_align
    tilingParam.movealignFlag = IsMoveAlign(context_, outSize, outStride, tilingParam);
    OP_LOGD(context_, "MovealignFlag: %u", tilingParam.movealignFlag);
    if (tilingParam.movealignFlag) {
        MoveAlignForAsStrided(context_, tilingParam, outSize, outStride, tiling);
        tilingParam.tilingKey += MOVEALIGN_KEY;
        return ge::GRAPH_SUCCESS;
    }

    // UB gather
    if (IsUbGather(context_, tilingParam)) {
        AsStridedUbGatherParam ubGatherParam;
        ComputeUbGatherParam(context_, outSize, outStride, tilingParam, ubGatherParam);
        if (CheckBndryForUint16GatherIdx(outSize, outStride, tilingParam, ubGatherParam) && 
            (ubGatherParam.mainBlockUbParam.ubFactor * tilingParam.sizeofDtype > GATHER_UB_SIZE_LOWER_LIMIT) ) {
            SetUbGatherTilingParam(outSize, outStride, tilingParam, ubGatherParam);
            SetWithGatherTilingData(ubGatherParam);
            tilingParam.blockNum = ubGatherParam.blockNum;
            tilingParam.tilingKey = WITH_GATHER_KEY;
            return ge::GRAPH_SUCCESS;
        } else {
            OP_LOGD(context_, "Exit ubGather template, because gather index exceeds uint16 boundary, or UB factor < 2048B.");
        }
    }

    // simt
    if (IsSmallShape(context_, outSize, tilingParam.sizeofDtype)) {
        SetSimtTilingParam(context_, outSize, outStride, tilingParam);
        SetSimtTilingData(tilingParam);
        tilingParam.tilingKey = SIMT_KEY;
        return ge::GRAPH_SUCCESS; 
    }

    // 走NDDMA时对b64数据进行处理
    if(tilingParam.sizeofDtype == INPUT_DTYPE_B64) {
        if(outSize.GetDimNum() < TILING_NDDMA_LEN) {
            return ge::GRAPH_SUCCESS;
        }
        tilingParam = tempTilingParam;
        ProcessB64Data(context_, outSize, outStride, tilingParam);
        // 重新进行单切分
        SingleCutOfNDDMAForAsStrided(tilingParam, outSize,  outStride, tiling);
        return ge::GRAPH_SUCCESS;
    }

    auto outShapeSize = outSize.GetDimNum();
    auto outStrideSize = outStride.GetDimNum();

    // DualCut
    tilingParam.dualCutFlag = CheckDualCut(outStride, tilingParam);
    bool dualFlag = (tilingParam.dualCutFlag) && (tilingParam.numCore > 0) && (tilingParam.tilingFlag != 0);
    OP_LOGD(
        context_, "dualFlag: %d, dualCutFlag = %d, numCore = %u, tilingFlag = %u", dualFlag,
        tilingParam.dualCutFlag, tilingParam.numCore, tilingParam.tilingFlag);
    
    if (dualFlag) {
        int64_t shape[outShapeSize];
        int64_t strides[outStrideSize];
        for (uint32_t i = 0; i < outShapeSize; i++) {
            shape[i] = outSize.GetDim(i);
            strides[i] = outStride.GetDim(i);
        }

        DualCutAxisSeeker seeker(shape, strides, outShapeSize, tilingParam.sizeofDtype, context_);
        bool cutSuccess = seeker.FindDualCutAxis(tilingParam.ubSizePlatForm, BUFFER_NUM);
        OP_LOGD(
        context_, "DualCutSuccess: %d", cutSuccess);
        if (cutSuccess) {
            seeker.GenTilingData();
            seeker.ComputeBlockTiling(tilingParam.numCore);
            seeker.PrintDebug();

            int64_t dualTileSize = tilingParam.sizeofDtype; // 双切分tileSize
            for (int64_t i = 0; i < TILING_NDDMA_LEN; i++) {
                dualTileSize *= static_cast<int64_t>(seeker.ubShape[i]);
            }

            int64_t singleTileSize = tilingParam.sizeofDtype;  // 单切分tileSize
            for (int64_t i = 0; i < TILING_NDDMA_LEN; i++) {
                singleTileSize *= static_cast<int64_t>(tilingParam.nddmaLoop[i]);
            }

            bool singleTailMoreDualTail = static_cast<int32_t>(tilingParam.nddmaLoop[TILING_NDDMA_LEN - 1]) > seeker.ubShape[TILING_NDDMA_LEN - 1];   // 单切分尾轴大于双切分尾轴，否则只可能相等
            bool isStrideAffect = IsStrideAffect(context_, tilingParam, outStride, seeker); // stride对单双切分是否有影响
            bool dualTileSizeSatifyCondition = dualTileSize >= singleTileSize || dualTileSize >= DUAL_CUT_CONDITION3; // 双切分是否满足基本的搬运tileSize，更好的利用搬运带宽

            if(!dualTileSizeSatifyCondition) {
                OP_LOGD(
                    context_, "dualTileSizeSatifyCondition: %d, dualTileSize = %ld, singleTileSize = %ld", dualTileSizeSatifyCondition, dualTileSize, singleTileSize);
                return ge::GRAPH_SUCCESS;
            }
            
            if(!isStrideAffect && !singleTailMoreDualTail) { // stride对搬运无影响并且单切分的尾轴不大于双切分，考虑连续搬出，走单切分方式
                OP_LOGD(
                    context_, "isStrideAffect: %d, singleTailMoreDualTail = %d", isStrideAffect, singleTailMoreDualTail);
                return ge::GRAPH_SUCCESS;
            }

            tilingParam.tilingKey = DUAL_CUT_KEY;
            SetTilingDataForDualCutting(tilingParam, seeker);
            for(int64_t idx = 0; idx < TILING_ARRAY_LEN; idx++) {
                tiling.gmOutStride[idx] = tilingParam.gmOutStride[idx];
            }
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool CheckInputInfo(gert::TilingContext *context, gert::Shape outSize, gert::Shape outStride, const gert::Shape& xShape,
                    AsStridedTilingParam& tilingParam)
{
    uint32_t requiredStorageSize = 0;
    uint32_t originalTensorStorageSize = 1;
    for (size_t i = 0; i < outSize.GetDimNum(); i++) {
        OP_CHECK_IF(outSize[i] < 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "output_size",
                        std::to_string(outSize.GetDim(i)).c_str(),
                        "The value of output_size must be greater than or equal to 0."),
                    return false);
        
        OP_CHECK_IF(outStride[i] < 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "output_stride",
                        std::to_string(outStride.GetDim(i)).c_str(),
                        "The value of output_stride must be greater than or equal to 0."),
                    return false);
        
        requiredStorageSize += (outSize[i] - 1) * outStride[i];
    }
    for (uint32_t i = 0; i < xShape.GetDimNum(); i++) {
        OP_CHECK_IF(xShape[i] < 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "input_size",
                        std::to_string(xShape.GetDim(i)).c_str(),
                        "The value of input_size must be greater than or equal to 0."),
                    return false);
        originalTensorStorageSize *= xShape.GetDim(i);
    }

    OP_CHECK_IF((tilingParam.storageOffset + static_cast<int64_t>(requiredStorageSize)) >= static_cast<int64_t>(originalTensorStorageSize),
                    OP_LOGE(context,
                    "The output element is out of input range!"), return false);
    
    tilingParam.inputSize = originalTensorStorageSize;
    return true;
}

ge::graphStatus AsStridedTilingClass::HandleEmptyTensor()
{
    context_->SetBlockDim(1);
    context_->SetTilingKey(EMPTY_TENSOR_KEY);
    emptyTilingData_ = context_->GetTilingData<AsStridedEmptyTilingData>();
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = WORKSPACE_SIZE_8MB;
    OP_LOGD(context_, "Output is an empty tensor.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AsStridedTilingClass::GetAndValidateDataType(AsStridedTilingParam& tilingParam, uint32_t ubSizePlatform)
{
    auto xTensorType = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xTensorType);
    auto dataType = xTensorType->GetDataType();
    OP_CHECK_IF(
        tilingTypeKeyMap.count(dataType) == 0, 
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
            Ops::Base::ToString(dataType).c_str(),
            "The dtype of x must be within the range [DT_INT64, DT_UINT64, DT_INT32, DT_UINT32, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_BF16, DT_INT16, DT_UINT16, DT_BOOL, DT_COMPLEX32, DT_COMPLEX64, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN]."),
        return ge::GRAPH_FAILED);
    tilingParam.ubSize = (ubSizePlatform / BUFFER_NUM) / tilingTypeKeyMap[dataType];
    tilingParam.sizeofDtype = tilingTypeKeyMap[dataType];
    tilingParam.tilingKey = tilingTypeKeyMap[dataType];
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AsStridedTilingClass::ExecuteTilingAndSetWorkspace(
    AsStridedTilingParam& tilingParam, AsStridedTilingData& tilingData,
    AsStridedRunInfo& runInfo)
{
    ge::graphStatus resOfTiling = NDDMAForAsStrided(tilingParam, runInfo.outputSize, runInfo.outputStride, tilingData);
    OP_CHECK_IF(
        resOfTiling != ge::GRAPH_SUCCESS, OP_LOGE(context_, "Tiling fail."), return ge::GRAPH_FAILED);

    resOfTiling = SetTilingData(tilingData, tilingParam);
    OP_CHECK_IF(resOfTiling != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "SetTilingData fail."), return ge::GRAPH_FAILED);

    size_t usrSize = 0;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    context_->SetBlockDim(tilingParam.blockNum);
    context_->SetTilingKey(tilingParam.tilingKey);
    OP_LOGI(context_, "TilingForAsStridedOfAsc success, blockNum:%u, tilingKey:%u.",
            tilingParam.blockNum, tilingParam.tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AsStridedTilingClass::TilingForAsStridedOfAsc(uint32_t maxCoreNum, uint32_t ubSizePlatform,
                                        AsStridedRunInfo& runInfo, int64_t storageOffset)
{
    OP_LOGD(context_, "Enter TilingForAsStridedOfAsc");
    AsStridedTilingParam tilingParam;
    AsStridedTilingData tilingData;

    tilingParam.numCore = maxCoreNum;
    tilingParam.ubSizePlatForm = ubSizePlatform;
    tilingParam.storageOffset = storageOffset;
    auto xTensorShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xTensorShape);
    const gert::Shape& xShape = xTensorShape->GetStorageShape();
    OP_CHECK_IF(runInfo.outputSize.GetDimNum() > VALID_DIM, 
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "output_size",
                        std::to_string(runInfo.outputSize.GetDimNum()).c_str(),
                        ("The shape dim of output_size must be within the range [0, " + std::to_string(VALID_DIM) + "].").c_str()),
                    return ge::GRAPH_FAILED);

    if (runInfo.outputSize.GetShapeSize() == 0) {
        return HandleEmptyTensor();
    }

    OP_CHECK_IF(!CheckInputInfo(context_, runInfo.outputSize, runInfo.outputStride, xShape, tilingParam),
                    OP_LOGE(context_, "The input info check failed!"), return ge::GRAPH_FAILED);

    if (GetAndValidateDataType(tilingParam, ubSizePlatform) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ExecuteTilingAndSetWorkspace(tilingParam, tilingData, runInfo);
}

static ge::graphStatus TilingForAsStridedArch35(gert::TilingContext* context)
{
    OP_LOGI(context, "[math] AsStrided tiling running begin");
    const AsStridedCompileInfo* compile_info = reinterpret_cast<const AsStridedCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    auto runtime_x_shape_ptr = context->GetInputShape(IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, runtime_x_shape_ptr);
    auto x_shape = Ops::Base::EnsureNotScalar(runtime_x_shape_ptr->GetStorageShape());

    // get const value of storage_offset
    int64_t storage_offset = 0;
    if (Ops::Base::GetConstInt(context, IN_OFFSET, storage_offset)) {
      OP_LOGI(context, "the storage_offset is const, get value is %ld", storage_offset);
      OP_CHECK_IF(
          storage_offset < 0,
          OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "storage_offset",
              std::to_string(storage_offset).c_str(),
              "The value of storage_offset must be greater than or equal to 0."),
          return ge::GRAPH_FAILED);
    } else {
      OP_LOGI(context, "the storage_offset is not const, will use default value 0");
      storage_offset = 0;
    }

    AsStridedRunInfo runInfo;
    OP_CHECK_IF(!GetSizeAndStride(context, runInfo),
                    OP_LOGE(context, "get const of Size/Stride failed"),
                    return ge::GRAPH_FAILED);
    
    OP_LOGI("AsStridedOutSizeInfoMath", "the out size is:[%s].", Ops::Base::ToString(runInfo.outputSize).c_str());
    OP_LOGI("AsStridedOutStridedInfoMath", "the out strided is:[%s].", Ops::Base::ToString(runInfo.outputStride).c_str());
    OP_LOGI("AsStridedStorageOffset", "the storage offset is:[%ld].", storage_offset);

    // do merge, stride all zero do not merge
    if (!IsAllStridesZero(context, runInfo.outputStride)) {
        MergeAxis(runInfo);
    }
    OP_LOGI(context, "the input shape is:[%s].", Ops::Base::ToString(x_shape).c_str());
    OP_LOGI(context, "the adjusted output shape is:[%s].", Ops::Base::ToString(runInfo.outputSize).c_str());
    OP_LOGI(context, "the adjusted output stride is:[%s].", Ops::Base::ToString(runInfo.outputStride).c_str());

    uint32_t maxCoreNum = compile_info->maxCoreNum;
    uint32_t ubSizePlatform = compile_info->ubSizePlatform;
    AsStridedTilingClass tiling (context);
    return tiling.TilingForAsStridedOfAsc(maxCoreNum, ubSizePlatform, runInfo, storage_offset);
}

static ge::graphStatus TilingPrepareForAsStridedArch35(gert::TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<AsStridedCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context, "AscendC tiling is starting!");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compile_info->maxCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compile_info->maxCoreNum <= 0), OP_LOGE(context, "The core num is invalid."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compile_info->ubSizePlatform = static_cast<uint32_t>(ubSize);
    OP_CHECK_IF(
        (compile_info->ubSizePlatform <= 0), OP_LOGE(context, "The ubSize is invalid."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AsStrided)
    .Tiling(TilingForAsStridedArch35)
    .TilingParse<AsStridedCompileInfo>(TilingPrepareForAsStridedArch35)
    .TilingInputsDataDependency({IN_SIZE, IN_STRIDE, IN_OFFSET});

} // namespace optiling