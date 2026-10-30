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
 * \file mul_addn_tiling.cc
 * \brief
 */

#include "mul_addn_tiling.h"
#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
namespace optiling {
constexpr int64_t INPUT_X1_IDX = 0;
constexpr int64_t INPUT_DIM_NUM = 3;
constexpr int64_t Tiling_KEY_FLOAT32 = 0;
constexpr int64_t Tiling_KEY_FLOAT16 = 1;
constexpr int64_t Tiling_KEY_BFLOAT16 = 2;

constexpr int64_t SHAPEN_MAX = 2040;
constexpr int64_t REMAINED_UB = 98304;
constexpr int64_t Dim_VAlUE = 1;
constexpr int32_t REPEAT_LIMITED = 255;

constexpr int64_t IR_IDX1 = 0;
constexpr int64_t IR_IDX2 = 1;
constexpr int64_t RALATIVE_IDX = 0;
constexpr int64_t IR_IDX = 0;

constexpr size_t DIM0 = 0;
constexpr size_t DIM1 = 1;
constexpr size_t DIM2 = 2;

constexpr int64_t ALIGN_COEF = 1;

constexpr int64_t DATA_ALIGN_BF16 = 16;
constexpr int64_t DATA_ALIGN_FLOAT16 = 16;
constexpr int64_t DATA_ALIGN_FLOAT32 = 8;

constexpr int64_t MNUM_BF16_MUL_COEF = 6;
constexpr int64_t MNUM_FLOAT16_MUL_COEF = 4;
constexpr int64_t MNUM_FLOAT32_MUL_COEF = 8;

constexpr int64_t MNUM_BF16_ADD_COEF = 20;
constexpr int64_t MNUM_FLOAT16_ADD_COEF = 18;
constexpr int64_t MNUM_FLOAT32_ADD_COEF = 36;

constexpr int64_t MNUM_BF16_SUB_COEF = 4;
constexpr int64_t MNUM_FLOAT16_SUB_COEF = 2;
constexpr int64_t MNUM_FLOAT32_SUB_COEF = 4;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t UbSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, UbSize);
    UbSize = UbSize - REMAINED_UB;

    MulAddnTilingData tiling;
    auto x1ShapePtr = context->GetDynamicInputShape(IR_IDX1, RALATIVE_IDX);
    auto x2ShapePtr = context->GetDynamicInputShape(IR_IDX2, RALATIVE_IDX);
    auto x1Shape = x1ShapePtr->GetStorageShape();
    auto x2Shape = x2ShapePtr->GetStorageShape();

    if (x1Shape.GetDimNum() != INPUT_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "Check x1Shape failed, the dims of x1 not equal 3.");
        return ge::GRAPH_FAILED;
    }
    if (x2Shape.GetDimNum() != INPUT_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "Check x2Shape failed, the dims of x2 not equal 3.");
        return ge::GRAPH_FAILED;
    }
    auto x1Dim3 = x1Shape.GetDim(DIM2);
    auto x2Dim2 = x2Shape.GetDim(DIM1);
    if (x1Dim3 != Dim_VAlUE) {
        OP_LOGE(context->GetNodeName(), "Check x1Shape failed, the 3rd dimension of x1 must be 1.");
        return ge::GRAPH_FAILED;
    }
    if (x2Dim2 != Dim_VAlUE) {
        OP_LOGE(context->GetNodeName(), "Check x2Shape failed, the second dimension of x2 must be 1.");
        return ge::GRAPH_FAILED;
    }
    auto shapeB = x1Shape.GetDim(DIM0);
    auto shapeM = x1Shape.GetDim(DIM1);
    auto shapeN = x2Shape.GetDim(DIM2);
    if (shapeN > SHAPEN_MAX) {
        OP_LOGE(context->GetNodeName(), "Check Shape failed, the shapeN must be less than or equal to 2040.");
        return ge::GRAPH_FAILED;
    }

    uint64_t shapeNAlign = 0;
    uint64_t dataAlign = 0;
    int32_t mNum = 0;
    auto dataType = context->GetInputDesc(INPUT_X1_IDX)->GetDataType();
    if (dataType == ge::DT_BF16) {
        dataAlign = DATA_ALIGN_BF16;
        shapeNAlign = (shapeN + dataAlign - ALIGN_COEF) / dataAlign * dataAlign;
        mNum = (UbSize - MNUM_BF16_SUB_COEF * shapeNAlign) / (MNUM_BF16_ADD_COEF + MNUM_BF16_MUL_COEF * shapeNAlign);
    } else if (dataType == ge::DT_FLOAT16) {
        dataAlign = DATA_ALIGN_FLOAT16;
        shapeNAlign = (shapeN + dataAlign - ALIGN_COEF) / dataAlign * dataAlign;
        mNum = (UbSize - MNUM_FLOAT16_SUB_COEF * shapeNAlign) / (MNUM_FLOAT16_ADD_COEF + MNUM_FLOAT16_MUL_COEF * shapeNAlign);
    } else if (dataType == ge::DT_FLOAT) {
        dataAlign = DATA_ALIGN_FLOAT32;
        shapeNAlign = (shapeN + dataAlign - ALIGN_COEF) / dataAlign * dataAlign;
        mNum = (UbSize - MNUM_FLOAT32_SUB_COEF * shapeNAlign) / (MNUM_FLOAT32_ADD_COEF + MNUM_FLOAT32_MUL_COEF * shapeNAlign);
    }

    auto taskNum = shapeB;
    int64_t coreTaskNum;
    if (coreNum != 0){
        coreTaskNum = (taskNum + coreNum - 1) / coreNum;
    }else{
        return 0;
    }
    auto useCoreNum = (taskNum + coreTaskNum - 1) / coreTaskNum;
    auto lastCoreTaskNum = taskNum - (useCoreNum - 1) * coreTaskNum;

    mNum = std::min(mNum, REPEAT_LIMITED);
    int64_t mLoopNum;
    if (mNum != 0){
        mLoopNum = (shapeM + mNum - 1) / mNum;
    }else{
        return 0;
    }
    mNum = (shapeM + mLoopNum - 1) / mLoopNum;
    int64_t mNumTail = shapeM - mNum * (mLoopNum - 1);
    int64_t N = *context->GetAttrs()->GetInt(0);

    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = Tiling_KEY_FLOAT16;
    } else if (dataType == ge::DT_FLOAT) {
        tilingKey = Tiling_KEY_FLOAT32;
    } else if (dataType == ge::DT_BF16) {
        tilingKey = Tiling_KEY_BFLOAT16;
    }

    tiling.set_N(N);
    tiling.set_shapeB(shapeB);
    tiling.set_shapeM(shapeM);
    tiling.set_shapeN(shapeN);
    tiling.set_shapeNAlign(shapeNAlign);
    tiling.set_useCoreNum(useCoreNum);
    tiling.set_coreTaskNum(coreTaskNum);
    tiling.set_lastCoreTaskNum(lastCoreTaskNum);
    tiling.set_mNum(mNum);
    tiling.set_mLoopNum(mLoopNum);
    tiling.set_mNumTail(mNumTail);
    context->SetBlockDim(useCoreNum);
    context->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForMulAddn(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct MulAddnCompileInfo {
};

IMPL_OP_OPTILING(MulAddn).Tiling(TilingFunc).TilingParse<MulAddnCompileInfo>(TilingPrepareForMulAddn);
} // namespace optiling