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
 * \file prod_env_mat_a_calc_rij_infershape.cpp
 * \brief InferShape for ProdEnvMatACalcRij. Output shapes depend on the const value of natoms and the sel_a attribute:
 *        rij is [nsample, nloc * nnei * 3], the other five outputs are [nsample, nloc * nnei]. When coord shape is
 *        unknown or natoms is not a compile-time constant, all outputs fall back to a dynamic 2D shape.
 */

#include <cstdint>

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t kInputCoordIdx = 0U;
constexpr size_t kInputNatomsIdx = 2U;
constexpr size_t kAttrSelAIdx = 3U;
constexpr size_t kOutputRijIdx = 0U;
constexpr size_t kOutputNlistIdx = 1U;
constexpr size_t kOutputDistanceIdx = 2U;
constexpr size_t kOutputRijXIdx = 3U;
constexpr size_t kOutputRijYIdx = 4U;
constexpr size_t kOutputRijZIdx = 5U;
constexpr size_t kOutputNum = 6U;
constexpr size_t kOutputDimNum = 2U;
constexpr int64_t kCoordinateXyzNum = 3;
constexpr int64_t kUnknownDim = -1;
} // namespace

static bool IsShapeUnknown(const gert::Shape* shape)
{
    for (size_t i = 0; i < shape->GetDimNum(); i++) {
        if (shape->GetDim(i) < 0) {
            return true;
        }
    }
    return false;
}

static ge::graphStatus SetAllOutputsDynamic(gert::InferShapeContext* context)
{
    for (size_t i = 0; i < kOutputNum; i++) {
        gert::Shape* outShape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
        outShape->SetDimNum(kOutputDimNum);
        outShape->SetDim(0U, kUnknownDim);
        outShape->SetDim(1U, kUnknownDim);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetNnei(gert::InferShapeContext* context, int64_t& nnei)
{
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* selA = attrs->GetAttrPointer<gert::ContinuousVector>(kAttrSelAIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, selA);
    const int64_t* selAData = reinterpret_cast<const int64_t*>(selA->GetData());
    nnei = 0;
    for (size_t i = 0; i < selA->GetSize(); i++) {
        nnei += selAData[i];
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ProdEnvMatACalcRij(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShape4ProdEnvMatACalcRij");

    const gert::Shape* coordShape = context->GetInputShape(kInputCoordIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, coordShape);

    int64_t nnei = 0;
    if (GetNnei(context, nnei) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Output sizes need nloc, which comes from the const value of natoms. If coord shape is unknown or natoms is not a
    // compile-time constant, the concrete size cannot be inferred, so fall back to a dynamic 2D shape.
    const gert::Tensor* natomsTensor = context->GetInputTensor(kInputNatomsIdx);
    if (IsShapeUnknown(coordShape) || natomsTensor == nullptr || natomsTensor->GetData<int32_t>() == nullptr) {
        return SetAllOutputsDynamic(context);
    }

    const int64_t nsample = coordShape->GetDim(0);
    const int64_t nloc = static_cast<int64_t>(natomsTensor->GetData<int32_t>()[0]);
    const int64_t rijLen = nloc * nnei * kCoordinateXyzNum;
    const int64_t otherLen = nloc * nnei;

    gert::Shape* rijShape = context->GetOutputShape(kOutputRijIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, rijShape);
    rijShape->SetDimNum(kOutputDimNum);
    rijShape->SetDim(0U, nsample);
    rijShape->SetDim(1U, rijLen);

    const size_t otherOutputs[] = {kOutputNlistIdx, kOutputDistanceIdx, kOutputRijXIdx, kOutputRijYIdx, kOutputRijZIdx};
    for (size_t outIdx : otherOutputs) {
        gert::Shape* outShape = context->GetOutputShape(outIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
        outShape->SetDimNum(kOutputDimNum);
        outShape->SetDim(0U, nsample);
        outShape->SetDim(1U, otherLen);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ProdEnvMatACalcRij)
    .InferShape(InferShape4ProdEnvMatACalcRij)
    .InputsDataDependency({kInputNatomsIdx});
} // namespace ops
