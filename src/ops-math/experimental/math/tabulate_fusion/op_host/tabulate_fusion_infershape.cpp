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
 * \file tabulate_fusion_infershape.cpp
 * \brief InferShape / InferDataType implementation for tabulate_fusion operator
 *
 * Output shape: descriptor = (nloc, 4, last_layer_size)
 *   nloc = em.shape[0]
 *   dim1 = 4 (4 ll channels)
 *   dim2 = last_layer_size (ATTR)
 * Output dtype: em.dtype (explicitly set in InferDataType callback, per MDE 7.2)
 *
 * Validation rules mirror MDE 3.3 (consistent with tiling side):
 *   - Shape checks are performed in InferShape callback.
 *   - Dtype checks (consistency + support) are performed in InferDataType callback,
 *     because SE 5.9 declares no value dependency and InferShapeContext does not
 *     expose GetInputDesc/SetDataType (MDE 7.2 example uses non-existent API;
 *     corrected here to use InferDataTypeContext::SetOutputDataType).
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;
static constexpr int64_t IDX_2 = 2;
static constexpr int64_t INDEX_TABLE = 0;
static constexpr int64_t INDEX_TABLE_INFO = 1;
static constexpr int64_t INDEX_EM_X = 2;
static constexpr int64_t INDEX_EM = 3;
static constexpr int64_t INDEX_DESCRIPTOR = 0;
static constexpr int64_t DESC_DIM1 = 4;
static constexpr int64_t TABLE_INFO_MIN_SIZE = 5;
static constexpr int64_t ALIGN_64 = 64;
static constexpr int64_t COEFF_COUNT = 6; // polynomial coefficient count a0~a5

// Validate em shape, em_x total elements, and last_layer_size attr (subset of MDE 3.3)
static ge::graphStatus ValidateInferShapeEmAndAttrs(gert::InferShapeContext* context, int64_t& nloc, int64_t& nnei,
                                                    int64_t& lastLayerSize)
{
    // em shape: [nloc, nnei, 4]
    const gert::Shape* emShape = context->GetInputShape(INDEX_EM);
    OP_CHECK_NULL_WITH_CONTEXT(context, emShape);
    OP_CHECK_IF(emShape->GetDimNum() != 3,
                OP_LOGE(context, "TabulateFusion: em should be 3D, got %zu", emShape->GetDimNum()),
                return GRAPH_FAILED);
    OP_CHECK_IF(emShape->GetDim(IDX_2) != 4,
                OP_LOGE(context, "TabulateFusion: em.shape[2] should be 4, got %ld", emShape->GetDim(IDX_2)),
                return GRAPH_FAILED);
    nloc = emShape->GetDim(IDX_0);
    nnei = emShape->GetDim(IDX_1);
    OP_CHECK_IF(nloc <= 0, OP_LOGE(context, "TabulateFusion: nloc should be > 0, got %ld", nloc), return GRAPH_FAILED);
    OP_CHECK_IF(nnei <= 0, OP_LOGE(context, "TabulateFusion: nnei should be > 0, got %ld", nnei), return GRAPH_FAILED);

    // em_x: total elements should == nloc * nnei
    const gert::Shape* emXShape = context->GetInputShape(INDEX_EM_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, emXShape);
    int64_t emXTotal = emXShape->GetShapeSize();
    OP_CHECK_IF(emXTotal != nloc * nnei,
                OP_LOGE(context, "TabulateFusion: em_x total %ld != nloc*nnei %ld*%ld", emXTotal, nloc, nnei),
                return GRAPH_FAILED);

    // required attr last_layer_size
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* lastLayerSizePtr = attrs->GetAttrPointer<int64_t>(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, lastLayerSizePtr);
    lastLayerSize = *lastLayerSizePtr;
    OP_CHECK_IF(lastLayerSize <= 0,
                OP_LOGE(context, "TabulateFusion: last_layer_size should be > 0, got %ld", lastLayerSize),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

// Validate table shape (2D, shape[1]==lastSizeAlign*COEFF_COUNT) and table_info size (subset of MDE 3.3)
static ge::graphStatus ValidateInferShapeTables(gert::InferShapeContext* context, int64_t lastLayerSize)
{
    // table: 2D, shape[1] == lastSizeAlign * COEFF_COUNT
    const gert::Shape* tableShape = context->GetInputShape(INDEX_TABLE);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableShape);
    OP_CHECK_IF(tableShape->GetDimNum() != 2,
                OP_LOGE(context, "TabulateFusion: table should be 2D, got %zu", tableShape->GetDimNum()),
                return GRAPH_FAILED);
    int64_t lastSizeAlign = ((lastLayerSize + ALIGN_64 - 1) / ALIGN_64) * ALIGN_64;
    OP_CHECK_IF(tableShape->GetDim(IDX_1) != lastSizeAlign * COEFF_COUNT,
                OP_LOGE(context, "TabulateFusion: table.shape[1] %ld != lastSizeAlign*6 %ld", tableShape->GetDim(IDX_1),
                        lastSizeAlign * COEFF_COUNT),
                return GRAPH_FAILED);

    // table_info: 1D, total elements should >= TABLE_INFO_MIN_SIZE
    const gert::Shape* tableInfoShape = context->GetInputShape(INDEX_TABLE_INFO);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableInfoShape);
    int64_t tableInfoTotal = tableInfoShape->GetShapeSize();
    OP_CHECK_IF(tableInfoTotal < TABLE_INFO_MIN_SIZE,
                OP_LOGE(context, "TabulateFusion: size of table_info should be >= 5, got %ld", tableInfoTotal),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

// InferShape entry point (orchestrates validation then sets output shape)
static ge::graphStatus InferShapeTabulateFusion(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeTabulateFusion");

    int64_t nloc = 0, nnei = 0, lastLayerSize = 0;
    OP_CHECK_IF(ValidateInferShapeEmAndAttrs(context, nloc, nnei, lastLayerSize) != GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateInferShapeEmAndAttrs error"), return GRAPH_FAILED);
    OP_CHECK_IF(ValidateInferShapeTables(context, lastLayerSize) != GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateInferShapeTables error"), return GRAPH_FAILED);

    // output shape: [nloc, 4, last_layer_size]
    gert::Shape* descShape = context->GetOutputShape(INDEX_DESCRIPTOR);
    OP_CHECK_NULL_WITH_CONTEXT(context, descShape);
    descShape->SetDimNum(3);
    descShape->SetDim(IDX_0, nloc);
    descShape->SetDim(IDX_1, DESC_DIM1);
    descShape->SetDim(IDX_2, lastLayerSize);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeTabulateFusion");
    return GRAPH_SUCCESS;
}

// InferDataType callback: dtype consistency + support check + explicit output dtype set
// (per MDE 3.3 dtype rules and MDE 7.2 "explicitly set output dtype" requirement;
//  MDE 7.2 example uses non-existent InferShapeContext::GetInputDesc/SetDataType,
//  corrected here to use InferDataTypeContext::GetInputDataType/SetOutputDataType)
static ge::graphStatus InferDataTypeTabulateFusion(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeTabulateFusion");

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};
    ge::DataType tableDtype = context->GetInputDataType(INDEX_TABLE);
    OP_CHECK_IF(supportedDtype.count(tableDtype) == 0,
                OP_LOGE(context, "TabulateFusion: unsupported dtype at input %ld", INDEX_TABLE), return GRAPH_FAILED);

    // dtype consistency: inputs 1..3 must match input 0
    for (int64_t i = INDEX_TABLE_INFO; i <= INDEX_EM; i++) {
        ge::DataType dt = context->GetInputDataType(static_cast<size_t>(i));
        OP_CHECK_IF(supportedDtype.count(dt) == 0,
                    OP_LOGE(context, "TabulateFusion: unsupported dtype at input %ld", i), return GRAPH_FAILED);
        OP_CHECK_IF(dt != tableDtype, OP_LOGE(context, "TabulateFusion: dtype of inputs should be the same"),
                    return GRAPH_FAILED);
    }

    // explicitly set output dtype = em.dtype
    ge::DataType emDtype = context->GetInputDataType(INDEX_EM);
    OP_CHECK_IF(context->SetOutputDataType(static_cast<size_t>(INDEX_DESCRIPTOR), emDtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "TabulateFusion: SetOutputDataType failed"), return GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeTabulateFusion");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TabulateFusion).InferShape(InferShapeTabulateFusion).InferDataType(InferDataTypeTabulateFusion);
} // namespace ops
