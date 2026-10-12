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
 * \file tabulate_fusion_grad_infershape.cpp
 * \brief InferShape / InferDataType implementation for tabulate_fusion_grad operator
 *
 * Output shape:
 *   dy_dem_x.shape = em_x.shape   (对 em_x 的梯度, 逐元素对应)
 *   dy_dem.shape   = em.shape     (对 em 的梯度, 逐元素对应)
 * Output dtype:
 *   dy_dem_x.dtype = em_x.dtype
 *   dy_dem.dtype   = em.dtype
 *
 * Validation rules (mirror tiling side, per MDE 3.3):
 *   - em rank=3, shape[2]=4
 *   - dy/descriptor rank=3, shape[1]=4, shape[2] 相等
 *   - table shape[1] == 6*sizeAlign64
 *   - table_info size >= 5
 *   - em_x total == nloc * nnei
 *   - dtype == float32
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
static constexpr int64_t INDEX_DY = 4;
static constexpr int64_t INDEX_DESCRIPTOR = 5;
static constexpr int64_t INDEX_OUT_DY_DEM_X = 0;
static constexpr int64_t INDEX_OUT_DY_DEM = 1;
static constexpr int64_t DESC_DIM1 = 4;
static constexpr int64_t TABLE_INFO_MIN_SIZE = 5;
static constexpr int64_t ALIGN_64 = 64;

// ============================================================================
// InferShape: output shape = input shape (em_x -> dy_dem_x, em -> dy_dem)
// ============================================================================

static ge::graphStatus InferShapeTabulateFusionGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeTabulateFusionGrad");

    // em: [nloc, nnei, 4]
    const gert::Shape* emShape = context->GetInputShape(INDEX_EM);
    OP_CHECK_NULL_WITH_CONTEXT(context, emShape);
    OP_CHECK_IF(emShape->GetDimNum() != 3,
                OP_LOGE(context, "TabulateFusionGrad: em should be 3D, got %zu", emShape->GetDimNum()),
                return GRAPH_FAILED);
    OP_CHECK_IF(emShape->GetDim(IDX_2) != 4,
                OP_LOGE(context, "TabulateFusionGrad: em.shape[2] should be 4, got %ld", emShape->GetDim(IDX_2)),
                return GRAPH_FAILED);
    int64_t nloc = emShape->GetDim(IDX_0);
    int64_t nnei = emShape->GetDim(IDX_1);

    // dy: [nloc, 4, L]
    const gert::Shape* dyShape = context->GetInputShape(INDEX_DY);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyShape);
    OP_CHECK_IF(dyShape->GetDimNum() != 3,
                OP_LOGE(context, "TabulateFusionGrad: dy should be 3D, got %zu", dyShape->GetDimNum()),
                return GRAPH_FAILED);
    OP_CHECK_IF(dyShape->GetDim(IDX_1) != 4,
                OP_LOGE(context, "TabulateFusionGrad: dy.shape[1] should be 4, got %ld", dyShape->GetDim(IDX_1)),
                return GRAPH_FAILED);
    int64_t lastLayerSize = dyShape->GetDim(IDX_2);
    OP_CHECK_IF(lastLayerSize <= 0,
                OP_LOGE(context, "TabulateFusionGrad: last_layer_size should be > 0, got %ld", lastLayerSize),
                return GRAPH_FAILED);

    // descriptor: [nloc, 4, L] -> shape[2] should == dy.shape[2]
    const gert::Shape* descShape = context->GetInputShape(INDEX_DESCRIPTOR);
    OP_CHECK_NULL_WITH_CONTEXT(context, descShape);
    OP_CHECK_IF(descShape->GetDimNum() != 3,
                OP_LOGE(context, "TabulateFusionGrad: descriptor should be 3D, got %zu", descShape->GetDimNum()),
                return GRAPH_FAILED);
    OP_CHECK_IF(
        descShape->GetDim(IDX_1) != 4,
        OP_LOGE(context, "TabulateFusionGrad: descriptor.shape[1] should be 4, got %ld", descShape->GetDim(IDX_1)),
        return GRAPH_FAILED);
    OP_CHECK_IF(descShape->GetDim(IDX_2) != lastLayerSize,
                OP_LOGE(context, "TabulateFusionGrad: descriptor.shape[2] %ld != dy.shape[2] %ld",
                        descShape->GetDim(IDX_2), lastLayerSize),
                return GRAPH_FAILED);

    // table: [N_table, 6*sizeAlign64]
    const gert::Shape* tableShape = context->GetInputShape(INDEX_TABLE);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableShape);
    OP_CHECK_IF(tableShape->GetDimNum() != 2,
                OP_LOGE(context, "TabulateFusionGrad: table should be 2D, got %zu", tableShape->GetDimNum()),
                return GRAPH_FAILED);
    int64_t sizeAlign64 = ((lastLayerSize + ALIGN_64 - 1) / ALIGN_64) * ALIGN_64;
    OP_CHECK_IF(tableShape->GetDim(IDX_1) != sizeAlign64 * 6,
                OP_LOGE(context, "TabulateFusionGrad: table.shape[1] %ld != sizeAlign64*6 %ld",
                        tableShape->GetDim(IDX_1), sizeAlign64 * 6),
                return GRAPH_FAILED);

    // table_info: size >= 5
    const gert::Shape* tableInfoShape = context->GetInputShape(INDEX_TABLE_INFO);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableInfoShape);
    OP_CHECK_IF(tableInfoShape->GetShapeSize() < TABLE_INFO_MIN_SIZE,
                OP_LOGE(context, "TabulateFusionGrad: size of table_info should be >= 5, got %ld",
                        tableInfoShape->GetShapeSize()),
                return GRAPH_FAILED);

    // em_x: total == nloc * nnei
    const gert::Shape* emXShape = context->GetInputShape(INDEX_EM_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, emXShape);
    OP_CHECK_IF(emXShape->GetShapeSize() != nloc * nnei,
                OP_LOGE(context, "TabulateFusionGrad: em_x total %ld != nloc*nnei %ld*%ld", emXShape->GetShapeSize(),
                        nloc, nnei),
                return GRAPH_FAILED);

    // 输出 shape: dy_dem_x = em_x.shape, dy_dem = em.shape
    gert::Shape* outDyDemX = context->GetOutputShape(INDEX_OUT_DY_DEM_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDyDemX);
    outDyDemX->SetDimNum(emXShape->GetDimNum());
    for (size_t i = 0; i < emXShape->GetDimNum(); i++) {
        outDyDemX->SetDim(i, emXShape->GetDim(i));
    }

    gert::Shape* outDyDem = context->GetOutputShape(INDEX_OUT_DY_DEM);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDyDem);
    outDyDem->SetDimNum(emShape->GetDimNum());
    for (size_t i = 0; i < emShape->GetDimNum(); i++) {
        outDyDem->SetDim(i, emShape->GetDim(i));
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeTabulateFusionGrad");
    return GRAPH_SUCCESS;
}

// ============================================================================
// InferDataType: dtype consistency + support check + explicit output dtype set
// ============================================================================

static ge::graphStatus InferDataTypeTabulateFusionGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeTabulateFusionGrad");

    // dtype check: only float32 supported
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT};
    for (int64_t i = INDEX_TABLE; i <= INDEX_DESCRIPTOR; i++) {
        ge::DataType dt = context->GetInputDataType(static_cast<size_t>(i));
        OP_CHECK_IF(supportedDtype.count(dt) == 0,
                    OP_LOGE(context, "TabulateFusionGrad: unsupported dtype at input %ld (only float32)", i),
                    return GRAPH_FAILED);
        OP_CHECK_IF(dt != context->GetInputDataType(static_cast<size_t>(INDEX_TABLE)),
                    OP_LOGE(context, "TabulateFusionGrad: dtype of inputs should be the same"), return GRAPH_FAILED);
    }

    // dy_dem_x.dtype = em_x.dtype
    ge::DataType emXDtype = context->GetInputDataType(static_cast<size_t>(INDEX_EM_X));
    OP_CHECK_IF(context->SetOutputDataType(static_cast<size_t>(INDEX_OUT_DY_DEM_X), emXDtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "TabulateFusionGrad: SetOutputDataType for dy_dem_x failed"), return GRAPH_FAILED);

    // dy_dem.dtype = em.dtype
    ge::DataType emDtype = context->GetInputDataType(static_cast<size_t>(INDEX_EM));
    OP_CHECK_IF(context->SetOutputDataType(static_cast<size_t>(INDEX_OUT_DY_DEM), emDtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "TabulateFusionGrad: SetOutputDataType for dy_dem failed"), return GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeTabulateFusionGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TabulateFusionGrad)
    .InferShape(InferShapeTabulateFusionGrad)
    .InferDataType(InferDataTypeTabulateFusionGrad);
} // namespace ops
