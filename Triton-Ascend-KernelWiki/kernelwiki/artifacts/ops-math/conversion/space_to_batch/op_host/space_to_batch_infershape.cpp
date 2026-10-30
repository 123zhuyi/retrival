/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/util/const_util.h"
#include "op_host/util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_PADDINGS = 1;
static constexpr size_t OUTPUT_IDX_Y = 0;
static constexpr int64_t UNKNOWN_DIM = -1;
static constexpr size_t PADDINGS_ROWS = 2;
static constexpr size_t PADDINGS_COLS = 2;

class SpaceToBatchInferShapeHelper {
public:
    explicit SpaceToBatchInferShapeHelper(gert::InferShapeContext* context) : context_(context) {}

    ge::graphStatus Inference();

private:
    ge::graphStatus Init();
    int64_t GetBlockSize();

private:
    gert::InferShapeContext* context_;
    const gert::Shape* xShape_{nullptr};
    gert::Shape* yShape_{nullptr};
    int64_t blockSize_{0};
    bool isConstPaddings_{false};
    gert::Shape paddingsVec_;
};

ge::graphStatus SpaceToBatchInferShapeHelper::Init()
{
    xShape_ = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape_);

    yShape_ = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape_);

    blockSize_ = GetBlockSize();
    OP_CHECK_IF(blockSize_ <= 0, OP_LOGE(context_, "block_size must be positive, but got %ld", blockSize_),
                return ge::GRAPH_FAILED);

    const gert::Tensor* paddingsTensor = context_->GetInputTensor(INPUT_IDX_PADDINGS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, paddingsTensor);

    isConstPaddings_ = Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context_, INPUT_IDX_PADDINGS,
                                                                              paddingsVec_);

    return ge::GRAPH_SUCCESS;
}

int64_t SpaceToBatchInferShapeHelper::GetBlockSize()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* blockSizePtr = attrs->GetInt(0);
    OP_CHECK_IF(blockSizePtr == nullptr, OP_LOGE(context_, "get block_size attr failed"), return -1);
    return *blockSizePtr;
}

ge::graphStatus SpaceToBatchInferShapeHelper::Inference()
{
    auto ret = Init();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    if (Ops::Base::IsUnknownRank(*xShape_)) {
        Ops::Base::SetUnknownRank(*yShape_);
        return ge::GRAPH_SUCCESS;
    }

    yShape_->SetDimNum(0);

    int64_t batch = xShape_->GetDim(0);
    if (batch != UNKNOWN_DIM) {
        batch = batch * blockSize_ * blockSize_;
    }
    yShape_->AppendDim(batch);

    if (isConstPaddings_) {
        int64_t padTop = paddingsVec_.GetDim(0);
        int64_t padBottom = paddingsVec_.GetDim(1);
        int64_t hIn = xShape_->GetDim(1);
        if (hIn != UNKNOWN_DIM) {
            yShape_->AppendDim((hIn + padTop + padBottom) / blockSize_);
        } else {
            yShape_->AppendDim(UNKNOWN_DIM);
        }
    } else {
        yShape_->AppendDim(UNKNOWN_DIM);
    }

    if (isConstPaddings_) {
        int64_t padLeft = paddingsVec_.GetDim(2);
        int64_t padRight = paddingsVec_.GetDim(3);
        int64_t wIn = xShape_->GetDim(2);
        if (wIn != UNKNOWN_DIM) {
            yShape_->AppendDim((wIn + padLeft + padRight) / blockSize_);
        } else {
            yShape_->AppendDim(UNKNOWN_DIM);
        }
    } else {
        yShape_->AppendDim(UNKNOWN_DIM);
    }

    yShape_->AppendDim(xShape_->GetDim(3));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Infershape4SpaceToBatch(gert::InferShapeContext* context)
{
    SpaceToBatchInferShapeHelper helper(context);
    return helper.Inference();
}

IMPL_OP_INFERSHAPE(SpaceToBatch).InferShape(Infershape4SpaceToBatch).InputsDataDependency({INPUT_IDX_PADDINGS});
} // namespace ops
