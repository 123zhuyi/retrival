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
static constexpr size_t INPUT_IDX_BS = 1;
static constexpr size_t INPUT_IDX_PADS = 2;
static constexpr size_t OUTPUT_IDX_Y = 0;
static constexpr size_t PADS_DIM2 = 2;
static constexpr int64_t MAX_RANK = 8;

class SpaceToBatchNDInferShapeHelper {
public:
    explicit SpaceToBatchNDInferShapeHelper(gert::InferShapeContext* context) : context_(context) {}

    ge::graphStatus Inference();

private:
    ge::graphStatus Init();
    ge::graphStatus CheckAndInfer();

    gert::InferShapeContext* context_;
    const gert::Shape* xShape_{nullptr};
    gert::Shape* yShape_{nullptr};
    size_t N_{0};
    gert::Shape blockVec_;
    gert::Shape padsVec_;
};

ge::graphStatus SpaceToBatchNDInferShapeHelper::Init()
{
    xShape_ = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape_);

    yShape_ = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape_);

    const gert::Tensor* bsTensor = context_->GetInputTensor(INPUT_IDX_BS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, bsTensor);
    N_ = bsTensor->GetShapeSize();
    OP_CHECK_IF(static_cast<int64_t>(N_) < 0, OP_LOGE(context_, "block_shape element count is negative"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context_, INPUT_IDX_BS, blockVec_),
                OP_LOGE(context_, "get block_shape const data failed"), return ge::GRAPH_FAILED);

    const gert::Tensor* padsTensor = context_->GetInputTensor(INPUT_IDX_PADS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, padsTensor);
    auto padsShape = context_->GetInputShape(INPUT_IDX_PADS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, padsShape);
    OP_CHECK_IF(padsShape->GetDimNum() != 2 || padsShape->GetDim(0) != static_cast<int64_t>(N_) ||
                    padsShape->GetDim(1) != static_cast<int64_t>(PADS_DIM2),
                OP_LOGE(context_, "paddings shape must be [%zu, 2]", N_), return ge::GRAPH_FAILED);

    if (!Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context_, INPUT_IDX_PADS, padsVec_)) {
        OP_LOGE(context_, "get paddings const data failed");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToBatchNDInferShapeHelper::CheckAndInfer()
{
    auto rank = xShape_->GetDimNum();
    OP_CHECK_IF(static_cast<size_t>(rank) < N_ + 1 || rank > MAX_RANK,
                OP_LOGE(context_, "invalid rank %zu, N=%zu", rank, N_), return ge::GRAPH_FAILED);

    int64_t batchMul = 1;
    for (size_t i = 0; i < N_; i++) {
        int64_t bs = blockVec_.GetDim(i);
        OP_CHECK_IF(bs <= 0, OP_LOGE(context_, "block_shape[%zu]=%ld must be > 0", i, bs), return ge::GRAPH_FAILED);
        batchMul *= bs;
    }

    yShape_->SetDimNum(0);

    int64_t firstDim = xShape_->GetDim(0);
    firstDim = firstDim == -1 ? -1 : firstDim * batchMul;
    yShape_->AppendDim(firstDim);

    for (size_t i = 0; i < N_; i++) {
        int64_t dim = xShape_->GetDim(i + 1) == -1 ?
                          -1 :
                          ((xShape_->GetDim(i + 1) + padsVec_.GetDim(i * 2) + padsVec_.GetDim(i * 2 + 1)) /
                           blockVec_.GetDim(i));
        yShape_->AppendDim(dim);
    }
    for (size_t i = N_ + 1; i < static_cast<size_t>(rank); i++) {
        yShape_->AppendDim(xShape_->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SpaceToBatchNDInferShapeHelper::Inference()
{
    OP_CHECK_IF(Init() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "init failed"), return ge::GRAPH_FAILED);

    if (Ops::Base::IsUnknownRank(*xShape_)) {
        Ops::Base::SetUnknownRank(*yShape_);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(CheckAndInfer() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "infer failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Infershape4SpaceToBatchND(gert::InferShapeContext* context)
{
    SpaceToBatchNDInferShapeHelper helper(context);
    return helper.Inference();
}

IMPL_OP_INFERSHAPE(SpaceToBatchND)
    .InferShape(Infershape4SpaceToBatchND)
    .InputsDataDependency({INPUT_IDX_BS, INPUT_IDX_PADS});
} // namespace ops
