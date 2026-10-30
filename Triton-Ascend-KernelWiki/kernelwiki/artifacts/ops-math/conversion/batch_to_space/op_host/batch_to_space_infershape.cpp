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
static constexpr size_t INPUT_IDX_CROPS = 1;
static constexpr size_t OUTPUT_IDX_Y = 0;
static constexpr size_t DIM_N = 0;
static constexpr size_t DIM_H = 1;
static constexpr size_t DIM_W = 2;
static constexpr size_t DIM_C = 3;
static constexpr int64_t UNKNOWN_DIM = -1;
static constexpr size_t CROPS_SIZE = 4;

class BatchToSpaceInferShapeHelper {
public:
    explicit BatchToSpaceInferShapeHelper(gert::InferShapeContext* context) : context_(context) {}

    ge::graphStatus Inference();

private:
    ge::graphStatus Init();

    gert::InferShapeContext* context_;
    const gert::Shape* xShape_{nullptr};
    gert::Shape* yShape_{nullptr};
    int64_t blockSize_{0};
    int64_t crops_[CROPS_SIZE]{0};
    bool hasCropsConst_{false};
};

ge::graphStatus BatchToSpaceInferShapeHelper::Init()
{
    xShape_ = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape_);

    yShape_ = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape_);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto blockSizePtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, blockSizePtr);
    blockSize_ = *blockSizePtr;
    OP_CHECK_IF(blockSize_ <= 0, OP_LOGE(context_, "block_size must be positive"), return ge::GRAPH_FAILED);

    gert::Shape cropsShape;
    if (Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context_, INPUT_IDX_CROPS, cropsShape)) {
        for (size_t i = 0; i < CROPS_SIZE; ++i) {
            crops_[i] = cropsShape[i];
        }
        hasCropsConst_ = true;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchToSpaceInferShapeHelper::Inference()
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

    // N = N_in / (block_size * block_size)
    int64_t bs2 = blockSize_ * blockSize_;
    if (xShape_->GetDim(DIM_N) != UNKNOWN_DIM) {
        yShape_->AppendDim(xShape_->GetDim(DIM_N) / bs2);
    } else {
        yShape_->AppendDim(UNKNOWN_DIM);
    }

    // H_out = H_in * block_size - crop_top - crop_bottom
    if (xShape_->GetDim(DIM_H) != UNKNOWN_DIM && hasCropsConst_) {
        int64_t hOut = xShape_->GetDim(DIM_H) * blockSize_ - crops_[0] - crops_[1];
        yShape_->AppendDim(hOut);
    } else {
        yShape_->AppendDim(UNKNOWN_DIM);
    }

    // W_out = W_in * block_size - crop_left - crop_right
    if (xShape_->GetDim(DIM_W) != UNKNOWN_DIM && hasCropsConst_) {
        int64_t wOut = xShape_->GetDim(DIM_W) * blockSize_ - crops_[2] - crops_[3];
        yShape_->AppendDim(wOut);
    } else {
        yShape_->AppendDim(UNKNOWN_DIM);
    }

    // C: unchanged
    if (xShape_->GetDim(DIM_C) != UNKNOWN_DIM) {
        yShape_->AppendDim(xShape_->GetDim(DIM_C));
    } else {
        yShape_->AppendDim(UNKNOWN_DIM);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Infershape4BatchToSpace(gert::InferShapeContext* context)
{
    BatchToSpaceInferShapeHelper helper(context);
    return helper.Inference();
}

IMPL_OP_INFERSHAPE(BatchToSpace).InferShape(Infershape4BatchToSpace).InputsDataDependency({INPUT_IDX_CROPS});
} // namespace ops
