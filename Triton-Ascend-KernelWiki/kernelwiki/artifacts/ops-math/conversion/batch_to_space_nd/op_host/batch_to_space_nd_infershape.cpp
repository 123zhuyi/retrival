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
#include "op_api/op_util.h"
#include "op_host/util/const_util.h"
#include "op_host/util/shape_util.h"

using namespace ge;
namespace ops {
// 输入索引
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_BLOCK_SHAPE = 1;
static constexpr size_t INPUT_IDX_CROPS = 2;
// 输出索引
static constexpr size_t OUTPUT_IDX_Y = 0;
// 未知维度值
static constexpr int64_t UNKNOWN_DIM = -1;
// crops 第二维的长度
static constexpr size_t CROPS_LENGTH = 2;

class BatchToSpaceNDInferShapeHelper {
public:
    explicit BatchToSpaceNDInferShapeHelper(gert::InferShapeContext* context) : context_(context) {}

    ge::graphStatus Inference();

private:
    ge::graphStatus Init();

private:
    gert::InferShapeContext* context_;
    const gert::Shape* xShape_{nullptr};
    gert::Shape* yShape_{nullptr};
    size_t blockNum_{0};
    gert::Shape blockVec_;
    gert::Shape cropsVec_;
    bool isConstBlock_{false};
    bool isConstCrops_{false};
};

ge::graphStatus BatchToSpaceNDInferShapeHelper::Init()
{
    xShape_ = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape_);

    const gert::Tensor* blockTensor = context_->GetInputTensor(INPUT_IDX_BLOCK_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, blockTensor);
    blockNum_ = static_cast<size_t>(blockTensor->GetShapeSize());

    const gert::Tensor* cropsTensor = context_->GetInputTensor(INPUT_IDX_CROPS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cropsTensor);

    yShape_ = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape_);

    if (IsConstTensor(blockTensor)) {
        isConstBlock_ = true;
        OP_CHECK_IF(!Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context_, INPUT_IDX_BLOCK_SHAPE, blockVec_),
                    OP_LOGE(context_, "Get block_shape tensor data failed!"), return ge::GRAPH_FAILED);
    }

    if (IsConstTensor(cropsTensor)) {
        isConstCrops_ = true;
        OP_CHECK_IF(!Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context_, INPUT_IDX_CROPS, cropsVec_),
                    OP_LOGE(context_, "Get crops tensor data failed!"), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchToSpaceNDInferShapeHelper::Inference()
{
    auto ret = Init();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    if (Ops::Base::IsUnknownRank(*xShape_) || !isConstBlock_) {
        Ops::Base::SetUnknownRank(*yShape_);
        return ge::GRAPH_SUCCESS;
    }

    yShape_->SetDimNum(0);

    // batch
    int64_t batch = xShape_->GetDim(0);
    if (batch != UNKNOWN_DIM) {
        for (size_t i = 0; i < blockNum_; ++i) {
            OP_CHECK_IF(blockVec_.GetDim(i) == 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_shape", "0",
                                                              "The value of block_shape cannot be zero"),
                        return ge::GRAPH_FAILED);
            batch = batch / blockVec_.GetDim(i);
        }
    }
    yShape_->AppendDim(batch);

    // spatial shape
    for (size_t i = 1; i <= blockNum_; ++i) {
        size_t j = i - 1;
        if (xShape_->GetDim(i) != UNKNOWN_DIM && isConstCrops_) {
            int64_t totalCrop = cropsVec_.GetDim(CROPS_LENGTH * j) + cropsVec_.GetDim(CROPS_LENGTH * j + 1);
            yShape_->AppendDim(xShape_->GetDim(i) * blockVec_.GetDim(j) - totalCrop);
        } else {
            yShape_->AppendDim(UNKNOWN_DIM);
        }
    }

    // remain shape
    for (size_t i = blockNum_ + 1; i < xShape_->GetDimNum(); ++i) {
        yShape_->AppendDim(xShape_->GetDim(i));
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Infershape4BatchToSpaceND(gert::InferShapeContext* context)
{
    BatchToSpaceNDInferShapeHelper helper(context);
    return helper.Inference();
}

IMPL_OP_INFERSHAPE(BatchToSpaceND)
    .InferShape(Infershape4BatchToSpaceND)
    .InputsDataDependency({INPUT_IDX_BLOCK_SHAPE, INPUT_IDX_CROPS});
} // namespace ops
