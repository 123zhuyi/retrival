/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_api/op_util.h"
#include "op_host/util/const_util.h"
#include "op_host/util/shape_util.h"
#include "common/inc/op_host/math_log.h"

namespace ops {
// 输入索引
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_DIAG = 1;
static constexpr size_t INPUT_IDX_K = 2;
// 输出索引
static constexpr size_t OUTPUT_IDX_Y = 0;

static constexpr uint8_t MIN_INPUT_DIMNUM = 2;
static constexpr uint8_t MAX_K_DIMNUM = 2;
// 尾轴(行、列)维度数
static constexpr size_t TAIL_AXIS_DIM_NUM = 2;

class MatrixSetDiagV2InferShapeHelper {
public:
    explicit MatrixSetDiagV2InferShapeHelper(gert::InferShapeContext* context) : context_(context) {}

    ge::graphStatus Inference();

private:
    ge::graphStatus Init();
    ge::graphStatus CheckShape();
    ge::graphStatus CheckK();
    ge::graphStatus SetOutputShape();

private:
    gert::InferShapeContext* context_;
    const gert::Shape* xShape_{nullptr};
    const gert::Shape* diagShape_{nullptr};
    const gert::Shape* kShape_{nullptr};
    gert::Shape* yShape_{nullptr};
    const gert::Tensor* kTensor_{nullptr};
    gert::Shape kVec_;
    int64_t upper_{0};
    int64_t lower_{0};
    int64_t row_{0};
    int64_t col_{0};
    size_t xDimNum_{0};
    size_t diagDimNum_{0};
    bool isConstK_{false};
};

ge::graphStatus MatrixSetDiagV2InferShapeHelper::Init()
{
    xShape_ = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape_);

    diagShape_ = context_->GetInputShape(INPUT_IDX_DIAG);
    OP_CHECK_NULL_WITH_CONTEXT(context_, diagShape_);

    kShape_ = context_->GetInputShape(INPUT_IDX_K);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kShape_);

    yShape_ = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape_);

    kTensor_ = context_->GetInputTensor(INPUT_IDX_K);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kTensor_);

    isConstK_ = false;
    if (IsConstTensor(kTensor_)) {
        isConstK_ = true;
        OP_CHECK_IF(!Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context_, INPUT_IDX_K, kVec_),
                    OP_LOGE(context_, "get const k data failed!"), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatrixSetDiagV2InferShapeHelper::CheckK()
{
    if (kShape_->GetDim(0) == MAX_K_DIMNUM) {
        upper_ = kVec_.GetDim(1);
        lower_ = kVec_.GetDim(0);
        OP_CHECK_IF(upper_ < lower_,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "k", Ops::Base::ToString(kVec_),
                                                          "The value of k[1] must be greater than k[0]"),
                    return ge::GRAPH_FAILED);
    } else {
        lower_ = kVec_.GetDim(0);
        upper_ = lower_;
    }

    row_ = xShape_->GetDim(xDimNum_ - TAIL_AXIS_DIM_NUM);
    col_ = xShape_->GetDim(xDimNum_ - 1);
    OP_CHECK_IF(row_ != ge::UNKNOWN_DIM && lower_ <= -row_,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "k", Ops::Base::ToString(kVec_),
                                                      "The value of k[1] must be less than last axis of input"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(col_ != ge::UNKNOWN_DIM && upper_ >= col_,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "k", Ops::Base::ToString(kVec_),
                                                      "The value of -k[0] must be less than -2nd axis of input"),
                return ge::GRAPH_FAILED);

    if (lower_ == upper_) {
        OP_CHECK_IF(diagDimNum_ != xDimNum_ - 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        context_->GetNodeName(), "input and diagonal", Ops::Math::Join(xDimNum_, diagDimNum_),
                        "The StorageShape dim of diagonal must be equal to the StorageShape dim of input plus -1"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(diagDimNum_ != xDimNum_,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "input and diagonal",
                                                              Ops::Math::Join(xDimNum_, diagDimNum_),
                                                              "The shape dims of input and diagonal must be the same"),
                    return ge::GRAPH_FAILED);
        int64_t numDiags = diagShape_->GetDim(diagDimNum_ - 2);
        int64_t kNum = upper_ - lower_ + 1;
        OP_CHECK_IF(numDiags != kNum,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "diagonal", std::to_string(numDiags),
                                                          "-2nd axis of diagonal must be " + std::to_string(kNum) +
                                                              ", when the value of k is " + Ops::Base::ToString(kVec_)),
                    return ge::GRAPH_FAILED);
    }

    int64_t rowLen = xShape_->GetDim(xDimNum_ - 2) + std::min(upper_, static_cast<int64_t>(0));
    int64_t colLen = xShape_->GetDim(xDimNum_ - 1) - std::max(lower_, static_cast<int64_t>(0));
    int64_t diagLenComputed = std::min(rowLen, colLen);
    int64_t maxDiagLen = diagShape_->GetDim(diagDimNum_ - 1);
    OP_CHECK_IF(
        maxDiagLen != diagLenComputed,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "diagonal", std::to_string(maxDiagLen),
                                              "-1st axis of diagonal must be " + std::to_string(diagLenComputed) +
                                                  ", when the value of k is " + Ops::Base::ToString(kVec_)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatrixSetDiagV2InferShapeHelper::CheckShape()
{
    xDimNum_ = xShape_->GetDimNum();
    OP_CHECK_IF(xDimNum_ < MIN_INPUT_DIMNUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "input", std::to_string(xDimNum_),
                                                         "The shape dim of input must be >= 2"),
                return ge::GRAPH_FAILED);

    diagDimNum_ = diagShape_->GetDimNum();
    OP_CHECK_IF(
        diagDimNum_ < 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "diagonal", std::to_string(diagDimNum_),
                                                 "The shape dim of diagonal must be >= 1"),
        return ge::GRAPH_FAILED);

    size_t kDimNum = kShape_->GetDimNum();
    OP_CHECK_IF(kDimNum > 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "k", std::to_string(kDimNum),
                                                         "The shape dim of k must be <= 1"),
                return ge::GRAPH_FAILED);

    if (Ops::Base::IsUnknownShape(*kShape_) || !isConstK_) {
        OP_CHECK_IF(xDimNum_ - 1 > diagDimNum_,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "input and diagonal",
                                                              Ops::Math::Join(xDimNum_, diagDimNum_),
                                                              "The shape dim of diagonal must be >= that of input - 1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(diagDimNum_ > xDimNum_,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "input and diagonal",
                                                              Ops::Math::Join(xDimNum_, diagDimNum_),
                                                              "The shape dim of diagonal must be <= that of input"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    return CheckK();
}

ge::graphStatus MatrixSetDiagV2InferShapeHelper::SetOutputShape()
{
    *yShape_ = *xShape_;

    // 根据 diag shape 推导
    for (size_t i = 0; i < xDimNum_ - TAIL_AXIS_DIM_NUM; i++) {
        if (diagShape_->GetDim(i) == ge::UNKNOWN_DIM) {
            continue;
        }
        if (xShape_->GetDim(i) == ge::UNKNOWN_DIM) {
            yShape_->SetDim(i, diagShape_->GetDim(i));
        } else {
            OP_CHECK_IF(xShape_->GetDim(i) != diagShape_->GetDim(i),
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            context_->GetNodeName(), "input and diagonal",
                            Ops::Math::Join(xShape_->GetDim(i), diagShape_->GetDim(i)),
                            std::to_string(i) + "th axis of diagonal must be equal to same axis of input"),
                        return ge::GRAPH_FAILED);
        }
    }

    OP_LOGD(context_, "out shape: %s", Ops::Base::ToString(*yShape_).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatrixSetDiagV2InferShapeHelper::Inference()
{
    CHECK_RET_SUCC(Init());

    if (Ops::Base::IsUnknownRank(*xShape_) || Ops::Base::IsUnknownRank(*diagShape_)) {
        Ops::Base::SetUnknownRank(*yShape_);
        return ge::GRAPH_SUCCESS;
    }

    CHECK_RET_SUCC(CheckShape());
    CHECK_RET_SUCC(SetOutputShape());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Infershape4MatrixSetDiagV2(gert::InferShapeContext* context)
{
    MatrixSetDiagV2InferShapeHelper helper(context);
    return helper.Inference();
}

IMPL_OP_INFERSHAPE(MatrixSetDiagV2).InferShape(Infershape4MatrixSetDiagV2).TilingInputsDataDependency({INPUT_IDX_K});
} // namespace ops
