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
#include "conversion/matrix_set_diag_v2/op_kernel/arch35/matrix_set_diag_v2_tilingdata.h"
#include "conversion/matrix_set_diag_v2/op_host/arch35/matrix_set_diag_v2_tiling_arch35_base.h"
#include "log/log.h"

namespace optiling {
// 输入索引
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_DIAG = 1;
static constexpr uint32_t COL_DIM_OFFSET = 1; // 列维度（-1轴）距shape末尾的偏移量
static constexpr uint32_t ROW_DIM_OFFSET = 2; // 行维度（-2轴）距shape末尾的偏移量

static constexpr uint8_t MIN_INPUT_DIMNUM = 2;
static constexpr uint8_t MAX_INPUT_DIMNUM = 8;

class MatrixSetDiagTiling {
private:
    // tiling context
    gert::TilingContext* context_;

    // 输入参数
    uint32_t dimNum_{1};
    uint32_t diagDimNum_{1};
    MatrixSetDiagInputInfo inputInfo_;

public:
    explicit MatrixSetDiagTiling(gert::TilingContext* context) : context_(context) {};
    ~MatrixSetDiagTiling();

    ge::graphStatus DoTiling();

private:
    // 参数检查、获取
    ge::graphStatus ParamCheck();
};

MatrixSetDiagTiling::~MatrixSetDiagTiling() {}

ge::graphStatus MatrixSetDiagTiling::DoTiling()
{
    // 校验属性
    auto ret = ParamCheck();
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTiling ParamCheck failed"), return ge::GRAPH_FAILED);

    inputInfo_.k0 = 0;
    inputInfo_.k1 = 0;
    inputInfo_.diagNum = 1;
    MatrixSetDiagTilingBase tiling{context_, inputInfo_};
    return tiling.DoTiling();
}

ge::graphStatus MatrixSetDiagTiling::ParamCheck()
{
    auto inputValueDesc = context_->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueDesc);

    auto inputDataType = inputValueDesc->GetDataType();
    inputInfo_.dSize = ge::GetSizeByDataType(inputDataType);
    OP_CHECK_IF(inputInfo_.dSize <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x dtype size",
                                                      std::to_string(inputInfo_.dSize).c_str(), "must be positive"),
                return ge::GRAPH_FAILED);

    // 校验输入shape
    auto inputShape = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);

    auto inputShapeVal = inputShape->GetStorageShape();
    dimNum_ = inputShapeVal.GetDimNum();
    OP_CHECK_IF(dimNum_ < MIN_INPUT_DIMNUM || dimNum_ > MAX_INPUT_DIMNUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "input", std::to_string(dimNum_).c_str(),
                                             "between [2, 8]"),
                return ge::GRAPH_FAILED);

    auto diagValueDesc = context_->GetInputDesc(INPUT_IDX_DIAG);
    OP_CHECK_NULL_WITH_CONTEXT(context_, diagValueDesc);

    auto diagDataType = diagValueDesc->GetDataType();
    OP_CHECK_IF(inputDataType != diagDataType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "input and diagonal",
                    (Ops::Base::ToString(inputDataType) + " and " + Ops::Base::ToString(diagDataType)).c_str(),
                    "dtypes of input and diagonal must be the same"),
                return ge::GRAPH_FAILED);

    // 校验输入shape
    auto diagShape = context_->GetInputShape(INPUT_IDX_DIAG);
    OP_CHECK_NULL_WITH_CONTEXT(context_, diagShape);

    auto diagShapeVal = diagShape->GetStorageShape();
    diagDimNum_ = diagShapeVal.GetDimNum();
    OP_CHECK_IF(diagDimNum_ < 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "diagonal", std::to_string(diagDimNum_).c_str(),
                                             "greater than or equal to 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dimNum_ != diagDimNum_ + 1,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context_->GetNodeName(), "input and diagonal",
                    (std::to_string(dimNum_) + " and " + std::to_string(diagDimNum_)).c_str(),
                    "diagonal dim num must equal input dim num minus 1"),
                return ge::GRAPH_FAILED);

    inputInfo_.xColNum = inputShapeVal.GetDim(dimNum_ - COL_DIM_OFFSET);
    inputInfo_.xRowNum = inputShapeVal.GetDim(dimNum_ - ROW_DIM_OFFSET);
    uint64_t maxDiagLen = static_cast<uint64_t>(diagShapeVal.GetDim(diagDimNum_ - 1));
    inputInfo_.maxDiagLen = static_cast<uint32_t>(maxDiagLen);
    OP_CHECK_IF((inputInfo_.maxDiagLen != std::min(inputInfo_.xColNum, inputInfo_.xRowNum)) ||
                    (inputInfo_.maxDiagLen != maxDiagLen),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "diagonal",
                                                      Ops::Base::ToString(diagShapeVal).c_str(),
                                                      "diagonal length must equal min(row, col) of input"),
                return ge::GRAPH_FAILED);
    if (diagDimNum_ > 1) {
        for (int32_t i = diagDimNum_ - 2; i >= 0; i--) {
            OP_CHECK_IF(diagShapeVal.GetDim(i) != inputShapeVal.GetDim(i),
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            context_->GetNodeName(), "input and diagonal",
                            (Ops::Base::ToString(inputShapeVal) + " and " + Ops::Base::ToString(diagShapeVal)).c_str(),
                            ("dim " + std::to_string(i) + " of diagonal must match input").c_str()),
                        return ge::GRAPH_FAILED);
            inputInfo_.mergeDimSize *= static_cast<uint64_t>(diagShapeVal.GetDim(i));
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4MatrixSetDiag(gert::TilingContext* context)
{
    // DoTiling
    MatrixSetDiagTiling tiling{context};
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForMatrixSetDiag([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatrixSetDiag)
    .Tiling(Tiling4MatrixSetDiag)
    .TilingParse<MatrixSetDiagCompileInfo>(TilingPrepareForMatrixSetDiag);
} // namespace optiling
