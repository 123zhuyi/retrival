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
#include "op_host/util/const_util.h"
#include "matrix_set_diag_v2_tiling_arch35_base.h"
#include "common/inc/op_host/math_log.h"
#include "log/log.h"

namespace optiling {
// 输入索引
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_DIAG = 1;
static constexpr size_t INPUT_IDX_K = 2;

static constexpr uint8_t MIN_INPUT_DIMNUM = 2;
static constexpr uint8_t MAX_INPUT_DIMNUM = 8;

// 公共常量
static constexpr std::array VALUE_DATA_TYPE_ALL{
    ge::DT_BOOL,   ge::DT_INT8,   ge::DT_INT16, ge::DT_INT32,   ge::DT_INT64, ge::DT_UINT8,  ge::DT_UINT16,
    ge::DT_UINT32, ge::DT_UINT64, ge::DT_BF16,  ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_COMPLEX64,
};

class MatrixSetDiagV2Tiling {
private:
    // tiling context
    gert::TilingContext* context_ = nullptr;

    /* data */
    uint32_t dimNum_{1};
    uint32_t diagDimNum_{1};
    gert::Shape inputShapeVal_;
    gert::Shape diagShapeVal_;
    ge::DataType inputDataType_ = ge::DT_UNDEFINED;

    // 输入参数
    MatrixSetDiagInputInfo inputInfo_;

public:
    explicit MatrixSetDiagV2Tiling(gert::TilingContext* context) : context_(context) {};
    ~MatrixSetDiagV2Tiling();

    ge::graphStatus DoTiling();

private:
    // 参数检查，数据获取
    ge::graphStatus ParamCheck();
    ge::graphStatus CheckX();
    ge::graphStatus CheckDiag();
    ge::graphStatus CheckK();
};

MatrixSetDiagV2Tiling::~MatrixSetDiagV2Tiling() {}

ge::graphStatus MatrixSetDiagV2Tiling::DoTiling()
{
    // 校验属性
    CHECK_RET_SUCC(ParamCheck());

    MatrixSetDiagTilingBase tiling{context_, inputInfo_};
    return tiling.DoTiling();
}

ge::graphStatus MatrixSetDiagV2Tiling::CheckX()
{
    auto inputValueDesc = context_->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueDesc);

    inputDataType_ = inputValueDesc->GetDataType();
    inputInfo_.dSize = ge::GetSizeByDataType(inputDataType_);
    OP_CHECK_IF(inputInfo_.dSize <= 0,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "input",
                                          ge::TypeUtils::DataTypeToSerialString(inputDataType_),
                                          Ops::Math::JoinArray<Ops::Math::ItemConj::OR>(VALUE_DATA_TYPE_ALL)),
                return ge::GRAPH_FAILED);

    // 校验输入shape
    auto inputShape = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);

    inputShapeVal_ = inputShape->GetStorageShape();
    dimNum_ = inputShapeVal_.GetDimNum();
    OP_CHECK_IF(dimNum_ < MIN_INPUT_DIMNUM || dimNum_ > MAX_INPUT_DIMNUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "input", std::to_string(dimNum_),
                                                         "The shape dim of input must be within the range [2, 8]"),
                return ge::GRAPH_FAILED);

    int32_t lastIdx = static_cast<int32_t>(dimNum_) - 1;
    inputInfo_.xColNum = inputShapeVal_.GetDim(lastIdx);
    inputInfo_.xRowNum = inputShapeVal_.GetDim(lastIdx - 1);
    // 非尾轴合轴
    for (int32_t i = 0; i < lastIdx - 1; ++i) {
        inputInfo_.mergeDimSize *= static_cast<uint64_t>(inputShapeVal_.GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatrixSetDiagV2Tiling::CheckDiag()
{
    auto diagValueDesc = context_->GetInputDesc(INPUT_IDX_DIAG);
    OP_CHECK_NULL_WITH_CONTEXT(context_, diagValueDesc);

    auto diagDataType = diagValueDesc->GetDataType();
    OP_CHECK_IF(inputDataType_ != diagDataType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "input and diagonal",
                                                       Ops::Math::Join(inputDataType_, diagDataType),
                                                       "The dtypes of input and diagonal must be the same"),
                return ge::GRAPH_FAILED);

    // 校验输入shape
    auto diagShape = context_->GetInputShape(INPUT_IDX_DIAG);
    OP_CHECK_NULL_WITH_CONTEXT(context_, diagShape);

    diagShapeVal_ = diagShape->GetStorageShape();
    diagDimNum_ = diagShapeVal_.GetDimNum();

    OP_CHECK_IF(diagDimNum_ > dimNum_,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "input and diagonal",
                                                          Ops::Math::Join(dimNum_, diagDimNum_),
                                                          "The shape dim of diagonal must be <= that of input"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<int64_t>(diagDimNum_) < static_cast<int64_t>(dimNum_) - 1,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "input and diagonal",
                                                          Ops::Math::Join(dimNum_, diagDimNum_),
                                                          "The shape dim of diagonal must be >= that of input - 1"),
                return ge::GRAPH_FAILED);
    int32_t lastIdx = static_cast<int32_t>(dimNum_) - 1;
    for (int32_t i = 0; i < lastIdx - 1; ++i) {
        OP_CHECK_IF(diagShapeVal_.GetDim(i) != inputShapeVal_.GetDim(i),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context_->GetNodeName(), "input and diagonal",
                        Ops::Math::Join(inputShapeVal_.GetDim(i), diagShapeVal_.GetDim(i)),
                        std::to_string(i) + "th axis of diagonal must be equal to the same axis of input"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        diagDimNum_ < 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "diagonal", std::to_string(diagDimNum_),
                                                 "The shape dim of diagonal must be at least 1"),
        return ge::GRAPH_FAILED);
    inputInfo_.maxDiagLen = static_cast<uint32_t>(diagShapeVal_.GetDim(static_cast<size_t>(diagDimNum_) - 1));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatrixSetDiagV2Tiling::CheckK()
{
    auto kShape = context_->GetInputShape(INPUT_IDX_K);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kShape);

    auto kShapeVal = kShape->GetStorageShape();
    auto kDimNum = kShapeVal.GetDimNum();
    OP_CHECK_IF(kDimNum > 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "k", std::to_string(kDimNum),
                                                         "The shape dim of k must be <= 1"),
                return ge::GRAPH_FAILED);

    // 获取 k 值 check k
    gert::Shape k;
    OP_CHECK_IF(!Ops::Base::GetConstIntToShape(context_, INPUT_IDX_K, k), OP_LOGE(context_, "get k tensor failed"),
                return ge::GRAPH_FAILED);
    inputInfo_.k0 = k[0];
    inputInfo_.k1 = k.GetDimNum() > 1 ? k[1] : k[0];
    OP_CHECK_IF(inputInfo_.k1 < inputInfo_.k0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "k", Ops::Base::ToString(k),
                                                      "The value of k[1] must be greater than k[0]"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(static_cast<int64_t>(inputInfo_.k1) >= static_cast<int64_t>(inputInfo_.xColNum),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "k", Ops::Base::ToString(k),
                                                      "The value of k[1] must be less than last axis of input"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<int64_t>(inputInfo_.k0) <= -static_cast<int64_t>(inputInfo_.xRowNum),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "k", Ops::Base::ToString(k),
                                                      "The value of -k[0] must be less than -2nd axis of input"),
                return ge::GRAPH_FAILED);
    inputInfo_.diagNum = inputInfo_.k1 - inputInfo_.k0 + 1;

    if (inputInfo_.k0 == inputInfo_.k1) {
        OP_CHECK_IF(dimNum_ != diagDimNum_ + 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        context_->GetNodeName(), "input and diagonal", Ops::Math::Join(dimNum_, diagDimNum_),
                        "The StorageShape dim of diagonal must be equal to the StorageShape dim of input plus -1"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(dimNum_ != diagDimNum_,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "input and diagonal",
                                                              Ops::Math::Join(dimNum_, diagDimNum_),
                                                              "The shape dims of input and diagonal must be the same"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            diagDimNum_ < 2,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "diagonal", std::to_string(diagDimNum_),
                                                     "The shape dim of diagonal must be at least 2"),
            return ge::GRAPH_FAILED);

        uint64_t numDiags = static_cast<uint64_t>(diagShapeVal_.GetDim(static_cast<size_t>(diagDimNum_) - 2));
        OP_CHECK_IF(inputInfo_.diagNum != numDiags,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "diagonal", std::to_string(numDiags),
                                                          "-2nd axis of diagonal must be " +
                                                              std::to_string(inputInfo_.diagNum) +
                                                              ", when the value of k is " + Ops::Base::ToString(k)),
                    return ge::GRAPH_FAILED);
    }

    uint64_t diagRowLenComputed = inputInfo_.xRowNum + std::min(inputInfo_.k1, 0);
    uint64_t diagColLenComputed = inputInfo_.xColNum - std::max(inputInfo_.k0, 0);
    uint64_t diagLenComputed = std::min(diagRowLenComputed, diagColLenComputed);
    OP_CHECK_IF(inputInfo_.maxDiagLen != diagLenComputed,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "diagonal", std::to_string(inputInfo_.maxDiagLen),
                    "-1st axis of diagonal must be " + std::to_string(diagLenComputed) + ", when the value of k is " +
                        Ops::Base::ToString(k)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatrixSetDiagV2Tiling::ParamCheck()
{
    CHECK_RET_SUCC(CheckX());
    CHECK_RET_SUCC(CheckDiag());
    CHECK_RET_SUCC(CheckK());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4MatrixSetDiag(gert::TilingContext* context)
{
    // DoTiling
    MatrixSetDiagV2Tiling tiling{context};
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForMatrixSetDiag([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatrixSetDiagV2)
    .Tiling(Tiling4MatrixSetDiag)
    .TilingInputsDataDependency({INPUT_IDX_K})
    .TilingParse<MatrixSetDiagCompileInfo>(TilingPrepareForMatrixSetDiag);
} // namespace optiling
