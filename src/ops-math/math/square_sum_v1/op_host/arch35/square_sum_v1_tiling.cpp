/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file square_sum_v1_tiling.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../../op_kernel/square_sum_v1_dag.h"
#include "../../op_kernel/square_sum_v1_tiling_key.h"
#include "../../op_kernel/square_sum_v1_tiling_data.h"
#include "square_sum_v1_tiling.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "op_host/tiling_base_util.h"

using namespace ge;
using namespace Ops::Base;

namespace optiling {
static const int64_t ASCEND_WORKSPACE = 16 * 1024 * 1024;
static constexpr int32_t SIZE4 = 4;
static constexpr int32_t SIZE2 = 2;

class SquareSumV1Tiling {
public:
    explicit SquareSumV1Tiling(gert::TilingContext* context) : tilingContext_(context) {};
    ge::graphStatus RunTiling(const ReduceOpCompileInfo* compileInfo);

protected:
    ge::graphStatus DoEleTiling(ReduceOpInputParam& opInput);
    ge::graphStatus DoReduceTiling(ReduceOpInputParam& opInput, ReduceTilingKey& key,
                                   const ReduceOpCompileInfo* compileInfo);
    ge::graphStatus SetTilingData();

private:
    gert::TilingContext* tilingContext_;
    SquareSumV1TilingKey key_;
    SquareSumV1TilingData* tilingData_ = nullptr;
};

ge::graphStatus SquareSumV1Tiling::SetTilingData()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Enter SetTilingData");
    uint64_t tilingKey;
    GEN_REDUCE_TILING_KEY(tilingKey, key_.reduceTiling, key_.noop);
    OP_LOGI(tilingContext_->GetNodeName(),
            "patternID:%u, loopARCount:%u, loopInnerARCount:%u, noop:%u, Tiling Key is:%lu",
            key_.reduceTiling.patternID, key_.reduceTiling.loopARCount, key_.reduceTiling.loopInnerARCount, key_.noop,
            tilingKey);
    if (key_.noop == 1) {
        size_t* currentWorkspace = tilingContext_->GetWorkspaceSizes(1);
        currentWorkspace[0] = ASCEND_WORKSPACE;
        tilingContext_->SetBlockDim(tilingData_->elewiseTiling.blockNum);
    }
    tilingContext_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SquareSumV1Tiling::DoEleTiling(ReduceOpInputParam& opInput)
{
    ElewiseBaseTiling eleBaseTiling(tilingContext_);
    ge::graphStatus status = ge::GRAPH_FAILED;
    if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
        status = eleBaseTiling.DoTiling<SquareSumV1::SquareSumV1NoopDag<float, float>::OpDag>(
            tilingData_->elewiseTiling);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
        status = eleBaseTiling.DoTiling<SquareSumV1::SquareSumV1NoopDag<half, float>::OpDag>(
            tilingData_->elewiseTiling);
    } else {
        OP_CHECK_IF(
            (status == ge::GRAPH_FAILED),
            OP_LOGE_FOR_INVALID_DTYPE(tilingContext_->GetNodeName(), "x",
                                      Ops::Base::ToString(opInput.inputDtype).c_str(), "bfloat16, float16 or float"),
            return ge::GRAPH_FAILED);
    }
    return status;
}

ge::graphStatus SquareSumV1Tiling::DoReduceTiling(ReduceOpInputParam& opInput, ReduceTilingKey& key,
                                                  const ReduceOpCompileInfo* compileInfo)
{
    ge::graphStatus status = ge::GRAPH_FAILED;
    if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
        status = Tiling4ReduceOp<SquareSumV1::SquareSumV1Dag<float, float>::OpDag>(
            tilingContext_, opInput, key, compileInfo, &(tilingData_->reduceTiling));
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
        status = Tiling4ReduceOp<SquareSumV1::SquareSumV1Dag<half, float>::OpDag>(
            tilingContext_, opInput, key, compileInfo, &(tilingData_->reduceTiling));
    }
    OP_CHECK_IF(
        (status == ge::GRAPH_FAILED),
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext_->GetNodeName(), "x", Ops::Base::ToString(opInput.inputDtype).c_str(),
                                  "bfloat16, float16 or float"),
        return ge::GRAPH_FAILED);
    return status;
}

ge::graphStatus SquareSumV1Tiling::RunTiling(const ReduceOpCompileInfo* compileInfo)
{
    tilingData_ = tilingContext_->GetTilingData<SquareSumV1TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, tilingData_);

    ReduceOpInputParam opInput;
    bool isNoop = false;
    OP_CHECK_IF((ReduceOpTmpl::GetInputParam(tilingContext_, opInput, 0) == ge::GRAPH_FAILED),
                OP_LOGE(tilingContext_->GetNodeName(), "ReduceOp get x input failed"), return ge::GRAPH_FAILED);
    auto out = tilingContext_->GetOutputShape(0);
    OP_CHECK_IF(out == nullptr, OP_LOGE(tilingContext_, "out is nullptr"), return ge::GRAPH_FAILED);
    gert::Shape outShape = EnsureNotScalar(out->GetStorageShape());
    if (outShape.GetShapeSize() == 1L) {
        opInput.axes.resize(opInput.shape.size());
        for (size_t i = 0; i < opInput.shape.size(); i++) {
            opInput.axes[i] = i;
        }
        isNoop = false;
    }
    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);
    auto axis = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, axis);
    auto axisData = static_cast<const int64_t*>(axis->GetData());
    if (axis->GetSize() == 0) {
        const bool isNoopWithEmpty = *(attrs->GetAttrPointer<bool>(2));
        isNoop = isNoopWithEmpty ? true : false;
        if (!isNoopWithEmpty) {
            opInput.axes.resize(opInput.shape.size());
            for (size_t i = 0; i < opInput.shape.size(); i++) {
                opInput.axes[i] = i;
            }
        }
    } else {
        size_t size = axis->GetSize();
        opInput.axes.resize(size);
        for (size_t i = 0; i < size; i++) {
            opInput.axes[i] = axisData[i];
        }
        isNoop = false;
    }
    key_.noop = isNoop ? 1 : 0;
    if (isNoop) {
        OP_CHECK_IF((DoEleTiling(opInput) == ge::GRAPH_FAILED),
                    OP_LOGE(tilingContext_->GetNodeName(), "DoEleTiling Failed for SquareSumV1"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((DoReduceTiling(opInput, key_.reduceTiling, compileInfo) == ge::GRAPH_FAILED),
                    OP_LOGE(tilingContext_->GetNodeName(), "DoReduceTiling Failed for SquareSumV1"),
                    return ge::GRAPH_FAILED);
    }
    return SetTilingData();
}

ge::graphStatus Tiling4SquareSumV1(gert::TilingContext* context)
{
    OP_LOGD("SquareSumV1Tiling", "Enter Tiling4SquareSumV1");
    if (context == nullptr) {
        OP_LOGE("SquareSumV1Tiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const ReduceOpCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    SquareSumV1Tiling tiling(context);
    return tiling.RunTiling(compileInfo);
}

static ge::graphStatus TilingPrepareForSquareSumV1([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the SquareSumV1 op.
IMPL_OP_OPTILING(SquareSumV1).Tiling(Tiling4SquareSumV1).TilingParse<ReduceOpCompileInfo>(TilingPrepareForSquareSumV1);
} // namespace optiling
