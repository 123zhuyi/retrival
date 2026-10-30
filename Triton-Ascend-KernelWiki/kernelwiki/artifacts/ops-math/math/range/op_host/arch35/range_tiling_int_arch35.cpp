/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file range_tiling_int_arch35.cpp
 * \brief
 */
#include "range_tiling.h"
#include "range_tiling_arch35.h"

namespace optiling {
class RangeRegBaseIntTilingClass : public RangeRegBaseTilingClass {
public:
    explicit RangeRegBaseIntTilingClass(gert::TilingContext* context) : RangeRegBaseTilingClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    RangeTilingDataInt tilingData_;
};

bool RangeRegBaseIntTilingClass::IsCapable()
{
    return true;
}

template <typename T>
ge::graphStatus SetStartStep(gert::TilingContext* context, RangeTilingDataInt& tilingData)
{
    T start;
    T delta;
    auto tensorStart = context->GetInputTensor(0);
    auto tensorDelta = context->GetInputTensor(2);
    OP_CHECK_IF(
        RangeGetConstValue<T>(context, tensorStart, start) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "get start const value fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        RangeGetConstValue<T>(context, tensorDelta, delta) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "get delta const value fail."), return ge::GRAPH_FAILED);
    tilingData.set_start(start);
    tilingData.set_delta(delta);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RangeRegBaseIntTilingClass::PostTiling()
{
    tilingData_.set_usedCoreNum(tilingParam_.usedCoreNum);
    tilingData_.set_totalCoreNum(tilingParam_.totalCoreNum);
    tilingData_.set_hardwareUbSize(tilingParam_.hardwareUbSize);
    tilingData_.set_totalElementNum(tilingParam_.totalElementNum);
    tilingData_.set_numOfPerCore(tilingParam_.numOfPerCore);
    tilingData_.set_loopOfPerCore(tilingParam_.loopOfPerCore);
    tilingData_.set_perOfPerCore(tilingParam_.perOfPerCore);
    tilingData_.set_tailOfPerCore(tilingParam_.tailOfPerCore);
    tilingData_.set_numOfTailCore(tilingParam_.numOfTailCore);
    tilingData_.set_loopOfTailCore(tilingParam_.loopOfTailCore);
    tilingData_.set_perOfTailCore(tilingParam_.perOfTailCore);
    tilingData_.set_tailOfTailCore(tilingParam_.tailOfTailCore);
    tilingData_.set_ubOneLoopNum(tilingParam_.ubOneLoopNum);

    if (outDataType_ == ge::DataType::DT_INT32) {
        SetStartStep<int32_t>(context_, tilingData_);
    } else {
        SetStartStep<int64_t>(context_, tilingData_);
    }

    // 设置userWorkspace
    size_t* userWorkspaceSize = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, userWorkspaceSize);
    userWorkspaceSize[0] = RESERVED_WORKSPACE;
    tilingData_.set_workspaceSize(userWorkspaceSize[0]);

    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    context_->SetBlockDim(tilingData_.get_usedCoreNum());
    context_->SetTilingKey(GetTilingKey());

    return ge::GRAPH_SUCCESS;
}

uint64_t RangeRegBaseIntTilingClass::GetTilingKey() const
{
    if (outDataType_ == ge::DataType::DT_INT32) {
        return INT32_TILING_KEY;
    } else {
        return INT64_TILING_KEY;
    }
}

REGISTER_OPS_TILING_TEMPLATE(Range, RangeRegBaseIntTilingClass, 3000);
} // namespace optiling