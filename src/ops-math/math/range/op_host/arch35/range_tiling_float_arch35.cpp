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
 * \file range_tiling_float_arch35.cpp
 * \brief
 */
#include "range_tiling.h"
#include "range_tiling_arch35.h"

namespace optiling {
class RangeRegBaseFloatTilingClass : public RangeRegBaseTilingClass {
public:
    explicit RangeRegBaseFloatTilingClass(gert::TilingContext* context) : RangeRegBaseTilingClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    RangeTilingDataFloat tilingData_;
};

bool RangeRegBaseFloatTilingClass::IsCapable()
{
    return outDataType_ == ge::DT_FLOAT || outDataType_ == ge::DT_FLOAT16 || outDataType_ == ge::DT_BF16;
}

ge::graphStatus RangeRegBaseFloatTilingClass::PostTiling()
{
    RangeTilingDataFloat tilingData;
    tilingData.set_usedCoreNum(tilingParam_.usedCoreNum);
    tilingData.set_totalCoreNum(tilingParam_.totalCoreNum);
    tilingData.set_hardwareUbSize(tilingParam_.hardwareUbSize);
    tilingData.set_totalElementNum(tilingParam_.totalElementNum);
    tilingData.set_numOfPerCore(tilingParam_.numOfPerCore);
    tilingData.set_loopOfPerCore(tilingParam_.loopOfPerCore);
    tilingData.set_perOfPerCore(tilingParam_.perOfPerCore);
    tilingData.set_tailOfPerCore(tilingParam_.tailOfPerCore);
    tilingData.set_numOfTailCore(tilingParam_.numOfTailCore);
    tilingData.set_loopOfTailCore(tilingParam_.loopOfTailCore);
    tilingData.set_perOfTailCore(tilingParam_.perOfTailCore);
    tilingData.set_tailOfTailCore(tilingParam_.tailOfTailCore);
    tilingData.set_ubOneLoopNum(tilingParam_.ubOneLoopNum);

    auto tensorStart = context_->GetInputTensor(0);
    auto tensorDelta = context_->GetInputTensor(2);
    float start(0);
    float delta(0);
    OP_CHECK_IF(
        RangeGetConstValue<float>(context_, tensorStart, start) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "get start const value fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        RangeGetConstValue<float>(context_, tensorDelta, delta) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "get delta const value fail."), return ge::GRAPH_FAILED);
    tilingData.set_start(start);
    tilingData.set_delta(delta);

    // 设置userWorkspace
    size_t* userWorkspaceSize = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, userWorkspaceSize);
    userWorkspaceSize[0] = RESERVED_WORKSPACE;
    tilingData.set_workspaceSize(userWorkspaceSize[0]);

    if (tilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    context_->SetBlockDim(tilingData.get_usedCoreNum());
    context_->SetTilingKey(GetTilingKey());

    return ge::GRAPH_SUCCESS;
}

uint64_t RangeRegBaseFloatTilingClass::GetTilingKey() const
{
    if (outDataType_ == ge::DataType::DT_FLOAT16) {
        return FP16_TILING_KEY;
    } else if (outDataType_ == ge::DataType::DT_BF16) {
        return BF16_TILING_KEY;
    } else {
        return FP32_TILING_KEY;
    }
}

REGISTER_OPS_TILING_TEMPLATE(Range, RangeRegBaseFloatTilingClass, 2000);
} // namespace optiling