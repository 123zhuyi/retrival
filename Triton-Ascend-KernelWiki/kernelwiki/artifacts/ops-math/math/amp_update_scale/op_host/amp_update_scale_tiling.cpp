/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file amp_update_scale.cc
 * \brief
 */

#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "util/math_util.h"
#include "log/log.h"
#include "amp_update_scale_tiling.h"

namespace {
constexpr uint64_t GROWTH_FACTOR_ID = 0;
constexpr uint64_t BACKOFF_FACTOR_ID = 1;
constexpr uint64_t GROWTH_INTERVAL = 2;
constexpr uint64_t TILING_KEY_FLOAT = 0;
constexpr uint64_t TILING_KEY_FLOAT16 = 1;
constexpr uint64_t TILING_KEY_BF16 = 2;
} // namespace

namespace optiling {

class AmpUpdateScaleTiling {
public:
    explicit AmpUpdateScaleTiling(gert::TilingContext* context) : TilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint() const;

private:
    AmpUpdateScaleTilingData TilingData;
    gert::TilingContext* TilingContext = nullptr;
    float growthFactor = 0;
    float backoffFactor = 0;
    int32_t growthInterval = 0;
};

ge::graphStatus AmpUpdateScaleTiling::Init()
{
    auto attrs = TilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(TilingContext, attrs);

    auto growthFactorPtr = attrs->GetAttrPointer<float>(GROWTH_FACTOR_ID);
    OP_CHECK_NULL_WITH_CONTEXT(TilingContext, growthFactorPtr);
    auto backoffFactorPtr = attrs->GetAttrPointer<float>(BACKOFF_FACTOR_ID);
    OP_CHECK_NULL_WITH_CONTEXT(TilingContext, backoffFactorPtr);
    auto growthIntervalPtr = attrs->GetAttrPointer<int64_t>(GROWTH_INTERVAL);
    OP_CHECK_NULL_WITH_CONTEXT(TilingContext, growthIntervalPtr);

    growthFactor = *growthFactorPtr;
    backoffFactor = *backoffFactorPtr;

    int64_t growthIntervalRaw = *growthIntervalPtr;
    // 校验 growthInterval 取值范围 [1, INT32_MAX]
    OP_CHECK_IF(growthIntervalRaw < 1 || growthIntervalRaw > INT32_MAX,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(TilingContext->GetNodeName(), "growthInterval",
                                                      std::to_string(growthIntervalRaw).c_str(),
                                                      "growthInterval must be in range [1, INT32_MAX]"),
                return ge::GRAPH_FAILED);
    growthInterval = static_cast<int32_t>(growthIntervalRaw);

    // 校验输入 tensor 的 shape 均为标量 [1]
    constexpr int kInputNum = 3;
    const char* kInputNames[kInputNum] = {"currentScale", "growthTracker", "foundInf"};
    constexpr int64_t kScalarDimNum = 1;
    for (int i = 0; i < kInputNum; i++) {
        auto inputShape = TilingContext->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(TilingContext, inputShape);
        auto storageShape = inputShape->GetStorageShape();
        OP_CHECK_IF(storageShape.GetDimNum() != kScalarDimNum,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(TilingContext->GetNodeName(), kInputNames[i],
                                                 std::to_string(storageShape.GetDimNum()).c_str(), "scalar [1]"),
                    return ge::GRAPH_FAILED);
        int64_t shapeSize = storageShape.GetShapeSize();
        OP_CHECK_IF(shapeSize != 1,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(TilingContext->GetNodeName(), kInputNames[i],
                                                              std::to_string(shapeSize).c_str(),
                                                              "The shape of input must be scalar [1]"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AmpUpdateScaleTiling::RunKernelTiling()
{
    TilingContext->SetBlockDim(1);

    auto currentScaleDtype = TilingContext->GetInputDesc(0)->GetDataType();
    uint64_t tilingKey = 0;
    if (currentScaleDtype == ge::DT_FLOAT) {
        tilingKey = TILING_KEY_FLOAT;
    } else if (currentScaleDtype == ge::DT_FLOAT16) {
        tilingKey = TILING_KEY_FLOAT16;
    } else if (currentScaleDtype == ge::DT_BF16) {
        tilingKey = TILING_KEY_BF16;
    }
    TilingContext->SetTilingKey(tilingKey);

    TilingData.set_growthFactor(growthFactor);
    TilingData.set_backoffFactor(backoffFactor);
    TilingData.set_growthInterval(growthInterval);

    TilingData.SaveToBuffer(TilingContext->GetRawTilingData()->GetData(),
                            TilingContext->GetRawTilingData()->GetCapacity());
    TilingContext->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

void AmpUpdateScaleTiling::TilingDataPrint() const
{
    OP_LOGI(TilingContext->GetNodeName(), "growthFactor:%f.", growthFactor);
    OP_LOGI(TilingContext->GetNodeName(), "backoffFactor:%f.", backoffFactor);
    OP_LOGI(TilingContext->GetNodeName(), "growthInterval:%d.", growthInterval);
}

static ge::graphStatus TilingAmpUpdateScale(gert::TilingContext* context)
{
    AmpUpdateScaleTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}

static ge::graphStatus TilingPrepareForAmpUpdateScale(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForAmpUpdateScale start.");
    return ge::GRAPH_SUCCESS;
}
struct AmpUpdateScaleCompileInfo {};
IMPL_OP_OPTILING(AmpUpdateScale)
    .Tiling(TilingAmpUpdateScale)
    .TilingParse<AmpUpdateScaleCompileInfo>(TilingPrepareForAmpUpdateScale);
} // namespace optiling
