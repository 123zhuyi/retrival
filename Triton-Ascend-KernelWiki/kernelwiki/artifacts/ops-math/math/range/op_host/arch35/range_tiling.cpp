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
 * \file range_tiling.cpp
 * \brief
 */
#include "range_tiling.h"
#include <cmath>
#include <type_traits>
#include "log/log.h"
#include "platform/platform_info.h"
#include "range_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "util/math_util.h"
#include "util/fp16.h"

namespace optiling {
constexpr size_t INPUT_IDX_START = 0;
constexpr size_t INPUT_IDX_LIMIT = 1;
constexpr size_t INPUT_IDX_DELTA = 2;
constexpr int32_t INT16_BITS_NUM = 16;

class RangeMemBaseTilingClass : public Ops::Base::TilingBaseClass {
public:
    explicit RangeMemBaseTilingClass(gert::TilingContext* context) : TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    ge::graphStatus GetPlatformInfo() override
    {
        auto platformInfo = context_->GetPlatformInfo();
        if (platformInfo != nullptr) {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
            socVersion_ = ascendcPlatform.GetSocVersion();
        } else {
            auto compileInfoPtr = reinterpret_cast<const RangeCompileInfo*>(context_->GetCompileInfo());
            OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
            socVersion_ = compileInfoPtr->socVersion;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    bool IsCapable() override
    {
        return !Ops::Base::IsRegbaseSocVersion(context_);
    }

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;

    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {
        return context_->GetTilingKey();
    }

    platform_ascendc::SocVersion socVersion_;
};

static bool InputTypeIsInvalid(ge::DataType start, ge::DataType limit, ge::DataType delta, ge::DataType output)
{
    std::set<ge::DataType> outputDtype = {ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT64};
    std::set<ge::DataType> inputDtype = {ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16,
                                         ge::DT_BF16,  ge::DT_INT64, ge::DT_DOUBLE};

    bool startInvalid = (inputDtype.count(start) == 0);
    bool limitInvalid = (inputDtype.count(limit) == 0);
    bool deltaInvalid = (inputDtype.count(delta) == 0);
    bool outputInvalid = (outputDtype.count(output) == 0);
    return startInvalid || limitInvalid || deltaInvalid || outputInvalid;
}

template <typename T>
ge::graphStatus RangeGetConstValue(gert::TilingContext* context, const gert::Tensor* tensor, T& value)
{
    if (tensor->GetDataType() == ge::DT_INT32) {
        const int32_t* constDataPtr = tensor->GetData<int32_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, constDataPtr);
        value = static_cast<T>(*constDataPtr);
        OP_LOGD(context->GetNodeName(), "range get const value:%d", *constDataPtr);
    } else if (tensor->GetDataType() == ge::DT_FLOAT) {
        const float* constDataPtr = tensor->GetData<float>();
        OP_CHECK_NULL_WITH_CONTEXT(context, constDataPtr);
        value = static_cast<T>(*constDataPtr);
        OP_LOGD(context->GetNodeName(), "range get const value:%f", *constDataPtr);
    } else if (tensor->GetDataType() == ge::DT_FLOAT16) {
        const Ops::Base::fp16_t* constDataPtr = tensor->GetData<Ops::Base::fp16_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, constDataPtr);
        float f32 = static_cast<float>(*constDataPtr);
        value = static_cast<T>(f32);
        OP_LOGD(context->GetNodeName(), "range get const value:%f", static_cast<float>(*constDataPtr));
    } else if (tensor->GetDataType() == ge::DT_BF16) {
        const int16_t* constDataPtr = tensor->GetData<int16_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, constDataPtr);
        int32_t f32hex = (static_cast<int32_t>(*constDataPtr)) << INT16_BITS_NUM;
        value = static_cast<T>((reinterpret_cast<float&>(f32hex)));
        OP_LOGD(context->GetNodeName(), "range get const value:%d", f32hex);
    } else if (tensor->GetDataType() == ge::DT_INT64) {
        const int64_t* constDataPtr = tensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, constDataPtr);
        value = static_cast<T>(*constDataPtr);
        OP_LOGD(context->GetNodeName(), "range get const value:%ld", *constDataPtr);
    } else if (tensor->GetDataType() == ge::DT_DOUBLE) {
        const double* constDataPtr = tensor->GetData<double>();
        OP_CHECK_NULL_WITH_CONTEXT(context, constDataPtr);
        value = static_cast<T>(*constDataPtr);
        OP_LOGD(context->GetNodeName(), "range get const value:%f", *constDataPtr);
    } else {
        // do nothing, impossible
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus AppendTilingData(gert::TilingData* tilingData, T value)
{
    return tilingData->Append<T>(value);
}

template <typename T>
static ge::graphStatus CheckStep(gert::TilingContext* context, T start, T limit, T delta)
{
    OP_CHECK_IF(
        !(delta > (static_cast<T>(0)) || delta < (static_cast<T>(0))),
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "delta", "0", "non zero"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((limit > start) && (delta < 0)),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "start, limit and delta",
            (std::to_string(start) + ", " + std::to_string(limit) + " and " + std::to_string(delta)).c_str(),
            "When limit is greater than start, delta must be greater than or equal to 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((limit < start) && (delta > 0)),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "start, limit and delta",
            (std::to_string(start) + ", " + std::to_string(limit) + " and " + std::to_string(delta)).c_str(),
            "When limit is less than start, delta must be less than or equal to 0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CalculateOutputSize(
    gert::TilingContext* context, const gert::Tensor* tensorStart, const gert::Tensor* tensorLimit,
    const gert::Tensor* tensorDelta, uint64_t& outputSize)
{
    T start(0);
    T limit(0);
    T delta(0);
    OP_CHECK_IF(
        RangeGetConstValue<T>(context, tensorStart, start) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "get start const value fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        RangeGetConstValue<T>(context, tensorLimit, limit) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "get limit const value fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        RangeGetConstValue<T>(context, tensorDelta, delta) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "get delta const value fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckStep<T>(context, start, limit, delta) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "CheckStep fail."), return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(0));
    auto dtypeOutput = context->GetOutputDesc(0)->GetDataType();
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* isClosed = attrs->GetAttrPointer<bool>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, isClosed);
    if (*isClosed) {
        outputSize = static_cast<uint64_t>((limit - start) / delta + 1);
    } else {
        if (dtypeOutput == ge::DT_INT64) {
            outputSize = static_cast<uint64_t>(Ops::Base::CeilDiv(
                static_cast<int64_t>(limit) - static_cast<int64_t>(start), static_cast<int64_t>(delta)));
        } else {
            double startDouble = 0.0;
            double limitDouble = 0.0;
            double deltaDouble = 0.0;
            OP_CHECK_IF(
                RangeGetConstValue<double>(context, tensorStart, startDouble) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "get start value fail."), return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                RangeGetConstValue<double>(context, tensorLimit, limitDouble) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "get limit value fail."), return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                RangeGetConstValue<double>(context, tensorDelta, deltaDouble) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "get delta value fail."), return ge::GRAPH_FAILED);
            outputSize = static_cast<uint64_t>(std::ceil((limitDouble - startDouble) / deltaDouble));
        }
    }
    OP_LOGD(
        context->GetNodeName(), "CalculateOutputSize: start: %lf, limit: %lf, delta: %lf, outputSize: %lu",
        static_cast<double>(start), static_cast<double>(limit), static_cast<double>(delta), outputSize);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    uint64_t outSizeFromFramework = out_shape->GetStorageShape().GetShapeSize();
    OP_LOGD(context->GetNodeName(), "OFF: %lu", outSizeFromFramework);
    if (outputSize != outSizeFromFramework) {
        OP_LOGW(context->GetNodeName(), "OFF is %lu, but OFT is %lu", outSizeFromFramework, outputSize);
        outputSize = (outputSize > outSizeFromFramework) ? outSizeFromFramework : outputSize;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus AppendTilingArgs(
    gert::TilingContext* context, const gert::Tensor* tensorStart, const gert::Tensor* tensorDelta,
    const uint64_t outputSize, gert::TilingData* tilingData)
{
    T start(0);
    T delta(0);
    OP_CHECK_IF(
        RangeGetConstValue<T>(context, tensorStart, start) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "get start const value fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        RangeGetConstValue<T>(context, tensorDelta, delta) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "get delta const value fail."), return ge::GRAPH_FAILED);

    auto compileInfo = reinterpret_cast<const RangeCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_CHECK_IF(
        AppendTilingData<uint64_t>(tilingData, outputSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "append  fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        AppendTilingData<uint64_t>(tilingData, static_cast<uint64_t>(compileInfo->running_core_num)) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "append running_core_num fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        AppendTilingData<T>(tilingData, start) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "append start fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        AppendTilingData<T>(tilingData, delta) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "append delta fail."), return ge::GRAPH_FAILED);

    context->SetBlockDim(compileInfo->running_core_num);

    OP_LOGD(
        context->GetNodeName(), "range get output_total_num:%lu, compile_info->running_core_num:%u", outputSize,
        compileInfo->running_core_num);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareRangeForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareRangeForAscendC entering.");
    auto compileInfo = context->GetCompiledInfo<RangeCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Range(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("range", "context is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "TilingPrepare4Range regbase enter.");
    return TilingPrepareRangeForAscendC(context);
}

ge::graphStatus OpTilingCalculateOutputSize(
    gert::TilingContext* context, const gert::Tensor* start, const gert::Tensor* tensorLimit,
    const gert::Tensor* tensorDelta, const ge::DataType dtypeOutput)
{
    auto tilingData = context->GetRawTilingData();
    uint64_t outputSize = 0;

    switch (dtypeOutput) {
        case ge::DT_INT32: {
            OP_CHECK_IF(
                CalculateOutputSize<int64_t>(context, start, tensorLimit, tensorDelta, outputSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CalculateOutputSize fail."), return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                AppendTilingArgs<int32_t>(context, start, tensorDelta, outputSize, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "append tiling args fail."), return ge::GRAPH_FAILED);
            break;
        }
        case ge::DT_INT64: {
            OP_CHECK_IF(
                CalculateOutputSize<int64_t>(context, start, tensorLimit, tensorDelta, outputSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CalculateOutputSize fail."), return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                AppendTilingArgs<int64_t>(context, start, tensorDelta, outputSize, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "append tiling args fail."), return ge::GRAPH_FAILED);
            break;
        }
        case ge::DT_FLOAT: {
            OP_CHECK_IF(
                CalculateOutputSize<double>(context, start, tensorLimit, tensorDelta, outputSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CalculateOutputSize fail."), return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                AppendTilingArgs<float>(context, start, tensorDelta, outputSize, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "append tiling args fail."), return ge::GRAPH_FAILED);
            break;
        }
        default: {
            OP_CHECK_IF(
                CalculateOutputSize<float>(context, start, tensorLimit, tensorDelta, outputSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "append tiling args fail."), return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                AppendTilingArgs<float>(context, start, tensorDelta, outputSize, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "append tiling args fail."), return ge::GRAPH_FAILED);
            break;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RangeMemBaseTilingClass::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Tiling4Range enter.");

    auto tensorStart = context_->GetInputTensor(INPUT_IDX_START);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensorStart);
    auto tensorLimit = context_->GetInputTensor(INPUT_IDX_LIMIT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensorLimit);
    auto tensorDelta = context_->GetInputTensor(INPUT_IDX_DELTA);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensorDelta);

    auto dtypeStart = tensorStart->GetDataType();
    auto dtypeLimit = tensorLimit->GetDataType();
    auto dtypeDelta = tensorDelta->GetDataType();
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    auto dtypeOutput = context_->GetOutputDesc(0)->GetDataType();
    OP_CHECK_IF(
        InputTypeIsInvalid(dtypeStart, dtypeLimit, dtypeDelta, dtypeOutput),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "start, limit, delta and y",
            (Ops::Base::ToString(dtypeStart) + ", " + Ops::Base::ToString(dtypeLimit) + ", " +
             Ops::Base::ToString(dtypeDelta) + " and " + Ops::Base::ToString(dtypeOutput))
                .c_str(),
            "The dtype of start, limit and delta must be int32, int64, float, float16, bf16 or double, "
            "and the dtype of y must be int32, int64, float, float16 or bf16"),
        return ge::GRAPH_FAILED);

    auto ret = OpTilingCalculateOutputSize(context_, tensorStart, tensorLimit, tensorDelta, dtypeOutput);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    OP_LOGD(context_->GetNodeName(), "Tiling4Range exit.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Range(gert::TilingContext* context)
{
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

REGISTER_OPS_TILING_TEMPLATE(Range, RangeMemBaseTilingClass, 1000);

IMPL_OP_OPTILING(Range).Tiling(Tiling4Range).TilingParse<RangeCompileInfo>(TilingPrepare4Range);
} // namespace optiling
