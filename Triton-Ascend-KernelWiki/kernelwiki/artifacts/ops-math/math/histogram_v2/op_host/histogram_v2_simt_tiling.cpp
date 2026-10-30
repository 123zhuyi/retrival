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
 * \file histogram_v2_simt_tiling.cpp
 * \brief
 */
#include "histogram_v2_tiling.h"
#include "op_host/math_tiling_templates_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_base_util.h"

namespace optiling {
static const std::unordered_map<ge::DataType, uint32_t> INPUT_DATA_TYPE_TO_INT{
    {ge::DataType::DT_FLOAT, 1}, {ge::DataType::DT_INT32, 2}, {ge::DataType::DT_INT8, 3},   {ge::DataType::DT_UINT8, 4},
    {ge::DataType::DT_INT16, 5}, {ge::DataType::DT_INT64, 6}, {ge::DataType::DT_FLOAT16, 7}};
constexpr int64_t INPUT_NUM = 3;
constexpr int64_t INPUT_IDX_X = 0;
constexpr int64_t OUTPUT_IDX = 0;
constexpr int64_t BINS_IDX = 0;
constexpr int64_t SIZE_OF_INT32 = 4;
constexpr int64_t SIZE_OF_FLOAT32 = 4;
constexpr int64_t GM_ATOMIC_ADD_FACTOR = 100;
constexpr int64_t DEFAULT_BINS = 100;
constexpr int64_t TILING_KEY_UB_FULL = 100;
constexpr int64_t TILING_KEY_UB_NOT_FULL = 200;
constexpr int64_t TILING_KEY_UB_NOT_FULL_SIMT = 300;
// fp32 output key offset: set tens digit to 1 (e.g. 101->111, 201->211, 301->311)
constexpr int64_t OUTPUT_FP32_KEY_OFFSET = 10;
// deterministic key offset: set thousands digit to 1 (e.g. 111->1111, 117->1117)
constexpr int64_t DETERM_OFFSET = 1000;
constexpr uint64_t SIMT_DCACHE_SIZE = 32 * 1024;

class HistogramV2SimtTiling : public HistogramV2BaseClass
{
public:
    explicit HistogramV2SimtTiling(gert::TilingContext* context) : HistogramV2BaseClass(context) {};
    ~HistogramV2SimtTiling() override = default;
    void Reset(gert::TilingContext* context) override
    {
        HistogramV2BaseClass::Reset(context);
    }

protected:
    bool IsCapable() override
    {
        if (!Ops::Base::IsRegbaseSocVersion(context_)) {
            return false;
        }
        return true;
    }

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    uint64_t GetTilingKey() const override;

private:
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    ge::graphStatus TilingDataForCore();
    void TilingDataPrint() const;
    inline void SetTilingKeyMode(int64_t inputDtypeVal) const;

    HistogramV2SimtTilingData tilingData;
    int64_t inputDtypeVal_ = 0;
    int64_t coreNum_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t totalLength_ = 0; // the length of input
    int64_t tailNum_ = 0;
    bool isFp32Output_ = false;
    int64_t isDeterministic_ = 0;

private:
    // kernel needed.
    int64_t bins_ = 0;
    int64_t ubNumCanUse_ = 0;
    int64_t ubLoopNum_ = 0;

    int64_t needXCoreNum_ = 0;
    int64_t formerLength_ = 0;
    int64_t tailLength_ = 0;

    int64_t clearYCoreNum_ = 0;
    int64_t clearYFactor_ = 0;
    int64_t clearYTail_ = 0;
};

ge::graphStatus HistogramV2SimtTiling::DoOpTiling()
{
    aicoreParams_.ubSize = aicoreParams_.ubSize - SIMT_DCACHE_SIZE;
    Init();
    return SetKernelTiling();
}

ge::graphStatus HistogramV2SimtTiling::PostTiling()
{
    context_->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HistogramV2SimtTiling::GetShapeAttrsInfo()
{
    auto inputShape = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    totalLength_ = inputShape->GetStorageShape().GetShapeSize();

    auto compileInfo = reinterpret_cast<const HistogramV2CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->totalCoreNum;
    OP_CHECK_IF(
        coreNum_ <= 0, OP_LOGE(context_, "coreNum must be > 0, but got %ld.", coreNum_), return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* binsPtr = attrs->GetAttrPointer<int64_t>(BINS_IDX);
    bins_ = (binsPtr == nullptr) ? DEFAULT_BINS : *binsPtr;
    OP_CHECK_IF(
        bins_ <= 0, 
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "bins", std::to_string(bins_).c_str(), "a positive integer"),
        return ge::GRAPH_FAILED);

    auto outputShape = context_->GetOutputShape(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto outputDataLength = outputShape->GetStorageShape().GetShapeSize();
    if (outputDataLength != bins_) {
        std::string sizeStr = std::to_string(outputDataLength) + " and " + std::to_string(bins_);
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "y and bins",
            sizeStr.c_str(), "The shape size of y should be the same as bins");
        return ge::GRAPH_FAILED;
    }

    auto inputDesc = context_->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    auto dType = inputDesc->GetDataType();

    for (int64_t i = 1; i < INPUT_NUM; i++) {
        auto inputMinMaxDesc = context_->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputMinMaxDesc);
        auto minMaxDtype = inputMinMaxDesc->GetDataType();
        if (minMaxDtype != dType) {
            std::string dtypesStr = Ops::Base::ToString(dType) + " and " + Ops::Base::ToString(minMaxDtype);
            std::string paramNames = std::string("x and ") + (i == 1 ? "min" : "max");
            std::string reason = std::string("The dtypes of ") + paramNames + " must be the same";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), paramNames.c_str(),
                dtypesStr.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
        auto minMaxShape = context_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, minMaxShape);
        auto minMaxLength = minMaxShape->GetStorageShape().GetShapeSize();
        if (minMaxLength != 1) {
            std::string paramName = (i == 1) ? "min" : "max";
            OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), paramName.c_str(),
                std::to_string(minMaxLength).c_str(), "1");
            return ge::GRAPH_FAILED;
        }
    }

    auto iter = INPUT_DATA_TYPE_TO_INT.find(dType);
    OP_CHECK_IF(
        (iter == INPUT_DATA_TYPE_TO_INT.end()),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", Ops::Base::ToString(dType).c_str(),
            "float, int32, int8, uint8, int16, int64 or float16"),
        return ge::GRAPH_FAILED);
    inputDtypeVal_ = iter->second;

    auto outputDesc = context_->GetOutputDesc(OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    auto outputDType = outputDesc->GetDataType();

    // Determine fp32 output flag directly from output dtype
    isFp32Output_ = (outputDType == ge::DT_FLOAT);

    // Validate fp32 output: only allowed for fp16 or fp32 input on ascend950
    if (isFp32Output_) {
        if (inputDtypeVal_ != 1 && inputDtypeVal_ != 7) { // 1=fp32, 7=fp16
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y", Ops::Base::ToString(outputDType).c_str(), "int32");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (outputDType != ge::DT_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y", Ops::Base::ToString(outputDType).c_str(), "int32, float");
            return ge::GRAPH_FAILED;
        }
    }

    // Parse deterministic flag from context
    isDeterministic_ = (context_->GetDeterministic() == 1) ? 1 : 0;

    return ge::GRAPH_SUCCESS;
}

uint64_t HistogramV2SimtTiling::GetTilingKey() const
{
    return context_->GetTilingKey();
}

ge::graphStatus HistogramV2SimtTiling::Init()
{
    OP_LOGD(context_, "Tiling initing.");
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkSpace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkSpace);
    currentWorkSpace[0] = sysWorkspaceSize;
    // For deterministic fp32 output path
    int64_t ubElemSize = (isFp32Output_ && isDeterministic_ == 1) ? SIZE_OF_FLOAT32 : SIZE_OF_INT32;
    ubNumCanUse_ = aicoreParams_.ubSize / ubElemSize;
    ubLoopNum_ = Ops::Base::CeilDiv(bins_, ubNumCanUse_);
    SetTilingKeyMode(inputDtypeVal_);
    TilingDataForCore();
    OP_LOGD(context_, "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

inline void HistogramV2SimtTiling::SetTilingKeyMode(int64_t inputDtypeVal) const
{
    int64_t outputOffset = isFp32Output_ ? OUTPUT_FP32_KEY_OFFSET : 0;
    // 只有输出为fp32才会走确定性逻辑的tilingKey
    int64_t determOffset = isFp32Output_ ? isDeterministic_ * DETERM_OFFSET : 0;
    
    if (bins_ < ubNumCanUse_) {
        context_->SetLocalMemorySize(aicoreParams_.ubSize);
        context_->SetTilingKey(TILING_KEY_UB_FULL + outputOffset + determOffset + inputDtypeVal);
    } else if (totalLength_ > bins_ / GM_ATOMIC_ADD_FACTOR) {
        context_->SetLocalMemorySize(aicoreParams_.ubSize);
        context_->SetTilingKey(TILING_KEY_UB_NOT_FULL + outputOffset + determOffset + inputDtypeVal);
    } else {
        context_->SetTilingKey(TILING_KEY_UB_NOT_FULL_SIMT + outputOffset + determOffset + inputDtypeVal);
    }
}

ge::graphStatus HistogramV2SimtTiling::TilingDataForCore()
{
    OP_LOGD(context_, "TilingDataForCore start.");
    formerLength_ = Ops::Base::CeilDiv(totalLength_, coreNum_);
    needXCoreNum_ = Ops::Base::CeilDiv(totalLength_, formerLength_);
    tailLength_ = totalLength_ - (needXCoreNum_ - 1) * formerLength_;

    clearYFactor_ = Ops::Base::CeilDiv(bins_, coreNum_);
    OP_CHECK_IF(clearYFactor_ == 0, OP_LOGE(context_, "clearYFactor must not be 0."), return ge::GRAPH_FAILED);
    clearYCoreNum_ = Ops::Base::CeilDiv(bins_, clearYFactor_);
    clearYTail_ = bins_ - (clearYCoreNum_ - 1) * clearYFactor_;
    needCoreNum_ = std::max(needXCoreNum_, clearYCoreNum_);
    OP_LOGD(context_, "TilingDataForCore end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HistogramV2SimtTiling::SetKernelTiling()
{
    context_->SetBlockDim(needCoreNum_);

    tilingData.set_bins(bins_);
    tilingData.set_ubNumCanUse(ubNumCanUse_);
    tilingData.set_ubLoopNum(ubLoopNum_);

    tilingData.set_needXCoreNum(needXCoreNum_);
    tilingData.set_formerLength(formerLength_);
    tilingData.set_tailLength(tailLength_);

    tilingData.set_clearYCoreNum(clearYCoreNum_);
    tilingData.set_clearYFactor(clearYFactor_);
    tilingData.set_clearYTail(clearYTail_);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

void HistogramV2SimtTiling::TilingDataPrint() const
{
    OP_LOGD(context_, "tilingKey_: %lu.", context_->GetTilingKey());
    OP_LOGD(context_, "needCoreNum_: %ld.", needCoreNum_);
    OP_LOGD(context_, "totalLength_: %ld.", totalLength_);
    OP_LOGD(context_, "bins_: %ld.", bins_);
    OP_LOGD(context_, "ubNumCanUse_: %ld.", ubNumCanUse_);
    OP_LOGD(context_, "ubLoopNum_: %ld.", ubLoopNum_);
    OP_LOGD(context_, "needXCoreNum_: %ld.", needXCoreNum_);
    OP_LOGD(context_, "formerLength_: %ld.", formerLength_);
    OP_LOGD(context_, "tailLength_: %ld.", tailLength_);
    OP_LOGD(context_, "clearYCoreNum_: %ld.", clearYCoreNum_);
    OP_LOGD(context_, "clearYFactor_: %ld.", clearYFactor_);
    OP_LOGD(context_, "clearYTail_: %ld.", clearYTail_);
    OP_LOGD(context_, "isFp32Output_: %d.", isFp32Output_);
    OP_LOGD(context_, "isDeterministic_: %ld.", isDeterministic_);
}

REGISTER_OPS_TILING_TEMPLATE(HistogramV2, HistogramV2SimtTiling, 30000);
} // namespace optiling