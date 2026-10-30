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
 * \file histogram_v2_op_tiling.cpp
 * \brief
 */
#include "histogram_v2_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "op_host/math_tiling_templates_registry.h"
#include "op_host/tiling_base_util.h"

namespace optiling {
constexpr int64_t SIZE_OF_FP32 = 4L;
constexpr int64_t BYTE_BLOCK = 32L;

constexpr int64_t HISTOGRAM_V2_FP32 = 0L;
constexpr int64_t HISTOGRAM_V2_INT32 = 1L;
constexpr int64_t HISTOGRAM_V2_INT8 = 2L;
constexpr int64_t HISTOGRAM_V2_UINT8 = 3L;
constexpr int64_t HISTOGRAM_V2_INT16 = 4L;
constexpr int64_t HISTOGRAM_V2_INT64 = 5L;
constexpr int64_t HISTOGRAM_V2_FP16 = 6L;
constexpr int64_t HISTOGRAM_V2_NOT_SUPPORT = -1L;

constexpr int64_t UB_SELF_LENGTH = 16320L; // 64 * 255
constexpr int64_t UB_BINS_LENGTH = 16320L; // 16320 * 4 = 65280 < 65535，结果可一次性搬出
constexpr int64_t UB_SELF_LENGTH_310P = 16000L;
constexpr int64_t UB_BINS_LENGTH_310P = 16320L;

class HistogramV2Tiling
{
public:
    explicit HistogramV2Tiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint() const;

private:
    inline void SetTilingKeyMode(ge::DataType dType) const;
    inline void TilingDataForCore();
    inline void TilingDataInCore(ge::DataType dType);

    HistogramV2TilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    int64_t coreNum = 0;
    int64_t totalLength = 0; // the length of input
    int64_t tailNum = 0;
    int64_t userWorkspaceSize = 0;

private:
    // kernel needed.
    int64_t bins = 0;
    int64_t ubSelfLength = UB_SELF_LENGTH;
    int64_t ubBinsLength = UB_BINS_LENGTH;

    int64_t formerNum = 0;
    int64_t formerLength = 0;
    int64_t formerLengthAligned = 0;
    int64_t tailLength = 0;
    int64_t tailLengthAligned = 0;

    int64_t formerTileNum = 0;
    int64_t formerTileDataLength = 0;
    int64_t formerTileLeftDataLength = 0;
    int64_t formerTileLeftDataLengthAligned = 0;

    int64_t tailTileNum = 0;
    int64_t tailTileDataLength = 0;
    int64_t tailTileLeftDataLength = 0;
    int64_t tailTileLeftDataLengthAligned = 0;

    bool is310Soc = false;
};

inline void HistogramV2Tiling::SetTilingKeyMode(ge::DataType dType) const
{
    switch (dType) {
        case ge::DT_FLOAT:
            tilingContext->SetTilingKey(HISTOGRAM_V2_FP32);
            break;
        case ge::DT_INT32:
            tilingContext->SetTilingKey(HISTOGRAM_V2_INT32);
            break;
        case ge::DT_INT8:
            tilingContext->SetTilingKey(HISTOGRAM_V2_INT8);
            break;
        case ge::DT_UINT8:
            tilingContext->SetTilingKey(HISTOGRAM_V2_UINT8);
            break;
        case ge::DT_INT16:
            tilingContext->SetTilingKey(HISTOGRAM_V2_INT16);
            break;
        case ge::DT_INT64:
            tilingContext->SetTilingKey(HISTOGRAM_V2_INT64);
            break;
        case ge::DT_FLOAT16:
            tilingContext->SetTilingKey(HISTOGRAM_V2_FP16);
            break;
        default:
            tilingContext->SetTilingKey(HISTOGRAM_V2_NOT_SUPPORT);
            break;
    }
}

inline void HistogramV2Tiling::TilingDataForCore()
{
    OP_LOGD(tilingContext, "TilingDataForCore start.");
    auto alignNum = BYTE_BLOCK / SIZE_OF_FP32;
    formerNum = 1;
    tailNum = coreNum - formerNum;
    tailLength = totalLength / coreNum;
    formerLength = totalLength - (tailLength * coreNum) + tailLength;
    formerLengthAligned = ((formerLength + alignNum - 1) / alignNum) * alignNum;
    tailLengthAligned = ((tailLength + alignNum - 1) / alignNum) * alignNum;
    if (tailLength == 0) {
        coreNum = 1;
    }
    if (totalLength == 0) {
        coreNum = 0;
    }
    OP_LOGD(tilingContext, "TilingDataForCore end.");
}

inline void HistogramV2Tiling::TilingDataInCore(ge::DataType dType)
{
    int64_t tileLength = ubSelfLength;
    switch (dType) {
        case ge::DT_INT64:
            tileLength = ubSelfLength / HISTOGRAM_V2_INT8;
            break;
        default:
            break;
    }
    int64_t alignNum = BYTE_BLOCK / SIZE_OF_FP32;
    formerTileNum = formerLength / tileLength;
    formerTileDataLength = tileLength;
    formerTileLeftDataLength = formerLength - formerTileNum * formerTileDataLength;
    formerTileLeftDataLengthAligned = ((formerTileLeftDataLength + alignNum - 1) / alignNum) * alignNum;

    tailTileNum = tailLength / tileLength;
    tailTileDataLength = tileLength;
    tailTileLeftDataLength = tailLength - tailTileNum * tailTileDataLength;
    tailTileLeftDataLengthAligned = ((tailTileLeftDataLength + alignNum - 1) / alignNum) * alignNum;
}

ge::graphStatus HistogramV2Tiling::Init()
{
    OP_LOGD(tilingContext, "Tiling initing.");

    auto selfShape = tilingContext->GetInputShape(0)->GetStorageShape();
    totalLength = selfShape.GetShapeSize();

    auto compileInfo = reinterpret_cast<const HistogramV2CompileInfo*>(tilingContext->GetCompileInfo());
    coreNum = compileInfo->totalCoreNum;
    OP_LOGD(tilingContext, "coreNum %ld.", coreNum);
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    auto dType = tilingContext->GetInputDesc(0)->GetDataType();

    auto attrs = tilingContext->GetAttrs();
    int32_t binsIndex = 0;
    bins = *(attrs->GetAttrPointer<int64_t>(binsIndex));
    SetTilingKeyMode(dType);
    tilingContext->SetNeedAtomic(true);
    TilingDataForCore();
    if (compileInfo->npuArch == NpuArch::DAV_2002) {
        is310Soc = true;
        ubSelfLength = UB_SELF_LENGTH_310P;
        ubBinsLength = UB_BINS_LENGTH_310P;
    }
    TilingDataInCore(dType);
    // Sync workspace size and kernel result size
    if (compileInfo->npuArch == NpuArch::DAV_2002) {
        int64_t alignNum = BYTE_BLOCK / SIZE_OF_FP32;
        userWorkspaceSize = coreNum * BYTE_BLOCK + coreNum * (bins + alignNum) * SIZE_OF_FP32;
    }
    size_t* currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    currentWorkSpace[0] = compileInfo->sysWorkspaceSize + userWorkspaceSize;
    OP_LOGD(tilingContext, "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HistogramV2Tiling::SetKernelTiling()
{
    tilingContext->SetBlockDim(coreNum);
    if (is310Soc) {
        tilingContext->SetScheduleMode(1);
    }
    tilingData.set_bins(bins);
    tilingData.set_ubBinsLength(ubBinsLength);

    tilingData.set_formerNum(formerNum);
    tilingData.set_formerLength(formerLength);
    tilingData.set_formerLengthAligned(formerLengthAligned);
    tilingData.set_tailLength(tailLength);
    tilingData.set_tailLengthAligned(tailLengthAligned);

    tilingData.set_formerTileNum(formerTileNum);
    tilingData.set_formerTileDataLength(formerTileDataLength);
    tilingData.set_formerTileLeftDataLength(formerTileLeftDataLength);
    tilingData.set_formerTileLeftDataLengthAligned(formerTileLeftDataLengthAligned);

    tilingData.set_tailTileNum(tailTileNum);
    tilingData.set_tailTileDataLength(tailTileDataLength);
    tilingData.set_tailTileLeftDataLength(tailTileLeftDataLength);
    tilingData.set_tailTileLeftDataLengthAligned(tailTileLeftDataLengthAligned);

    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

void HistogramV2Tiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext, "tilingKey: %lu.", tilingContext->GetTilingKey());
    OP_LOGD(tilingContext, "coreNum: %ld.", coreNum);
    OP_LOGD(tilingContext, "totalLength: %ld.", totalLength);
    OP_LOGD(tilingContext, "bins: %ld.", bins);
    OP_LOGD(tilingContext, "formerNum: %ld.", formerNum);
    OP_LOGD(tilingContext, "formerLength: %ld.", formerLength);
    OP_LOGD(tilingContext, "formerLengthAligned: %ld.", formerLengthAligned);
    OP_LOGD(tilingContext, "tailLength: %ld.", tailLength);
    OP_LOGD(tilingContext, "tailLengthAligned: %ld.", tailLengthAligned);
    OP_LOGD(tilingContext, "formerTileNum: %ld.", formerTileNum);
    OP_LOGD(tilingContext, "formerTileDataLength: %ld.", formerTileDataLength);
    OP_LOGD(tilingContext, "formerTileLeftDataLength: %ld.", formerTileLeftDataLength);
    OP_LOGD(tilingContext, "formerTileLeftDataLengthAligned: %ld.", formerTileLeftDataLengthAligned);
    OP_LOGD(tilingContext, "tailTileNum: %ld.", tailTileNum);
    OP_LOGD(tilingContext, "tailTileDataLength: %ld.", tailTileDataLength);
    OP_LOGD(tilingContext, "tailTileLeftDataLength: %ld.", tailTileLeftDataLength);
    OP_LOGD(tilingContext, "tailTileLeftDataLengthAligned: %ld.", tailTileLeftDataLengthAligned);
}

class HistogramV2MembaseTiling : public HistogramV2BaseClass
{
public:
    explicit HistogramV2MembaseTiling(gert::TilingContext* context) : HistogramV2BaseClass(context) {};
    ~HistogramV2MembaseTiling() override = default;
    void Reset(gert::TilingContext* context) override
    {
        HistogramV2BaseClass::Reset(context);
    }

protected:
    bool IsCapable() override
    {
        if (Ops::Base::IsRegbaseSocVersion(context_)) {
            return false;
        }
        return true;
    }

    ge::graphStatus DoOpTiling() override
    {
        HistogramV2Tiling tilingObject(context_);
        tilingObject.Init();
        return tilingObject.SetKernelTiling();
    }

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
};

REGISTER_OPS_TILING_TEMPLATE(HistogramV2, HistogramV2MembaseTiling, 10000);
} // namespace optiling