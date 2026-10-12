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
 * \file hans_encode.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "log/log.h"
#include "hans_encode_tiling.h"

namespace optiling {

constexpr uint64_t TILING_KEY_HALF = 2;
constexpr uint64_t TILING_KEY_FLOAT = 4;
constexpr uint64_t TILING_KEY_BFLOAT16 = 2;
constexpr int64_t PROCESS_MIN_SIZE_PER_CORE = 32768;
constexpr int64_t ENCODE_META_INFO_BYTES = 512;
constexpr int64_t ENCODE_TAIL_INFO_BYTES_PER_CORE = 8448;
constexpr int64_t PROCESS_SIZE_PER_LOOP = 64;
constexpr int64_t PDF_NUMEL_LENGTH = 256;

static ge::graphStatus TilingPrepare4HansEncodeTiling([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

class HansEncodeTiling
{
public:
    explicit HansEncodeTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus ParamCheck();
    ge::graphStatus SetTilingData();

private:
    ge::DataType dataType = ge::DT_UNDEFINED;
    gert::TilingContext* tilingContext = nullptr;
    HansEncodeTilingData tilingData;
    uint64_t sysWorkspaceSize;
    int64_t aivNum;
    int64_t dtypeBytes;
    int64_t inputSize;
    int64_t pdfNumel = 1;
    int64_t fixedByteSize = 1;
    int64_t varByteSize = 1;
    int64_t mantissaSize = 1;
    int64_t processCoreDim;
    int64_t compressUpperBoundBytes;
    bool reshuff;
    bool statistic;

    inline int64_t GetProcessBlockDim(const int64_t dataSize, const int64_t maxUseAivNum)
    {
        int64_t properAivNum = dataSize / PROCESS_MIN_SIZE_PER_CORE;
        return properAivNum > maxUseAivNum ? maxUseAivNum : properAivNum;
    }

    inline int64_t GetSizeByStorageShape(const gert::StorageShape* shape, int64_t initValue)
    {
        for (int dim = 0; dim < static_cast<int>(shape->GetStorageShape().GetDimNum()); dim++) {
            initValue = initValue * shape->GetStorageShape().GetDim(dim);
        }
        return initValue;
    }
};

ge::graphStatus HansEncodeTiling::Init()
{
    OP_LOGD(tilingContext->GetNodeName(), "HansEncodeTiling tiling starts running");
    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    sysWorkspaceSize = platformInfo.GetLibApiWorkSpaceSize();
    aivNum = static_cast<int64_t>(platformInfo.GetCoreNumAiv());
    // check input
    auto inputTensor = tilingContext->GetInputTensor(0);
    OP_CHECK_IF(inputTensor == nullptr, OP_LOGE("HansEncode", "inputTensor is nullptr."), return ge::GRAPH_FAILED);
    auto pdfTensor = tilingContext->GetInputTensor(1);
    OP_CHECK_IF(pdfTensor == nullptr, OP_LOGE("HansEncode", "pdfTensor is nullptr."), return ge::GRAPH_FAILED);
    // check tiling
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_IF(inputDesc == nullptr, OP_LOGE("HansEncode", "inputDesc is nullptr."), return ge::GRAPH_FAILED);
    const gert::RuntimeAttrs* attrs = tilingContext->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("HansEncode", "attrs is nullptr."), return ge::GRAPH_FAILED);
    statistic = *attrs->GetAttrPointer<bool>(0);
    reshuff = *attrs->GetAttrPointer<bool>(1);
    OP_LOGD("HansEncode", "reshuff: %d statistic: %d", reshuff, statistic);
    const gert::StorageShape* pdfShape = tilingContext->GetInputShape(1);
    const gert::StorageShape* mantissaShape = tilingContext->GetOutputShape(1);
    const gert::StorageShape* fixedShape = tilingContext->GetOutputShape(2);
    const gert::StorageShape* varShape = tilingContext->GetOutputShape(3);
    dataType = inputDesc->GetDataType();
    dtypeBytes = GetSizeByDataType(dataType);
    inputSize = tilingContext->GetInputTensor(0)->GetShapeSize();
    OP_CHECK_IF(pdfShape == nullptr, OP_LOGE("HansEncode", "pdfShape is nullptr."), return ge::GRAPH_FAILED);
    pdfNumel = GetSizeByStorageShape(pdfShape, pdfNumel);
    OP_CHECK_IF(mantissaShape == nullptr, OP_LOGE("HansEncode", "mantissaShape is nullptr."), return ge::GRAPH_FAILED);
    mantissaSize = GetSizeByStorageShape(mantissaShape, dtypeBytes);
    OP_CHECK_IF(fixedShape == nullptr, OP_LOGE("HansEncode", "fixedShape is nullptr."), return ge::GRAPH_FAILED);
    fixedByteSize = GetSizeByStorageShape(fixedShape, dtypeBytes);
    OP_CHECK_IF(varShape == nullptr, OP_LOGE("HansEncode", "varShape is nullptr."), return ge::GRAPH_FAILED);
    varByteSize = GetSizeByStorageShape(varShape, dtypeBytes);
    processCoreDim = GetProcessBlockDim(inputSize, aivNum);
    compressUpperBoundBytes = inputSize + inputSize / PROCESS_SIZE_PER_LOOP +
                              ENCODE_TAIL_INFO_BYTES_PER_CORE * processCoreDim + ENCODE_META_INFO_BYTES;
    OP_LOGD(tilingContext->GetNodeName(), "HansEncodeTiling tiling end running.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HansEncodeTiling::ParamCheck()
{
    if (pdfNumel != PDF_NUMEL_LENGTH) {
        OP_LOGE(tilingContext->GetNodeType(), "pdf length must equal to 256.");
        return ge::GRAPH_FAILED;
    }
    if ((inputSize % PROCESS_SIZE_PER_LOOP != 0) || (inputSize < PROCESS_MIN_SIZE_PER_CORE)) {
        OP_LOGE(
            tilingContext->GetNodeType(),
            "The number of input tensors must be a multiple of 64 and greater than 32768.");
        return ge::GRAPH_FAILED;
    }
    if (mantissaSize != (dtypeBytes - 1) * inputSize) {
        OP_LOGE(tilingContext->GetNodeType(), "Insufficient size for mantissa.");
        return ge::GRAPH_FAILED;
    }
    if (reshuff) {
        // reshuff only support for addr1
        if (fixedByteSize < compressUpperBoundBytes) {
            OP_LOGE(
                tilingContext->GetNodeType(), "If reshuff, the space of fixed must be greater than the upper bound.");
            return ge::GRAPH_FAILED;
        } else {
            fixedByteSize = compressUpperBoundBytes;
        }
    }
    if (fixedByteSize < ENCODE_META_INFO_BYTES) {
        OP_LOGE(tilingContext->GetNodeType(), "The fixed space must be greater than 512.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HansEncodeTiling::SetTilingData()
{
    int64_t prepareOutputBytes = fixedByteSize + varByteSize;
    if (prepareOutputBytes < compressUpperBoundBytes) {
        return ge::GRAPH_FAILED;
    }
    uint64_t tilingKeyNum = 0;
    if (dataType == ge::DT_FLOAT16) {
        tilingKeyNum = TILING_KEY_HALF;
    } else if (dataType == ge::DT_FLOAT) {
        tilingKeyNum = TILING_KEY_FLOAT;
    } else if (dataType == ge::DT_BF16) {
        tilingKeyNum = TILING_KEY_BFLOAT16;
    } else {
        return ge::GRAPH_FAILED;
    }
    tilingContext->SetTilingKey(tilingKeyNum);
    int64_t outputBytes = fixedByteSize - ENCODE_META_INFO_BYTES;
    int64_t processBlockLoopNum = inputSize / PROCESS_SIZE_PER_LOOP;
    int64_t processLoopPerCore = processBlockLoopNum / processCoreDim;
    int64_t processLoopLastCore = processLoopPerCore + (processBlockLoopNum % processCoreDim);
    int64_t fixedLengthPerCore = outputBytes / processCoreDim;
    int64_t fixedLengthLastCore = fixedLengthPerCore + outputBytes % processCoreDim;
    uint64_t opWorkspaceSize = reshuff ? static_cast<uint64_t>(compressUpperBoundBytes) : 0;

    tilingData.set_processCoreDim(processCoreDim);
    tilingData.set_processLoopPerCore(processLoopPerCore);
    tilingData.set_processLoopLastCore(processLoopLastCore);
    tilingData.set_fixedLengthPerCore(fixedLengthPerCore);
    tilingData.set_fixedLengthLastCore(fixedLengthLastCore);
    tilingData.set_varLength(varByteSize);
    tilingData.set_statistic(statistic);
    tilingData.set_reshuff(reshuff);
    tilingContext->SetBlockDim(processCoreDim);
    OP_LOGD(tilingContext->GetNodeName(), "processCoreDim: %ld.", processCoreDim);
    OP_LOGD(tilingContext->GetNodeName(), "processLoopPerCore: %ld.", processLoopPerCore);
    OP_LOGD(tilingContext->GetNodeName(), "processLoopLastCore: %ld.", processLoopLastCore);
    OP_LOGD(tilingContext->GetNodeName(), "fixedLengthPerCore: %ld.", fixedLengthPerCore);
    OP_LOGD(tilingContext->GetNodeName(), "fixedLengthLastCore: %ld.", fixedLengthLastCore);
    OP_LOGD(tilingContext->GetNodeName(), "varByteSize: %ld.", varByteSize);
    OP_LOGD(tilingContext->GetNodeName(), "statistic: %d.", statistic);
    OP_LOGD(tilingContext->GetNodeName(), "reshuff: %d.", reshuff);
    OP_LOGD(tilingContext->GetNodeName(), "opWorkspaceSize: %lu.", opWorkspaceSize);
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize + opWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingHansEncodeTiling(gert::TilingContext* context)
{
    HansEncodeTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Init failed!");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.ParamCheck() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Check Param failed!");
        return ge::GRAPH_FAILED;
    }
    return tilingObject.SetTilingData();
}

IMPL_OP_OPTILING(HansEncode)
    .Tiling(TilingHansEncodeTiling)
    .TilingParse<HansEncodeCompileInfo>(TilingPrepare4HansEncodeTiling);

} // namespace optiling
