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
 * \file hans_decode_tiling.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "log/log.h"
#include "hans_decode_tiling.h"

namespace optiling {

constexpr uint64_t TILING_KEY_HALF = 2;
constexpr uint64_t TILING_KEY_FLOAT = 4;
constexpr uint64_t TILING_KEY_BFLOAT16 = 2;
constexpr int64_t PDF_NUMEL_LENGTH = 256;

static ge::graphStatus TilingPrepare4HansDecodeTiling([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

class HansDecodeTiling
{
public:
    explicit HansDecodeTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus SetTilingData();

private:
    ge::DataType dataType = ge::DT_UNDEFINED;
    gert::TilingContext* tilingContext = nullptr;
    gert::Shape inputShape;
    HansDecodeTilingData tilingData;
    uint64_t sysWorkspaceSize;
    uint64_t aivNum;
    int64_t dtypeBytes;
    int64_t inputSize;
    int64_t pdfNumel;
    int64_t fixedSize;
    int64_t mantissaSize;
    int64_t processCoreDim;
    int64_t recoverSize;
    bool reshuff;

    inline int64_t GetSizeByStorageShape(const gert::StorageShape* shape, int64_t initValue)
    {
        for (int dim = 0; dim < static_cast<int>(shape->GetStorageShape().GetDimNum()); dim++) {
            initValue = initValue * shape->GetStorageShape().GetDim(dim);
        }
        return initValue;
    }
};

ge::graphStatus HansDecodeTiling::Init()
{
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    aivNum = ascendcPlatform.GetCoreNumAiv();
    sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // check input
    auto mantissa = tilingContext->GetInputTensor(0);
    OP_CHECK_IF(mantissa == nullptr, OP_LOGE("HansDecode", "mantissa is nullptr."), return ge::GRAPH_FAILED);
    auto fixed = tilingContext->GetInputTensor(1);
    OP_CHECK_IF(fixed == nullptr, OP_LOGE("HansDecode", "fixed is nullptr."), return ge::GRAPH_FAILED);
    auto var = tilingContext->GetInputTensor(2);
    OP_CHECK_IF(var == nullptr, OP_LOGE("HansDecode", "var is nullptr."), return ge::GRAPH_FAILED);
    auto pdf = tilingContext->GetInputTensor(3);
    OP_CHECK_IF(pdf == nullptr, OP_LOGE("HansDecode", "pdf is nullptr."), return ge::GRAPH_FAILED);
    const gert::RuntimeAttrs* attrs = tilingContext->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("HansDecode", "attrs is nullptr."), return ge::GRAPH_FAILED);
    reshuff = *attrs->GetAttrPointer<bool>(0);
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_IF(inputDesc == nullptr, OP_LOGE("HansDecode", "inputDesc is nullptr."), return ge::GRAPH_FAILED);
    dataType = tilingContext->GetInputDesc(0)->GetDataType();
    dtypeBytes = GetSizeByDataType(dataType);
    mantissaSize = tilingContext->GetInputTensor(0)->GetShapeSize();
    fixedSize = tilingContext->GetInputTensor(1)->GetShapeSize();
    const gert::StorageShape* recoverShape = tilingContext->GetOutputShape(0);
    OP_CHECK_IF(recoverShape == nullptr, OP_LOGE("HansDecode", "recoverShape is nullptr."), return ge::GRAPH_FAILED);
    recoverSize = GetSizeByStorageShape(recoverShape, 1);
    const gert::StorageShape* pdfShape = tilingContext->GetInputShape(3);
    OP_CHECK_IF(pdfShape == nullptr, OP_LOGE("HansDecode", "pdfShape is nullptr."), return ge::GRAPH_FAILED);
    pdfNumel = GetSizeByStorageShape(pdfShape, 1);
    if (mantissaSize != (dtypeBytes - 1) * recoverSize / dtypeBytes) {
        OP_LOGE(tilingContext, "Insufficient size for mantissa.");
        return ge::GRAPH_FAILED;
    }
    if (pdfNumel != PDF_NUMEL_LENGTH) {
        OP_LOGE(tilingContext, "pdf length must equal to 256.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HansDecodeTiling::SetTilingData()
{
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = TILING_KEY_HALF;
    } else if (dataType == ge::DT_FLOAT) {
        tilingKey = TILING_KEY_FLOAT;
    } else if (dataType == ge::DT_BF16) {
        tilingKey = TILING_KEY_BFLOAT16;
    } else {
        return ge::GRAPH_FAILED;
    }
    tilingContext->SetTilingKey(tilingKey);
    recoverSize = recoverSize * dtypeBytes;
    tilingData.set_mantissaByteSize(mantissaSize * dtypeBytes);
    tilingData.set_fixedByteSize(fixedSize * dtypeBytes);
    tilingData.set_recoverExpByteSize(recoverSize);
    tilingData.set_recoverByteSize(recoverSize * dtypeBytes);
    tilingData.set_reshuff(reshuff);

    OP_LOGD(tilingContext->GetNodeName(), "tilingKey: %lu.", tilingKey);
    OP_LOGD(tilingContext->GetNodeName(), "aivNum: %lu.", aivNum);
    OP_LOGD(tilingContext->GetNodeName(), "mantissaByteSize: %ld.", mantissaSize * dtypeBytes);
    OP_LOGD(tilingContext->GetNodeName(), "fixedByteSize: %ld.", fixedSize * dtypeBytes);
    OP_LOGD(tilingContext->GetNodeName(), "recoverExpByteSize: %ld.", recoverSize);
    OP_LOGD(tilingContext->GetNodeName(), "recoverByteSize: %ld.", recoverSize * dtypeBytes);
    OP_LOGD(tilingContext->GetNodeName(), "reshuff: %d.", reshuff);
    tilingContext->SetBlockDim(aivNum);
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingHansDecodeTiling(gert::TilingContext* context)
{
    HansDecodeTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Init failed!");
        return ge::GRAPH_FAILED;
    }
    return tilingObject.SetTilingData();
}

IMPL_OP_OPTILING(HansDecode)
    .Tiling(TilingHansDecodeTiling)
    .TilingParse<HansDecodeCompileInfo>(TilingPrepare4HansDecodeTiling);
} // namespace optiling
