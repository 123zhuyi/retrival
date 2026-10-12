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
 * \file transform_bias_rescale_qkv_tiling.cpp
 * \brief transform_bias_rescale_qkv_tiling source file
 */

#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "log/log.h"
#include "transform_bias_rescale_qkv_tiling.h"

namespace optiling {

constexpr int32_t MAX_ELEMENT_NUM_EACH_CORE = 12 * 1024;

constexpr uint64_t TILING_KEY_DEFAULT = 1;

const static int32_t SIZE_16 = 16;
const static int32_t LENGTH_1024 = 1024;
const static int32_t NUM_THREE = 3;
constexpr int32_t MAX_BLOCK_CNT = 4095;
// 32字节对齐
constexpr int32_t DATA_BLOCK = 32;
constexpr int32_t HALF_SIZE = sizeof(float) / 2;

constexpr int32_t IDX_0 = 0;
constexpr int32_t IDX_1 = 1;
constexpr int32_t IDX_2 = 2;

std::string TAG = "TransformBiasRescaleQkvTiling";

struct TilingInfo {
    ge::DataType dataType;
    int64_t qkvShapeSize;
    int64_t batch;
    int64_t token;
    int64_t tripleDim;
    int64_t dim;
    int64_t numHeads;
    int64_t tilingKey;
    int64_t dataTypeSize;
    int64_t dimPerHead;
    uint32_t needCoreNum;
};

class TransformBiasRescaleQkvTiling
{
public:
    explicit TransformBiasRescaleQkvTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling();

private:
    gert::TilingContext* tilingContext = nullptr;
    TransformBiasRescaleQkvTilingData tilingData;
    TilingInfo tilingInfo = {ge::DT_UNDEFINED, 0, 0, 0, 0, 0, 0, 0, sizeof(float), 0, 0};

    const int32_t workspaceSize_ = SIZE_16 * LENGTH_1024 * LENGTH_1024;

    inline int64_t CeilA2B(const int64_t a, const int64_t b)
    {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    int64_t GetNeedCoreNum(const int64_t bt, const int32_t coreNumPlatform)
    {
        if (bt * NUM_THREE <= coreNumPlatform) {
            return NUM_THREE * bt;
        }
        return coreNumPlatform;
    }

    ge::graphStatus InitTilingData();

    ge::graphStatus Check();

    void SaveTilingData();
};

ge::graphStatus TransformBiasRescaleQkvTiling::InitTilingData()
{
    const gert::RuntimeAttrs* attrs = tilingContext->GetAttrs();

    if (attrs == nullptr || attrs->GetInt(IDX_0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingInfo.numHeads = *(attrs->GetInt(IDX_0));

    if (tilingContext->GetInputDesc(IDX_0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingInfo.dataType = tilingContext->GetInputDesc(IDX_0)->GetDataType();

    auto qkvInputShape = tilingContext->GetInputShape(IDX_0);
    if (qkvInputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tilingInfo.qkvShapeSize = qkvInputShape->GetOriginShape().GetShapeSize();
    tilingInfo.batch = qkvInputShape->GetOriginShape().GetDim(IDX_0);
    tilingInfo.token = qkvInputShape->GetOriginShape().GetDim(IDX_1);
    tilingInfo.tripleDim = qkvInputShape->GetOriginShape().GetDim(IDX_2);
    tilingInfo.dim = tilingInfo.tripleDim / NUM_THREE;
    tilingInfo.dimPerHead = tilingInfo.dim / tilingInfo.numHeads;
    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    tilingInfo.needCoreNum = GetNeedCoreNum(tilingInfo.batch * tilingInfo.token, platformInfo.GetCoreNumAiv());

    tilingInfo.tilingKey = TILING_KEY_DEFAULT;
    if (tilingInfo.dataType == ge::DT_FLOAT16) {
        tilingInfo.dataTypeSize = HALF_SIZE;
    } else if (tilingInfo.dataType == ge::DT_FLOAT) {
        tilingInfo.dataTypeSize = sizeof(float);
    } else if (tilingInfo.dataType == ge::DT_BF16) {
        tilingInfo.dataTypeSize = HALF_SIZE;
    } else {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransformBiasRescaleQkvTiling::Check()
{
    auto qkv = tilingContext->GetInputTensor(IDX_0);
    if (qkv == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto qkvBias = tilingContext->GetInputTensor(IDX_1);
    if (qkvBias == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo.numHeads <= 0) {
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo.dim % tilingInfo.numHeads != 0) {
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo.tripleDim % NUM_THREE != 0) {
        return ge::GRAPH_FAILED;
    }

    // 获取输入qkv_bias的shape
    auto qkvBiasInputShape = tilingContext->GetInputShape(1);
    const gert::Shape qkvBiasOriginShape = qkvBiasInputShape->GetOriginShape();
    const auto qkvBiasDimNum = qkvBiasOriginShape.GetDimNum();
    if (qkvBiasDimNum != 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t qkvBiasShapeSize = qkvBiasOriginShape.GetShapeSize();
    if (qkvBiasShapeSize != tilingInfo.tripleDim) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void TransformBiasRescaleQkvTiling::SaveTilingData()
{
    tilingContext->SetTilingKey(tilingInfo.tilingKey);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(IDX_1);
    currentWorkspace[IDX_0] = workspaceSize_;

    tilingData.set_qkvShapeSize(tilingInfo.qkvShapeSize);
    tilingData.set_needCoreNum(tilingInfo.needCoreNum);
    tilingData.set_batch(tilingInfo.batch);
    tilingData.set_token(tilingInfo.token);
    tilingData.set_dimension(tilingInfo.dim);
    tilingData.set_numHeads(tilingInfo.numHeads);
    tilingData.set_dimPerHead(tilingInfo.dimPerHead);
    tilingData.set_maxEleNumUB(MAX_ELEMENT_NUM_EACH_CORE);
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    tilingContext->SetBlockDim(tilingInfo.needCoreNum);
}

ge::graphStatus TransformBiasRescaleQkvTiling::RunBigKernelTiling()
{
    ge::graphStatus ret = InitTilingData();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = Check();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    SaveTilingData();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4TransformBiasRescaleQkvTiling(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4TransformBiasRescaleQkvTiling tiling end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingTransformBiasRescaleQkvTiling(gert::TilingContext* context)
{
    TransformBiasRescaleQkvTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(TransformBiasRescaleQkv)
    .Tiling(TilingTransformBiasRescaleQkvTiling)
    .TilingParse<TransformBiasRescaleQkvCompileInfo>(TilingPrepare4TransformBiasRescaleQkvTiling);
} // namespace optiling