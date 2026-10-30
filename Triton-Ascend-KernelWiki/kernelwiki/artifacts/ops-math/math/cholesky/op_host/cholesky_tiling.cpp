/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "cholesky_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "log/log.h"

namespace optiling {
constexpr uint32_t TILING_KEY_FALSE = 1;
constexpr uint32_t TILING_KEY_TRUE = 2;
constexpr uint32_t MINIMUM_DIMENSION = 2;
constexpr uint32_t UPPER_INDEX = 0;
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
constexpr uint32_t MAX_BLOCK_SIZE = 256;

class CholeskyTiling {
public:
    explicit CholeskyTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus Init();
    ge::graphStatus RunBigKernelTiling();

private:
    uint32_t GetTilingKeyVal() const;
    void PrintTilingData();

private:
    gert::TilingContext* tilingContext = nullptr;
    CholeskyTilingData tilingData;
    uint32_t matSizeN = 0;
    uint64_t matrixNumCount = 1;
    uint32_t needCoreNum = 0;
    bool upper = false;
    uint32_t blockSize = 0;
    uint32_t blockNum = 0;
};

ge::graphStatus CholeskyTiling::Init() {
    auto inputTensor = tilingContext->GetInputTensor(0);
    if (inputTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = tilingContext->GetAttrs();
    const bool* ptrUpper = attrs->GetAttrPointer<bool>(UPPER_INDEX);
    if (ptrUpper == nullptr) {
        return ge::GRAPH_FAILED;
    }
    upper = *ptrUpper;

    auto matAShape = tilingContext->GetInputShape(0)->GetOriginShape();
    uint32_t inputDim = static_cast<uint32_t>(matAShape.GetDimNum());
    if (inputDim < MINIMUM_DIMENSION) {
        return ge::GRAPH_FAILED;
    }
    matSizeN = static_cast<uint32_t>(matAShape[inputDim-1]);
    for (uint32_t i = 0; i < (inputDim - MINIMUM_DIMENSION); i++) {
        matrixNumCount = matrixNumCount * static_cast<uint64_t>(matAShape[i]);
    }

    if (matSizeN <= MAX_BLOCK_SIZE) {
        blockSize = matSizeN;
        blockNum = 1;
    } else {
        blockSize = MAX_BLOCK_SIZE;
        blockNum = (matSizeN + blockSize - 1) / blockSize;
    }

    auto compileInfo = reinterpret_cast<const CholeskyCompileInfo*>(tilingContext->GetCompileInfo());
    uint32_t coreNumPlatForm = compileInfo->coreNum;
    needCoreNum = coreNumPlatForm < matrixNumCount ? coreNumPlatForm : matrixNumCount;

    size_t* currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkSpace);
    currentWorkSpace[0] = WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CholeskyTiling::RunBigKernelTiling() {
    tilingContext->SetBlockDim(needCoreNum);
    tilingContext->SetTilingKey(GetTilingKeyVal());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    
    tilingData.set_matrixNumCount(matrixNumCount);
    tilingData.set_matSizeN(matSizeN);
    tilingData.set_blockSize(blockSize);
    tilingData.set_blockNum(blockNum);
    
    if (tilingContext->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint32_t CholeskyTiling::GetTilingKeyVal() const {
    if (upper == true) {
        return TILING_KEY_TRUE;
    } else {
        return TILING_KEY_FALSE;
    }
}

void CholeskyTiling::PrintTilingData() {
    OP_LOGD(tilingContext, "matSizeN: %u", matSizeN);
    OP_LOGD(tilingContext, "matrixNumCount: %lu", matrixNumCount);
    OP_LOGD(tilingContext, "blockSize: %u", blockSize);
    OP_LOGD(tilingContext, "blockNum: %u", blockNum);
}

static ge::graphStatus CholeskyTilingFunc(gert::TilingContext* context)
{
    CholeskyTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus tilingPrepareTiling(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<CholeskyCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0),
        OP_LOGE(context->GetNodeName(), "Cholesky GetHardwareInfo Failed, vectorCoreNum: %u", compileInfo->coreNum), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Cholesky)
    .Tiling(CholeskyTilingFunc)
    .TilingParse<CholeskyCompileInfo>(tilingPrepareTiling);
}