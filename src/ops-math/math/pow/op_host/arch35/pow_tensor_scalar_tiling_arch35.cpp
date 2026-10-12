/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file pow_tensor_scalar_tiling_arch35.cpp
 * \brief
 */

#include "pow_tiling_arch35.h"
#include "tiling/math/power_tiling.h"
#include "op_host/math_tiling_templates_registry.h"

namespace optiling
{
constexpr int64_t BUFFER_ALIGN = 512;
constexpr uint64_t ASCENDC_API_TENSOR_SCALAR_FLOAT16_PATTERN = 1001;
constexpr uint64_t ASCENDC_API_TENSOR_SCALAR_BFLOAT16_PATTERN = 2001;
constexpr uint64_t ASCENDC_API_TENSOR_SCALAR_FLOAT32_PATTERN = 3001;
constexpr uint64_t ASCENDC_API_TENSOR_SCALAR_UINT8_PATTERN = 4001;
constexpr uint64_t ASCENDC_API_TENSOR_SCALAR_INT8_PATTERN = 5001;
constexpr uint64_t ASCENDC_API_TENSOR_SCALAR_INT16_PATTERN = 6001;
constexpr uint64_t ASCENDC_API_TENSOR_SCALAR_INT32_PATTERN = 7001;
constexpr int64_t POWS_MIN_TILING_BITS_SIZE_PER_CORE = 4096;

bool PowTensorScalarTiling::IsCapable()
{
    int64_t expShapeDims = params_.expShape.GetDimNum();
    int64_t expSize = 1;
    for (int64_t i = 0; i < expShapeDims; i++) {
        expSize = expSize * params_.expShape.GetDim(i);
    }
    return expSize == 1;
}

uint64_t PowTensorScalarTiling::GetTilingKey() const
{
    if (params_.baseDtype == ge::DT_FLOAT) {
        return ASCENDC_API_TENSOR_SCALAR_FLOAT32_PATTERN;
    } else if (params_.baseDtype == ge::DT_BF16) {
        return ASCENDC_API_TENSOR_SCALAR_BFLOAT16_PATTERN;
    } else if (params_.baseDtype == ge::DT_UINT8) {
        return ASCENDC_API_TENSOR_SCALAR_UINT8_PATTERN;
    } else if (params_.baseDtype == ge::DT_INT8) {
        return ASCENDC_API_TENSOR_SCALAR_INT8_PATTERN;
    } else if (params_.baseDtype == ge::DT_INT16) {
        return ASCENDC_API_TENSOR_SCALAR_INT16_PATTERN;
    } else if (params_.baseDtype == ge::DT_INT32) {
        return ASCENDC_API_TENSOR_SCALAR_INT32_PATTERN;
    }
    return ASCENDC_API_TENSOR_SCALAR_FLOAT16_PATTERN;
}

int64_t PowTensorScalarTiling::GetUbFactor(int64_t dim0, int64_t oriUbFactor) const
{
    int64_t blockFactor = (dim0 + params_.coreNum - 1) / params_.coreNum;
    if (blockFactor >= oriUbFactor) {
        return oriUbFactor;
    }
    int64_t refineUbFactor = std::max(blockFactor, POWS_MIN_TILING_BITS_SIZE_PER_CORE / params_.baseDtypeSize);
    refineUbFactor = std::min(refineUbFactor, oriUbFactor);
    int64_t bufferAlignElem = BUFFER_ALIGN / params_.computeDtypeSize;
    if (refineUbFactor >= bufferAlignElem) {
        refineUbFactor = refineUbFactor / bufferAlignElem * bufferAlignElem;
    }

    return refineUbFactor;
}

ge::graphStatus PowTensorScalarTiling::DoOpTiling()
{
    // compute graph node max value 5
    int64_t graphNode = 5;
    uint32_t powsApiNode;
    uint32_t powsApiTmpBuffer;
    bool isTypeInt = ((params_.baseDtype == ge::DT_UINT8)
                      || (params_.baseDtype == ge::DT_INT8)
                      || (params_.baseDtype == ge::DT_INT16)
                      || (params_.baseDtype == ge::DT_INT32));
    uint32_t baseTypeSize = static_cast<uint32_t>(params_.computeDtypeSize);
    AscendC::GetPowerTmpBufferFactorSize(true, false, isTypeInt, baseTypeSize, powsApiNode, powsApiTmpBuffer);
    OP_LOGD(context_->GetNodeName(),
            "Pows GetPowerTmpBufferFactorSize, isTypeInt:%d, baseTypeSize:%u, powsApiNode:%u, powsApiTmpBuffer:%u",
            isTypeInt, baseTypeSize, powsApiNode, powsApiTmpBuffer);
    // power tmpBuffer ：api buffer 8k + expQue(256) + expTmp(256)
    int64_t powerApiTmpBuffer = 8 * 1024 + 256 + 256;
    powerApiTmpBuffer = powerApiTmpBuffer + static_cast<int64_t>(powsApiTmpBuffer);
    graphNode = graphNode + static_cast<int64_t>(powsApiNode);
    // base counter
    int64_t baseSize = 1;
    int64_t baseShapeDims = params_.baseShape.GetDimNum();
    for (int64_t i = 0; i < baseShapeDims; i++) {
        baseSize = baseSize * params_.baseShape.GetDim(i);
    }

    // ub tiling
    int64_t ubSize = (params_.ubSize - powerApiTmpBuffer) / graphNode;
    if (ubSize >= BUFFER_ALIGN) {
        ubSize = ubSize / BUFFER_ALIGN * BUFFER_ALIGN;
    } else if (ubSize >= params_.blockSize) {
        ubSize = ubSize / params_.blockSize * params_.blockSize;
    }
    int64_t ubFactor = GetUbFactor(baseSize, ubSize / params_.computeDtypeSize);
    int64_t ubTail = baseSize % ubFactor;
    if (ubTail == 0) {
        ubTail = ubFactor;
    }
    int64_t ubOuter = (baseSize + ubFactor - 1) / ubFactor;

    // block tiling
    int64_t blockFactor = (ubOuter + params_.coreNum - 1) / params_.coreNum;
    int64_t blockTail = ubOuter % blockFactor;
    if (blockTail == 0) {
        blockTail = blockFactor;
    }
    int64_t blockNum = (ubOuter + blockFactor - 1) / blockFactor;

    // set tilingdata
    tilingData_.set_blockNum(blockNum);
    tilingData_.set_blockFactor(blockFactor);
    tilingData_.set_blockTail(blockTail);
    tilingData_.set_ubOuter(ubOuter);
    tilingData_.set_ubFactor(ubFactor);
    tilingData_.set_ubTail(ubTail);
    tilingData_.set_totalSize(baseSize);
    tilingData_.set_ubSize(ubSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PowTensorScalarTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.get_blockNum());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(Pow, PowTensorScalarTiling, 0);
}  // namespace optiling