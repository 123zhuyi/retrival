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
 * \file pow_tensor_tensor_tiling_arch35.cpp
 * \brief
 */

#include "pow_tiling_arch35.h"
#include "tiling/math/power_tiling.h"
#include "op_host/math_tiling_templates_registry.h"

namespace optiling
{
using namespace ge;
using namespace Ops::Base;
static constexpr uint64_t OP_KEY_INVALID = 0;
static constexpr uint64_t OP_KEY_1 = 1;
static constexpr uint64_t OP_KEY_2 = 2;
static constexpr uint64_t OP_KEY_3 = 3;
static constexpr uint64_t OP_KEY_4 = 4;
static constexpr uint64_t OP_KEY_5 = 5;
static constexpr uint64_t OP_KEY_6 = 6;
static constexpr uint64_t OP_KEY_7 = 7;
static constexpr uint64_t INDEX_0 = 0;
static constexpr uint64_t INDEX_1 = 1;
static constexpr uint64_t INDEX_2 = 2;
static constexpr int64_t BYTE2BITS = 8;
static constexpr int64_t BUFFER_DIVISOR_SIZE = 2;

bool PowTensorTensorTiling::IsCapable()
{
    return true;
}

void PowTensorTensorTiling::SetOpKey()
{
    opKeys[{DT_FLOAT16, DT_FLOAT16, DT_FLOAT16}] = OP_KEY_1;
    opKeys[{DT_BF16, DT_BF16, DT_BF16}] = OP_KEY_2;
    opKeys[{DT_FLOAT, DT_FLOAT, DT_FLOAT}] = OP_KEY_3;
    opKeys[{DT_UINT8, DT_UINT8, DT_UINT8}] = OP_KEY_4;
    opKeys[{DT_INT8, DT_INT8, DT_INT8}] = OP_KEY_5;
    opKeys[{DT_INT16, DT_INT16, DT_INT16}] = OP_KEY_6;
    opKeys[{DT_INT32, DT_INT32, DT_INT32}] = OP_KEY_7;
}

uint64_t PowTensorTensorTiling::GetOpKey(ge::DataType inputXDtype, ge::DataType inputYDtype, ge::DataType outputZDtype)
{
    std::tuple<ge::DataType, ge::DataType, ge::DataType> condition = {inputXDtype, inputYDtype, outputZDtype};
    if (opKeys.find(condition) != opKeys.end()) {
        return opKeys[condition];
    }
    return OP_KEY_INVALID;
}

uint64_t PowTensorTensorTiling::GenerateTilingKey(uint64_t innerKey)
{
    return opKey_ * BROADCAST_OP_KEY_OFFSET + innerKey;
}

std::map<uint64_t, BroadcastComputeParams> PowTensorTensorTiling::GetComputeMap(uint64_t opKey)
{
    BroadcastComputeParams computeParams0;
    switch (opKey) {
        case OP_KEY_1:
            computeParams0.maxDtypeBits = static_cast<int64_t>(BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {96, 96};
            return {{1, computeParams0}};
        case OP_KEY_2:
            computeParams0.maxDtypeBits = static_cast<int64_t>(BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {96, 96};
            return {{1, computeParams0}};
        case OP_KEY_3:
            computeParams0.maxDtypeBits = static_cast<int64_t>(BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {192, 192};
            return {{1, computeParams0}};
        case OP_KEY_4:
            computeParams0.maxDtypeBits = static_cast<int64_t>(BROADCAST_BITS_SIZE::BITS8_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(BROADCAST_BITS_SIZE::BITS8_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {48, 48};
            return {{1, computeParams0}};
        case OP_KEY_5:
            computeParams0.maxDtypeBits = static_cast<int64_t>(BROADCAST_BITS_SIZE::BITS8_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(BROADCAST_BITS_SIZE::BITS8_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {48, 48};
            return {{1, computeParams0}};
        case OP_KEY_6:
            computeParams0.maxDtypeBits = static_cast<int64_t>(BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {96, 96};
            return {{1, computeParams0}};
        case OP_KEY_7:
            computeParams0.maxDtypeBits = static_cast<int64_t>(BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {192, 192};
            return {{1, computeParams0}};
        default:
            return {};
    }
}

ge::graphStatus PowTensorTensorTiling::DoOpTiling()
{
    SetOpKey();
    opKey_ = GetOpKey(params_.baseDtype, params_.expDtype, params_.powDtype);
    OP_CHECK_IF((opKey_ == OP_KEY_INVALID),
                    OP_LOGE(context_->GetNodeName(), "can not get opKey_"),
                    return ge::GRAPH_FAILED);

    BroadcastTilingParams broadcastTilingParams;
    for (uint64_t i = 0; i < context_->GetComputeNodeInputNum(); i++) {
        auto shape = context_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, shape);
        broadcastTilingParams.inShape.push_back(shape->GetStorageShape());
    }
    broadcastTilingParams.outShape = context_->GetOutputShape(0)->GetStorageShape();
    broadcastTilingParams.computeMap = GetComputeMap(opKey_);
    broadcastTilingParams.coreNum = params_.coreNum;
    broadcastTilingParams.ubSize = params_.ubSize;
    broadcastTilingParams.preferMultiCore = true;

    uint32_t powApiNode;
    uint32_t powApiTmpBuffer;
    bool isTypeInt = ((params_.baseDtype == ge::DT_UINT8)
                      || (params_.baseDtype == ge::DT_INT8)
                      || (params_.baseDtype == ge::DT_INT16)
                      || (params_.baseDtype == ge::DT_INT32));
    uint32_t baseTypeSize = static_cast<uint32_t>(params_.baseDtypeSize);
    AscendC::GetPowerTmpBufferFactorSize(true, true, isTypeInt, baseTypeSize, powApiNode, powApiTmpBuffer);
    OP_LOGD(context_->GetNodeName(),
            "Pow GetPowerTmpBufferFactorSize, isTypeInt:%d, baseTypeSize:%u, powApiNode:%u, powApiTmpBuffer:%u",
            isTypeInt, baseTypeSize, powApiNode, powApiTmpBuffer);
    broadcastTilingParams.ubSize = broadcastTilingParams.ubSize - static_cast<int64_t>(powApiTmpBuffer);

    if ((broadcastTilingParams.computeMap.count(1) > 0)
        && (broadcastTilingParams.computeMap[1].bufferDivisor.size() == BUFFER_DIVISOR_SIZE))
    {
        int64_t bufferDivisorApiValue = static_cast<int64_t>(powApiNode) * params_.baseDtypeSize * BYTE2BITS;
        broadcastTilingParams.computeMap[1].bufferDivisor[0]
            = broadcastTilingParams.computeMap[1].bufferDivisor[0] + bufferDivisorApiValue;
        broadcastTilingParams.computeMap[1].bufferDivisor[1]
            = broadcastTilingParams.computeMap[1].bufferDivisor[1] + bufferDivisorApiValue;
    } else {
        OP_LOGE(context_->GetNodeName(), "broadcast tiling get computeMap faild.");
        return ge::GRAPH_FAILED;
    }

    BroadcastTilingData broadcastTilingData;
    ge::graphStatus status = BroadcastTiling(broadcastTilingParams, broadcastTilingData);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "broadcast tiling failed.");
        return ge::GRAPH_FAILED;
    }

    tilingKey_ = GenerateTilingKey(broadcastTilingData.innerKey);
    blockNum = broadcastTilingData.blockNum;
    tilingData_.set_blockFormer(broadcastTilingData.blockFormer);
    tilingData_.set_ubFormer(broadcastTilingData.ubFormer);
    tilingData_.set_ubOuter(broadcastTilingData.ubOuter);
    tilingData_.set_ubTail(broadcastTilingData.ubTail);
    tilingData_.set_blockTail(broadcastTilingData.blockTail);
    tilingData_.set_shapeLen(broadcastTilingData.shapeLen);
    tilingData_.set_ubSplitAxis(broadcastTilingData.ubSplitAxis);
    tilingData_.set_dimProductBeforeUbInner(broadcastTilingData.dimProductBeforeUbInner);
    tilingData_.set_elemNum(broadcastTilingData.elemNum);

    std::copy(broadcastTilingData.dims[INDEX_0].begin(), broadcastTilingData.dims[INDEX_0].end(), input0Dims);
    tilingData_.set_input0Dims(input0Dims);
    std::copy(broadcastTilingData.dims[INDEX_1].begin(), broadcastTilingData.dims[INDEX_1].end(), input1Dims);
    tilingData_.set_input1Dims(input1Dims);
    std::copy(broadcastTilingData.dims[INDEX_2].begin(), broadcastTilingData.dims[INDEX_2].end(), outputDims);
    tilingData_.set_outputDims(outputDims);
    std::copy(broadcastTilingData.strides[INDEX_0].begin(), broadcastTilingData.strides[INDEX_0].end(), input0Strides);
    tilingData_.set_input0Strides(input0Strides);
    std::copy(broadcastTilingData.strides[INDEX_1].begin(), broadcastTilingData.strides[INDEX_1].end(), input1Strides);
    tilingData_.set_input1Strides(input1Strides);
    std::copy(broadcastTilingData.strides[INDEX_2].begin(), broadcastTilingData.strides[INDEX_2].end(), outputStrides);
    tilingData_.set_outputStrides(outputStrides);

    return ge::GRAPH_SUCCESS;
}

uint64_t PowTensorTensorTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus PowTensorTensorTiling::PostTiling()
{
    context_->SetBlockDim(blockNum);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(Pow, PowTensorTensorTiling, 1);
}  // namespace optiling