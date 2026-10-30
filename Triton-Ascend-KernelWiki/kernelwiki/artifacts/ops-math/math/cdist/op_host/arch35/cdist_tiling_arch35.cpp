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
 * \file cdist_tiling.cpp
 * \brief
 */

#include <array>
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_base_util.h"
#include "op_host/math_tiling_templates_registry.h"
#include "cdist_tiling_arch35.h"

namespace optiling {
ge::graphStatus CdistTiling::CheckParams() {
    OP_LOGD(tilingContext_->GetNodeName(), "Start CheckParams.");
    auto x1 = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, x1);
    auto x1Desc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, x1Desc);
    ge::DataType dtype = x1Desc->GetDataType();
    dtypeSize_ = ge::GetSizeByDataType(dtype);
    auto x2 = tilingContext_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, x2);
    auto y = tilingContext_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, y);
    const gert::Shape& x1Shape = x1->GetStorageShape();
    const gert::Shape& x2Shape = x2->GetStorageShape();
    const gert::Shape& yShape = y->GetStorageShape();
    x1Shape_ = x1Shape;
    x2Shape_ = x2Shape;
    yShape_ = yShape;
    int64_t dimNum = x1Shape_.GetDimNum();
    OP_CHECK_IF(dimNum < MIN_DIM_LEN,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    tilingContext_->GetNodeName(), "x1", std::to_string(dimNum).c_str(),
                    "Input only supports at least 2D tensors"),
                return ge::GRAPH_FAILED);
    int64_t x2DimNum = x2Shape_.GetDimNum();
    if (x2DimNum != dimNum) {
        std::string reasonMsg = "The dim num of x1 and x2 must be the same, x1 got: " +
                                std::to_string(dimNum) + " ,x2 got: " + std::to_string(x2DimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    tilingContext_->GetNodeName(), "x2", std::to_string(x2DimNum).c_str(),
                    reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t yDimNum = yShape_.GetDimNum();
    if (yDimNum != dimNum) {
        std::string reasonMsg = "The dim num of input and output must be the same, x1 got: " +
                                std::to_string(dimNum) + " ,y got: " + std::to_string(yDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            tilingContext_->GetNodeName(), "y", std::to_string(yDimNum).c_str(),
            reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t M1 = x1Shape_.GetDim(dimNum - 1);
    int64_t M2 = x2Shape_.GetDim(dimNum - 1);
    if (M1 != M2) {
        std::string reasonMsg = "The last dim of x1 and x2 must be the same, x1 got: " +
                                std::to_string(M1) + " ,x2 got: " + std::to_string(M2);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            tilingContext_->GetNodeName(), "x2", std::to_string(M2).c_str(),
            reasonMsg.c_str());
            return ge::GRAPH_FAILED;
    }
    M_ = M1;
    P_ = x1Shape_.GetDim(dimNum - MIN_DIM_LEN);
    R_ = x2Shape_.GetDim(dimNum - MIN_DIM_LEN);
    if (P_ != yShape_.GetDim(dimNum - MIN_DIM_LEN) || R_ != yShape_.GetDim(dimNum - 1)) {
        std::string reasonMsg = "The last two dims of output are incorrect, output[-1] got: " +
                                std::to_string(yShape_.GetDim(dimNum - 1)) +
                                " ,output[-2] got: " + std::to_string(yShape_.GetDim(dimNum - MIN_DIM_LEN));
        std:: string errDimMsg = "output[-1]: " + std::to_string(yShape_.GetDim(dimNum - 1)) +
                                 " ,output[-2]: " + std::to_string(yShape_.GetDim(dimNum - MIN_DIM_LEN)); 
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            tilingContext_->GetNodeName(), "y", errDimMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);
    tilingData_.p = 2.0f;
    if (attrs->GetAttrNum() > 0) {
        const float* pAttr = attrs->GetAttrPointer<float>(0);
        tilingData_.p = pAttr == nullptr ? 2.0f : *pAttr;
        OP_CHECK_IF(tilingData_.p < 0,
            OP_LOGE_WITH_INVALID_ATTR(tilingContext_->GetNodeName(), "p",
                std::to_string(tilingData_.p).c_str(),
                "greater than or equal to 0"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CdistTiling::MergeBatchAxis() {
    OP_LOGD(tilingContext_->GetNodeName(), "Start MergeBatchAxis.");
    int64_t dimNum = x1Shape_.GetDimNum();
    B_ = 1;
    int64_t x2B = 1;
    int64_t yB = 1;
    for (int64_t i = 0; i < dimNum - MIN_DIM_LEN; i++) {
        B_ *= x1Shape_.GetDim(i);
        x2B *= x2Shape_.GetDim(i);
        yB *= yShape_.GetDim(i);
    }
    if (B_ != x2B || B_ != yB) {
        std::string reasonMsg = "The batch of input and output must be the same, but x1 got: " +
            std::to_string(B_) + ", x2 got: " + std::to_string(B_) + ", y got: " + std::to_string(yB);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            tilingContext_->GetNodeName(), "y", std::to_string(yB).c_str(),
            reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void CdistTiling::DoSimtTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start DoSimtTiling.");
    int64_t totalElements = B_ * P_ * R_;
    int64_t minPerCoreElement = SIMT_MIN_BYTE / dtypeSize_;
    if (totalElements <= minPerCoreElement) {
        tilingData_.realCoreNum = 1;
        tilingData_.blockFactor = totalElements;
        tilingData_.blockTailFactor = totalElements;
        return;
    }
    else {
        int64_t minRequiredCores = Ops::Base::CeilDiv(totalElements, minPerCoreElement);
        int64_t usedCoreNum = std::min(coreNum_, minRequiredCores);
        if (usedCoreNum == 0) {
            return;
        }
        int64_t perCoreElement = totalElements / usedCoreNum;
        int64_t tailCoreElement = totalElements - perCoreElement * (usedCoreNum - 1);
        if (perCoreElement < minPerCoreElement) {
            perCoreElement = minPerCoreElement;
            tailCoreElement = totalElements - perCoreElement * (usedCoreNum - 1);
        }
        tilingData_.realCoreNum = usedCoreNum;
        tilingData_.blockFactor = perCoreElement;
        tilingData_.blockTailFactor = tailCoreElement;
    }
}

void CdistTiling::SetDefaultBlockTiling()
{
    tilingData_.blockMainNumB = B_;
    tilingData_.blockMainFactorB = 1;
    tilingData_.blockMainNumP = P_;
    tilingData_.blockMainFactorP = 1;
    tilingData_.blockMainNumR = R_;
    tilingData_.blockMainFactorR = 1;
}

void CdistTiling::DoNormalBlockTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start DoNormalBlockTiling.");
    // block tiling: B -> P -> R
    SetDefaultBlockTiling();
    int64_t currentCoreNum = coreNum_;
    int64_t remainingCoreNum = currentCoreNum / B_;
    if (currentCoreNum < B_) {
        tilingData_.blockMainNumB = B_ % currentCoreNum == 0 ? currentCoreNum : B_ % currentCoreNum;
        tilingData_.blockTailNumB = currentCoreNum - tilingData_.blockMainNumB;
        tilingData_.blockMainFactorB = tilingData_.blockTailNumB == 0 ? B_ / currentCoreNum : B_ / currentCoreNum + 1;
        tilingData_.blockTailFactorB = tilingData_.blockTailNumB == 0 ? 0 : tilingData_.blockMainFactorB - 1;
    }
    if (remainingCoreNum <= 1) {
        tilingData_.blockMainNumP = 1;
        tilingData_.blockMainFactorP = P_;
        tilingData_.blockMainNumR = 1;
        tilingData_.blockMainFactorR = R_;
        return;
    }
    currentCoreNum = remainingCoreNum;
    remainingCoreNum = currentCoreNum / P_;
    if (currentCoreNum < P_) {
        tilingData_.blockMainNumP = P_ % currentCoreNum == 0 ? currentCoreNum : P_ % currentCoreNum;
        tilingData_.blockTailNumP = currentCoreNum - tilingData_.blockMainNumP;
        tilingData_.blockMainFactorP = tilingData_.blockTailNumP == 0 ? P_ / currentCoreNum : P_ / currentCoreNum + 1;
        tilingData_.blockTailFactorP = tilingData_.blockTailNumP == 0 ? 0 : tilingData_.blockMainFactorP - 1;
    }
    if (remainingCoreNum <= 1) {
        tilingData_.blockMainNumR = 1;
        tilingData_.blockMainFactorR = R_;
        return;
    }
    currentCoreNum = remainingCoreNum;
    remainingCoreNum = currentCoreNum / R_;
    if (currentCoreNum < R_) {
        tilingData_.blockMainNumR = R_ % currentCoreNum == 0 ? currentCoreNum : R_ % currentCoreNum;
        tilingData_.blockTailNumR = currentCoreNum - tilingData_.blockMainNumR;
        tilingData_.blockMainFactorR = tilingData_.blockTailNumR == 0 ? R_ / currentCoreNum : R_ / currentCoreNum + 1;
        tilingData_.blockTailFactorR = tilingData_.blockTailNumR == 0 ? 0 : tilingData_.blockMainFactorR - 1;
    }
}

void CdistTiling::SetDefaultUbTiling()
{
    tilingData_.ubLoopNumM = 1;
    tilingData_.ubFactorM = M_;
    tilingData_.ubTailFactorM = 0;
    tilingData_.ubLoopNumR = 1;
    tilingData_.ubFactorR = tilingData_.blockMainFactorR;
    tilingData_.ubTailFactorR = 0;
    tilingData_.ubLoopNumP = 1;
    tilingData_.ubFactorP = tilingData_.blockMainFactorP;
    tilingData_.ubTailFactorP = 0;
    tilingData_.ubLoopNumB = 1;
    tilingData_.ubFactorB = tilingData_.blockMainFactorB;
    tilingData_.ubTailFactorB = 0;
}

void CdistTiling::ProcessDimension(const DimConfig& config, int64_t availableUbElements, int64_t& findUbTilingIdx)
{   
    findUbTilingIdx++;
    int64_t totalElements = config.calcTotalElements(config.baseValue);
    if (totalElements > availableUbElements) {
        for (int64_t i = config.baseValue - 1; i >= 1; i--) {
            int64_t splitElements = config.calcTotalElements(i);
            if (splitElements <= availableUbElements) {
                SetDefaultUbTiling();
                *config.factor = i;
                *config.loopNum = Ops::Base::CeilDiv(config.baseValue, i);
                *config.tailFactor = config.baseValue % i;
                notFoundUbTilingAxis_ = 0;
                break;
            }
        }
    }
}

void CdistTiling::DoNormalUbTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start DoNormalUbTiling.");
    // Ub tiling: M -> R0 -> P0 -> B0
    int64_t availableUbElements = ubSize_ / dtypeSize_;
    int64_t blockElements = BLOCK_BYTES / dtypeSize_;
    int64_t B0 = tilingData_.blockMainFactorB;
    int64_t P0 = tilingData_.blockMainFactorP;
    int64_t R0 = tilingData_.blockMainFactorR;
    int64_t MBlockAlign = Ops::Base::CeilAlign(M_ * dtypeSize_, BLOCK_BYTES) / dtypeSize_;
    int64_t R0BlockAlign = Ops::Base::CeilAlign(R0 * dtypeSize_, BLOCK_BYTES) / dtypeSize_;

    std::vector<DimConfig> configs;
    if (dtypeSize_ == B4) {
        configs = {
            {&tilingData_.ubLoopNumM, &tilingData_.ubFactorM, &tilingData_.ubTailFactorM, M_,
            [this, blockElements](int64_t i)
                {
                    int64_t iBlockAlign = Ops::Base::CeilAlign(i * dtypeSize_, BLOCK_BYTES) / dtypeSize_;
                    return BUFFER_NUM * (iBlockAlign + iBlockAlign + blockElements) +
                           blockElements;
                }
            },
            {&tilingData_.ubLoopNumR, &tilingData_.ubFactorR, &tilingData_.ubTailFactorR, R0,
            [this, MBlockAlign, blockElements](int64_t i)
                {
                    int64_t iBlockAlign = Ops::Base::CeilAlign(i * dtypeSize_, BLOCK_BYTES) / dtypeSize_;
                    return BUFFER_NUM * (MBlockAlign + i * MBlockAlign + iBlockAlign) +
                           blockElements;
                }
            },
            {&tilingData_.ubLoopNumP, &tilingData_.ubFactorP, &tilingData_.ubTailFactorP, P0,
            [this, MBlockAlign, R0BlockAlign, R0, blockElements](int64_t i)
                {
                    return BUFFER_NUM * (i * MBlockAlign + R0 * MBlockAlign + i * R0BlockAlign) +
                           blockElements;
                }
            },
            {&tilingData_.ubLoopNumB, &tilingData_.ubFactorB, &tilingData_.ubTailFactorB, B0,
            [this, MBlockAlign, R0BlockAlign, R0, P0, blockElements](int64_t i)
                {   
                    return BUFFER_NUM * (i * P0 * MBlockAlign + i * R0 * MBlockAlign + i * P0 * R0BlockAlign) +
                           blockElements;
                }
            }
        };
    } else {
        configs = {
            {&tilingData_.ubLoopNumM, &tilingData_.ubFactorM, &tilingData_.ubTailFactorM, M_,
            [this, blockElements](int64_t i)
                {
                    int64_t iBlockAlign = Ops::Base::CeilAlign(i * dtypeSize_, BLOCK_BYTES) / dtypeSize_;
                    return BUFFER_NUM * (iBlockAlign + iBlockAlign + blockElements) +
                           CAST_BUFFER_RATIO * (iBlockAlign + iBlockAlign + blockElements) +
                           blockElements;
                }
            },
            {&tilingData_.ubLoopNumR, &tilingData_.ubFactorR, &tilingData_.ubTailFactorR, R0,
            [this, MBlockAlign, blockElements](int64_t i)
                {
                    int64_t iBlockAlign = Ops::Base::CeilAlign(i * dtypeSize_, BLOCK_BYTES) / dtypeSize_;
                    return BUFFER_NUM * (MBlockAlign + i * MBlockAlign + iBlockAlign) +
                           CAST_BUFFER_RATIO * (MBlockAlign + i * MBlockAlign + iBlockAlign) +
                           blockElements;
                }
            },
            {&tilingData_.ubLoopNumP, &tilingData_.ubFactorP, &tilingData_.ubTailFactorP, P0,
            [this, MBlockAlign, R0BlockAlign, R0, blockElements](int64_t i)
                {
                    return BUFFER_NUM * (i * MBlockAlign + R0 * MBlockAlign + i * R0BlockAlign) +
                           CAST_BUFFER_RATIO * (i * MBlockAlign + R0 * MBlockAlign + i * R0BlockAlign) +
                           blockElements;
                }
            },
            {&tilingData_.ubLoopNumB, &tilingData_.ubFactorB, &tilingData_.ubTailFactorB, B0,
            [this, MBlockAlign, R0BlockAlign, R0, P0, blockElements](int64_t i)
                {   
                    return BUFFER_NUM * (i * P0 * MBlockAlign + i * R0 * MBlockAlign + i * P0 * R0BlockAlign) +
                           CAST_BUFFER_RATIO * (i * P0 * MBlockAlign + i * R0 * MBlockAlign + i * P0 * R0BlockAlign) +
                           blockElements;
                }
            }
        };
    }

    SetDefaultUbTiling();
    int64_t findUbTilingIdx = 0;
    for (auto& config : configs) {
        if (notFoundUbTilingAxis_) {
            ProcessDimension(config, availableUbElements, findUbTilingIdx);
        }
    }
    switch (findUbTilingIdx) {
        case 1:
            tilingData_.ubLoopNumR = tilingData_.blockMainFactorR;
            tilingData_.ubFactorR = 1;
            tilingData_.ubLoopNumP = tilingData_.blockMainFactorP;
            tilingData_.ubFactorP = 1;
            tilingData_.ubLoopNumB = tilingData_.blockMainFactorB;
            tilingData_.ubFactorB = 1;
            break;
        case 2:
            tilingData_.ubLoopNumP = tilingData_.blockMainFactorP;
            tilingData_.ubFactorP = 1;
            tilingData_.ubLoopNumB = tilingData_.blockMainFactorB;
            tilingData_.ubFactorB = 1;
            break;
        case 3:
            tilingData_.ubLoopNumB = tilingData_.blockMainFactorB;
            tilingData_.ubFactorB = 1;
            break;
        default:
            break;
    }
}

void CdistTiling::DoNormalTiling()
{   
    OP_LOGD(tilingContext_->GetNodeName(), "Start DoNormalTiling.");
    DoNormalBlockTiling();
    tilingData_.realCoreNum = (tilingData_.blockMainNumB + tilingData_.blockTailNumB) *
                              (tilingData_.blockMainNumP + tilingData_.blockTailNumP) *
                              (tilingData_.blockMainNumR + tilingData_.blockTailNumR);
    DoNormalUbTiling();
}

void CdistTiling::DoTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start DoTiling.");
    if (M_ <= M_SIZE) {
        is_small_m_ = 1;
        DoSimtTiling();
    } else {
        is_small_m_ = 0;
        DoNormalTiling();
    }
}

ge::graphStatus CdistTiling::RunCdistTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start RunCdistTiling.");
    OP_CHECK_IF(CheckParams() != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext_->GetNodeName(),
                "RunCdistTiling check params failed!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(MergeBatchAxis() != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext_->GetNodeName(),
                "RunCdistTiling merge batch axis failed!"),
                return ge::GRAPH_FAILED);
    DoTiling();
    OP_CHECK_IF(SetTilingData() != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext_->GetNodeName(),
                "RunCdistTiling failed to set tiling data!"),
                return ge::GRAPH_FAILED);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CdistTiling::Init()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start init CdistTiling.");
    auto compileInfo = reinterpret_cast<const CdistCompileInfo*>(tilingContext_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, compileInfo);
    coreNum_ = compileInfo->coreNum;
    OP_CHECK_IF((coreNum_ <= 0),
                OP_LOGE(tilingContext_->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF((ubSize_ <= 0),
                OP_LOGE(tilingContext_->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext_->GetNodeName(), "Init CdistTiling sucess.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CdistTiling::SetTilingData()
{   
    OP_LOGD(tilingContext_->GetNodeName(), "Start SetTilingData.");
    tilingData_.B = B_;
    tilingData_.P = P_;
    tilingData_.R = R_;
    tilingData_.M = M_;
    auto ptrTilingData = tilingContext_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, ptrTilingData);
    auto capSize = ptrTilingData->GetCapacity();
    void* ptrData = ptrTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, ptrData);
    void* ptrStruct = static_cast<void*>(&tilingData_);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, ptrStruct);
    OP_CHECK_IF(
        memcpy_s(ptrData, capSize, ptrStruct, sizeof(tilingData_)) != 0,
        OP_LOGE(tilingContext_->GetNodeName(), "Set tiling data failed!"), return ge::GRAPH_FAILED);
    ptrTilingData->SetDataSize(sizeof(tilingData_));

    tilingContext_->SetBlockDim(tilingData_.realCoreNum);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(is_small_m_));
    tilingContext_->SetTilingKey(tilingKey);
    size_t* workspaces = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, workspaces);
    workspaces[0] = WORK_SPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

void CdistTiling::PrintTilingData()
{
    std::stringstream ss;
    ss << " realCoreNum: " << tilingData_.realCoreNum << " blockFactor: " << tilingData_.blockFactor
       << " blockTailFactor: " << tilingData_.blockTailFactor << " B: " << tilingData_.B
       << " P: " << tilingData_.P << " R: " << tilingData_.R << " M: " << tilingData_.M
       << " blockMainNumB: " << tilingData_.blockMainNumB << " blockTailNumB: " << tilingData_.blockTailNumB
       << " blockMainFactorB: " << tilingData_.blockMainFactorB << " blockTailFactorB: " << tilingData_.blockTailFactorB
       << " blockMainNumP: " << tilingData_.blockMainNumP << " blockTailNumP: " << tilingData_.blockTailNumP
       << " blockMainFactorP: " << tilingData_.blockMainFactorP << " blockTailFactorP: " << tilingData_.blockTailFactorP
       << " blockMainNumR: " << tilingData_.blockMainNumR << " blockTailNumR: " << tilingData_.blockTailNumR
       << " blockMainFactorR: " << tilingData_.blockMainFactorR << " blockTailFactorR: " << tilingData_.blockTailFactorR
       << " ubLoopNumB: " << tilingData_.ubLoopNumB << " ubFactorB: " << tilingData_.ubFactorB
       << " ubTailFactorB: " << tilingData_.ubTailFactorB << " ubLoopNumP: " << tilingData_.ubLoopNumP
       << " ubFactorP: " << tilingData_.ubFactorP << " ubTailFactorP: " << tilingData_.ubTailFactorP
       << " ubLoopNumR: " << tilingData_.ubLoopNumR << " ubFactorR: " << tilingData_.ubFactorR
       << " ubTailFactorR: " << tilingData_.ubTailFactorR << " ubLoopNumM: " << tilingData_.ubLoopNumM
       << " ubFactorM: " << tilingData_.ubFactorM << " ubTailFactorM: " << tilingData_.ubTailFactorM
       << " p: " << tilingData_.p;
    OP_LOGI(tilingContext_->GetNodeName(), "CdistTilingData: %s", ss.str().c_str());
}

static ge::graphStatus TilingParseForCdist([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start TilingParseForCdist");
    OP_CHECK_IF(
        context == nullptr, OP_LOGE("TilingParseForCdist", "TilingParseContext is nullptr!"), return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<CdistCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
                OP_LOGE(context->GetNodeName(),
                "Get hardwareInfo failed, coreNum:%ld.", compileInfo->coreNum),
                return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0),
                OP_LOGE(context->GetNodeName(),
                "Get hardwareInfo failed, ubSize:%ld.", compileInfo->ubSize),
                return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Get coreNum:%ld, ubSize:%ld.", compileInfo->coreNum, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CdistTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start CdistTilingFunc.");
    CdistTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunCdistTiling();
}

IMPL_OP_OPTILING(Cdist).Tiling(CdistTilingFunc).TilingParse<CdistCompileInfo>(TilingParseForCdist);
} // namespace optiling