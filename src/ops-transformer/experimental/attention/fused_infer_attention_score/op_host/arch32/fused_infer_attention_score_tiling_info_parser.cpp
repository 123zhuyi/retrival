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
 * \file fused_infer_attention_score_tiling_info_parser.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "tiling/tiling_api.h"
#include "../fused_infer_attention_score_tiling_index.h"
#include "fused_infer_attention_score_tiling_info_parser.h"


using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
namespace optiling {
ge::graphStatus FiaInfoParser::GetLegacyIfaFlag()
{
    uint32_t querySize = 0;
    std::string layout(opParamInfo_.layOut);
    if (layout == "BSH" || layout == "BSND" || layout == "BNSD") {
        querySize = static_cast<uint32_t>(queryShape_->GetShapeS());
        if (querySize == 1U &&
            qkHeadDim_ == vHeadDim_ &&
            opParamInfo_.queryRope.tensor == nullptr &&
            opParamInfo_.keyRope.tensor == nullptr) {
            isLegacyIfa_ = true;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetOpName()
{
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize_);

    return ge::GRAPH_SUCCESS;
}

void FiaInfoParser::GetOptionalInputParaInfo()
{
    GetOptionalInputParaRopeInfo();
}


void FiaInfoParser::GetOptionalInputParaRopeInfo()
{
    opParamInfo_.queryRope.tensor = context_->GetOptionalInputTensor(QUERY_ROPE_INDEX);
    opParamInfo_.queryRope.desc = context_->GetOptionalInputDesc(QUERY_ROPE_INDEX);
    opParamInfo_.keyRope.tensor = context_->GetOptionalInputTensor(KEY_ROPE_INDEX);
    opParamInfo_.keyRope.desc = context_->GetOptionalInputDesc(KEY_ROPE_INDEX);
}

void FiaInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.value.desc = context_->GetInputDesc(VALUE_INDEX);
    opParamInfo_.value.shape = context_->GetInputShape(VALUE_INDEX);
    GetOptionalInputParaInfo();
}

void FiaInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(ATTENTION_OUT_INDEX);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(ATTENTION_OUT_INDEX);
}

ge::graphStatus FiaInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    // sparseMode,preToken,nextToken在GetUpdateInfo()中获取
    opParamInfo_.numHeads = attrs->GetAttrPointer<int32_t>(ATTR_N_INDEX);
    opParamInfo_.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    opParamInfo_.layOut = attrs->GetStr(ATTR_INPUT_LAYOUT_INDEX);
    opParamInfo_.kvHeadNums = attrs->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
    opParamInfo_.innerPrecise = attrs->GetAttrPointer<int32_t>(ATTR_INNER_PRECISE_INDEX);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetKvCache()
{
    // 处理Key和value的TensorList场景
    kCache_.clear();
    uint32_t keyBIdx = 0;
    while ((context_->GetDynamicInputShape(KEY_INDEX, keyBIdx)) != nullptr) {
        kCache_.push_back(const_cast<gert::StorageShape *>(context_->GetDynamicInputShape(KEY_INDEX, keyBIdx)));
        keyBIdx++;
    }
    OP_CHECK_IF(keyBIdx == 0,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "tensor list of %s is empty.", KEY_NAME.c_str()),
        return ge::GRAPH_FAILED);
    kCache_.resize(keyBIdx);

    vCache_.clear();
    uint32_t valueBIdx = 0;
    while ((context_->GetDynamicInputShape(VALUE_INDEX, valueBIdx)) != nullptr) {
        vCache_.push_back(const_cast<gert::StorageShape *>(context_->GetDynamicInputShape(VALUE_INDEX, valueBIdx)));
        valueBIdx++;
    }
    vCache_.resize(valueBIdx);

    OP_CHECK_IF(kCache_.size() != vCache_.size(),
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
            "tensor list of %s has %zu tensor, but tensor list of %s has %zu tensor, they should be equal.",
            KEY_NAME.c_str(), kCache_.size(), VALUE_NAME.c_str(), vCache_.size()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKvType_ = opParamInfo_.key.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    if (opParamInfo_.queryRope.desc != nullptr) {
        inputQRopeType_ = opParamInfo_.queryRope.desc->GetDataType();
    }
    if (opParamInfo_.keyRope.desc != nullptr) {
        inputKRopeType_ = opParamInfo_.keyRope.desc->GetDataType();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    bSize_ = queryShape_->GetShapeB();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetQkHeadDim()
{
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    qkHeadDim_ = static_cast<uint32_t>(queryShape_->GetShapeD()); // 后面需要把qkHeadDim_改成uint64
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND/NTD时, 以query的S维度为基准;
    s1Size_ = static_cast<uint32_t>(queryShape_->GetShapeS());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetKvStorageMode()
{
    // kv存储模式基准值
    kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetKvLayout()
{
    // kv Layout基准值
    kvLayout_ = qLayout_;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus FiaInfoParser::GetS2Size()
{
    // 获取S2基准值
    s2Size_ = keyShape_->GetShapeS();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetValueHeadDim()
{
    // 获取vHeadDim基准值
    // 以value的D维度为基准
    vHeadDim_ = static_cast<uint32_t>(valueShape_->GetShapeD()); // 后面需要把vHeadDim_改成uint64
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetRopeMode()
{
    bool existSplitRopeTensor = ((opParamInfo_.queryRope.tensor != nullptr) && (opParamInfo_.queryRope.desc != nullptr));
    if (qkHeadDim_ < vHeadDim_) {
        OP_LOGE(opName_, "the query's head dim(%u) should be greater than or equal to the value's head dim(%u)",
            qkHeadDim_, vHeadDim_);
            return ge::GRAPH_FAILED;
    } else if (qkHeadDim_ > vHeadDim_) {
        if (existSplitRopeTensor) {
            OP_LOGE(opName_, "when %s exist, the query's head dim(%u) should be equal to the value's head dim(%u). ",
                QUERY_ROPE_NAME.c_str(), qkHeadDim_, vHeadDim_);
                return ge::GRAPH_FAILED;
        } else {
            ropeMode_ = RopeMode::ROPE_COMBINE;
        }
    } else {
        if (existSplitRopeTensor) {
            ropeMode_ = RopeMode::ROPE_SPLIT;
        } else {
            ropeMode_ = RopeMode::NO_ROPE;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetRopeHeadDim()
{
    if (ge::GRAPH_SUCCESS != GetRopeMode()) {
        return ge::GRAPH_FAILED;
    }
    if (ropeMode_ == RopeMode::NO_ROPE) {
        ropeHeadDim_ = 0U;
    } else if (ropeMode_ == RopeMode::ROPE_COMBINE) {
        ropeHeadDim_ = qkHeadDim_ - vHeadDim_;
    } else {
        queryRopeShape_ = std::make_shared<FiaTilingShape>(opParamInfo_.queryRope.tensor->GetStorageShape(),
            qLayout_, QUERY_ROPE_NAME, opName_, n1Size_);
        ropeHeadDim_ = static_cast<uint32_t>(queryRopeShape_->GetShapeD());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetQueryAndOutLayout()
{
    // 获取query和attentionOut的Layout基准值
    // inputLayout: {qLayout, outLayout}
    const map<string, pair<FiaLayout, FiaLayout>> layoutMap = {
        {"BSND",        {FiaLayout::BSND,    FiaLayout::BSND}},
        {"BNSD",        {FiaLayout::BNSD,    FiaLayout::BNSD}},
    };

    std::string layout(opParamInfo_.layOut);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second.first;
        outLayout_ = it->second.second;
    } else {
        OP_LOGE(opName_, "input layout is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetN1Size()
{
    // 获取N1基准值
    int32_t numHeads = *(opParamInfo_.numHeads);
    n1Size_ = static_cast<uint32_t>(numHeads);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::GetN2Size()
{
    // 获取N2基准值
    int32_t kvHeadNums = *(opParamInfo_.kvHeadNums);
    n2Size_ = (kvHeadNums == 0) ? n1Size_ : static_cast<uint32_t>(kvHeadNums);
    return ge::GRAPH_SUCCESS;
}

void FiaInfoParser::SetFiaShape()
{
    queryShape_ = std::make_shared<FiaTilingShape>(opParamInfo_.query.shape->GetStorageShape(),
        qLayout_, QUERY_NAME, opName_, n1Size_);
    keyShape_ = std::make_shared<FiaTilingShape>(kCache_[0]->GetStorageShape(),
        kvLayout_, KEY_NAME, opName_, n1Size_);
    valueShape_ = std::make_shared<FiaTilingShape>(vCache_[0]->GetStorageShape(),
        kvLayout_, VALUE_NAME, opName_, n2Size_);
}

ge::graphStatus FiaInfoParser::GetGSize()
{
    // 获取G基准值
    if (n1Size_ % n2Size_ != 0U) {
        OP_LOGE(opName_, "%s(%u) should be a multiple of %s(%u).", 
            QUERY_HEADS_NUM_NAME.c_str(), n1Size_,
            KV_HEADS_NUM_NAME.c_str(), n2Size_);
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;
    return ge::GRAPH_SUCCESS;
}

TilingKeyLayout FiaInfoParser::MapStringToLayout(FiaLayout &layoutString) const
{
    const std::map<FiaLayout, TilingKeyLayout> layoutMap = {
        {FiaLayout::BSND, TilingKeyLayout::BSH_BSND},
        {FiaLayout::BNSD, TilingKeyLayout::BNSD},
    };

    auto it = layoutMap.find(layoutString);
    if (it != layoutMap.end()) {
        return it->second;
    }
    return TilingKeyLayout::BSH_BSND;
}
 
void FiaInfoParser::GenerateLayoutInfo(FiaTilingInfo &fiaInfo)
{
    fiaInfo.qLayout = qLayout_;
    fiaInfo.kvLayout = kvLayout_;
    fiaInfo.outLayout = outLayout_;
    fiaInfo.inputKvLayout = MapStringToLayout(kvLayout_);
    fiaInfo.inputLayout = MapStringToLayout(qLayout_);
    fiaInfo.outputLayout = MapStringToLayout(outLayout_);
}

void FiaInfoParser::GenerateInfo(FiaTilingInfo &fiaInfo)
{
    fiaInfo.opName = opName_;
    fiaInfo.platformInfo = platformInfo_;
    fiaInfo.opParamInfo = opParamInfo_;
    fiaInfo.socVersion = socVersion_;
    GenerateAxisInfo(fiaInfo);
    GenerateDtypeInfo(fiaInfo);
    fiaInfo.kvStorageMode = kvStorageMode_;
    fiaInfo.batchContinuousFlag = (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS);
    fiaInfo.ropeMode = ropeMode_;
    fiaInfo.l2CacheSize = l2CacheSize_;

    fiaInfo.kCache = kCache_;
    fiaInfo.vCache = vCache_;

    fiaInfo.totalOutputSize = opParamInfo_.attenOut.shape->GetStorageShape().GetShapeSize();

    fiaInfo.l2CacheOffFlag = false;
    fiaInfo.totalBlockNum = kCache_[0]->GetStorageShape().GetDim(0);
    fiaInfo.scaleValue = *opParamInfo_.scaleValue;
    fiaInfo.needInit = needInit_;

    fiaInfo.actualLenQDims = actualLenQDims_;
    fiaInfo.actualLenDims = actualLenDims_;
    GenerateLayoutInfo(fiaInfo);
}

void FiaInfoParser::GenerateAxisInfo(FiaTilingInfo &fiaInfo)
{
    fiaInfo.bSize = bSize_;
    fiaInfo.n1Size = n1Size_;
    fiaInfo.n2Size = n2Size_;
    fiaInfo.s1Size = s1Size_;
    fiaInfo.s2Size = s2Size_;
    fiaInfo.gSize = gSize_;
    fiaInfo.qkHeadDim = qkHeadDim_;
    fiaInfo.vHeadDim = vHeadDim_;
    fiaInfo.ropeHeadDim = ropeHeadDim_;
}

void FiaInfoParser::GenerateDtypeInfo(FiaTilingInfo &fiaInfo)
{
    fiaInfo.inputQType = inputQType_;
    fiaInfo.inputKvType = inputKvType_;
    fiaInfo.inputQRopeType = inputQRopeType_;
    fiaInfo.inputKRopeType = inputKRopeType_;
    fiaInfo.outputType = outputType_;
}

ge::graphStatus FiaInfoParser::Parse(FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != GetOpName() ||
        ge::GRAPH_SUCCESS != GetNpuInfo() ||
        ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != GetKvCache()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetInOutDataType() ||
        ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetKvStorageMode() ||
        ge::GRAPH_SUCCESS != GetKvLayout()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ParseAxisInfo()) {
        return ge::GRAPH_FAILED;
    }
    GenerateInfo(fiaInfo);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaInfoParser::ParseAxisInfo()
{
    if (ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetN2Size()) {
        return ge::GRAPH_FAILED;
    }
    SetFiaShape();
    if (ge::GRAPH_SUCCESS != GetQkHeadDim() ||
        ge::GRAPH_SUCCESS != GetValueHeadDim() ||
        ge::GRAPH_SUCCESS != GetLegacyIfaFlag() ||
        ge::GRAPH_SUCCESS != GetBatchSize() ||
        ge::GRAPH_SUCCESS != GetS1Size()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetS2Size() ||
        ge::GRAPH_SUCCESS != GetRopeHeadDim()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
