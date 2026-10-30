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
 * \file add_lora_tiling.cpp
 * \brief
 */

#include "add_lora_tiling.h"
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"

namespace optiling {
void AddLoraTiling::Reset()
{
    tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
}

ge::graphStatus AddLoraTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum <= 0,
                OP_LOGE(context_->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_CHECK_IF(ubSizePlatForm <= 0,
                OP_LOGE(context_->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    coreNum_ = ascendcPlatform.GetCoreNumAic();
    isAscend310P_ = ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::GetShapeAttrsInfo()
{
    yParaInfo_.desc = context_->GetInputDesc(Y_INPUT_INDEX);
    yParaInfo_.shape = context_->GetInputShape(Y_INPUT_INDEX);
    xParaInfo_.desc = context_->GetInputDesc(X_INPUT_INDEX);
    xParaInfo_.shape = context_->GetInputShape(X_INPUT_INDEX);
    weightBParaInfo_.desc = context_->GetInputDesc(WEIGHTB_INPUT_INDEX);
    weightBParaInfo_.shape = context_->GetInputShape(WEIGHTB_INPUT_INDEX);

    indicesParaInfo_.desc = context_->GetInputDesc(INDICES_INPUT_INDEX);
    indicesParaInfo_.shape = context_->GetInputShape(INDICES_INPUT_INDEX);
    weightAParaInfo_.desc = context_->GetOptionalInputDesc(WEIGHTA_INPUT_INDEX);
    weightAParaInfo_.tensor = context_->GetOptionalInputTensor(WEIGHTA_INPUT_INDEX);
    yOutParaInfo_.desc = context_->GetOutputDesc(OUTPUT_INDEX);
    yOutParaInfo_.shape = context_->GetOutputShape(OUTPUT_INDEX);

    attrs_ = context_->GetAttrs();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::CheckInputDataType()
{
    const gert::CompileTimeTensorDesc *yDesc = yParaInfo_.desc;
    OP_CHECK_IF(yDesc == nullptr, OP_LOGE(context_->GetNodeName(), "AddLora desc of tensor y is nullptr."),
                return ge::GRAPH_FAILED);
    ge::DataType yDataType = yDesc->GetDataType();
    OP_CHECK_IF(yDataType != ge::DT_FLOAT16,
                OP_LOGE(context_->GetNodeName(),
                "AddLora y dtype current is %d, is invalid, should be float16.", yDataType),
                return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc *xDesc = xParaInfo_.desc;
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "AddLora desc of tensor x is nullptr."),
                return ge::GRAPH_FAILED);
    ge::DataType xDataType = xDesc->GetDataType();
    OP_CHECK_IF(xDataType != ge::DT_FLOAT16,
                OP_LOGE(context_->GetNodeName(),
                "AddLora x dtype current is %d, is invalid, should be float16.", xDataType),
                return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc *weightBDesc = weightBParaInfo_.desc;
    OP_CHECK_IF(weightBDesc == nullptr, OP_LOGE(context_->GetNodeName(), "AddLora desc of tensor weightB is nullptr."),
                return ge::GRAPH_FAILED);
    ge::DataType weightBDataType = weightBDesc->GetDataType();
    OP_CHECK_IF(weightBDataType != ge::DT_FLOAT16,
                OP_LOGE(context_->GetNodeName(),
                "AddLora weightB dtype current is %d, is invalid, should be float16.", weightBDataType),
                return ge::GRAPH_FAILED);

    const gert::Tensor *weightATensor = weightAParaInfo_.tensor;
    const gert::CompileTimeTensorDesc *weightADesc = weightAParaInfo_.desc;
    if (weightATensor != nullptr && weightADesc != nullptr) {
        ge::DataType weightADataType = weightADesc->GetDataType();
        OP_CHECK_IF(weightADataType != ge::DT_FLOAT16,
                    OP_LOGE(context_->GetNodeName(),
                    "AddLora weightA dtype current is %d, is invalid, should be float16.", weightADataType),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool AddLoraTiling::IsCapable()
{
    return true;
}

ge::graphStatus AddLoraTiling::CheckInputTensorShape()
{
    // Check base input tensors
    const gert::StorageShape *yShape = yParaInfo_.shape;
    OP_CHECK_IF(yShape == nullptr, OP_LOGE(context_->GetNodeName(), "AddLora shape of tensor y is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yShape->GetStorageShape().GetDimNum() != 2,
                OP_LOGE(context_->GetNodeName(),
                "AddLora tensor y dim number must be 2."),
                return ge::GRAPH_FAILED);
    
    const gert::StorageShape *xShape = xParaInfo_.shape;
    OP_CHECK_IF(xShape == nullptr, OP_LOGE(context_->GetNodeName(), "AddLora shape of tensor x is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape->GetStorageShape().GetDimNum() != 2,
                OP_LOGE(context_->GetNodeName(),
                "AddLora tensor x dim number must be 2."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(yShape->GetStorageShape().GetDim(DIM_INDEX_ZERO) != xShape->GetStorageShape().GetDim(DIM_INDEX_ZERO),
                OP_LOGE(context_->GetNodeName(), "AddLora tensor y dim 0 must be same with tensor x dim 0."),
                return ge::GRAPH_FAILED);

    batchSize_ = static_cast<uint32_t>(yShape->GetStorageShape().GetDim(DIM_INDEX_ZERO));
    H3_ = static_cast<uint32_t>(yShape->GetStorageShape().GetDim(DIM_INDEX_ONE));
    H1_ = static_cast<uint32_t>(xShape->GetStorageShape().GetDim(DIM_INDEX_ONE));

    const gert::StorageShape *weightBShape = weightBParaInfo_.shape;
    OP_CHECK_IF(weightBShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "AddLora shape of tensor weightB is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightBShape->GetStorageShape().GetDimNum() != 4,
                OP_LOGE(context_->GetNodeName(),
                "AddLora tensor weightB dim number must be 4."),
                return ge::GRAPH_FAILED);

    wBatch_ = static_cast<uint32_t>(weightBShape->GetStorageShape().GetDim(DIM_INDEX_ZERO));
    L_ = static_cast<uint32_t>(weightBShape->GetStorageShape().GetDim(DIM_INDEX_ONE));
    H2_ = static_cast<uint32_t>(weightBShape->GetStorageShape().GetDim(DIM_INDEX_TWO));
    R_ = static_cast<uint32_t>(weightBShape->GetStorageShape().GetDim(DIM_INDEX_THREE));

    OP_CHECK_IF(H2_ > H3_, OP_LOGE(context_->GetNodeName(),
                "AddLora tensor weightB dim 3 must less than or equal tensor y dim 1."),
                return ge::GRAPH_FAILED);

    const gert::StorageShape *indicesShape = indicesParaInfo_.shape;
    OP_CHECK_IF(indicesShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "AddLora shape of tensor indices is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesShape->GetStorageShape().GetDimNum() != 1,
                OP_LOGE(context_->GetNodeName(), "AddLora tensor indices dim number must be 1."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesShape->GetStorageShape().GetDim(DIM_INDEX_ZERO) !=
                yShape->GetStorageShape().GetDim(DIM_INDEX_ZERO),
                OP_LOGE(context_->GetNodeName(),
                "AddLora tensor indices dim 1 must same with tensor y and x dim 0."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::CheckOptionalInputTensorShape()
{
    const gert::StorageShape *xShape = xParaInfo_.shape;
    const gert::StorageShape *weightBShape = weightBParaInfo_.shape;
    const gert::Tensor *weightATensor = weightAParaInfo_.tensor;
    if (weightATensor != nullptr && weightBShape != nullptr) {
        OP_CHECK_IF(weightATensor->GetStorageShape().GetDim(DIM_INDEX_ZERO) !=
            weightBShape->GetStorageShape().GetDim(DIM_INDEX_ZERO) ||
            weightATensor->GetStorageShape().GetDim(DIM_INDEX_ONE) !=
            weightBShape->GetStorageShape().GetDim(DIM_INDEX_ONE),
            OP_LOGE(context_->GetNodeName(),
            "AddLora tensor weightA dim 0 or dim 1 must same with tensor weightB."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(weightATensor->GetStorageShape().GetDim(DIM_INDEX_TWO) !=
            weightBShape->GetStorageShape().GetDim(DIM_INDEX_THREE),
            OP_LOGE(context_->GetNodeName(),
            "AddLora tensor weightA dim 2 must same with tensor weightB dim 3, is shape param R."),
            return ge::GRAPH_FAILED);
        addLoraFlag_ = 1;
    }

    if (weightATensor != nullptr) {
        OP_CHECK_IF(weightATensor->GetStorageShape().GetDim(DIM_INDEX_THREE) !=
            xShape->GetStorageShape().GetDim(DIM_INDEX_ONE),
            OP_LOGE(context_->GetNodeName(),
            "AddLora tensor weightA dim 3 must same with tensor x dim 1."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::CheckInputAttrLimit()
{
    // check base input attrs
    OP_CHECK_IF(attrs_ == nullptr, OP_LOGE(context_->GetNodeName(), "AddLora attrs got from GE is nullptr."),
                return ge::GRAPH_FAILED);

    const int64_t *layerIdxPtr = attrs_->GetAttrPointer<int64_t>(LAYER_IDX_ATTR_INDEX);
    OP_CHECK_IF(layerIdxPtr == nullptr, OP_LOGE(context_->GetNodeName(), "AddLora attr layerIdx is nullptr."),
                return ge::GRAPH_FAILED);
    layerIdx_ = static_cast<uint32_t>(*layerIdxPtr);
    OP_CHECK_IF(layerIdx_ >= L_,
                OP_LOGE(context_->GetNodeName(),
                "AddLora attr layerIdx must less than weightB and weightA dim 1, current is %u.", layerIdx_),
                return ge::GRAPH_FAILED);

    const float *scalePtr = attrs_->GetAttrPointer<float>(SCALE_ATTR_INDEX);
    OP_CHECK_IF(scalePtr == nullptr, OP_LOGE(context_->GetNodeName(), "AddLora attr scale is nullptr."),
                return ge::GRAPH_FAILED);
    scale_ = *scalePtr;

    return CheckYSliceAttrs();
}

ge::graphStatus AddLoraTiling::CheckYSliceAttrs()
{
    const int64_t *yOffsetPtr = attrs_->GetAttrPointer<int64_t>(Y_OFFSET_ATTR_INDEX);
    OP_CHECK_IF(yOffsetPtr == nullptr, OP_LOGE(context_->GetNodeName(), "AddLora attr yOffset is nullptr."),
                return ge::GRAPH_FAILED);
    int64_t yOffsetValue = *yOffsetPtr;
    OP_CHECK_IF(yOffsetValue < 0,
                OP_LOGE(context_->GetNodeName(),
                "AddLora attr yOffset must be non-negative, current is %ld.", yOffsetValue),
                return ge::GRAPH_FAILED);
    yOffset_ = static_cast<uint32_t>(yOffsetValue);
    OP_CHECK_IF(yOffset_ >= H3_,
                OP_LOGE(context_->GetNodeName(),
                "AddLora attr yOffset must less than y dim 1, current is %u.", yOffset_),
                return ge::GRAPH_FAILED);

    const int64_t *ySliceSizePtr = attrs_->GetAttrPointer<int64_t>(Y_SLICE_SIZE_ATTR_INDEX);
    OP_CHECK_IF(ySliceSizePtr == nullptr, OP_LOGE(context_->GetNodeName(), "AddLora attr ySliceSize is nullptr."),
                return ge::GRAPH_FAILED);
    int64_t ySliceSizeValue = *ySliceSizePtr;
    if (ySliceSizeValue == Y_SLICE_SIZE_DEFAULT) {
        ySliceSize_ = H2_;
    } else {
        OP_CHECK_IF(ySliceSizeValue < 0,
                    OP_LOGE(context_->GetNodeName(),
                    "AddLora attr ySliceSize must be non-negative or %ld, current is %ld.",
                    Y_SLICE_SIZE_DEFAULT, ySliceSizeValue),
                    return ge::GRAPH_FAILED);
        ySliceSize_ = static_cast<uint32_t>(ySliceSizeValue);
    }
    OP_CHECK_IF(ySliceSize_ > H2_,
                OP_LOGE(context_->GetNodeName(),
                "AddLora attr ySliceSize must not exceed weightB dim 2 (H2), current ySliceSize is %u, H2 is %u.",
                ySliceSize_, H2_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yOffset_ + H2_ > H3_,
                OP_LOGE(context_->GetNodeName(),
                "AddLora yOffset + H2 must not exceed y dim 1, yOffset is %u, H2 is %u, y dim 1 is %u.",
                yOffset_, H2_, H3_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::CheckInputLimits()
{
    OP_CHECK_IF(addLoraFlag_ && R_ > MAX_RANK_SIZE,
                OP_LOGE(context_->GetNodeName(), "AddLora rank size R exceeds limit 128."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(R_ % CUBESIZE != 0,
                OP_LOGE(context_->GetNodeName(), "AddLora R should be align to 16."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(wBatch_ > MAX_WEIGHT_NUM,
                OP_LOGE(context_->GetNodeName(), "AddLora weight num wBatch exceeds limit 32."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(L_ > MAX_LAYER_NUM,
                OP_LOGE(context_->GetNodeName(), "AddLora layer num exceeds limit 32."),
                return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(H1_ > MAX_HIDDEN_SIZE,
                OP_LOGE(context_->GetNodeName(), "AddLora H1 exceeds limit 131072."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(H1_ % CUBESIZE != 0,
                OP_LOGE(context_->GetNodeName(), "AddLora H1 should be align to 16."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(H2_ > MAX_HIDDEN_SIZE,
                OP_LOGE(context_->GetNodeName(), "AddLora H2 exceeds limit 131072."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(H2_ % CUBESIZE != 0,
                OP_LOGE(context_->GetNodeName(), "AddLora H2 should be align to 16."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(H3_ > MAX_HIDDEN_SIZE,
                OP_LOGE(context_->GetNodeName(), "AddLora H3 exceeds limit 131072."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(H3_ % CUBESIZE != 0,
                OP_LOGE(context_->GetNodeName(), "AddLora H3 should be align to 16."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::DoOpTiling()
{
    if ((CheckInputDataType() != ge::GRAPH_SUCCESS) ||
        (CheckInputTensorShape() != ge::GRAPH_SUCCESS) ||
        (CheckOptionalInputTensorShape() != ge::GRAPH_SUCCESS) ||
        (CheckInputAttrLimit() != ge::GRAPH_SUCCESS) ||
        (CheckInputLimits() != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    
    uint32_t maxCore = static_cast<uint32_t>(coreNum_) * VECTORDOUBLE;
    if (isAscend310P_) {
        maxCore = static_cast<uint32_t>(coreNum_);
    }

    uint32_t usedCoreNum = static_cast<uint32_t>(coreNum_);
    uint32_t taskNumPerCore = batchSize_ / maxCore;
    uint32_t H2PerCore = Ops::Base::CeilDiv(H2_, CUBESIZE) / maxCore * CUBESIZE;
    tilingData_.set_layer(L_);
    tilingData_.set_batch(batchSize_);
    tilingData_.set_H2(H2_);
    tilingData_.set_H1(H1_);
    tilingData_.set_wBatch(wBatch_);
    tilingData_.set_usedCoreNum(usedCoreNum);
    tilingData_.set_R(R_);
    tilingData_.set_layer_idx(layerIdx_);
    tilingData_.set_scale(scale_);
    tilingData_.set_y_offset(yOffset_);
    tilingData_.set_y_slice_size(ySliceSize_);
    tilingData_.set_taskNumPerCore(taskNumPerCore);
    tilingData_.set_H2PerCore(H2PerCore);
    tilingData_.set_addLoraFlag(addLoraFlag_);
    tilingData_.set_y_column(H3_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::DoLibApiTiling()
{
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t AddLoraTiling::GetTilingKey() const
{
    uint64_t tilingKey = static_cast<uint64_t>(TilingKeyInfo::KEY_DEFAULT_SCENE);
    if (batchSize_ <= static_cast<uint32_t>(coreNum_)) {
        tilingKey = static_cast<uint64_t>(TilingKeyInfo::KEY_SPARSE_SCENE);
    }

    uint32_t socVersionFlag = static_cast<uint32_t>(SocVersionKey::KEY_SOC_VERSION_910);
    if (isAscend310P_) {
        socVersionFlag = static_cast<uint32_t>(SocVersionKey::KEY_SOC_VERSION_310);

        if (!addLoraFlag_) {
            tilingKey = static_cast<uint64_t>(TilingKeyInfo::KEY_BGMV_SCENE);
        }
    }

    tilingKey += static_cast<uint64_t>(socVersionFlag) * SOC_TILINGKEY_OFFSET;
    return tilingKey;
}

ge::graphStatus AddLoraTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    size_t sysWorkspaceSize = WORKSIZE;
    /* usrWorkspaceSize = workspace for other gm */
    size_t usrWorkspaceSize =
        tilingData_.get_batch() * tilingData_.get_H1() * sizeof(uint16_t) +
        Ops::Base::CeilDiv(tilingData_.get_batch(), CUBESIZE) * CUBESIZE * tilingData_.get_R() * sizeof(uint16_t) +
        Ops::Base::CeilDiv(tilingData_.get_batch(), CUBESIZE) * CUBESIZE * tilingData_.get_H2() * sizeof(uint16_t) +
        Ops::Base::CeilDiv(tilingData_.get_batch(), CUBESIZE) * CUBESIZE * sizeof(uint32_t) * 3 + 1024 * 1024;
    workspaces[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "final tiling data size: %zu", tilingData_.GetDataSize());
    OP_CHECK_IF(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
                OP_LOGE(context_->GetNodeName(), "tiling data size [%zu] is not aligned to 8",
                tilingData_.GetDataSize()),
                return ge::GRAPH_FAILED);
    
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    auto numBlocks = tilingData_.get_usedCoreNum();
    context_->SetBlockDim(numBlocks);
    context_->SetTilingKey(GetTilingKey());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void AddLoraTiling::PrintTilingData()
{
    std::stringstream ss;
    ss << " layer: " << tilingData_.get_layer() << " batch: " << tilingData_.get_batch()
       << " H2: " << tilingData_.get_H2() << " H1: " << tilingData_.get_H1() << " wBatch: " << tilingData_.get_wBatch()
       << " usedCoreNum: " << tilingData_.get_usedCoreNum() << " R: " << tilingData_.get_R()
       << " layer_idx: " << tilingData_.get_layer_idx() << " scale: " << tilingData_.get_scale()
       << " y_offset: " << tilingData_.get_y_offset() << " y_slice_size: " << tilingData_.get_y_slice_size()
       << " taskNumPerCore: " << tilingData_.get_taskNumPerCore() << " H2PerCore: " << tilingData_.get_H2PerCore()
       << " addLoraFlag: " << tilingData_.get_addLoraFlag() << " y_column: " << tilingData_.get_y_column();
    OP_LOGD(context_->GetNodeName(), "api tiling: %s", ss.str().c_str());
}

static ge::graphStatus AddLoraTilingFunc(gert::TilingContext* context)
{
    AddLoraTiling tiling(context);
    context->SetScheduleMode(BATCH_MODE);
    return tiling.DoTiling();
}

static ge::graphStatus TilingParseForAddLora([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AddLora, AddLoraTiling, 0);

IMPL_OP_OPTILING(AddLora).Tiling(AddLoraTilingFunc).TilingParse<Tiling4AddLoraCompileInfo>(TilingParseForAddLora);
} // namespace optiling