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
 * \file pad_v3_grad_replicate_tiling.cpp
 * \brief
 */

#include "pad_v3_grad_replicate_tiling.h"
#include "pad_grad_tiling_common.h"

namespace optiling {

// ======================== File-Specific Constants ========================
constexpr uint32_t FLOAT_H_W_ONE_TILING_KEY = 11111;
constexpr uint32_t FLOAT16_H_W_ONE_TILING_KEY = 22222;
constexpr uint32_t BFLOAT16_H_W_ONE_TILING_KEY = 33333;
constexpr uint32_t RESERVED_UB = 32 * 1024;
constexpr uint32_t SMALL_W_LIMIT = 64;
static std::map<ge::DataType, int32_t> DATATYPE_LEN_MAP = {{ge::DT_FLOAT, 4}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}};

template <typename TilingData, int32_t dataTypeLen>
class PadV3GradReplicateTiling
    : public PadGradTilingBase<PadV3GradReplicateTiling<TilingData, dataTypeLen>, TilingData, dataTypeLen> {
    using Base = PadGradTilingBase<PadV3GradReplicateTiling<TilingData, dataTypeLen>, TilingData, dataTypeLen>;

public:
    explicit PadV3GradReplicateTiling(InputParamsInfo& param, const uint32_t inputCoreNum, const uint32_t inputUbSize)
        : Base(param, inputCoreNum, inputUbSize)
    {
        this->wCalCount = this->CeilAlign(std::max(param.padLeft, param.padRight) + 1, BYTE_BLOCK);
        return;
    }

    void GetTilingKey();
    void SplitUb();
    void FillTilingData(TilingData* tilingData);

private:
    uint32_t wCalCount = 0;
};

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradReplicateTiling<TilingData, dataTypeLen>::GetTilingKey()
{
    if (this->dtype == FLOAT_DTYPE && this->mode == EDGE_MODE) {
        if (this->padLeft == 0 && this->padRight == 0 && (this->padTop != 0 || this->padBottom != 0)) {
            this->tilingKey = FLOAT_NO_W_PAD_TILING_KEY; // mode1: float, replicate, w dim no pad
            this->divideUbNum = CONST_VALUE_8;
        } else if (this->padTop == 0 && this->padBottom == 0 && (this->padLeft != 0 || this->padRight != 0)) {
            this->tilingKey = FLOAT_NO_H_PAD_TILING_KEY; // mode1: float, replicate, h dim no pad
            this->divideUbNum = CONST_VALUE_2;
        } else if (this->height <= SMALL_H_LIMIT && this->width <= SMALL_W_LIMIT) {
            this->tilingKey = FLOAT_MINI_SHAPE_TILING_KEY; // mode1: float, replicate, small shape
            this->divideUbNum = CONST_VALUE_4;
        } else if (this->height <= SMALL_H_LIMIT && this->width > SMALL_W_LIMIT) {
            this->tilingKey = FLOAT_SMALL_H_LARGE_W_TILING_KEY; // float, mini h dim
            this->divideUbNum = CONST_VALUE_3;
        } else if (this->height > SMALL_H_LIMIT && this->width <= SMALL_W_LIMIT) {
            this->tilingKey = FLOAT_LARGE_H_SMALL_W_TILING_KEY; // float, mini w dim
            this->divideUbNum = CONST_VALUE_3;
        } else if (this->outHeight == 1) {
            this->tilingKey = FLOAT_H_W_ONE_TILING_KEY; // mode1: float, replicate, outHeight == 1
            this->divideUbNum = CONST_VALUE_4;
        } else {
            this->tilingKey = FLOAT_H_W_PAD_TILING_KEY; // mode1: float, replicate, big shape
            this->divideUbNum = CONST_VALUE_4;
        }
    } else if (this->dtype == FLOAT16_DTYPE && this->mode == EDGE_MODE) {
        if (this->padLeft == 0 && this->padRight == 0 && (this->padTop != 0 || this->padBottom != 0)) {
            this->tilingKey = FLOAT16_NO_W_PAD_TILING_KEY; // mode1: float16, replicate, w dim no pad
            this->divideUbNum = CONST_VALUE_12;
        } else if (this->padTop == 0 && this->padBottom == 0 && (this->padLeft != 0 || this->padRight != 0)) {
            this->tilingKey = FLOAT16_NO_H_PAD_TILING_KEY; // mode1: float16, replicate, h dim no pad
            this->divideUbNum = CONST_VALUE_6;
        } else if (this->height <= SMALL_H_LIMIT && this->width <= SMALL_W_LIMIT) {
            this->tilingKey = FLOAT16_MINI_SHAPE_TILING_KEY; // mode1: float16, replicate, small shape
            this->divideUbNum = CONST_VALUE_8;
        } else if (this->height <= SMALL_H_LIMIT && this->width > SMALL_W_LIMIT) {
            this->tilingKey = FLOAT16_SMALL_H_LARGE_W_TILING_KEY; // float, mini h dim
            this->divideUbNum = CONST_VALUE_6;
        } else if (this->height > SMALL_H_LIMIT && this->width <= SMALL_W_LIMIT) {
            this->tilingKey = FLOAT16_LARGE_H_SMALL_W_TILING_KEY; // float, mini w dim
            this->divideUbNum = CONST_VALUE_6;
        } else if (this->outHeight == 1) {
            this->tilingKey = FLOAT16_H_W_ONE_TILING_KEY; // mode1: float16, replicate, outHeight == 1
            this->divideUbNum = CONST_VALUE_8;
        } else {
            this->tilingKey = FLOAT16_H_W_PAD_TILING_KEY; // mode1: float16, replicate, big shape
            this->divideUbNum = CONST_VALUE_8;
        }
    } else if (this->dtype == BF16_DTYPE && this->mode == EDGE_MODE) {
        if (this->padLeft == 0 && this->padRight == 0 && (this->padTop != 0 || this->padBottom != 0)) {
            this->tilingKey = BFLOAT16_NO_W_PAD_TILING_KEY; // mode1: bfloat16, replicate, w dim no pad
            this->divideUbNum = CONST_VALUE_12;
        } else if (this->padTop == 0 && this->padBottom == 0 && (this->padLeft != 0 || this->padRight != 0)) {
            this->tilingKey = BFLOAT16_NO_H_PAD_TILING_KEY; // mode1: bfloat16, replicate, h dim no pad
            this->divideUbNum = CONST_VALUE_6;
        } else if (this->height <= SMALL_H_LIMIT && this->width <= SMALL_W_LIMIT) {
            this->tilingKey = BFLOAT16_MINI_SHAPE_TILING_KEY; // mode1: bfloat16, replicate, small shape
            this->divideUbNum = CONST_VALUE_8;
        } else if (this->height <= SMALL_H_LIMIT && this->width > SMALL_W_LIMIT) {
            this->tilingKey = BFLOAT16_SMALL_H_LARGE_W_TILING_KEY; // bfloat16, mini h dim
            this->divideUbNum = CONST_VALUE_6;
        } else if (this->height > SMALL_H_LIMIT && this->width <= SMALL_W_LIMIT) {
            this->tilingKey = BFLOAT16_LARGE_H_SMALL_W_TILING_KEY; // bfloat16, mini w dim
            this->divideUbNum = CONST_VALUE_6;
        } else if (this->outHeight == 1) {
            this->tilingKey = BFLOAT16_H_W_ONE_TILING_KEY; // mode1: bfloat16, replicate, outHeight == 1
            this->divideUbNum = CONST_VALUE_8;
        } else {
            this->tilingKey = BFLOAT16_H_W_PAD_TILING_KEY; // mode1: bfloat16, replicate, big shape
            this->divideUbNum = CONST_VALUE_8;
        }
    }
}

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradReplicateTiling<TilingData, dataTypeLen>::SplitUb()
{
    uint32_t tilingDataSize = this->CeilAlign(sizeof(TilingData), BYTE_BLOCK);
    uint32_t canUseUbSize = this->FloorAlign(this->ubSize - tilingDataSize, BYTE_BLOCK);
    if (this->tilingKey == FLOAT_H_W_PAD_TILING_KEY || this->tilingKey == FLOAT16_H_W_PAD_TILING_KEY ||
        this->tilingKey == BFLOAT16_H_W_PAD_TILING_KEY || this->tilingKey == FLOAT_H_W_ONE_TILING_KEY ||
        this->tilingKey == FLOAT16_H_W_ONE_TILING_KEY || this->tilingKey == BFLOAT16_H_W_ONE_TILING_KEY) {
        this->ubFactorElement = this->FloorAlign(canUseUbSize / this->divideUbNum / TRANSPOSE_LINES, ALIGN_256_BYTES) /
                                dataTypeLen;
    } else if (this->tilingKey == FLOAT_MINI_SHAPE_TILING_KEY || this->tilingKey == FLOAT16_MINI_SHAPE_TILING_KEY ||
               this->tilingKey == BFLOAT16_MINI_SHAPE_TILING_KEY) {
        this->ubFactorElement = this->FloorAlign(canUseUbSize / this->divideUbNum / SMALL_H_LIMIT, ALIGN_256_BYTES) /
                                dataTypeLen;
    } else if (this->tilingKey == FLOAT_SMALL_H_LARGE_W_TILING_KEY ||
               this->tilingKey == FLOAT16_SMALL_H_LARGE_W_TILING_KEY ||
               this->tilingKey == BFLOAT16_SMALL_H_LARGE_W_TILING_KEY) {
        this->ubFactorElement = this->FloorAlign(
            this->FloorAlign(canUseUbSize / this->divideUbNum / SMALL_H_LIMIT, BYTE_BLOCK) / dataTypeLen, ALIGN_16);
    } else if (this->tilingKey == FLOAT_LARGE_H_SMALL_W_TILING_KEY ||
               this->tilingKey == FLOAT16_LARGE_H_SMALL_W_TILING_KEY ||
               this->tilingKey == BFLOAT16_LARGE_H_SMALL_W_TILING_KEY) {
        this->ubFactorElement = this->FloorAlign(
            this->FloorAlign(canUseUbSize / this->divideUbNum / SMALL_W_LIMIT, BYTE_BLOCK) / dataTypeLen, ALIGN_16);
    } else {
        this->ubFactorElement = this->FloorAlign(canUseUbSize / this->divideUbNum, ALIGN_256_BYTES) / dataTypeLen;
    }
}

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradReplicateTiling<TilingData, dataTypeLen>::FillTilingData(TilingData* tilingData)
{
    tilingData->set_batch(this->batch);
    tilingData->set_channel(this->channel);
    tilingData->set_height(this->height);
    tilingData->set_width(this->width);
    tilingData->set_alignHeight(this->alignHeight);
    tilingData->set_alignWidth(this->alignWidth);
    tilingData->set_outHeight(this->outHeight);
    tilingData->set_outWidth(this->outWidth);
    tilingData->set_alignOutHeight(this->alignOutHeight);
    tilingData->set_alignOutWidth(this->alignOutWidth);
    tilingData->set_padTop(this->padTop);
    tilingData->set_padBottom(this->padBottom);
    tilingData->set_padLeft(this->padLeft);
    tilingData->set_padRight(this->padRight);
    tilingData->set_blockNum(this->usedCoreNum);
    tilingData->set_ubFactorElement(this->ubFactorElement);
    tilingData->set_ncPerCore(this->ncPerCore);
    tilingData->set_tailNC(this->tailNC);
    tilingData->set_tilingKey(this->tilingKey);
    tilingData->set_wCalCount(wCalCount);
    if (this->tilingKey == FLOAT_NO_W_PAD_TILING_KEY || this->tilingKey == FLOAT16_NO_W_PAD_TILING_KEY ||
        this->tilingKey == BFLOAT16_NO_W_PAD_TILING_KEY) {
        this->workspacePerCore = 0;
    } else if (this->tilingKey == FLOAT_NO_H_PAD_TILING_KEY || this->tilingKey == FLOAT16_NO_H_PAD_TILING_KEY ||
               this->tilingKey == BFLOAT16_NO_H_PAD_TILING_KEY) {
        this->workspacePerCore = CONST_VALUE_2 * wCalCount * this->dataTypeSize;
    } else {
        this->workspacePerCore = std::max(this->alignHeight, this->alignWidth) * WORK_SPACE_PART * this->dataTypeSize;
    }
    tilingData->set_workspacePerCore(this->workspacePerCore);
}

template <typename TilingData, int32_t dataTypeLen>
void GetPadV3GradReplicateTiling(TilingData* tilingData, InputParamsInfo& params, uint32_t coreNum, uint32_t ubSize)
{
    class PadV3GradReplicateTiling<TilingData, dataTypeLen> tilingObj(params, coreNum, ubSize);
    tilingObj.GetTiling(tilingData);
}

static void PrintTilingData(gert::TilingContext* tilingContext, PadV3GradReplicateTilingData& tilingData,
                            const size_t usrWorkspace)
{
    PrintCommonTilingFields(tilingContext, tilingData);
    OP_LOGD(tilingContext->GetNodeName(), "padTop is %d.", tilingData.get_padTop());
    OP_LOGD(tilingContext->GetNodeName(), "padBottom is %d.", tilingData.get_padBottom());
    OP_LOGD(tilingContext->GetNodeName(), "padLeft is %d.", tilingData.get_padLeft());
    OP_LOGD(tilingContext->GetNodeName(), "padRight is %d.", tilingData.get_padRight());
    OP_LOGD(tilingContext->GetNodeName(), "wCalCount is %u.", tilingData.get_wCalCount());
    OP_LOGD(tilingContext->GetNodeName(), "usrWorkspace is %lu.", usrWorkspace);
    OP_LOGD(tilingContext->GetNodeName(), "End printing");
}

static ge::graphStatus Tiling4PadV3GradReplicate(gert::TilingContext* tilingContext)
{
    OP_LOGI(tilingContext->GetNodeName(), "PadV3GradReplicate tiling starts running");
    auto compileInfo = reinterpret_cast<const Tiling4PadV3GradReplicateCompileInfo*>(tilingContext->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    uint64_t ubSizePlatForm = compileInfo->ubSizePlatForm;
    uint32_t ubSize = static_cast<uint32_t>(ubSizePlatForm);
    uint32_t availableUb = ubSize - RESERVED_UB;
    uint32_t coreNum = compileInfo->coreNum;
    OP_LOGI(tilingContext->GetNodeName(), "ubSizePlatForm:%lu, coreNum:%u", ubSizePlatForm, coreNum);
    uint32_t sysWorkspaceSize = compileInfo->sysWorkspaceSize;
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(tilingContext->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSizePlatForm <= 0, OP_LOGE(tilingContext->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);

    ge::DataType inputDatatype = tilingContext->GetInputDesc(0)->GetDataType();
    OP_CHECK_IF(inputDatatype != ge::DT_FLOAT && inputDatatype != ge::DT_FLOAT16 && inputDatatype != ge::DT_BF16,
                OP_LOGE(tilingContext->GetNodeName(),
                        "the current x dtype is not in dtype support list [bfloat16, float16, float]."),
                return ge::GRAPH_FAILED);

    ge::DataType paddingDatatype = tilingContext->GetInputDesc(1)->GetDataType();
    OP_CHECK_IF(
        paddingDatatype != ge::DT_INT32 && paddingDatatype != ge::DT_INT64,
        OP_LOGE(tilingContext->GetNodeName(), "the current padding dtype is not in dtype support list [int32, int64]."),
        return ge::GRAPH_FAILED);
    InputParamsInfo params;
    params.dtype = GetDtypeMap()[inputDatatype];

    if (paddingDatatype == ge::DT_INT32) {
        OP_CHECK_IF(GetInputInfo<int32_t>(tilingContext, params, false) != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext->GetNodeName(), "get op inputs failed."), return ge::GRAPH_FAILED);
    } else if (paddingDatatype == ge::DT_INT64) {
        OP_CHECK_IF(GetInputInfo<int64_t>(tilingContext, params, false) != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext->GetNodeName(), "get op inputs failed."), return ge::GRAPH_FAILED);
    }

    PadV3GradReplicateTilingData tilingData;
    if (inputDatatype == ge::DT_FLOAT) {
        GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, FLOAT_BYTES>(&tilingData, params, coreNum,
                                                                               availableUb);
    } else {
        GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, FLOAT16_BYTES>(&tilingData, params, coreNum,
                                                                                 availableUb);
    }

    OP_CHECK_IF(tilingData.get_ubFactorElement() <= 0,
                OP_LOGE(tilingContext->GetNodeName(), "ub space is not enough, please check input."),
                return ge::GRAPH_FAILED);
    uint64_t workspacePerCore = tilingData.get_workspacePerCore();
    uint32_t tilingKey = tilingData.get_tilingKey();
    uint32_t blockNum = tilingData.get_blockNum();
    size_t usrWorkspace = workspacePerCore * blockNum;
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(blockNum);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = usrWorkspace + sysWorkspaceSize;
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    PrintTilingData(tilingContext, tilingData, usrWorkspace);
    OP_LOGI(tilingContext->GetNodeName(), "PadV3GradReplicate tiling end running");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4PadV3GradReplicate(gert::TilingParseContext* context)
{
    OP_LOGI(context->GetNodeName(), "TilingPrepare4PadV3GradReplicate start.");
    auto compileInfo = context->GetCompiledInfo<Tiling4PadV3GradReplicateCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_CHECK_IF(ubSizePlatForm <= 0, OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGI(context->GetNodeName(), "TilingPrepare4PadV3GradReplicate end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PadV3GradReplicate)
    .Tiling(Tiling4PadV3GradReplicate)
    .TilingParse<Tiling4PadV3GradReplicateCompileInfo>(TilingPrepare4PadV3GradReplicate)
    .TilingInputsDataDependency({1});
} // namespace optiling
