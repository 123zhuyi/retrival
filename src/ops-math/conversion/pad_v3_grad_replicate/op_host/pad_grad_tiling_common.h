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
 * \file pad_grad_tiling_common.h
 * \brief Common utilities, constants, and base tiling class shared by pad_v4_grad and pad_v3_grad_replicate
 */
#ifndef __PAD_GRAD_TILING_COMMON_H__
#define __PAD_GRAD_TILING_COMMON_H__

#include "log/log.h"
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "platform/platform_info.h"
#include <algorithm>
#include <map>
#include <string>

namespace optiling {

// ======================== Shared Constants ========================
constexpr uint32_t BYTE_BLOCK = 32;
constexpr size_t MODE_INDEX = 0;
constexpr int32_t X_INPUT_INDEX = 0;
constexpr int32_t PAD_INPUT_INDEX = 1;
constexpr int32_t FLOAT_BYTES = 4;
constexpr int32_t FLOAT16_BYTES = 2;
constexpr size_t CHECK_DIM_NUM = 4;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr uint32_t DIM_INDEX2 = 2;
constexpr uint32_t DIM_INDEX3 = 3;
constexpr uint32_t PADDING_NUM_INDEX4 = 4;
constexpr uint32_t PADDING_NUM_INDEX5 = 5;
constexpr uint32_t PADDING_NUM_INDEX6 = 6;
constexpr uint32_t PADDING_NUM_INDEX7 = 7;
constexpr uint32_t ALIGN_256_BYTES = 256;
constexpr uint32_t ALIGN_16 = 16;
constexpr uint32_t TRANSPOSE_LINES = 16;
constexpr uint32_t CAL_COUNT = 64;
constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t WORK_SPACE_PART = 64;
constexpr uint32_t SMALL_H_LIMIT = 64;
constexpr uint32_t CONST_VALUE_2 = 2;
constexpr uint32_t CONST_VALUE_3 = 3;
constexpr uint32_t CONST_VALUE_4 = 4;
constexpr uint32_t CONST_VALUE_5 = 5;
constexpr uint32_t CONST_VALUE_6 = 6;
constexpr uint32_t CONST_VALUE_8 = 8;
constexpr uint32_t CONST_VALUE_12 = 12;
constexpr uint32_t REFLECTION_MODE = 0;
constexpr uint32_t EDGE_MODE = 1;
constexpr uint32_t FLOAT_DTYPE = 1;
constexpr uint32_t FLOAT16_DTYPE = 2;
constexpr uint32_t BF16_DTYPE = 3;

// ======================== Shared Tiling Keys ========================
constexpr uint32_t FLOAT_MINI_SHAPE_TILING_KEY = 1000;
constexpr uint32_t FLOAT_SMALL_H_LARGE_W_TILING_KEY = 1100;
constexpr uint32_t FLOAT_LARGE_H_SMALL_W_TILING_KEY = 1010;
constexpr uint32_t FLOAT_NO_W_PAD_TILING_KEY = 1110;
constexpr uint32_t FLOAT_NO_H_PAD_TILING_KEY = 1101;
constexpr uint32_t FLOAT_H_W_PAD_TILING_KEY = 1111;
constexpr uint32_t FLOAT16_MINI_SHAPE_TILING_KEY = 2000;
constexpr uint32_t FLOAT16_SMALL_H_LARGE_W_TILING_KEY = 2100;
constexpr uint32_t FLOAT16_LARGE_H_SMALL_W_TILING_KEY = 2010;
constexpr uint32_t FLOAT16_NO_W_PAD_TILING_KEY = 2110;
constexpr uint32_t FLOAT16_NO_H_PAD_TILING_KEY = 2101;
constexpr uint32_t FLOAT16_H_W_PAD_TILING_KEY = 2111;
constexpr uint32_t BFLOAT16_MINI_SHAPE_TILING_KEY = 3000;
constexpr uint32_t BFLOAT16_SMALL_H_LARGE_W_TILING_KEY = 3100;
constexpr uint32_t BFLOAT16_LARGE_H_SMALL_W_TILING_KEY = 3010;
constexpr uint32_t BFLOAT16_NO_W_PAD_TILING_KEY = 3110;
constexpr uint32_t BFLOAT16_NO_H_PAD_TILING_KEY = 3101;
constexpr uint32_t BFLOAT16_H_W_PAD_TILING_KEY = 3111;

// ======================== Shared Static Maps ========================
inline std::map<std::string, int>& GetPaddingModeMap()
{
    static std::map<std::string, int> paddingModeMap = {{"reflect", 0}, {"edge", 1}, {"constant", 2}};
    return paddingModeMap;
}

inline std::map<ge::DataType, uint32_t>& GetDtypeMap()
{
    static std::map<ge::DataType, uint32_t> dtypeMap = {{ge::DT_FLOAT, 1}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 3}};
    return dtypeMap;
}

// ======================== Standalone Utility Functions ========================
template <typename T1, typename T2>
static inline T1 CeilAlignValue(T1 a, T2 b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

template <typename T1, typename T2>
static inline T1 FloorAlignValue(T1 a, T2 b)
{
    if (b == 0) {
        return a;
    }
    return a / b * b;
}

template <typename T1, typename T2>
static T1 CeilAlignStatus(T1 a, T2 b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

// ======================== Shared GetInputInfo ========================
template <typename T>
static ge::graphStatus GetInputInfo(gert::TilingContext* tilingContext, InputParamsInfo& params,
                                    bool allowConstantMode = true)
{
    OP_LOGI(tilingContext->GetNodeName(), "start to get input dims");
    const gert::StorageShape* xShape = tilingContext->GetInputShape(X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xShape);
    OP_CHECK_IF(xShape->GetStorageShape().GetDimNum() != CHECK_DIM_NUM,
                OP_LOGE(tilingContext->GetNodeName(), "input dim is not 4, please check input shape dimension."),
                return ge::GRAPH_FAILED);
    const gert::StorageShape* paddingShape = tilingContext->GetInputShape(PAD_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, paddingShape);
    OP_CHECK_IF(static_cast<int32_t>(xShape->GetStorageShape().GetDimNum() * 2) !=
                    static_cast<int32_t>(paddingShape->GetStorageShape().GetDim(0)),
                OP_LOGE(tilingContext->GetNodeName(),
                        "input dim num does not match padding shape size, expected padding size = input dim num * 2."),
                return ge::GRAPH_FAILED);
    const gert::Tensor* paddingsTensor = tilingContext->GetInputTensor(PAD_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, paddingsTensor);

    const T* paddingsValue = paddingsTensor->GetData<T>();

    params.padTop = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX4]);
    params.padBottom = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX5]);
    params.padLeft = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX6]);
    params.padRight = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX7]);

    const gert::StorageShape* outShape = tilingContext->GetOutputShape(0);
    uint32_t outHeight = outShape->GetStorageShape().GetDim(DIM_INDEX2);
    uint32_t outWidth = outShape->GetStorageShape().GetDim(DIM_INDEX3);
    params.batch = xShape->GetStorageShape().GetDim(DIM_INDEX0);
    params.channel = xShape->GetStorageShape().GetDim(DIM_INDEX1);
    params.height = xShape->GetStorageShape().GetDim(DIM_INDEX2);
    params.width = xShape->GetStorageShape().GetDim(DIM_INDEX3);
    params.outHeight = outHeight;
    params.outWidth = outWidth;

    OP_CHECK_IF((outHeight != (params.height - params.padTop - params.padBottom)) ||
                    (outWidth != (params.width - params.padLeft - params.padRight)),
                OP_LOGE(tilingContext->GetNodeName(), "Please check input or output shape"), return ge::GRAPH_FAILED);

    params.alignHeight = CeilAlignStatus(params.height, ALIGN_16);
    params.alignWidth = CeilAlignStatus(params.width, ALIGN_16);
    params.alignOutHeight = CeilAlignStatus(params.outHeight, ALIGN_16);
    params.alignOutWidth = CeilAlignStatus(params.outWidth, ALIGN_16);

    const gert::RuntimeAttrs* attrs = tilingContext->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(tilingContext->GetNodeName(), "Failed to get runtime attrs."),
                return ge::GRAPH_FAILED);
    const std::string mode = std::string(attrs->GetAttrPointer<char>(MODE_INDEX));
    if (allowConstantMode) {
        OP_CHECK_IF(mode != "reflect" && mode != "edge" && mode != "constant",
                    OP_LOGE(tilingContext->GetNodeName(), "%s is not supported", mode.c_str()),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(mode != "reflect" && mode != "edge",
                    OP_LOGE(tilingContext->GetNodeName(), "%s is not supported", mode.c_str()),
                    return ge::GRAPH_FAILED);
    }

    params.mode = GetPaddingModeMap()[mode];
    return ge::GRAPH_SUCCESS;
}

// ======================== Shared Print Helper ========================
template <typename TilingData>
static void PrintCommonTilingFields(gert::TilingContext* tilingContext, TilingData& tilingData)
{
    OP_LOGD(tilingContext->GetNodeName(), "Start printing");
    OP_LOGD(tilingContext->GetNodeName(), "batch is %u.", tilingData.get_batch());
    OP_LOGD(tilingContext->GetNodeName(), "channel is %u.", tilingData.get_channel());
    OP_LOGD(tilingContext->GetNodeName(), "height is %u.", tilingData.get_height());
    OP_LOGD(tilingContext->GetNodeName(), "width is %u.", tilingData.get_width());
    OP_LOGD(tilingContext->GetNodeName(), "alignHeight is %u.", tilingData.get_alignHeight());
    OP_LOGD(tilingContext->GetNodeName(), "alignWidth is %u.", tilingData.get_alignWidth());
    OP_LOGD(tilingContext->GetNodeName(), "outHeight is %u.", tilingData.get_outHeight());
    OP_LOGD(tilingContext->GetNodeName(), "outWidth is %u.", tilingData.get_outWidth());
    OP_LOGD(tilingContext->GetNodeName(), "alignOutHeight is %u.", tilingData.get_alignOutHeight());
    OP_LOGD(tilingContext->GetNodeName(), "alignOutWidth is %u.", tilingData.get_alignOutWidth());
    OP_LOGD(tilingContext->GetNodeName(), "blockNum is %u.", tilingData.get_blockNum());
    OP_LOGD(tilingContext->GetNodeName(), "ubFactorElement is %u.", tilingData.get_ubFactorElement());
    OP_LOGD(tilingContext->GetNodeName(), "ncPerCore is %u.", tilingData.get_ncPerCore());
    OP_LOGD(tilingContext->GetNodeName(), "tailNC is %u.", tilingData.get_tailNC());
    OP_LOGD(tilingContext->GetNodeName(), "tilingKey is %u.", tilingData.get_tilingKey());
}

// ======================== Shared Tiling Base Class ========================
template <typename Derived, typename TilingData, int32_t dataTypeLen>
class PadGradTilingBase {
public:
    explicit PadGradTilingBase(InputParamsInfo& param, const uint32_t inputCoreNum, const uint32_t inputUbSize)
    {
        this->batch = param.batch;
        this->channel = param.channel;
        this->height = param.height;
        this->width = param.width;
        this->alignHeight = param.alignHeight;
        this->alignWidth = param.alignWidth;
        this->outHeight = param.outHeight;
        this->outWidth = param.outWidth;
        this->alignOutHeight = param.alignOutHeight;
        this->alignOutWidth = param.alignOutWidth;
        this->padTop = param.padTop;
        this->padBottom = param.padBottom;
        this->padLeft = param.padLeft;
        this->padRight = param.padRight;
        this->mode = param.mode;
        this->dtype = param.dtype;
        this->ubSize = FloorAlignValue(inputUbSize, BYTE_BLOCK);
        this->dataTypeSize = dataTypeLen;
        this->elementsPerBlock = BYTE_BLOCK / dataTypeSize;
        this->coreNum = inputCoreNum;
    }

    void GetUsedCore()
    {
        uint64_t nMulC = batch * channel;
        if (tilingKey == FLOAT_NO_H_PAD_TILING_KEY || tilingKey == FLOAT16_NO_H_PAD_TILING_KEY ||
            tilingKey == BFLOAT16_NO_H_PAD_TILING_KEY) {
            nMulC = nMulC * height;
        }
        if (nMulC <= coreNum) {
            ncPerCore = 1;
            usedCoreNum = nMulC;
            tailNC = 0;
            return;
        }
        ncPerCore = nMulC / coreNum;
        tailNC = nMulC % coreNum;
        usedCoreNum = coreNum;
    }

    void GetTiling(TilingData* tilingData)
    {
        static_cast<Derived*>(this)->GetTilingKey();
        GetUsedCore();
        static_cast<Derived*>(this)->SplitUb();
        static_cast<Derived*>(this)->FillTilingData(tilingData);
    }

protected:
    template <typename T1, typename T2>
    inline auto CeilDiv(T1 a, T2 b) -> T1
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    }
    template <typename T1, typename T2>
    inline auto FloorDiv(T1 a, T2 b) -> T1
    {
        if (b == 0) {
            return a;
        }
        return a / b;
    }
    template <typename T1, typename T2>
    inline auto CeilAlign(T1 a, T2 b) -> T1
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b * b;
    }
    template <typename T1, typename T2>
    inline auto FloorAlign(T1 a, T2 b) -> T1
    {
        if (b == 0) {
            return a;
        }
        return a / b * b;
    }

    uint32_t batch = 0;
    uint32_t channel = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t alignHeight = 0;
    uint32_t alignWidth = 0;
    uint32_t outHeight = 0;
    uint32_t outWidth = 0;
    uint32_t alignOutHeight = 0;
    uint32_t alignOutWidth = 0;
    int32_t padTop = 0;
    int32_t padBottom = 0;
    int32_t padLeft = 0;
    int32_t padRight = 0;
    uint32_t mode = 0;
    uint32_t ubSize = 0;
    uint32_t usedCoreNum = 0;
    uint32_t coreNum = 0;
    uint32_t ncPerCore = 1;
    uint32_t tailNC = 0;
    uint32_t ubFactorElement = 0;
    uint32_t tilingKey = 0;
    uint32_t dtype = 1;
    uint8_t dataTypeSize = 0;
    uint8_t elementsPerBlock = 0;
    uint32_t divideUbNum = 1;
    uint64_t workspacePerCore = 0;
};

} // namespace optiling

#endif // __PAD_GRAD_TILING_COMMON_H__
