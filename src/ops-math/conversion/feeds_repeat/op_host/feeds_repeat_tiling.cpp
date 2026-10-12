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
 * \file feeds_repeat_tiling.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "feeds_repeat_tiling.h"

namespace optiling {

constexpr int64_t SIZE_OF_FP16 = 2;
constexpr int64_t SIZE_OF_FP32 = 4;
constexpr int64_t SIZE_OF_BF16 = 2;
constexpr int64_t SIZE_OF_INT32 = 4;
constexpr int64_t SIZE_OF_INT64 = 8;
constexpr int64_t ALIGN_NUM = 32;
constexpr int64_t ALIGN_FP16 = ALIGN_NUM / SIZE_OF_FP16;
constexpr int64_t ALIGN_FP32 = ALIGN_NUM / SIZE_OF_FP32;
constexpr int64_t ALIGN_BF16 = ALIGN_NUM / SIZE_OF_BF16;
constexpr int64_t ALIGN_INT32 = ALIGN_NUM / SIZE_OF_INT32;
constexpr int64_t ALIGN_INT64 = ALIGN_NUM / SIZE_OF_INT64;
constexpr uint32_t SPACE_USED_RATIO = 4;
constexpr uint32_t UB_BUFFER_USED = 128;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t TILING_KEY_100 = 100;

static void FeedsRepeatPrintParam(gert::TilingContext* context, FeedsRepeatTilingData& tiling)
{
    auto nodeName = context;
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print FeedsRepeat tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, ">>> op [TilingData]: elem_row = %ld", tiling.get_elem_row());
    OP_LOGD(nodeName, ">>> op [TilingData]: elem_per_loop = %ld", tiling.get_elem_per_loop());
    OP_LOGD(nodeName, ">>> op [TilingData]: length = %u", tiling.get_length());
    OP_LOGD(nodeName, ">>> op [TilingData]: length_aligned = %u", tiling.get_length_aligned());
    OP_LOGD(nodeName, ">>> op [TilingData]: max_core_num = %ld", tiling.get_max_core_num());
    OP_LOGD(nodeName, ">>> op [TilingData]: core_per_group = %ld", tiling.get_core_per_group());
    OP_LOGD(nodeName, ">>> op [TilingData]: core_moreover = %ld", tiling.get_core_moreover());
    OP_LOGD(nodeName, ">>> op [TilingData]: empty_size = %ld", tiling.get_empty_size());
    OP_LOGD(nodeName, ">>> op [TilingData]: row_per_core = %ld", tiling.get_row_per_core());
    OP_LOGD(nodeName, ">>> op [TilingData]: row_left = %ld", tiling.get_row_left());
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> End print FeedsRepeat tiling data <<<<<<<<<<<<<<<<");
}

static ge::graphStatus SetTilingBatch(const gert::TilingContext* context, FeedsRepeatTilingData& tiling)
{
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto max_core_num = ascendcPlatform.GetCoreNumAiv();
    const gert::Shape feeds_shape = context->GetInputShape(0)->GetStorageShape();
    int64_t batch_num = feeds_shape.GetDim(0);
    int64_t group = 0;
    int64_t core_per_group = 0;
    int64_t core_moreover = 0;
    OP_CHECK_IF(
        batch_num == 0, OP_LOGE(context->GetNodeName(), "[FeedsRepeat] The batch_num of feeds_shape should not be 0"),
        return ge::GRAPH_FAILED);
    if (batch_num < max_core_num) {
        group = batch_num;
        core_per_group = max_core_num / batch_num;
        core_moreover = max_core_num % batch_num;
    } else {
        group = max_core_num;
    }
    int64_t row_per_core = batch_num / group;
    int64_t row_left = batch_num % group;
    tiling.set_max_core_num(max_core_num);
    tiling.set_core_per_group(core_per_group);
    tiling.set_core_moreover(core_moreover);
    tiling.set_row_per_core(row_per_core);
    tiling.set_row_left(row_left);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingLength(
    const gert::TilingContext* context, FeedsRepeatTilingData& tiling, uint32_t& tiling_key, uint32_t& length_space)
{
    uint32_t length = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    uint32_t length_aligned = 0;
    OP_CHECK_IF(
        length != context->GetInputShape(1)->GetStorageShape().GetDim(0),
        OP_LOGE(
            context->GetNodeName(),
            "[FeedsRepeat] The length of feeds_repeat_times should be same as feeds' dim0 size"),
        return ge::GRAPH_FAILED);
    auto dtype_length = context->GetInputDesc(1)->GetDataType();
    if (dtype_length == ge::DT_INT32) {
        length_space = ((length + ALIGN_INT32 - 1) / ALIGN_INT32) * ALIGN_NUM;
        length_aligned = length_space / SIZE_OF_INT32;
    } else if (dtype_length == ge::DT_INT64) {
        tiling_key = TILING_KEY_100; // feeds_repeat_times为int64时tiling_key百位为1
        length_space = ((length + ALIGN_INT64 - 1) / ALIGN_INT64) * ALIGN_NUM;
        length_aligned = length_space / SIZE_OF_INT64;
    } else {
        OP_LOGD("feeds_repeat_times' dtype only support int32, int64.");
        return ge::GRAPH_FAILED;
    }
    tiling.set_length(length);
    tiling.set_length_aligned(length_aligned);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingElemRow(
    const gert::TilingContext* context, FeedsRepeatTilingData& tiling, uint32_t& tiling_key, uint32_t& length_space)
{
    const gert::Shape feeds_shape = context->GetInputShape(0)->GetStorageShape();
    int64_t elem_row = 1;
    int64_t dim_num = feeds_shape.GetDimNum();
    for (int64_t i = 1; i < dim_num; i++) {
        elem_row *= feeds_shape.GetDim(i);
    }
    auto dtype = context->GetInputDesc(0)->GetDataType();
    uint64_t max_ub_size;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, max_ub_size);
    int64_t elem_row_aligned = 0;
    int64_t elem_per_loop = 0;
    uint32_t align_data = 0;
    if (dtype == ge::DT_FLOAT) {
        tiling_key += 1; // feeds为fp32,tiling_key后两位为1
        align_data = ALIGN_FP32;
        elem_per_loop = (max_ub_size - length_space * SPACE_USED_RATIO - UB_BUFFER_USED) / SIZE_OF_FP32 / DOUBLE_BUFFER;
    } else if (dtype == ge::DT_FLOAT16) {
        tiling_key += 2; // feeds为fp16,tiling_key后两位为2
        align_data = ALIGN_FP16;
        elem_per_loop = (max_ub_size - length_space * SPACE_USED_RATIO - UB_BUFFER_USED) / SIZE_OF_FP16 / DOUBLE_BUFFER;
    } else if (dtype == ge::DT_BF16) {
        tiling_key += 3; // feeds为bf16,tiling_key后两位为3
        align_data = ALIGN_BF16;
        elem_per_loop = (max_ub_size - length_space * SPACE_USED_RATIO - UB_BUFFER_USED) / SIZE_OF_BF16 / DOUBLE_BUFFER;
    } else {
        OP_LOGD("feeds' dtype only support fp32, fp16, bf16 for now.");
        return ge::GRAPH_FAILED;
    }
    elem_row_aligned = (elem_row + align_data - 1) / align_data * align_data;
    OP_CHECK_IF(
        SPACE_USED_RATIO * length_space >=
            max_ub_size - UB_BUFFER_USED, // sum(), cast() and other buffers needs ub space
        OP_LOGE(context->GetNodeName(), "[FeedsRepeat] feeds_repeat_times is too large"), return ge::GRAPH_FAILED);
    elem_per_loop = elem_per_loop / align_data * align_data;
    elem_per_loop = elem_row_aligned > elem_per_loop ? elem_per_loop : elem_row_aligned;
    if (elem_per_loop == 0) {
        OP_LOGD("Tiling data error: elem_per_loop is 0 as a divisor.");
        return ge::GRAPH_FAILED;
    }
    tiling.set_elem_row(elem_row);
    tiling.set_elem_per_loop(elem_per_loop);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4FeedsRepeat(gert::TilingContext* context)
{
    OP_LOGD("FeedsRepeat tiling start");
    FeedsRepeatTilingData tiling;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto max_core_num = ascendcPlatform.GetCoreNumAiv();
    SetTilingBatch(context, tiling);
    uint32_t tiling_key = 0;
    uint32_t length_space = 0;
    SetTilingLength(context, tiling, tiling_key, length_space);
    SetTilingElemRow(context, tiling, tiling_key, length_space);
    int64_t empty_size = *context->GetAttrs()->GetInt(0);
    tiling.set_empty_size(empty_size);
    context->SetTilingKey(tiling_key);
    OP_LOGD(context, ">>> [FeedsRepeat] tilingKey: %u", tiling_key);
    context->SetBlockDim(max_core_num);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    FeedsRepeatPrintParam(context, tiling);
    OP_LOGD("Tiling4FeedsRepeat tiling end");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4FeedsRepeat(gert::TilingParseContext* context)
{
    OP_LOGD("FeedsRepeat:", "TilingPrepareForFeedsRepeat start.");
    auto compileInfo = context->GetCompiledInfo<FeedsRepeatCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->total_core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->total_core_num <= 0), // 0 negative number
        OP_LOGE(context->GetNodeName(), "Failed to get core num."), return ge::GRAPH_FAILED);

    uint64_t ub_size_platform = 0U; // 0, init
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size_platform);
    compileInfo->ub_size_platform = static_cast<int64_t>(ub_size_platform);
    OP_CHECK_IF(
        (compileInfo->ub_size_platform <= 0), // 0
        OP_LOGE(context->GetNodeName(), "Failed to get ub size"), return ge::GRAPH_FAILED);
    OP_LOGD("FeedsRepeat:", "TilingPrepareForFeedsRepeat end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FeedsRepeat).Tiling(Tiling4FeedsRepeat).TilingParse<FeedsRepeatCompileInfo>(TilingPrepare4FeedsRepeat);
} // namespace optiling