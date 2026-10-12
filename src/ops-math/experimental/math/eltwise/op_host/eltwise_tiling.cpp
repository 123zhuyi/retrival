/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file eltwise_tiling.cpp
 * \brief Eltwise tiling implementation (arch35)
 *
 * Tiling strategy:
 *   1. Multi-core: divide total elements evenly across AI Cores
 *   2. UB: divide per-core elements into UB-sized chunks
 *   3. Buffer layout depends on dtype:
 *      - FP32:    inputBuf(1) + accBuf(1) + outputBuf(1) = 3 buffers
 *      - FP16/BF16: inputBuf(1) + castBuf(1) + accBuf(1) + outputBuf(1) = 4 buffers
 *
 * UB factor formula:
 *   FP32:    ubFactor = FloorAlign(ubSize / (3 * 4), ubBlockSize)
 *   FP16/BF16: ubFactor = FloorAlign(ubSize / (2 + 4 + 4 + 2), ubBlockSize)
 *            = FloorAlign(ubSize / 12, ubBlockSize)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/eltwise_tiling_data.h"
#include "../op_kernel/eltwise_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint32_t MAX_INPUT_NUM = 32;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        if (coreNum == 0) {
            coreNum = ascendcPlatform.GetCoreNum();
        }
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    }
    // Ascend950 fallback values
    constexpr int64_t FALLBACK_CORE_NUM = 40;
    constexpr uint64_t FALLBACK_UB_SIZE = 253952; // 248 KB
    if (coreNum == 0) {
        OP_LOGW(context, "Eltwise: failed to get core num, using fallback %ld", FALLBACK_CORE_NUM);
        coreNum = FALLBACK_CORE_NUM;
    }
    if (ubSize == 0) {
        OP_LOGW(context, "Eltwise: failed to get ub size, using fallback %lu", FALLBACK_UB_SIZE);
        ubSize = FALLBACK_UB_SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EltwiseTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get input info (DYNAMIC_INPUT: actual count via compute node info)
    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);
    uint32_t inputNum = static_cast<uint32_t>(computeNodeInfo->GetInputsNum());
    OP_CHECK_IF(
        inputNum == 0 || inputNum > 32, OP_LOGE(context, "Eltwise: invalid inputNum %u", inputNum),
        return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();

    auto inputShape0 = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape0);
    auto shape = EnsureNotScalar(inputShape0->GetStorageShape());
    int64_t totalNum = shape.GetShapeSize();
    OP_CHECK_IF(
        totalNum < 0, OP_LOGE(context, "Eltwise: invalid totalNum %ld (negative shape)", totalNum),
        return ge::GRAPH_FAILED);

    // 3. Get mode attribute
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t mode = 1;                                           // default SUM
    const int64_t* modePtr = attrs->GetAttrPointer<int64_t>(0); // "mode" is first attr
    if (modePtr != nullptr) {
        mode = *modePtr;
    }
    OP_CHECK_IF(mode < 0 || mode > 2, OP_LOGE(context, "Eltwise: invalid mode %ld", mode), return ge::GRAPH_FAILED);

    // 4. Get workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 5. Compute tiling parameters
    EltwiseTilingData* tiling = context->GetTilingData<EltwiseTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(EltwiseTilingData), 0, sizeof(EltwiseTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // Determine type size
    int64_t typeSize = 4;
    switch (dtype) {
        case ge::DT_FLOAT:
            typeSize = 4;
            break;
        case ge::DT_FLOAT16:
            typeSize = 2;
            break;
        case ge::DT_BF16:
            typeSize = 2;
            break;
        default:
            OP_LOGE(context, "Eltwise: unsupported dtype %d", static_cast<int>(dtype));
            return ge::GRAPH_FAILED;
    }

    int64_t ubBlockSize = 32 / typeSize; // 32-byte alignment in elements

    // Handle empty tensor
    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        tiling->inputNum = inputNum;
        context->SetBlockDim(1);
        uint32_t dType = static_cast<uint32_t>(dtype);
        uint32_t modeVal = static_cast<uint32_t>(mode);
        ASCENDC_TPL_SEL_PARAM(context, dType, modeVal);
        return ge::GRAPH_SUCCESS;
    }

    // Multi-core split
    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    blockFactor = ((blockFactor + ubBlockSize - 1) / ubBlockSize) * ubBlockSize;
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);

    // UB split
    // Reserve system overhead for TPipe internal management (queues, barriers, etc.)
    constexpr int64_t UB_SYS_OVERHEAD = 2048; // 2 KB
    int64_t availUbSize = static_cast<int64_t>(ubSize) - UB_SYS_OVERHEAD;
    if (availUbSize < 0) {
        availUbSize = static_cast<int64_t>(ubSize);
    }

    int64_t bytesPerElem;
    if (dtype == ge::DT_FLOAT) {
        // FP32: inputBuf + accBuf + outputBuf = 3 * sizeof(float)
        bytesPerElem = 3 * static_cast<int64_t>(sizeof(float)); // 12
    } else {
        // FP16/BF16: inputBuf(T) + castBuf(fp32) + accBuf(fp32) + outputBuf(T)
        bytesPerElem = 2 * typeSize + 2 * static_cast<int64_t>(sizeof(float)); // 12
    }

    int64_t ubFactor = FloorAlign(availUbSize / bytesPerElem, ubBlockSize);

    // Cap ubFactor so that (ubFactor * typeSize) fits in uint16_t.
    // DataCopyParams.blockLen is uint16_t (max 65535).  The kernel uses
    // blockLen = currentNum * sizeof(T), where currentNum <= ubFactor.
    // Without this cap, large fp32 tiles (>16K elements) overflow blockLen
    // and only a partial copy occurs, corrupting output.
    constexpr int64_t MAX_BLOCK_LEN = 65535;
    int64_t maxUbFactor = FloorAlign(MAX_BLOCK_LEN / typeSize, ubBlockSize);
    if (ubFactor > maxUbFactor) {
        ubFactor = maxUbFactor;
    }

    OP_CHECK_IF(
        ubFactor <= 0, OP_LOGE(context, "Eltwise: ubFactor=%ld, UB too small", ubFactor), return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->inputNum = inputNum;

    // Get coeff values (mode=1 only)
    if (mode == 1) {
        // coeff is the second attr (index 1), it's a listFloat
        const auto* coeffList = attrs->GetListFloat(1);
        if (coeffList != nullptr && coeffList->GetSize() > 0) {
            size_t coeffSize = coeffList->GetSize();
            const float* coeffData = coeffList->GetData();
            if (coeffData != nullptr) {
                for (size_t i = 0; i < coeffSize && i < MAX_INPUT_NUM; i++) {
                    tiling->coeff[i] = coeffData[i];
                }
            }
        } else {
            // Default: all coeff = 1.0
            for (uint32_t i = 0; i < inputNum; i++) {
                tiling->coeff[i] = 1.0f;
            }
        }
    }

    context->SetBlockDim(usedCoreNum);

    // Set TilingKey
    uint32_t dType = static_cast<uint32_t>(dtype);
    uint32_t modeVal = static_cast<uint32_t>(mode);
    ASCENDC_TPL_SEL_PARAM(context, dType, modeVal);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForEltwise([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct EltwiseCompileInfo {};

IMPL_OP_OPTILING(Eltwise).Tiling(EltwiseTilingFunc).TilingParse<EltwiseCompileInfo>(TilingParseForEltwise);

} // namespace optiling
