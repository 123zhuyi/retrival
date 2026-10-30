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
 * \file trilu.cc
 * \brief
 */

#include <algorithm>
#include "triu_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "assert.h"
#include "platform/platform_ascendc.h"

using namespace Ops::Base;

namespace optiling {
constexpr int64_t MIN_ELT_NUM_PER_CORE = 256;

template <typename T>
inline static T* GetCompileInfoPtr(gert::TilingParseContext* context)
{
    return context->GetCompiledInfo<T>();
}

int64_t GetEltNumPerCore(
    int64_t availableUBSize, int32_t availableCoreNum, int64_t unitNum, uint8_t dtypeBytes, int64_t totalEltNum)
{
    assert(unitNum > 0);
    assert(dtypeBytes > 0);
    assert(availableCoreNum > 0);
    // we wanna use all available cores, and make sure there're enough(256 at least) elements on each core as well
    int64_t eltNumPerCore =
        (std::max(totalEltNum / availableCoreNum, MIN_ELT_NUM_PER_CORE) + unitNum - 1) / unitNum * unitNum;
    // next, we need to make sure there's avalible ub space to use on each core
    eltNumPerCore = std::min(availableUBSize / dtypeBytes / unitNum * unitNum, eltNumPerCore);
    // also, the input tensor may be too samll
    eltNumPerCore = std::min(totalEltNum, eltNumPerCore);
    return eltNumPerCore;
}

int64_t GetAlignedSize(int64_t unitSize)
{
    int64_t alignedSize = 32;
    for (auto i = 1; i < BLOCK_BYTES; ++i) {
        if (unitSize * i % BLOCK_BYTES == 0) {
            alignedSize = i;
            break;
        }
    }
    return alignedSize;
}

void PrintInfo(gert::TilingContext* context)
{
    auto params = context->GetTilingData<optiling::TriluTilingParams>();
    OP_LOGI(context->GetNodeName(), "op [TilingData] : taskNum=%ld", params->taskNum);
    OP_LOGI(context->GetNodeName(), "op [TilingData] : mask=%ld", params->mask);
    OP_LOGI(context->GetNodeName(), "op [TilingData] : matrixNum=%ld", params->matrixNum);
    OP_LOGI(context->GetNodeName(), "op [TilingData] : row=%ld", params->row);
    OP_LOGI(context->GetNodeName(), "op [TilingData] : col=%ld", params->col);
    OP_LOGI(context->GetNodeName(), "op [TilingData] : tilingMode=%ld", params->tilingMode);
    OP_LOGI(context->GetNodeName(), "op [TilingData] : diagonal=%ld", params->diagonal);
    OP_LOGI(context->GetNodeName(), "op [TilingData] : eleNumPerCore=%ld", params->eltNumPerCore);
    OP_LOGI(context->GetNodeName(), "op [TilingData] : usedCoreNum=%ld", params->usedCoreNum);
}

static ge::graphStatus TilingPrepareTrilForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareTrilForAscendC entering.");
    auto compileInfo = GetCompileInfoPtr<TriluCompileInfo>(context);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->availableAICoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->availableAICoreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to core num."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->availableUBSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (compileInfo->availableUBSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4Trilu(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4Tril in");
    return TilingPrepareTrilForAscendC(context);
}

IMPL_OP_OPTILING(Triu).Tiling(Tiling4Trilu<true, false>).TilingParse<TriluCompileInfo>(TilingPrepare4Trilu);

} // namespace optiling
