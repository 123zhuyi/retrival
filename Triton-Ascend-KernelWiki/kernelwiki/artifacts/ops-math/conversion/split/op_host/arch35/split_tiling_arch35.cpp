/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/split_v/op_host/arch35/split_v_tiling_arch35.h"
#include "op_host/tiling_base_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace gert;
using namespace ge;
using namespace AscendC;

namespace optiling {

graphStatus Tiling4Split(TilingContext* context) {
    OP_LOGD(context->GetNodeName(), "begin to do Tiling4Split");
    auto compile_info = reinterpret_cast<const SplitVCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    uint32_t maxCoreNum = compile_info->maxCoreNum;
    uint32_t ubSizePlatform = compile_info->ubSizePlatform;
    OP_CHECK_IF(ubSizePlatform <= 0, OP_LOGE(context->GetNodeName(),
                    "get ubSize <= 0 error"),
                    return GRAPH_FAILED);
    int32_t isSameLen = 1;
    OP_CHECK_IF(SplitVTilingAscendC(context, maxCoreNum, ubSizePlatform, isSameLen) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(),
                    "AscendC split tiling function call failed"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

graphStatus TilingPrepare4Split(TilingParseContext* context) {
    OP_LOGD(context->GetNodeName(), "AscendC Tiling Starting!");
    auto compile_info = context->GetCompiledInfo<SplitVCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compile_info->maxCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compile_info->maxCoreNum <= 0),
                    OP_LOGE(context->GetNodeName(), "The core num is invalid."),
                    return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compile_info->ubSizePlatform = static_cast<uint32_t>(ubSize);
    OP_CHECK_IF((compile_info->ubSizePlatform <= 0),
                    OP_LOGE(context->GetNodeName(), "The ubSize is invalid."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Split).Tiling(Tiling4Split)
              .TilingParse<SplitVCompileInfo>(TilingPrepare4Split);
}  // namespace optiling
