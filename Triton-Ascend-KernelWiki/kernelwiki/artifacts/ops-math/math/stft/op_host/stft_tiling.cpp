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
 * \file stft_tiling.cc
 * \brief
 */
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "exe_graph/runtime/shape.h"
#include "op_host/math_tiling_templates_registry.h"
#include "tiling/tiling_api.h"
#include "stft_tiling.h"

using namespace AscendC;
using namespace matmul_tiling;

namespace optiling {

static ge::graphStatus Tiling4STFT(gert::TilingContext* context)
{
    // 初始化算子Tiling类
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4STFT(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(
        (platformInfoPtr == nullptr), OP_LOGE(context->GetNodeName(), "platformInfoPtr is null"),
        return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<STFTCompileInfo>();
    OP_CHECK_IF(
        (compileInfoPtr == nullptr), OP_LOGE(context->GetNodeName(), "compileInfoPtr is null"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
    compileInfoPtr->aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicCoreNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(STFT).Tiling(Tiling4STFT).TilingParse<STFTCompileInfo>(TilingPrepare4STFT);

} // namespace optiling
