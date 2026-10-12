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
 * \file all_gather_matmul_v2_tiling.cpp
 * \brief
 */

#include "all_gather_matmul_tiling_v2.h"
#include "mc2_log.h"
#include "tiling_base/tiling_templates_registry.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"

using namespace AscendC;
using namespace ge;

namespace optiling
{
ge::graphStatus AllGatherMatmulTilingV2Func(gert::TilingContext* context);
ge::graphStatus TilingParseForAllGatherMatmulV2(gert::TilingParseContext* context);
constexpr uint32_t ATTR_COMMMODE = 11;

ge::graphStatus AllGatherMatmulTilingV2Func(gert::TilingContext* context)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    if (socVersion == "Ascend910B" || socVersion == "Ascend910_93") {
        auto attrs = context->GetAttrs();
        auto commModePtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMMMODE));
        OP_TILING_CHECK((commModePtr == nullptr || !(std::strcmp(commModePtr, "aiv") == 0)),
            OP_LOGE(context->GetNodeName(), "AivModeTiling commMode is invalid. commMode is %s", commModePtr), return ge::GRAPH_FAILED);
        if (std::strcmp(commModePtr, "aiv") == 0) {
            return AllGatherMatmulTilingAIVModeFunc(context);
        }
        return Ops::Transformer::OpTiling::TilingRegistryNew::GetInstance().DoTilingImpl(context);
    }
    return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

struct AllGatherMatmulCompileInfo {
};
ge::graphStatus TilingParseForAllGatherMatmulV2(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AllGatherMatmulV2)
    .Tiling(AllGatherMatmulTilingV2Func)
    .TilingParse<AllGatherMatmulCompileInfo>(TilingParseForAllGatherMatmulV2);
}  // namespace optiling
