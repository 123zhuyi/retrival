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
 * \file moe_distribute_dispatch_v3_tiling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>

#include "tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../../moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_tiling_v2.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "mc2_hcom_topo_info.h"

#ifdef MC2_EXCEPTION_HANDLER
#include "mc2_exception_dump.h"
using namespace Mc2Exception;
#endif

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;


namespace optiling {
static ge::graphStatus MoeDistributeDispatchV3TilingFunc(gert::TilingContext* context)
{
    DispatchV2Config config;
    config.contextIndex = 0;  // 0: 根据dispatchV3算子原型标志位初始化context索引
    config.xIndex = 1; // 1: 根据dispatchV3算子原型标志位初始化groupEp索引
    config.expertIdsIndex = 2; // 2: 根据dispatchV3算子原型标志位初始化expertIds索引
    config.scalesIndex = 3; // 3: 根据dispatchV3算子原型标志位初始化scales索引
    config.xActiveMaskIndex = 4; // 4: 根据dispatchV3算子原型标志位初始化xActiveMask索引
    config.expertScalesIndex = 5; // 5: 根据dispatchV3算子原型标志位初始化expertScales索引
    config.elasticInfoIndex = 6; // 6: 根据dispatchV3算子原型标志位初始化elasticInfo索引
    config.performanceInfoIndex = 7; // 7: 根据dispatchV3算子原型标志位初始化performanceInfo索引
    config.attrEpWorldSizeIndex = 0; // 0: 根据dispatchV3算子原型标志位初始化epWorldSize索引
    config.attrEpRankIdIndex = 1; // 1: 根据dispatchV3算子原型标志位初始化epRankId索引
    config.attrMoeExpertNumIndex = 2;  // 2: 根据dispatchV3算子原型标志位初始化moeExpertNum索引
    config.attrCclBufferSizeIndex = 3; // 3: 根据dispatchV3算子原型标志位初始化cclBufferSize索引
    config.attrTpWorldSizeIndex = 4; // 4: 根据dispatchV3算子原型标志位初始化tpWorldSize索引
    config.attrTpRankIdIndex = 5; // 5: 根据dispatchV3算子原型标志位初始化tpRankId索引
    config.attrExpertSharedTypeIndex = 6; // 6: 根据dispatchV3算子原型标志位初始化expertSharedType索引
    config.attrSharedExpertNumIndex = 7; // 7: 根据dispatchV3算子原型标志位初始化sharedExpertNum索引
    config.attrSharedExpertRankNumIndex = 8; // 8: 根据dispatchV3算子原型标志位初始化sharedExpertRankNum索引
    config.attrQuantModeIndex = 9; // 9: 根据dispatchV3算子原型标志位初始化quantMode索引
    config.attrGlobalBsIndex = 10; // 10: 根据dispatchV3算子原型标志位初始化globalBs索引
    config.attrExpertTokenNumsTypeIndex = 11; // 11: 根据dispatchV3算子原型标志位初始化expertTokenNumType索引
    config.attrCommAlgIndex = 12; // 12: 根据dispatchV3算子原型标志位初始化commAlg索引
    config.attrZeroExpertNumIndex = 13; // 13: 根据dispatchV3算子原型标志位初始化zeroExpertNumIndex索引
    config.attrCopyExpertNumIndex = 14; // 14: 根据dispatchV3算子原型标志位初始化copyExpertNumIndex索引
    config.attrConstExpertNumIndex = 15; // 15: 根据dispatchV3算子原型标志位初始化constExpertNumIndex索引
    config.isMc2Context = true;

    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Enter MoeDistributeDispatchV3 tiling");
    ge::graphStatus ret = MoeDistributeDispatchA3TilingFuncImplPublic(context, config);
    return ret;
}

struct MoeDistributeDispatchCompileInfo {};
static ge::graphStatus TilingParseForMoeDistributeDispatchV3(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeDispatchV3)
    .Tiling(MoeDistributeDispatchV3TilingFunc)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatchV3);

#ifdef MC2_EXCEPTION_HANDLER
// Register exception func
inline void MoeDistributeDispatchV3ExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    Mc2ExceptionImpl(args, userdata, "MoeDistributeDispatchV3");
}

IMPL_OP(MoeDistributeDispatchV3)
    .ExceptionDumpParseFunc(MoeDistributeDispatchV3ExceptionImplWrapper);
#endif
} // namespace optiling