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
 * \file moe_token_permute_with_routing_map_grad.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_GRAD_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_GRAD_H

#include "moe_token_permute_grad_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MoeTokenpermuteWithRoutingMapDropPadTilingData)
TILING_DATA_FIELD_DEF(int64_t, capacity)
TILING_DATA_FIELD_DEF(int64_t, expertNum)
TILING_DATA_FIELD_DEF(int64_t, tokenNum)
TILING_DATA_FIELD_DEF(int64_t, singleCoreLen)
TILING_DATA_FIELD_DEF(int64_t, lastCoreLen)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(
    MoeTokenpermuteWithRoutingMapDropPadTilingDataOp, MoeTokenpermuteWithRoutingMapDropPadTilingData)

REGISTER_TILING_DATA_CLASS(MoeTokenPermuteWithRoutingMapGradUnpermuteTilingDataOp, MoeTokenPermuteWithRoutingMapGradUnpermuteTilingData)

BEGIN_TILING_DATA_DEF(MoeTokenPermuteWithRoutingMapGradTilingData)
TILING_DATA_FIELD_DEF_STRUCT(MoeTokenPermuteWithRoutingMapGradUnpermuteTilingData, moeTokenPermuteWithRoutingMapGradUnpermuteTilingData)
TILING_DATA_FIELD_DEF_STRUCT(
    MoeTokenpermuteWithRoutingMapDropPadTilingData, moeTokenpermuteWithRoutingMapDropPadTilingData)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(MoeTokenPermuteWithRoutingMapGrad, MoeTokenPermuteWithRoutingMapGradTilingData)

struct MoeTokenPermuteWithRoutingMapGradCompileInfo {
};

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_PERMUTE_GRAD_H
