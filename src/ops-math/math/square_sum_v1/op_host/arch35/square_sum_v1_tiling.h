/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file square_sum_v1_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_SQUARE_SUM_V1_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_SQUARE_SUM_V1_TILING_H_

#include "register/tilingdata_base.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "../../op_kernel/square_sum_v1_tiling_data.h"

namespace optiling {

using namespace Ops::Base;

struct SquareSumV1TilingKey {
    ReduceTilingKey reduceTiling;
    uint32_t noop = 0;
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_SQUARE_SUM_V1_H_
