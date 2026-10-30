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
 * \file round_optiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ROUND_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ROUND_TILING_H

#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "../../op_kernel/arch35/round_tiling_struct.h"

// 声明具有外部链接的对象或函数
namespace optiling {
extern void Tiling4Round();
extern void TilingPrepareForRound();
}

namespace optiling {
using namespace Ops::Base;

struct RoundCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_ROUND_TILING_H
