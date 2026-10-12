/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kl_div_v2_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_KL_DIV_V2_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_KL_DIV_V2_H
#include "atvoss/reduce/reduce_tiling.h"

namespace optiling {
struct KLDivV2CompileInfo {
  Ops::Base::ReduceOpCompileInfo opInfo;
}; // namespace optiling
}
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_KL_DIV_V2_H_
