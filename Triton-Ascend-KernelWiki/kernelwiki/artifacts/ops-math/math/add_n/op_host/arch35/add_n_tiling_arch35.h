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
 * \file add_n_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ADD_N_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ADD_N_TILING_H

#include "atvoss/elewise/elewise_tiling.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_base_class.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AddNTilingData)
TILING_DATA_FIELD_DEF(int64_t, dim0);
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, elemNum);
TILING_DATA_FIELD_DEF(int64_t, sizeN);
TILING_DATA_FIELD_DEF(int64_t, firstN);
TILING_DATA_FIELD_DEF(int64_t, loopN);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddN, AddNTilingData)

struct AddNCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ADD_N_TILING_H