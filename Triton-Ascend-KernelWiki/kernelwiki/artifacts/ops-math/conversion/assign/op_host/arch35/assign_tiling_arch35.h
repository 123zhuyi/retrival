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
 * \file assign_tiling_arch35.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_ASSIGN_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_ASSIGN_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "conversion/tensor_move/op_host/arch35/tensor_move_tiling_arch35.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(Assign, TensorMoveTilingData)

struct AssignTilingParam {
    int64_t totalCoreNum;
    int64_t ubSize;
    int64_t uo;
    int64_t usedCoreNum;
    int64_t bytesForOneData;
    int64_t ubFactor;
    int64_t tailBlockTailUbFactor;
    int64_t blockFactor;
    int64_t tailBlockFactor;
    int64_t tilingKey;
};

struct AssignCompileInfo {
  int64_t coreNum;
  int64_t ubSize;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_ASSIGN_H_