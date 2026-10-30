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
 * \file fused_floyd_attention_grad_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(FFAGEmptyTensorTilingData)
TILING_DATA_FIELD_DEF(uint32_t, formerDqNum);
TILING_DATA_FIELD_DEF(uint32_t, formerDkNum);
TILING_DATA_FIELD_DEF(uint32_t, res);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreDqNum);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreDqNum);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreDkNum);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreDkNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FFAGEmptyTensorTilingDataOp, FFAGEmptyTensorTilingData)

BEGIN_TILING_DATA_DEF(FusedFloydAttentionGradTilingData)
TILING_DATA_FIELD_DEF_STRUCT(FFAGEmptyTensorTilingData, emptyTensorTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FusedFloydAttentionGrad_90, FusedFloydAttentionGradTilingData)
} // namespace optiling
