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
 * \file cumsum_tiling.h
 * \brief calc corenum and threadnum for AscendC kernel
 */
#ifndef _CUMSUM_TILING_H
#define _CUMSUM_TILING_H
#include <cstdint>
#include "op_host/tiling_base_class.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(CumsumSimtTilingData)
TILING_DATA_FIELD_DEF(uint32_t, block_factor);
TILING_DATA_FIELD_DEF(int64_t, outer_axis);
TILING_DATA_FIELD_DEF(int64_t, inner_axis);
TILING_DATA_FIELD_DEF(int64_t, target_axis);
TILING_DATA_FIELD_DEF(int64_t, exclusive);
TILING_DATA_FIELD_DEF(int64_t, reverse);
TILING_DATA_FIELD_DEF(uint32_t, use_core_num);
TILING_DATA_FIELD_DEF(uint32_t, mode);
TILING_DATA_FIELD_DEF(uint32_t, thread_dim_x);
TILING_DATA_FIELD_DEF(uint32_t, thread_dim_y);
TILING_DATA_FIELD_DEF(uint32_t, thread_dim_z);
TILING_DATA_FIELD_DEF(uint32_t, block_loop_num);
TILING_DATA_FIELD_DEF(uint32_t, outer_irows);
TILING_DATA_FIELD_DEF(int64_t, axis);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Cumsum, CumsumSimtTilingData)

} // namespace optiling
#endif // _CUMSUM_TILING_H
