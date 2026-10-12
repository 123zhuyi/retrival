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
 * \file trace_apt.cpp
 * \brief Kernel entry for trace operator
 *
 * Single template parameter:
 *   schMode (uint32_t): execution mode (only DEFAULT=0)
 * DTYPE_X macro auto-instantiates for each dtype combination from def.cpp
 */

#include "arch35/trace_simt.h"

template <uint32_t schMode>
__global__ __aicore__ void trace(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TraceTilingData);
    GET_TILING_DATA_WITH_STRUCT(TraceTilingData, tilingData, tiling);
    NsTrace::Process<DTYPE_X>(x, y, &tilingData);
}
