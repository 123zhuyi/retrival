/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file cosh_v2_arch32.cpp
 * \brief CoshV2 operator kernel entry (arch32 / Ascend910B)
 *
 * Template parameters (matching ASCENDC_TPL_ARGS_DECL in cosh_v2_tiling_key.h):
 *   - D_T: Data type (half / float / bfloat16_t)
 *   - BUFFER_MODE: Buffer mode (0=single buffer, 1=double buffer)
 */

#include "cosh_v2.h"

template <typename D_T, int BUFFER_MODE>
__global__ __aicore__ void cosh_v2(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CoshV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(CoshV2TilingData, tilingData, tiling);
    NsCoshV2::CoshV2Op<D_T, BUFFER_MODE> op;
    op.Init(self, out, &tilingData);
    op.Process();
}
