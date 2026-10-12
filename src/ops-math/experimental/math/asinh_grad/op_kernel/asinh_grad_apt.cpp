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

/*!
 * \file asinh_grad_apt.cpp
 * \brief AsinhGrad kernel entry point (arch35 / Ascend950)
 *
 * Template parameters (matching asinh_grad_tiling_key.h):
 *   - D_T_Y: data type of y/dy
 *   - BUFFER_MODE: 0=single buffer, 1=double buffer
 */

#include "arch35/asinh_grad.h"

template <typename D_T_Y, int BUFFER_MODE>
__global__ __aicore__ void asinh_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AsinhGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(AsinhGradTilingData, tilingData, tiling);
    NsAsinhGrad::AsinhGrad<D_T_Y, BUFFER_MODE> op;
    op.Init(y, dy, z, &tilingData);
    op.Process();
}
