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
 * \file complex_v3_arch32.cpp
 * \brief ComplexV3 operator kernel entry (arch32)
 *
 * Template parameters (matching complex_v3_tiling_key.h ASCENDC_TPL_ARGS_DECL):
 *   - D_T: Data type, mapped from ASCENDC_TPL_DATATYPE_DECL
 *   - BROADCAST_MODE: Broadcast mode (0=no broadcast, 1=broadcast), from ASCENDC_TPL_UINT_DECL
 */

#include "complex_v3.h"

template <typename D_T, int BROADCAST_MODE>
__global__ __aicore__ void complex_v3(GM_ADDR real, GM_ADDR imag, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ComplexV3TilingData);
    GET_TILING_DATA_WITH_STRUCT(ComplexV3TilingData, tilingData, tiling);

    NsComplexV3::ComplexV3<D_T, BROADCAST_MODE> op;
    op.Init(real, imag, out, &tilingData);
    op.Process();
}
