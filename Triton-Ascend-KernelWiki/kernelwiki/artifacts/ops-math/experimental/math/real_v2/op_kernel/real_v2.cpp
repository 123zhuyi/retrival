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
 * \file real_v2.cpp
 * \brief RealV2 operator kernel entry
 *
 * Template parameters (matching real_v2_tiling_key.h ASCENDC_TPL_ARGS_DECL):
 *   - D_T: Output data type, from ASCENDC_TPL_DATATYPE_DECL
 *   - IS_COMPLEX: 0=real passthrough, 1=complex extract real, from ASCENDC_TPL_UINT_DECL
 */

#include "real_v2.h"

template <typename D_T, int IS_COMPLEX>
__global__ __aicore__ void real_v2(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RealV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(RealV2TilingData, tilingData, tiling);
    NsRealV2::RealV2Op<D_T, IS_COMPLEX> op;
    op.Init(self, out, workspace, &tilingData);
    op.Process();
}
