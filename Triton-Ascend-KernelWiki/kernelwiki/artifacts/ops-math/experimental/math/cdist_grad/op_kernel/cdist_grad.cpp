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
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file cdist_grad_arch32.cpp
 * \brief CdistGrad Kernel entry point (arch32)
 *
 * Template parameters (matching cdist_grad_tiling_key.h):
 *   - D_T: Data type (float, half)
 *   - P_MODE: p value mode (0=p1, 1=p2, 2=pinf, 3=general)
 *   - SCH_MODE: Schedule mode (0=FullM, 1=SplitM reserved)
 */

#include "common/cdist_grad.h"

template <typename D_T, int P_MODE, int SCH_MODE>
__global__ __aicore__ void cdist_grad(
    GM_ADDR gradOutput, GM_ADDR x1, GM_ADDR x2, GM_ADDR cdistResult, GM_ADDR gradX1, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CdistGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(CdistGradTilingData, tilingData, tiling);

    if constexpr (SCH_MODE == 0) {
        // FullM mode
        if constexpr (P_MODE == 0) {
            NsCdistGrad::CdistGradP1<D_T> op;
            op.Init(gradOutput, x1, x2, cdistResult, gradX1, &tilingData);
            op.Process();
        } else if constexpr (P_MODE == 1) {
            NsCdistGrad::CdistGradP2<D_T> op;
            op.Init(gradOutput, x1, x2, cdistResult, gradX1, &tilingData);
            op.Process();
        } else if constexpr (P_MODE == 2) {
            NsCdistGrad::CdistGradPInf<D_T> op;
            op.Init(gradOutput, x1, x2, cdistResult, gradX1, &tilingData);
            op.Process();
        } else if constexpr (P_MODE == 3) {
            NsCdistGrad::CdistGradPGeneral<D_T> op;
            op.Init(gradOutput, x1, x2, cdistResult, gradX1, &tilingData);
            op.Process();
        }
    }
    // SCH_MODE == 1 (SplitM) reserved for iteration 3
}
