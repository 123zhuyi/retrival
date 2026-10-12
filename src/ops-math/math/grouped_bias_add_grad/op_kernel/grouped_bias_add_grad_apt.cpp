/**

Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
/*!

\file grouped_bias_add_grad.cpp
\brief grouped bias add grad
*/
#include "arch35/grouped_bias_add_grad_dag.h"
#include "arch35/grouped_bias_add_grad_struct.h"
#include "arch35/grouped_bias_add_grad_tilingkey.h"
#include "arch35/grouped_bias_add_grad_cut_h.h"
#include "arch35/grouped_bias_add_grad_cut_gh.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atvoss/reduce/reduce_sch.h"

using namespace AscendC;
using namespace GBAGradTPL;
using namespace Ops::Base::ReduceOpTmpl;
using namespace GroupedBiasAddGrad;

template <REDUCE_TPL_PARAM, uint32_t TemplateNum, uint32_t GroupIdxDtype>
__global__ __aicore__ void grouped_bias_add_grad(
    GM_ADDR grad_y, GM_ADDR group_idx, GM_ADDR grad_bias, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_NONE_TILING;
    if constexpr (TemplateNum == static_cast<uint32_t>(GroupedBiasAddGradTilingModeArch35::IS_REDUCE_T)) {
        if (g_coreType == AIC) {
            return;
        }
        if (workspace == nullptr) {
            return;
        }
        SetSysWorkspace(workspace);
        GM_ADDR userWS = GetUserWorkspace(workspace);
        if (userWS == nullptr) {
            return;
        }
        GET_TILING_DATA_WITH_STRUCT(GroupedBiasAddGradARATilingData, tilingData, tiling);

        TPipe pipe;
        using Op = ReduceSch<REDUCE_TPL_VALUE, GBAGradTPL::GBAGDag<DTYPE_GRAD_Y>::OpDag>;
        Op op((ReduceOpTilingData*)&tilingData.reduceTiling);
        op.Init(&pipe, grad_y, grad_bias, userWS);
        op.Process();
    } else if constexpr (TemplateNum == static_cast<uint32_t>(GroupedBiasAddGradTilingModeArch35::CUT_H_MODE)) {
        GET_TILING_DATA_WITH_STRUCT(GroupedBiasAddGradCutHTilingData, tilingData, tiling);
        if constexpr (GroupIdxDtype == 0) {
            GroupedBiasAddGradSplitH<DTYPE_GRAD_Y, int32_t> gbag;
            gbag.Init(grad_y, group_idx, grad_bias, &tilingData);
            gbag.Process();
        } else {
            GroupedBiasAddGradSplitH<DTYPE_GRAD_Y, int64_t> gbag;
            gbag.Init(grad_y, group_idx, grad_bias, &tilingData);
            gbag.Process();
        }
    } else if constexpr (TemplateNum == static_cast<uint32_t>(GroupedBiasAddGradTilingModeArch35::CUT_G_MODE)) {
        GET_TILING_DATA_WITH_STRUCT(GroupedBiasAddGradCutGTilingData, tilingData, tiling);
        TPipe pipe;
        if constexpr (GroupIdxDtype == 0) {
            GroupedBiasAddGradCutGH<DTYPE_GRAD_Y, int32_t> op(pipe);
            op.Init(grad_y, group_idx, grad_bias, &tilingData);
            op.Process();
        } else {
            GroupedBiasAddGradCutGH<DTYPE_GRAD_Y, int64_t> op(pipe);
            op.Init(grad_y, group_idx, grad_bias, &tilingData);
            op.Process();
        }
    } else if constexpr (TemplateNum == static_cast<uint32_t>(GroupedBiasAddGradTilingModeArch35::EMPTY_TENSOR)) {
        GET_TILING_DATA_WITH_STRUCT(GroupedBiasAddGradEmptyTensorTilingData, tilingData, tiling);
        //空tensor
    }
}