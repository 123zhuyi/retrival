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
 * \file sign.cpp
 * \brief
 */
#include "arch35/sign.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/elewise/elewise_base_struct.h"
#include "sign_struct.h"

using namespace Ops::Base;
using namespace SignNs;

extern "C" __global__ __aicore__ void sign(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SignTilingData);
    GET_TILING_DATA_WITH_STRUCT(SignTilingData, tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(101UL)) {
        ElementwiseSch<0UL, SignDag::SignForNumber<half>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
        return;
    } else if (TILING_KEY_IS(102UL)) {
        ElementwiseSch<0UL, SignDag::SignForBf<bfloat16_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
        return;
    } else if (TILING_KEY_IS(103UL)) {
        ElementwiseSch<0UL, SignDag::SignForNumber<float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
        return;
    } else if (TILING_KEY_IS(104UL)) {
        ElementwiseSch<0UL, SignDag::SignForNumber<int32_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
        return;
    } else if (TILING_KEY_IS(105UL)) {
        ElementwiseSch<0UL, SignDag::SignForInt64<int64_t>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
        return;
    }
    return;
}