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
 * \file masked_select_v3.cpp
 * \brief
 */
#include "masked_select_v3.h"

extern "C" __global__ __aicore__ void masked_select_v3(GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR shapeout, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace); // 获取用户workspace指针。

    if (TILING_KEY_IS(8)) {
        AscendC::KernelMaskedSelectV3<uint64_t> op;
        op.Init(x, mask, y, shapeout, usrWorkspace,
            tiling_data.formerNum,
            tiling_data.formerLength,
            tiling_data.formertileNum,
            tiling_data.formertileLength,
            tiling_data.formerlasttileLength,
            tiling_data.tailNum,
            tiling_data.tailLength,
            tiling_data.tailtileNum,
            tiling_data.tailtileLength,
            tiling_data.taillasttileLength);
        op.Process(y, shapeout);
    } else if (TILING_KEY_IS(4)) {
        AscendC::KernelMaskedSelectV3<uint32_t> op;
        op.Init(x, mask, y, shapeout, usrWorkspace,
            tiling_data.formerNum,
            tiling_data.formerLength,
            tiling_data.formertileNum,
            tiling_data.formertileLength,
            tiling_data.formerlasttileLength,
            tiling_data.tailNum,
            tiling_data.tailLength,
            tiling_data.tailtileNum,
            tiling_data.tailtileLength,
            tiling_data.taillasttileLength);
        op.Process(y, shapeout);
    } else if (TILING_KEY_IS(2)) {
        AscendC::KernelMaskedSelectV3<uint16_t> op;
        op.Init(x, mask, y, shapeout, usrWorkspace,
            tiling_data.formerNum,
            tiling_data.formerLength,
            tiling_data.formertileNum,
            tiling_data.formertileLength,
            tiling_data.formerlasttileLength,
            tiling_data.tailNum,
            tiling_data.tailLength,
            tiling_data.tailtileNum,
            tiling_data.tailtileLength,
            tiling_data.taillasttileLength);
        op.Process(y, shapeout);
    } else if (TILING_KEY_IS(1)) {
        AscendC::KernelMaskedSelectV3<uint8_t> op;
        op.Init(x, mask, y, shapeout, usrWorkspace,
            tiling_data.formerNum,
            tiling_data.formerLength,
            tiling_data.formertileNum,
            tiling_data.formertileLength,
            tiling_data.formerlasttileLength,
            tiling_data.tailNum,
            tiling_data.tailLength,
            tiling_data.tailtileNum,
            tiling_data.tailtileLength,
            tiling_data.taillasttileLength);
        op.Process(y, shapeout);
    }
}