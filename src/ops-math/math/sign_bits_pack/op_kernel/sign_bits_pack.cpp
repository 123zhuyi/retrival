/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "arch35/sign_bits_pack_struct.h"
#include "arch35/sign_bits_pack_tiling_data.h"
#include "arch35/sign_bits_pack_kernel.h"

template <int UB_AXES_IN_BLOCK>
__global__ __aicore__ void sign_bits_pack(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(SignBitsPackTilingData);
    GET_TILING_DATA_WITH_STRUCT(SignBitsPackTilingData, tilingData, tiling);
    SignBitsPackKernel<DTYPE_X, UB_AXES_IN_BLOCK> kernel;
    kernel.Init(x, y, &tilingData);
    kernel.Process();
}
