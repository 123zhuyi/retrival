/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <type_traits>
#include "im2col_kernel.h"
#include "im2col_channel_transpose.h"
#include "im2col_tiling_data.h"
#include "im2col_tiling_key.h"

template <typename D_T, uint32_t PATH>
__global__ __aicore__ void im2col(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(Im2colTilingHeader);
    GET_TILING_DATA_WITH_STRUCT(Im2colTilingHeader, tilingData, tiling);

    using StorageT = std::conditional_t<sizeof(D_T) == sizeof(int8_t), int8_t,
                                        std::conditional_t<sizeof(D_T) == sizeof(uint16_t), uint16_t, uint32_t>>;
    AscendC::TPipe pipe;
    if constexpr (PATH == IM2COL_PATH_CHANNEL_TRANSPOSE) {
        NsIm2col::Im2colChannelTransposeKernel<StorageT> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else {
        NsIm2col::Im2colKernel<StorageT, PATH> op;
        op.Init(x, y, tiling, &tilingData, &pipe);
        op.Process();
    }
}
