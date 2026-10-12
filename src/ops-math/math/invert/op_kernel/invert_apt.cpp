/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file invert.cpp
 * \brief invert kernel
 */

#include "atvoss/elewise/elewise_sch.h"
#include "arch35/invert_dag.h"
#include "atvoss/elewise/elewise_base_struct.h"

using namespace AscendC;
using namespace Ops::Base;

extern "C" __global__ __aicore__ void invert(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(EleBaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(EleBaseTilingData, tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(101UL)) {
        if constexpr (std::is_same<DTYPE_X, int8_t>::value) {
            ElementwiseSch<0UL, InvertDag<int8_t>::OpDag> sch(&tilingData, &pipe);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr (std::is_same<DTYPE_X, int16_t>::value) {
            ElementwiseSch<0UL, InvertDag<int16_t>::OpDag> sch(&tilingData, &pipe);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr (std::is_same<DTYPE_X, int32_t>::value) {
            ElementwiseSch<0UL, InvertDag<int32_t>::OpDag> sch(&tilingData, &pipe);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr (std::is_same<DTYPE_X, int64_t>::value) {
            ElementwiseSch<0UL, InvertDag<int64_t>::OpDag> sch(&tilingData, &pipe);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr (std::is_same<DTYPE_X, uint8_t>::value) {
            ElementwiseSch<0UL, InvertDag<uint8_t>::OpDag> sch(&tilingData, &pipe);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr (std::is_same<DTYPE_X, uint16_t>::value) {
            ElementwiseSch<0UL, InvertDag<uint16_t>::OpDag> sch(&tilingData, &pipe);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr (std::is_same<DTYPE_X, uint32_t>::value) {
            ElementwiseSch<0UL, InvertDag<uint32_t>::OpDag> sch(&tilingData, &pipe);
            sch.Init(x, y);
            sch.Process();
        } else if constexpr (std::is_same<DTYPE_X, uint64_t>::value) {
            ElementwiseSch<0UL, InvertDag<uint64_t>::OpDag> sch(&tilingData, &pipe);
            sch.Init(x, y);
            sch.Process();
        }
    }
    return;
}