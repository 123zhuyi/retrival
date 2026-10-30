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
 * \file hans_decode.cpp
 * \brief
 */
#include "hans_decode_base.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void hans_decode(
    GM_ADDR mantissa, GM_ADDR fixed, GM_ADDR var, GM_ADDR pdf, GM_ADDR recover, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    SetSysWorkspace(workspace);
    TPipe pipein;
    HansDecodeNS::HansDecodeInitConfig config = {mantissa, fixed, var, pdf, recover, &tilingData};
#if ORIG_DTYPE_MANTISSA != DT_FLOAT
    if (TILING_KEY_IS(2)) {
#ifdef __DAV_C220_VEC__
        HansDecodeNS::HansDecode<true> op;
        op.Init(&pipein, config);
        op.Process();
#endif
    }
#else
    if (TILING_KEY_IS(4)) {
#ifdef __DAV_C220_VEC__
        HansDecodeNS::HansDecode<false> op;
        op.Init(&pipein, config);
        op.Process();
#endif
    }
#endif
}