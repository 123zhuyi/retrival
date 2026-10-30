/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gcd.cpp
 * \brief Gcd AiCore entry
 */

#include "gcd.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void gcd(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)x1;
    (void)x2;
    (void)y;
    (void)workspace;
    (void)tiling;

    REGISTER_TILING_DEFAULT(GcdTilingData);

    GM_ADDR* paramBase = reinterpret_cast<GM_ADDR*>(get_para_base());
    constexpr uint32_t X1_PARAM_INDEX = 0;
    constexpr uint32_t X2_PARAM_INDEX = 1;
    constexpr uint32_t Y_PARAM_INDEX = 2;
    constexpr uint32_t INLINE_TILING_PARAM_INDEX = 5;
    const GcdTilingData* tilingData = reinterpret_cast<const GcdTilingData*>(paramBase + INLINE_TILING_PARAM_INDEX);

    if constexpr (AscendC::IsSameType<DTYPE_X1, uint8_t>::value && AscendC::IsSameType<DTYPE_X2, bfloat16_t>::value &&
                  AscendC::IsSameType<DTYPE_Y, uint8_t>::value) {
        GcdKernelUint8Bf16ToUint8<false> op;
        op.Init(paramBase[X1_PARAM_INDEX], paramBase[X2_PARAM_INDEX], paramBase[Y_PARAM_INDEX], tilingData);
        op.Process();
    } else if constexpr (AscendC::IsSameType<DTYPE_X1, bfloat16_t>::value &&
                         AscendC::IsSameType<DTYPE_X2, uint8_t>::value &&
                         AscendC::IsSameType<DTYPE_Y, uint8_t>::value) {
        GcdKernelUint8Bf16ToUint8<true> op;
        op.Init(paramBase[X1_PARAM_INDEX], paramBase[X2_PARAM_INDEX], paramBase[Y_PARAM_INDEX], tilingData);
        op.Process();
    } else if constexpr (AscendC::IsSameType<DTYPE_X1, int8_t>::value && AscendC::IsSameType<DTYPE_X2, float>::value &&
                         AscendC::IsSameType<DTYPE_Y, int8_t>::value) {
        GcdKernelInt8FloatToInt8<false> op;
        op.Init(paramBase[X1_PARAM_INDEX], paramBase[X2_PARAM_INDEX], paramBase[Y_PARAM_INDEX], tilingData);
        op.Process();
    } else if constexpr (AscendC::IsSameType<DTYPE_X1, float>::value && AscendC::IsSameType<DTYPE_X2, int8_t>::value &&
                         AscendC::IsSameType<DTYPE_Y, int8_t>::value) {
        GcdKernelInt8FloatToInt8<true> op;
        op.Init(paramBase[X1_PARAM_INDEX], paramBase[X2_PARAM_INDEX], paramBase[Y_PARAM_INDEX], tilingData);
        op.Process();
    } else if constexpr (AscendC::IsSameType<DTYPE_X1, int16_t>::value && AscendC::IsSameType<DTYPE_X2, half>::value &&
                         AscendC::IsSameType<DTYPE_Y, int16_t>::value) {
        GcdKernelInt16Fp16ToInt16<false> op;
        op.Init(paramBase[X1_PARAM_INDEX], paramBase[X2_PARAM_INDEX], paramBase[Y_PARAM_INDEX], tilingData);
        op.Process();
    } else if constexpr (AscendC::IsSameType<DTYPE_X1, half>::value && AscendC::IsSameType<DTYPE_X2, int16_t>::value &&
                         AscendC::IsSameType<DTYPE_Y, int16_t>::value) {
        GcdKernelInt16Fp16ToInt16<true> op;
        op.Init(paramBase[X1_PARAM_INDEX], paramBase[X2_PARAM_INDEX], paramBase[Y_PARAM_INDEX], tilingData);
        op.Process();
    } else {
        GcdKernel<DTYPE_X1> op;
        op.Init(paramBase[X1_PARAM_INDEX], paramBase[X2_PARAM_INDEX], paramBase[Y_PARAM_INDEX], tilingData);
        op.Process();
    }
}
