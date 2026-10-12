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
 * \file mod_dispatch_impl.h
 * \brief ModKernelImpl's five same-dtype TilingKey dispatch lanes.
 *
 * This file is included from mod.h inside namespace ModNs.
 */
#ifndef MOD_DISPATCH_IMPL_H
#define MOD_DISPATCH_IMPL_H

template <int D_T_X1, int D_T_X2, int D_T_Y>
__aicore__ inline void ModKernelDispatchSameDtype(__gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* y,
                                                  GM_ADDR userWS, const ModNs::ModTilingData* tilingData)
{
    if constexpr (D_T_X1 == MOD_TPL_INT32 && D_T_X2 == MOD_TPL_INT32 && D_T_Y == MOD_TPL_INT32) {
        ModNs::Mod<int> op;
        op.Init(x1, x2, y, userWS, tilingData);
        op.Process();
    } else if constexpr (D_T_X1 == MOD_TPL_FP16 && D_T_X2 == MOD_TPL_FP16 && D_T_Y == MOD_TPL_FP16) {
        ModNs::Mod<half> op;
        op.Init(x1, x2, y, userWS, tilingData);
        op.Process();
    } else if constexpr (D_T_X1 == MOD_TPL_FP32 && D_T_X2 == MOD_TPL_FP32 && D_T_Y == MOD_TPL_FP32) {
        ModNs::Mod<float> op;
        op.Init(x1, x2, y, userWS, tilingData);
        op.Process();
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    } else if constexpr (D_T_X1 == MOD_TPL_BF16 && D_T_X2 == MOD_TPL_BF16 && D_T_Y == MOD_TPL_BF16) {
        ModNs::Mod<bfloat16_t> op;
        op.Init(x1, x2, y, userWS, tilingData);
        op.Process();
#endif
#if MOD_ENH_ARCH22
    } else if constexpr (D_T_X1 == MOD_TPL_INT16 && D_T_X2 == MOD_TPL_INT16 && D_T_Y == MOD_TPL_INT16) {
        ModNs::Mod<int16_t> op;
        op.Init(x1, x2, y, userWS, tilingData);
        op.Process();
#endif
    }
}

template <int D_T_X1, int D_T_X2, int D_T_Y>
__aicore__ inline void ModKernelImpl(__gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* y,
                                     const ModNs::ModTilingData* tilingData)
{
    GM_ADDR userWS = nullptr;
    ModKernelDispatchSameDtype<D_T_X1, D_T_X2, D_T_Y>(x1, x2, y, userWS, tilingData);
}

#endif // MOD_DISPATCH_IMPL_H
