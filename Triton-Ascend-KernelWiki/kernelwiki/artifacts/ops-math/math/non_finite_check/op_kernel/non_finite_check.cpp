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
 * \file non_finite_check.cpp
 * \brief
 */
 
#include "non_finite_check_n_d.h"

using namespace NonFiniteCheck;

extern "C" __global__ __aicore__ void non_finite_check(
    GM_ADDR tensor_list, GM_ADDR found_flag, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(101)) {
        NonFiniteCheckND<half> op;
        op.Init(tensor_list, found_flag, &tilingData);
        op.Process();
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    } else if (TILING_KEY_IS(201)) {
        NonFiniteCheckND<bfloat16_t> op;
        op.Init(tensor_list, found_flag, &tilingData);
        op.Process();
#endif
    } else if (TILING_KEY_IS(301)) {
        NonFiniteCheckND<float> op;
        op.Init(tensor_list, found_flag, &tilingData);
        op.Process();
    }
}