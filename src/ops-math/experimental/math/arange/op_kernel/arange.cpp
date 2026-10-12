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
 * \file arange.cpp
 * \brief
 */

#include "arange.h"

// schMode 取值引用 tiling_key.h 的 MODE 宏：
//   MODE_0 → KernelArange_Cast（Cast 路径，窄整型/半精/int32/int64）；
//   MODE_1 → KernelArange（FP32 直算）。
//   与 host 侧 arange_tiling.cpp 写入 tilingKey 的口径一致（仅 DT_FLOAT→MODE_1，其余→MODE_0）。

template <uint32_t schMode>
__global__ __aicore__ void arange(GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out, GM_ADDR workspace,
                                  GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ArangeTilingData);
    GET_TILING_DATA_WITH_STRUCT(ArangeTilingData, tilingData, tiling);

    // 形参 end 透传给 op.Init 但 kernel 不读取：N 由 caller 经 out 张量 shape 决定，
    //   end 形参保留是 aclnn 自动生成的核函数四 IO 接口签名要求。

    // 场景1：Cast 路径（FP16/BF16/INT8/UINT8/INT16/INT32/INT64 → FP32 中间域算 → Cast 回 out 类型）
    if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_0) {
        NsArange::KernelArange_Cast<DTYPE_START, DTYPE_STEP, DTYPE_OUT> op;

        op.Init(start, end, step, out, tilingData);

        op.Process();
    }

    // 场景2：FP32 直算路径（无 Cast）
    if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_1) {
        NsArange::KernelArange<float, float, float> op;

        op.Init(start, end, step, out, tilingData);

        op.Process();
    }
}
