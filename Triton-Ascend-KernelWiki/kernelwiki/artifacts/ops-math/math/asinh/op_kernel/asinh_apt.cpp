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
 * \file asinh_arch35.cpp
 * \brief Asinh 算子 Kernel 入口（arch35 / Ascend950）
 *
 * 与 DESIGN.md v1.1 §3.7 对齐：
 *   - 模板参数 D_T_X 由 asinh_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 定义
 *   - dtype 静态分发：Asinh<float> / Asinh<half> / Asinh<bfloat16_t> 共 3 个实例
 *     由构建系统按 ASCENDC_TPL_SEL 自动生成
 *   - if constexpr 分支在 asinh.h::Compute() 内部实现
 *
 * 迭代一范围：FP32 主线骨架；FP16/BF16 实例的 Compute 已包含 Cast 路径骨架，迭代二验证落地
 */

#include "arch35/asinh.h"

template <typename D_T_X>
__global__ __aicore__ void asinh(GM_ADDR input, GM_ADDR out,
                                  GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    REGISTER_TILING_DEFAULT(AsinhTilingData);
    GET_TILING_DATA_WITH_STRUCT(AsinhTilingData, tilingData, tiling);

    NsAsinh::Asinh<D_T_X> op;
    op.Init(input, out, &tilingData);
    op.Process();
}
