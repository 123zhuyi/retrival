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
 * \file adds_apt.cpp
 * \brief Adds 算子 Kernel 入口（atvoss 框架 - Elewise 模式）
 */

#include "kernel_operator.h"
#include "atvoss/elewise/elewise_sch.h"
#include "arch35/adds_dag.h"
#include "arch35/adds_tiling_data.h"
#include "arch35/adds_struct.h"

using namespace Ops::Base;
using namespace AscendC;

/**
 * \brief Adds Kernel 入口（Elewise 模式）
 * 
 * 执行流程：
 * 1. REGISTER_TILING_DEFAULT 注册自定义 TilingData 类型
 * 2. GET_TILING_DATA_WITH_STRUCT 获取 TilingData
 * 3. 创建 TPipe（atvoss Elewise 模式要求）
 * 4. 创建 ElementwiseSch 调度器
 * 5. SetVar<T, 0> 注入标量参数（对应 DAG 中的 Placeholder::Var<T, 0>）
 * 6. Init + Process 执行计算
 *
 * 模板参数说明：
 * - schMode: 调度模式（从 TilingKey 获取）
 * - dtype: 数据类型 ID（DT_FLOAT, DT_FLOAT16 等），通过 TypeFromId 转换为实际类型 T
 */
template <uint64_t schMode, uint64_t dtype>
__global__ __aicore__ void adds(GM_ADDR x, GM_ADDR y,
                                GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    
    // 从 dtype ID 获取实际数据类型
    using T = typename TypeFromId<dtype>::type;
    
    // 注册 TilingData 类型（Elewise 模式必须）
    REGISTER_TILING_DEFAULT(AddsTilingData);
    
    // 获取 TilingData（使用 WITH_STRUCT 获取自定义结构体）
    GET_TILING_DATA_WITH_STRUCT(AddsTilingData, tilingData, tiling);
    
    // 创建 TPipe（atvoss Elewise 模式要求使用 TPipe，不是 Pipe）
    TPipe pipe;

    if constexpr (dtype == TPL_FP16) {
        using T = half;
        ElementwiseSch<schMode, typename NsAdds::AddsOp<T, NsAdds::CAST_MODE_NONE, NsAdds::CAST_MODE_RINT>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, 0>(tilingData.scalarValue);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dtype == TPL_FP32) {
        using T = float;
        ElementwiseSch<schMode, typename NsAdds::AddsOp<T, NsAdds::CAST_MODE_NONE, NsAdds::CAST_MODE_RINT>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, 0>(tilingData.scalarValue);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dtype == TPL_BF16) {
        using T = bfloat16_t;
        ElementwiseSch<schMode, typename NsAdds::AddsOp<T, NsAdds::CAST_MODE_NONE, NsAdds::CAST_MODE_RINT>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, 0>(tilingData.scalarValue);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dtype == TPL_INT16) {
        using T = int16_t;
        ElementwiseSch<schMode, NsAdds::AddsInt16Op::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, 0>(tilingData.scalarValue);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dtype == TPL_INT32) {
        using T = int32_t;
        ElementwiseSch<schMode, NsAdds::AddsOp<T,NsAdds::CAST_MODE_RINT, NsAdds::CAST_MODE_TRUNC>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, 0>(tilingData.scalarValue);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dtype == TPL_INT64) {
        using T = int64_t;
        ElementwiseSch<schMode,NsAdds::AddsOp<T,NsAdds::CAST_MODE_RINT, NsAdds::CAST_MODE_TRUNC>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.template SetVar<float, 0>(tilingData.scalarValue);
        sch.Init(x, y);
        sch.Process();
    }
}