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
 * \file addr_v2_struct.h
 * \brief addr_v2 tiling data struct and tiling key definition (arch22 / Ascend910B)
 *
 * Design ref: DESIGN.md §7 (TilingData structure), §6.2 (tilingKey dispatch)
 *   - TilingKey 维度: 仅 dtype（6 种），通过 ASCENDC_TPL_DATATYPE_DECL 编译期实例化
 *   - 分支分派 (alpha==0 / beta==0 / both!=0) 通过 TilingData 中的 tilingKey 字段运行时 switch
 *   - TilingData 为纯 C++ struct（与 acosh 等算子一致的现代 CANN 模式）
 */
#ifndef ADDR_V2_STRUCT_H_
#define ADDR_V2_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

// ============================================================================
// TilingKey 模板参数定义：仅 dtype 维度
// 编译期实例化 6 个 dtype 变体，分支分派在 Kernel 运行时通过 tilingKey 字段 switch
// ============================================================================
ASCENDC_TPL_ARGS_DECL(AddrV2,
                      ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, C_DT_INT8, C_DT_UINT8,
                                                C_DT_BOOL, ASCENDC_TPL_INPUT(0)), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_INT8)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_UINT8)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BOOL)), );

// ============================================================================
// tilingKey 分支定义（DESIGN.md §6.2，运行时分派）
// ============================================================================
// 优先级: alpha==0 最高（跳过外积），其次 beta==0（跳过 self）
constexpr uint32_t ADDR_V2_TILING_KEY_WITHOUT_ALPHA = 0;   // alpha==0: out = beta * self
constexpr uint32_t ADDR_V2_TILING_KEY_WITHOUT_BETA = 1;    // beta==0, alpha!=0: out = alpha * (vec1⊗vec2)
constexpr uint32_t ADDR_V2_TILING_KEY_WITH_BETA_ALPHA = 2; // alpha!=0, beta!=0: 完整公式

// ============================================================================
// self 广播模式（DESIGN.md §8.2）
// ============================================================================
constexpr uint32_t ADDR_V2_BCAST_NONE = 0;   // [N,M]
constexpr uint32_t ADDR_V2_BCAST_ROW = 1;    // [1,M] 或 1D [M]
constexpr uint32_t ADDR_V2_BCAST_COL = 2;    // [N,1]
constexpr uint32_t ADDR_V2_BCAST_SCALAR = 3; // [1,1]

// ============================================================================
// 对齐常量
// ============================================================================
constexpr uint32_t ADDR_V2_ALIGN_BYTES = 256;     // vector repeat 对齐字节数
constexpr uint32_t ADDR_V2_SCALAR_BUF_BYTES = 32; // 标量搬运的最小 buffer

// ============================================================================
// TilingData 结构（DESIGN.md §7）
// ============================================================================
struct AddrV2TilingData {
    uint32_t blockDim;          // 使用的核数
    uint32_t formerNum;         // 前组核数（处理 formerRows 行）
    uint32_t formerRows;        // 前组每核行数
    uint32_t tailRows;          // 后组每核行数（<= formerRows）
    uint32_t totalRows;         // N（输出行数）
    uint32_t totalCols;         // M（输出列数）
    uint32_t tileM;             // UB 内 M 方向 tile 大小（元素数）
    uint32_t tileMLoop;         // M 方向 tile 循环次数
    uint32_t tileMTail;         // M 方向尾 tile 大小
    uint32_t tilingKey;         // 分支选择键（0/1/2，运行时 switch）
    uint32_t selfBroadcastMode; // self 广播模式（0/1/2/3）
    float betaValue;            // beta 标量值（float 表示）
    float alphaValue;           // alpha 标量值（float 表示）
};

#endif // ADDR_V2_STRUCT_H_
