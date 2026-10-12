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
 * \file reduce_var_tiling_key.h
 * \brief ReduceVar/ReduceStdV2 independent TilingKey definition
 * 不再依赖 atvoss/reduce/reduce_tiling_key_decl.h
 * 注意: 已移除 IsContiguous — reduce_var 不需要编译期连续/非连续分发
 */
#ifndef _REDUCE_VAR_TILING_KEY_H_
#define _REDUCE_VAR_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// ========== 1. PatternID 常量 ==========
// patternID = Pattern::ID × 10 + (TailA ? 0 : 1)
#define RV_PATTERN_EMPTY 0
#define RV_PATTERN_A 100
#define RV_PATTERN_AR_TailR 11
#define RV_PATTERN_AR_TailA 10
#define RV_PATTERN_ARA_TailR 21
#define RV_PATTERN_ARA_TailA 20
#define RV_PATTERN_ARAR_TailR 31
#define RV_PATTERN_ARAR_TailA 30
#define RV_PATTERN_ARARARAR_TailR 71
#define RV_PATTERN_ARARARAR_TailA 70
#define RV_PATTERN_ARARARARA_TailR 81
#define RV_PATTERN_ARARARARA_TailA 80

// ========== 2. LoopARCount 常量 (aCount × 10 + rCount) ==========
#define RV_LOOP_A1R0 10
#define RV_LOOP_A2R0 20
#define RV_LOOP_A3R0 30
#define RV_LOOP_A4R0 40
#define RV_LOOP_A5R0 50
#define RV_LOOP_A1R2 12
#define RV_LOOP_A1R3 13
#define RV_LOOP_A1R4 14
#define RV_LOOP_A1R5 15
#define RV_LOOP_A2R3 23
#define RV_LOOP_A2R4 24
#define RV_LOOP_A2R5 25
#define RV_LOOP_A2R6 26
#define RV_LOOP_A3R4 34
#define RV_LOOP_A3R5 35
#define RV_LOOP_A3R6 36
#define RV_LOOP_A3R7 37
#define RV_LOOP_A4R5 45
#define RV_LOOP_A4R6 46
#define RV_LOOP_A4R7 47
#define RV_LOOP_A4R8 48
#define RV_LOOP_A5R6 56
#define RV_LOOP_A5R7 57
#define RV_LOOP_A5R8 58
#define RV_LOOP_A5R9 59

// ========== 3. 位宽常量 ==========
#define RV_BIT_WIDTH 8
#define RV_RANGE_MIN 0
#define RV_RANGE_MAX 100

// ========== 4. TilingKey 模板参数声明（替代 REDUCE_TPL_KEY_DECL()）==========
// 参数: BatchInvariant, PatternID, LoopARCount, LoopInnerARCount
#define REDUCE_VAR_TPL_KEY_DECL()                                                                              \
    ASCENDC_TPL_BOOL_DECL(BatchInvariant, 0),                                                                  \
        ASCENDC_TPL_UINT_DECL(PatternID, RV_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, RV_RANGE_MIN, RV_RANGE_MAX),   \
        ASCENDC_TPL_UINT_DECL(LoopARCount, RV_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, RV_RANGE_MIN, RV_RANGE_MAX), \
        ASCENDC_TPL_UINT_DECL(LoopInnerARCount, RV_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, RV_RANGE_MIN, RV_RANGE_MAX)

// ========== 5. C++ 模板参数宏（替代 REDUCE_TPL_PARAM / REDUCE_TPL_VALUE）==========
#define REDUCE_VAR_TPL_PARAM bool BatchInvariant, uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount

#define REDUCE_VAR_TPL_VALUE BatchInvariant, PatternID, LoopARCount, LoopInnerARCount

// ========== 6. TilingKey SEL 规则 ==========
// 空 tensor
#define RV_TPL_KEY_SEL_EMPTY()                                                                  \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0), \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, 0),                                \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, 0),                              \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

// Pure Move
#define RV_TPL_KEY_SEL_PURE_MOVE()                                                              \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0), \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, 100),                              \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, 10),                             \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

// A pattern — 全规约
#define RV_TPL_KEY_SEL_A()                                                                      \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0), \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, 100),                              \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RV_LOOP_A1R0),                   \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

// AR — normal
#define RV_TPL_KEY_SEL_AR_NORMAL()                                                                      \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),         \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RV_PATTERN_AR_TailR, RV_PATTERN_AR_TailA), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RV_LOOP_A1R0),                           \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, 0, 1)

// AR — group
#define RV_TPL_KEY_SEL_AR_GROUP()                                                                       \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),      \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RV_PATTERN_AR_TailR, RV_PATTERN_AR_TailA), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RV_LOOP_A1R2),                           \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

// ARA — normal
#define RV_TPL_KEY_SEL_ARA_NORMAL()                                                                       \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),           \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RV_PATTERN_ARA_TailR, RV_PATTERN_ARA_TailA), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RV_LOOP_A1R0, RV_LOOP_A2R0),               \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, 0, 1)

// ARA — group
#define RV_TPL_KEY_SEL_ARA_GROUP()                                                                        \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),        \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RV_PATTERN_ARA_TailR, RV_PATTERN_ARA_TailA), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RV_LOOP_A1R2, RV_LOOP_A2R3),               \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

// ARAR — normal
#define RV_TPL_KEY_SEL_ARAR_NORMAL()                                                                        \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),             \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RV_PATTERN_ARAR_TailR, RV_PATTERN_ARAR_TailA), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RV_LOOP_A1R0, RV_LOOP_A2R0),                 \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, 0, 2)

// ARAR — group
#define RV_TPL_KEY_SEL_ARAR_GROUP()                                                                         \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),          \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RV_PATTERN_ARAR_TailR, RV_PATTERN_ARAR_TailA), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RV_LOOP_A1R2, RV_LOOP_A1R3, RV_LOOP_A2R3,    \
                             RV_LOOP_A2R4),                                                                 \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

// ARARARAR (TailR) — normal
#define RV_TPL_KEY_SEL_ARARARAR_NORMAL()                                                                 \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),          \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RV_PATTERN_ARARARAR_TailR),                 \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RV_LOOP_A1R0, RV_LOOP_A2R0, RV_LOOP_A3R0, \
                             RV_LOOP_A4R0),                                                              \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, 0, 4)

// ARARARAR (TailR) — group
#define RV_TPL_KEY_SEL_ARARARAR_GROUP()                                                                                \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),                     \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RV_PATTERN_ARARARAR_TailR),                               \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RV_LOOP_A1R2, RV_LOOP_A1R3, RV_LOOP_A1R4, RV_LOOP_A1R5, \
                             RV_LOOP_A2R3, RV_LOOP_A2R4, RV_LOOP_A2R5, RV_LOOP_A2R6, RV_LOOP_A3R4, RV_LOOP_A3R5,       \
                             RV_LOOP_A3R6, RV_LOOP_A3R7, RV_LOOP_A4R5, RV_LOOP_A4R6, RV_LOOP_A4R7, RV_LOOP_A4R8),      \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

// ARARARARA (TailA) — normal
#define RV_TPL_KEY_SEL_ARARARARA_NORMAL()                                                                              \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),                        \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RV_PATTERN_ARARARARA_TailR, RV_PATTERN_ARARARARA_TailA),  \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RV_LOOP_A1R0, RV_LOOP_A2R0, RV_LOOP_A3R0, RV_LOOP_A4R0, \
                             RV_LOOP_A5R0),                                                                            \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, 0, 4)

// ARARARARA (TailA) — group
#define RV_TPL_KEY_SEL_ARARARARA_GROUP()                                                                               \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),                     \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RV_PATTERN_ARARARARA_TailR, RV_PATTERN_ARARARARA_TailA),  \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RV_LOOP_A1R2, RV_LOOP_A1R3, RV_LOOP_A1R4, RV_LOOP_A1R5, \
                             RV_LOOP_A2R3, RV_LOOP_A2R4, RV_LOOP_A2R5, RV_LOOP_A2R6, RV_LOOP_A3R4, RV_LOOP_A3R5,       \
                             RV_LOOP_A3R6, RV_LOOP_A3R7, RV_LOOP_A4R5, RV_LOOP_A4R6, RV_LOOP_A4R7, RV_LOOP_A4R8,       \
                             RV_LOOP_A5R6, RV_LOOP_A5R7, RV_LOOP_A5R8, RV_LOOP_A5R9),                                  \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

// ========== 7. 聚合 SEL 表 ==========
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(RV_TPL_KEY_SEL_EMPTY()), ASCENDC_TPL_ARGS_SEL(RV_TPL_KEY_SEL_PURE_MOVE()),
                ASCENDC_TPL_ARGS_SEL(RV_TPL_KEY_SEL_AR_NORMAL()), ASCENDC_TPL_ARGS_SEL(RV_TPL_KEY_SEL_AR_GROUP()),
                ASCENDC_TPL_ARGS_SEL(RV_TPL_KEY_SEL_ARA_NORMAL()), ASCENDC_TPL_ARGS_SEL(RV_TPL_KEY_SEL_ARA_GROUP()),
                ASCENDC_TPL_ARGS_SEL(RV_TPL_KEY_SEL_ARAR_NORMAL()), ASCENDC_TPL_ARGS_SEL(RV_TPL_KEY_SEL_ARAR_GROUP()),
                ASCENDC_TPL_ARGS_SEL(RV_TPL_KEY_SEL_ARARARAR_NORMAL()),
                ASCENDC_TPL_ARGS_SEL(RV_TPL_KEY_SEL_ARARARAR_GROUP()),
                ASCENDC_TPL_ARGS_SEL(RV_TPL_KEY_SEL_ARARARARA_NORMAL()),
                ASCENDC_TPL_ARGS_SEL(RV_TPL_KEY_SEL_ARARARARA_GROUP()));

// ========== 8. TilingKey 打包宏（替代 GEN_REDUCE_TILING_KEY）==========
#define GEN_REDUCE_VAR_TILING_KEY(result, tilingKey)                                                  \
    result = GET_TPL_TILING_KEY(tilingKey.batchInvariant, tilingKey.patternID, tilingKey.loopARCount, \
                                tilingKey.loopInnerARCount)

// ========== 9. TilingKey 声明 ==========
ASCENDC_TPL_ARGS_DECL(ReduceVar, REDUCE_VAR_TPL_KEY_DECL());

#endif // _REDUCE_VAR_TILING_KEY_H_
