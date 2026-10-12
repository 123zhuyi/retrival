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
 * \file reduce_std_v2_tiling_key.h
 * \brief ReduceStdV2 independent TilingKey definition
 * 不再依赖 atvoss/reduce/reduce_tiling_key_decl.h
 * 结构与 reduce_var_tiling_key.h 一致，op name 独立为 ReduceStdV2
 */
#ifndef _REDUCE_STD_V2_TILING_KEY_H_
#define _REDUCE_STD_V2_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// ========== 1. PatternID 常量 ==========
// patternID = Pattern::ID × 10 + (TailA ? 0 : 1)
#define RSV2_PATTERN_EMPTY 0
#define RSV2_PATTERN_A 100
#define RSV2_PATTERN_AR_TailR 11
#define RSV2_PATTERN_AR_TailA 10
#define RSV2_PATTERN_ARA_TailR 21
#define RSV2_PATTERN_ARA_TailA 20
#define RSV2_PATTERN_ARAR_TailR 31
#define RSV2_PATTERN_ARAR_TailA 30
#define RSV2_PATTERN_ARARARAR_TailR 71
#define RSV2_PATTERN_ARARARAR_TailA 70
#define RSV2_PATTERN_ARARARARA_TailR 81
#define RSV2_PATTERN_ARARARARA_TailA 80

// ========== 2. LoopARCount 常量 (aCount × 10 + rCount) ==========
#define RSV2_LOOP_A1R0 10
#define RSV2_LOOP_A2R0 20
#define RSV2_LOOP_A3R0 30
#define RSV2_LOOP_A4R0 40
#define RSV2_LOOP_A5R0 50
#define RSV2_LOOP_A1R2 12
#define RSV2_LOOP_A1R3 13
#define RSV2_LOOP_A1R4 14
#define RSV2_LOOP_A1R5 15
#define RSV2_LOOP_A2R3 23
#define RSV2_LOOP_A2R4 24
#define RSV2_LOOP_A2R5 25
#define RSV2_LOOP_A2R6 26
#define RSV2_LOOP_A3R4 34
#define RSV2_LOOP_A3R5 35
#define RSV2_LOOP_A3R6 36
#define RSV2_LOOP_A3R7 37
#define RSV2_LOOP_A4R5 45
#define RSV2_LOOP_A4R6 46
#define RSV2_LOOP_A4R7 47
#define RSV2_LOOP_A4R8 48
#define RSV2_LOOP_A5R6 56
#define RSV2_LOOP_A5R7 57
#define RSV2_LOOP_A5R8 58
#define RSV2_LOOP_A5R9 59

// ========== 3. 位宽常量 ==========
#define RSV2_BIT_WIDTH 8
#define RSV2_RANGE_MIN 0
#define RSV2_RANGE_MAX 100

// ========== 4. TilingKey 模板参数声明（替代 REDUCE_TPL_KEY_DECL()）==========
// 参数: BatchInvariant, PatternID, LoopARCount, LoopInnerARCount
#define REDUCE_STD_V2_TPL_KEY_DECL()                                                                                 \
    ASCENDC_TPL_BOOL_DECL(BatchInvariant, 0),                                                                        \
        ASCENDC_TPL_UINT_DECL(PatternID, RSV2_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, RSV2_RANGE_MIN, RSV2_RANGE_MAX),   \
        ASCENDC_TPL_UINT_DECL(LoopARCount, RSV2_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, RSV2_RANGE_MIN, RSV2_RANGE_MAX), \
        ASCENDC_TPL_UINT_DECL(LoopInnerARCount, RSV2_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, RSV2_RANGE_MIN,             \
                              RSV2_RANGE_MAX)

// ========== 5. C++ 模板参数宏（替代 REDUCE_TPL_PARAM / REDUCE_TPL_VALUE）==========
#define REDUCE_STD_V2_TPL_PARAM bool BatchInvariant, uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount

#define REDUCE_STD_V2_TPL_VALUE BatchInvariant, PatternID, LoopARCount, LoopInnerARCount

// ========== 6. TilingKey SEL 规则 ==========
// 空 tensor
#define RSV2_TPL_KEY_SEL_EMPTY()                                                                \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0), \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, 0),                                \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, 0),                              \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

// Pure Move
#define RSV2_TPL_KEY_SEL_PURE_MOVE()                                                            \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0), \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, 100),                              \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, 10),                             \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

// A pattern — 全规约
#define RSV2_TPL_KEY_SEL_A()                                                                    \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0), \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, 100),                              \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RSV2_LOOP_A1R0),                 \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

// AR — normal
#define RSV2_TPL_KEY_SEL_AR_NORMAL()                                                                        \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),             \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RSV2_PATTERN_AR_TailR, RSV2_PATTERN_AR_TailA), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RSV2_LOOP_A1R0),                             \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, 0, 1)

// AR — group
#define RSV2_TPL_KEY_SEL_AR_GROUP()                                                                         \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),          \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RSV2_PATTERN_AR_TailR, RSV2_PATTERN_AR_TailA), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RSV2_LOOP_A1R2),                             \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

// ARA — normal
#define RSV2_TPL_KEY_SEL_ARA_NORMAL()                                                                         \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),               \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RSV2_PATTERN_ARA_TailR, RSV2_PATTERN_ARA_TailA), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RSV2_LOOP_A1R0, RSV2_LOOP_A2R0),               \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, 0, 1)

// ARA — group
#define RSV2_TPL_KEY_SEL_ARA_GROUP()                                                                          \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),            \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RSV2_PATTERN_ARA_TailR, RSV2_PATTERN_ARA_TailA), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RSV2_LOOP_A1R2, RSV2_LOOP_A2R3),               \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

// ARAR — normal
#define RSV2_TPL_KEY_SEL_ARAR_NORMAL()                                                                          \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),                 \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RSV2_PATTERN_ARAR_TailR, RSV2_PATTERN_ARAR_TailA), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RSV2_LOOP_A1R0, RSV2_LOOP_A2R0),                 \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, 0, 2)

// ARAR — group
#define RSV2_TPL_KEY_SEL_ARAR_GROUP()                                                                           \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),              \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RSV2_PATTERN_ARAR_TailR, RSV2_PATTERN_ARAR_TailA), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RSV2_LOOP_A1R2, RSV2_LOOP_A1R3, RSV2_LOOP_A2R3,  \
                             RSV2_LOOP_A2R4),                                                                   \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

// ARARARAR (TailR) — normal
#define RSV2_TPL_KEY_SEL_ARARARAR_NORMAL()                                                                     \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),                \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RSV2_PATTERN_ARARARAR_TailR),                     \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RSV2_LOOP_A1R0, RSV2_LOOP_A2R0, RSV2_LOOP_A3R0, \
                             RSV2_LOOP_A4R0),                                                                  \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, 0, 4)

// ARARARAR (TailR) — group
#define RSV2_TPL_KEY_SEL_ARARARAR_GROUP()                                                                      \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),             \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RSV2_PATTERN_ARARARAR_TailR),                     \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RSV2_LOOP_A1R2, RSV2_LOOP_A1R3, RSV2_LOOP_A1R4, \
                             RSV2_LOOP_A1R5, RSV2_LOOP_A2R3, RSV2_LOOP_A2R4, RSV2_LOOP_A2R5, RSV2_LOOP_A2R6,   \
                             RSV2_LOOP_A3R4, RSV2_LOOP_A3R5, RSV2_LOOP_A3R6, RSV2_LOOP_A3R7, RSV2_LOOP_A4R5,   \
                             RSV2_LOOP_A4R6, RSV2_LOOP_A4R7, RSV2_LOOP_A4R8),                                  \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

// ARARARARA (TailA) — normal
#define RSV2_TPL_KEY_SEL_ARARARARA_NORMAL()                                                                    \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),                \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RSV2_PATTERN_ARARARARA_TailR,                     \
                             RSV2_PATTERN_ARARARARA_TailA),                                                    \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RSV2_LOOP_A1R0, RSV2_LOOP_A2R0, RSV2_LOOP_A3R0, \
                             RSV2_LOOP_A4R0, RSV2_LOOP_A5R0),                                                  \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, 0, 4)

// ARARARARA (TailA) — group
#define RSV2_TPL_KEY_SEL_ARARARARA_GROUP()                                                                     \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0), ASCENDC_TPL_BOOL_SEL(BatchInvariant, 0),             \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, RSV2_PATTERN_ARARARARA_TailR,                     \
                             RSV2_PATTERN_ARARARARA_TailA),                                                    \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, RSV2_LOOP_A1R2, RSV2_LOOP_A1R3, RSV2_LOOP_A1R4, \
                             RSV2_LOOP_A1R5, RSV2_LOOP_A2R3, RSV2_LOOP_A2R4, RSV2_LOOP_A2R5, RSV2_LOOP_A2R6,   \
                             RSV2_LOOP_A3R4, RSV2_LOOP_A3R5, RSV2_LOOP_A3R6, RSV2_LOOP_A3R7, RSV2_LOOP_A4R5,   \
                             RSV2_LOOP_A4R6, RSV2_LOOP_A4R7, RSV2_LOOP_A4R8, RSV2_LOOP_A5R6, RSV2_LOOP_A5R7,   \
                             RSV2_LOOP_A5R8, RSV2_LOOP_A5R9),                                                  \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

// ========== 7. 聚合 SEL 表 ==========
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(RSV2_TPL_KEY_SEL_EMPTY()), ASCENDC_TPL_ARGS_SEL(RSV2_TPL_KEY_SEL_PURE_MOVE()),
                ASCENDC_TPL_ARGS_SEL(RSV2_TPL_KEY_SEL_AR_NORMAL()), ASCENDC_TPL_ARGS_SEL(RSV2_TPL_KEY_SEL_AR_GROUP()),
                ASCENDC_TPL_ARGS_SEL(RSV2_TPL_KEY_SEL_ARA_NORMAL()), ASCENDC_TPL_ARGS_SEL(RSV2_TPL_KEY_SEL_ARA_GROUP()),
                ASCENDC_TPL_ARGS_SEL(RSV2_TPL_KEY_SEL_ARAR_NORMAL()),
                ASCENDC_TPL_ARGS_SEL(RSV2_TPL_KEY_SEL_ARAR_GROUP()),
                ASCENDC_TPL_ARGS_SEL(RSV2_TPL_KEY_SEL_ARARARAR_NORMAL()),
                ASCENDC_TPL_ARGS_SEL(RSV2_TPL_KEY_SEL_ARARARAR_GROUP()),
                ASCENDC_TPL_ARGS_SEL(RSV2_TPL_KEY_SEL_ARARARARA_NORMAL()),
                ASCENDC_TPL_ARGS_SEL(RSV2_TPL_KEY_SEL_ARARARARA_GROUP()));

// ========== 8. TilingKey 打包宏（替代 GEN_REDUCE_TILING_KEY）==========
#define GEN_REDUCE_STD_V2_TILING_KEY(result, tilingKey)                                               \
    result = GET_TPL_TILING_KEY(tilingKey.batchInvariant, tilingKey.patternID, tilingKey.loopARCount, \
                                tilingKey.loopInnerARCount)

// ========== 9. TilingKey 声明 ==========
ASCENDC_TPL_ARGS_DECL(ReduceStdV2, REDUCE_STD_V2_TPL_KEY_DECL());

#endif // _REDUCE_STD_V2_TILING_KEY_H_
