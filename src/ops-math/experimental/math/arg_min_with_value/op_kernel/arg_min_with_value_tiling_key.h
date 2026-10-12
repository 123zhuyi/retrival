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
 * \file arg_min_with_value_tiling_key.h
 * \brief Compile-time template-argument (schMode) declaration so the kernel codegen emits one binary
 *        per algorithm domain. The host selects exactly once; the kernel dispatch is compile-time only.
 */
#include "ascendc/host_api/tiling/template_argument.h"

#define ARG_SCH_COPY 0
#define ARG_SCH_LAST_DIRECT 1
#define ARG_SCH_LAST_TINY 2
#define ARG_SCH_LAST_PACK2 3
#define ARG_SCH_LAST_PACK3 4
#define ARG_SCH_LAST_PACK4 5
#define ARG_SCH_LAST_PACK5 6
#define ARG_SCH_LAST_PACKN 7
#define ARG_SCH_LAST_SEG 8
#define ARG_SCH_LAST_PIECE 9
#define ARG_SCH_LAST_SPLIT1 10
#define ARG_SCH_LAST_SPLIT2 11
#define ARG_SCH_NLAST_OUTPUT 12
#define ARG_SCH_NLAST_BATCH 13
#define ARG_SCH_NLAST_TREE 14
#define ARG_SCH_NLAST_SPLIT 15
#define ARG_SCH_LAST_LONG 16
#define ARG_SCH_LAST_LONG_PACKED 17

ASCENDC_TPL_ARGS_DECL(ArgMinWithValue,
                      ASCENDC_TPL_UINT_DECL(schMode, 5, ASCENDC_TPL_UI_LIST, ARG_SCH_COPY, ARG_SCH_LAST_DIRECT,
                                            ARG_SCH_LAST_TINY, ARG_SCH_LAST_PACK2, ARG_SCH_LAST_PACK3,
                                            ARG_SCH_LAST_PACK4, ARG_SCH_LAST_PACK5, ARG_SCH_LAST_PACKN,
                                            ARG_SCH_LAST_SEG, ARG_SCH_LAST_PIECE, ARG_SCH_LAST_SPLIT1,
                                            ARG_SCH_LAST_SPLIT2, ARG_SCH_NLAST_OUTPUT, ARG_SCH_NLAST_BATCH,
                                            ARG_SCH_NLAST_TREE, ARG_SCH_NLAST_SPLIT, ARG_SCH_LAST_LONG,
                                            ARG_SCH_LAST_LONG_PACKED),
                      ASCENDC_TPL_BOOL_DECL(gather, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, ARG_SCH_COPY,
                                                          ARG_SCH_LAST_DIRECT, ARG_SCH_LAST_TINY, ARG_SCH_LAST_PACK2,
                                                          ARG_SCH_LAST_PACK3, ARG_SCH_LAST_PACK4, ARG_SCH_LAST_PACK5,
                                                          ARG_SCH_LAST_PACKN, ARG_SCH_LAST_SEG, ARG_SCH_LAST_PIECE,
                                                          ARG_SCH_NLAST_OUTPUT, ARG_SCH_NLAST_BATCH, ARG_SCH_NLAST_TREE,
                                                          ARG_SCH_LAST_LONG, ARG_SCH_LAST_LONG_PACKED),
                                     ASCENDC_TPL_BOOL_SEL(gather, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, ARG_SCH_LAST_TINY,
                                                          ARG_SCH_LAST_PACK2, ARG_SCH_LAST_PACK3, ARG_SCH_LAST_PACK4,
                                                          ARG_SCH_LAST_PACK5, ARG_SCH_LAST_PACKN, ARG_SCH_LAST_SEG,
                                                          ARG_SCH_LAST_PIECE),
                                     ASCENDC_TPL_BOOL_SEL(gather, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                                     ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, ARG_SCH_LAST_SPLIT1,
                                                          ARG_SCH_LAST_SPLIT2, ARG_SCH_NLAST_SPLIT),
                                     ASCENDC_TPL_BOOL_SEL(gather, 0)), );
