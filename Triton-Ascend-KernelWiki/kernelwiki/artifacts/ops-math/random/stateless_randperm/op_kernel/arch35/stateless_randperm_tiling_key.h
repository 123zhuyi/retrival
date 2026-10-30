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
 * \file stateless_randperm_tiling_key.h
 * \brief stateless randperm key declare
 */

#ifndef _STATELESS_RANDPERM_TILING_KEY_DECL_H_
#define _STATELESS_RANDPERM_TILING_KEY_DECL_H_
#include "ascendc/host_api/tiling/template_argument.h"

#define StatelessRandperm_TPL_KEY_DECL()                                                               \
    ASCENDC_TPL_UINT_DECL(randomType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3)                    \
        ASCENDC_TPL_UINT_DECL(nIsInt32, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1),                       \
        ASCENDC_TPL_UINT_DECL(schId, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),                      \
        ASCENDC_TPL_UINT_DECL(isInt32, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1),                   \
        ASCENDC_TPL_UINT_DECL(isDescend, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1)                          

#define StatelessRandperm_TPL_KEY_SEL()                                             \
    ASCENDC_TPL_UINT_SEL(randomType, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3)                    \
        ASCENDC_TPL_UINT_SEL(nIsInt32, ASCENDC_TPL_UI_LIST, 0, 1),                       \
        ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),                      \
        ASCENDC_TPL_UINT_SEL(isInt32, ASCENDC_TPL_UI_LIST, 0, 1),                   \
        ASCENDC_TPL_UINT_SEL(isDescend, ASCENDC_TPL_UI_LIST, 0) 

ASCENDC_TPL_ARGS_DECL(StatelessRandperm, StatelessRandperm_TPL_KEY_DECL());

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(StatelessRandperm_TPL_KEY_SEL())
);

#endif