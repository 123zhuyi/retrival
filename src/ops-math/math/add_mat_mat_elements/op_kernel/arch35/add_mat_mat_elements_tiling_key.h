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
 * \file add_mat_mat_elements_tiling_key.h
 * \brief AddMatMatElements TilingKey template parameters (arch35)
 */

#ifndef ADD_MAT_MAT_ELEMENTS_TILING_KEY_H_
#define ADD_MAT_MAT_ELEMENTS_TILING_KEY_H_

#ifndef __CCE_KT_TEST__
#include "ascendc/host_api/tiling/template_argument.h"

#define ADD_MAT_MAT_ELEMENTS_RANK_4 4
#define ADD_MAT_MAT_ELEMENTS_RANK_8 8

ASCENDC_TPL_ARGS_DECL(AddMatMatElements, ASCENDC_TPL_DATATYPE_DECL(D_T, C_DT_FLOAT16, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
                      ASCENDC_TPL_UINT_DECL(RANK, 8, ASCENDC_TPL_UI_LIST, ADD_MAT_MAT_ELEMENTS_RANK_4,
                                            ADD_MAT_MAT_ELEMENTS_RANK_8));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16),
                                     ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, ADD_MAT_MAT_ELEMENTS_RANK_4)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT),
                                     ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, ADD_MAT_MAT_ELEMENTS_RANK_4)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16),
                                     ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, ADD_MAT_MAT_ELEMENTS_RANK_8)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT),
                                     ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, ADD_MAT_MAT_ELEMENTS_RANK_8)));
#endif

#endif // ADD_MAT_MAT_ELEMENTS_TILING_KEY_H_
