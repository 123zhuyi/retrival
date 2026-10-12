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
 * \file log_add_exp_struct.h
 * \brief log_add_exp struct
 *
 * formulaType: 0 = simplified (base=-1, scale=1.0, shift=0.0)
 *              1 = full (any non-default base/scale/shift)
 */
#ifndef LOG_ADD_EXP_STRUCT_H_
#define LOG_ADD_EXP_STRUCT_H_

#include "atvoss/broadcast/broadcast_base_struct.h"

ASCENDC_TPL_ARGS_DECL(
    LogAddExp,
    BRC_TEMP_SCH_MODE_KEY_DECL(schMode),
    ASCENDC_TPL_UINT_DECL(formulaType, 1, ASCENDC_TPL_UI_LIST, 0, 1)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        BRC_TEMP_SCH_MODE_KEY_SEL(schMode),
        ASCENDC_TPL_UINT_SEL(formulaType, ASCENDC_TPL_UI_LIST, 0, 1)
    ),
);
#endif // LOG_ADD_EXP_STRUCT_H_
