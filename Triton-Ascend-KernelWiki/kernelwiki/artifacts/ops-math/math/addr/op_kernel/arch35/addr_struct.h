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
 * \file addr_struct.h
 * \brief addr struct
 */
#ifndef ADDR_STRUCT_H_
#define ADDR_STRUCT_H_

#include "atvoss/broadcast/broadcast_base_struct.h"
#include "ascendc/host_api/tiling/template_argument.h"

using namespace Ops::Base;

// 算子自定义的tiling key字段
ASCENDC_TPL_ARGS_DECL(
    Addr, BRC_TEMP_SCH_MODE_KEY_DECL(schMode), ASCENDC_TPL_UINT_DECL(betaZeroOrNot, 1, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(alphaZeroOrNot, 1, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        BRC_TEMP_SCH_MODE_KEY_SEL(schMode), ASCENDC_TPL_UINT_SEL(betaZeroOrNot, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(alphaZeroOrNot, ASCENDC_TPL_UI_LIST, 0, 1)), );
#endif // ADDR_STRUCT_H_
