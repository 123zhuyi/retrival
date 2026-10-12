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
 * \file truncate_div_tiling_key.h
 * \brief TruncateDiv tiling key (schMode) template argument declaration (12 dtype combos).
 */
#ifndef __TRUNCATEDIV_TILING_KEY_H__
#define __TRUNCATEDIV_TILING_KEY_H__

#include "truncate_div_tiling_data.h"
#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(TruncateDiv, ASCENDC_TPL_UINT_DECL(schMode, 4, ASCENDC_TPL_UI_LIST, TRUNCATEDIV_TPL_SCH_MODE_0,
                                                         TRUNCATEDIV_TPL_SCH_MODE_1, TRUNCATEDIV_TPL_SCH_MODE_2,
                                                         TRUNCATEDIV_TPL_SCH_MODE_3, TRUNCATEDIV_TPL_SCH_MODE_4,
                                                         TRUNCATEDIV_TPL_SCH_MODE_5, TRUNCATEDIV_TPL_SCH_MODE_6,
                                                         TRUNCATEDIV_TPL_SCH_MODE_7, TRUNCATEDIV_TPL_SCH_MODE_8,
                                                         TRUNCATEDIV_TPL_SCH_MODE_9, TRUNCATEDIV_TPL_SCH_MODE_10,
                                                         TRUNCATEDIV_TPL_SCH_MODE_11));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(
    schMode, ASCENDC_TPL_UI_LIST, TRUNCATEDIV_TPL_SCH_MODE_0, TRUNCATEDIV_TPL_SCH_MODE_1, TRUNCATEDIV_TPL_SCH_MODE_2,
    TRUNCATEDIV_TPL_SCH_MODE_3, TRUNCATEDIV_TPL_SCH_MODE_4, TRUNCATEDIV_TPL_SCH_MODE_5, TRUNCATEDIV_TPL_SCH_MODE_6,
    TRUNCATEDIV_TPL_SCH_MODE_7, TRUNCATEDIV_TPL_SCH_MODE_8, TRUNCATEDIV_TPL_SCH_MODE_9, TRUNCATEDIV_TPL_SCH_MODE_10,
    TRUNCATEDIV_TPL_SCH_MODE_11)));
#endif
