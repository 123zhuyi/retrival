/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file sinh_v2_tiling_key.h
 * \brief SinhV2 TilingKey definition
 *
 * TilingKey mapping:
 * - SINHV2_TPL_SCH_MODE_0 (0): FLOAT32 type
 * - SINHV2_TPL_SCH_MODE_1 (1): FLOAT16 type
 */

#ifndef __SINH_V2_TILING_KEY_H__
#define __SINH_V2_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define SINHV2_TPL_SCH_MODE_0 0 // FLOAT32 type
#define SINHV2_TPL_SCH_MODE_1 1 // FLOAT16 type

ASCENDC_TPL_ARGS_DECL(
    SinhV2, ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, SINHV2_TPL_SCH_MODE_0, SINHV2_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, SINHV2_TPL_SCH_MODE_0, SINHV2_TPL_SCH_MODE_1)));

#endif
