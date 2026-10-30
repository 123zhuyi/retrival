/**
 * This file is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 Yang Zhenze, Chongqing University of Posts and Telecommunications (CQUPT).
 * All Rights Reserved.
 *
 * Author (account):
 * - Yang Zhenze <@gcw_5x5Ew5Ms>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BIAS_ADD_TILING_KEY_H_
#define BIAS_ADD_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define BIAS_ADD_TPL_SCH_MODE_BASE 0
#define BIAS_ADD_TPL_SCH_MODE_TINY_NOQUEUE 1
#define BIAS_ADD_TPL_SCH_MODE_THIN_TINY_VECTOR_BROADCAST 2
#define BIAS_ADD_TPL_SCH_MODE_BROADCAST_UB_TILE 3

ASCENDC_TPL_ARGS_DECL(BiasAdd,
                      ASCENDC_TPL_UINT_DECL(schMode, 3, ASCENDC_TPL_UI_LIST, BIAS_ADD_TPL_SCH_MODE_BASE,
                                            BIAS_ADD_TPL_SCH_MODE_TINY_NOQUEUE,
                                            BIAS_ADD_TPL_SCH_MODE_THIN_TINY_VECTOR_BROADCAST,
                                            BIAS_ADD_TPL_SCH_MODE_BROADCAST_UB_TILE), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, BIAS_ADD_TPL_SCH_MODE_BASE,
                                                          BIAS_ADD_TPL_SCH_MODE_TINY_NOQUEUE,
                                                          BIAS_ADD_TPL_SCH_MODE_THIN_TINY_VECTOR_BROADCAST,
                                                          BIAS_ADD_TPL_SCH_MODE_BROADCAST_UB_TILE)), );

#endif // BIAS_ADD_TILING_KEY_H_
