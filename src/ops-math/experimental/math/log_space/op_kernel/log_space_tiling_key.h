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
 * \file log_space_tiling_key.h
 * \brief LogSpace TilingKey 模板参数定义
 *
 * 模板参数：
 *   - D_T_Y: 输出数据类型 (C_DT_FLOAT / C_DT_FLOAT16 / C_DT_BF16 /
 *            C_DT_INT8 / C_DT_INT16 / C_DT_INT32 / C_DT_UINT8)
 *   - MODE:  0 = NORMAL（steps >= 2），1 = SINGLE（steps == 0 / 1）
 *
 * 14 条路径全部启用（7 dtype × NORMAL/SINGLE）。整型在 kernel 内
 * 以 fp32 计算 base^x 后末步 Cast(CAST_TRUNC) 落整型（向零取整，匹配 torch）。
 */

#ifndef __LOG_SPACE_TILING_KEY_H__
#define __LOG_SPACE_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(LogSpace,
                      ASCENDC_TPL_DATATYPE_DECL(D_T_Y, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, C_DT_INT8, C_DT_INT16,
                                                C_DT_INT32, C_DT_UINT8, ASCENDC_TPL_OUTPUT(0)),
                      ASCENDC_TPL_UINT_DECL(MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_FLOAT16),
                                     ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_FLOAT),
                                     ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_BF16),
                                     ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_INT8),
                                     ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_INT16),
                                     ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_INT32),
                                     ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_UINT8),
                                     ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1)), );

#endif
