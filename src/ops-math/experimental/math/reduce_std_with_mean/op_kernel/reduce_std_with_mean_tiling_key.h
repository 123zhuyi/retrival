/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __REDUCE_STD_WITH_MEAN_TILING_KEY_H__
#define __REDUCE_STD_WITH_MEAN_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define REDUCE_STD_SCH_FP16 0
#define REDUCE_STD_SCH_FP32 1
#define REDUCE_STD_SCH_BF16 2

ASCENDC_TPL_ARGS_DECL(ReduceStdWithMean,
                      ASCENDC_TPL_UINT_DECL(schMode, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, REDUCE_STD_SCH_FP16,
                                            REDUCE_STD_SCH_FP32, REDUCE_STD_SCH_BF16));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, REDUCE_STD_SCH_FP16,
                                                          REDUCE_STD_SCH_FP32, REDUCE_STD_SCH_BF16)));

#endif
