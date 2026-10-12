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
 * \file ragged_bin_count_tiling_key.h
 * \brief Tiling key declaration for RaggedBinCount.
 *
 * schMode = mappingMode * 4 + binaryOutput * 2 + hasWeights.
 * mappingMode: 0=row ownership, 1=value-centric.
 */
#ifndef RAGGED_BIN_COUNT_TILING_KEY_H
#define RAGGED_BIN_COUNT_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define RBC_SCH_ROW_COUNT_NO_WEIGHT 0
#define RBC_SCH_ROW_COUNT_WEIGHT 1
#define RBC_SCH_ROW_BINARY_NO_WEIGHT 2
#define RBC_SCH_ROW_BINARY_WEIGHT 3
#define RBC_SCH_VALUE_COUNT_NO_WEIGHT 4
#define RBC_SCH_VALUE_COUNT_WEIGHT 5
#define RBC_SCH_VALUE_BINARY_NO_WEIGHT 6
#define RBC_SCH_VALUE_BINARY_WEIGHT 7

ASCENDC_TPL_ARGS_DECL(RaggedBinCount,
                      ASCENDC_TPL_UINT_DECL(schMode, 8, ASCENDC_TPL_UI_LIST, RBC_SCH_ROW_COUNT_NO_WEIGHT,
                                            RBC_SCH_ROW_COUNT_WEIGHT, RBC_SCH_ROW_BINARY_NO_WEIGHT,
                                            RBC_SCH_ROW_BINARY_WEIGHT, RBC_SCH_VALUE_COUNT_NO_WEIGHT,
                                            RBC_SCH_VALUE_COUNT_WEIGHT, RBC_SCH_VALUE_BINARY_NO_WEIGHT,
                                            RBC_SCH_VALUE_BINARY_WEIGHT));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, RBC_SCH_ROW_COUNT_NO_WEIGHT,
                                                          RBC_SCH_ROW_COUNT_WEIGHT, RBC_SCH_ROW_BINARY_NO_WEIGHT,
                                                          RBC_SCH_ROW_BINARY_WEIGHT, RBC_SCH_VALUE_COUNT_NO_WEIGHT,
                                                          RBC_SCH_VALUE_COUNT_WEIGHT, RBC_SCH_VALUE_BINARY_NO_WEIGHT,
                                                          RBC_SCH_VALUE_BINARY_WEIGHT)));

#endif // RAGGED_BIN_COUNT_TILING_KEY_H
