/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 /* !
 * \file moe_update_expert_tiling_key.h
 * \brief
 */

#ifndef MOE_UPDATE_EXPERT_TILING_KEY_H
#define MOE_UPDATE_EXPERT_TILING_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"

#define TILINGKEY_FLOAT 0
#define TILINGKEY_HALF 1
#define TILINGKEY_BFLOAT16 2

#define RANK_ID_BALANCING_MODE 0
#define TOKEN_ID_BALANCING_MODE 1
namespace Mc2Tiling {
ASCENDC_TPL_ARGS_DECL(MoeUpdateExpert, ASCENDC_TPL_UINT_DECL(TILINGKEY_BALANCE_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                                             RANK_ID_BALANCING_MODE, TOKEN_ID_BALANCING_MODE),
                      ASCENDC_TPL_DTYPE_DECL(TILINGKEY_DATA_TYPE, TILINGKEY_FLOAT, TILINGKEY_HALF, TILINGKEY_BFLOAT16),
);
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(TILINGKEY_BALANCE_MODE,  ASCENDC_TPL_UI_LIST, 
                                                          RANK_ID_BALANCING_MODE, TOKEN_ID_BALANCING_MODE),
                                     ASCENDC_TPL_DTYPE_SEL(TILINGKEY_DATA_TYPE, TILINGKEY_FLOAT),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(MoeUpdateExpertTilingData)), 
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(TILINGKEY_BALANCE_MODE,  ASCENDC_TPL_UI_LIST, 
                                                          TOKEN_ID_BALANCING_MODE),
                                     ASCENDC_TPL_DTYPE_SEL(TILINGKEY_DATA_TYPE, TILINGKEY_HALF),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(MoeUpdateExpertTilingData)), 
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(TILINGKEY_BALANCE_MODE,  ASCENDC_TPL_UI_LIST, 
                                                          TOKEN_ID_BALANCING_MODE),
                                     ASCENDC_TPL_DTYPE_SEL(TILINGKEY_DATA_TYPE, TILINGKEY_BFLOAT16),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(MoeUpdateExpertTilingData)), );
}
#endif