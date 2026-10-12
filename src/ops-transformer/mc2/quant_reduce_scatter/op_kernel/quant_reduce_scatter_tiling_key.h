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
 * \file quant_reduce_scatter_tiling_key.h
 * \brief
 */
#ifndef QUANT_REDUCE_SCATTER_TILING_KEY_H
#define QUANT_REDUCE_SCATTER_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"
#define MTE_COMM 1
// 模板参数
ASCENDC_TPL_ARGS_DECL(QuantReduceScatter,
    ASCENDC_TPL_UINT_DECL(quantReduceScatterCommMode, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, MTE_COMM),
);

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(quantReduceScatterCommMode, ASCENDC_TPL_UI_LIST, MTE_COMM),
    ),
);

#endif // QUANT_REDUCE_SCATTER_TILING_KEY_H