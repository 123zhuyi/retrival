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
 * \file quant_all_reduce_tiling_key.h
 * \brief
 */
#ifndef QUANT_ALL_REDUCE_TILING_KEY_H
#define QUANT_ALL_REDUCE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define MTE_ONE_SHOT 1 // 通过1步allgather的方法实现
// 模板参数
ASCENDC_TPL_ARGS_DECL(QuantAllReduce,
    ASCENDC_TPL_UINT_DECL(quantAllReduceCommMode, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, MTE_ONE_SHOT), // LIST模式，穷举
);

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(quantAllReduceCommMode, ASCENDC_TPL_UI_LIST, MTE_ONE_SHOT),
    ),
);

#endif // QUANT_ALL_REDUCE_TILING_KEY_H