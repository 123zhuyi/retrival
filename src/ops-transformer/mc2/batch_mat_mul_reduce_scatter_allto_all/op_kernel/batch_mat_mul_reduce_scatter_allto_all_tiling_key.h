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
 * \file batch_mat_mul_reduce_scatter_allto_all_tiling_key.h
 * \brief
 */
#ifndef MC2_BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_TILING_KET_H
#define MC2_BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_TILING_KET_H

#include "ascendc/host_api/tiling/template_argument.h"

// 模板参数
ASCENDC_TPL_ARGS_DECL( 
    BatchMatMulReduceScatterAlltoAll,
    ASCENDC_TPL_UINT_DECL(  // 按枚举值给出，0表示输出在H维度按tp分片，1表示输出在C维度按tp分片。
        BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_Y_SHARD, 
        ASCENDC_TPL_2_BW,
        ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_BOOL_DECL(  // 按布尔值给出，权重矩阵是否转置
        BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_WEIGHT_TRANSPOSE, 0, 1),
    ASCENDC_TPL_BOOL_DECL(  // 按布尔值给出，是否包含偏矩阵
        BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_IS_BIAS, 0, 1),
    ASCENDC_TPL_BOOL_DECL(  // 按布尔值给出，当前算子是否为轻量级(输出通道数小于阈值且输出在C维度按tp分片时启用)
        BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_LITE, 0, 1)
);

// 模板参数组合
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(  
            BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_Y_SHARD, 
            ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_BOOL_SEL(  
            BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_WEIGHT_TRANSPOSE, 0, 1),
        ASCENDC_TPL_BOOL_SEL(  
            BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_IS_BIAS, 0, 1),
        ASCENDC_TPL_BOOL_SEL(  
            BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_LITE, 0, 1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(  
            BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_Y_SHARD, 
            ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_BOOL_SEL(  
            BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_WEIGHT_TRANSPOSE, 0, 1),
        ASCENDC_TPL_BOOL_SEL(  
            BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_IS_BIAS, 0, 1),
        ASCENDC_TPL_BOOL_SEL(  
            BATCH_MM_REDUCE_SCATTER_ALLTO_ALL_LITE, 0)
    ),
);

#endif // MC2_BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_TILING_KET_H