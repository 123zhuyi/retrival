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
 * \file allto_all_all_gather_batch_mat_mul_tiling_key.h
 * \brief
 */
#ifndef MC2_ALLTOALL_ALLGATHER_BATCHMATMUL_TILING_KEY_H
#define MC2_ALLTOALL_ALLGATHER_BATCHMATMUL_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// 模板参数
ASCENDC_TPL_ARGS_DECL( 
    AlltoAllAllGatherBatchMatMul,
    ASCENDC_TPL_UINT_DECL(  // 按枚举值给出，0表示在H维度按tp域进行allgather，1表示在C维度上按tp域进行allgather
        ALL_TO_ALL_ALL_GATHER_BATCH_MM_X_SHARD, 
        ASCENDC_TPL_2_BW,
        ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_BOOL_DECL(  // 按布尔值给出，权重矩阵是否转置
        ALL_TO_ALL_ALL_GATHER_BATCH_MM_WEIGHT_TRANSPOSE, 0, 1),
    ASCENDC_TPL_BOOL_DECL(  // 按布尔值给出，是否包含偏矩阵
        ALL_TO_ALL_ALL_GATHER_BATCH_MM_IS_BIAS, 0, 1),
    ASCENDC_TPL_BOOL_DECL(  // 按布尔值给出，是否包含AllGather的输出
        ALL_TO_ALL_ALL_GATHER_BATCH_MM_Y2_NEED, 0, 1), 
    ASCENDC_TPL_BOOL_DECL(  // 按布尔值给出，有激活函数时，是否包含BatchMatMul的输出
        ALL_TO_ALL_ALL_GATHER_BATCH_MM_Y3_NEED, 0, 1), 
);

// 模板参数组合
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(  
            ALL_TO_ALL_ALL_GATHER_BATCH_MM_X_SHARD, 
            ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_BOOL_SEL(  
            ALL_TO_ALL_ALL_GATHER_BATCH_MM_WEIGHT_TRANSPOSE, 0, 1),
        ASCENDC_TPL_BOOL_SEL(  
            ALL_TO_ALL_ALL_GATHER_BATCH_MM_IS_BIAS, 0, 1),
        ASCENDC_TPL_BOOL_SEL(  
            ALL_TO_ALL_ALL_GATHER_BATCH_MM_Y2_NEED, 0, 1), 
        ASCENDC_TPL_BOOL_SEL(  
            ALL_TO_ALL_ALL_GATHER_BATCH_MM_Y3_NEED, 0, 1), 
    )
);

#endif // MC2_ALLTOALL_ALLGATHER_BATCHMATMUL_TILING_KEY_H
