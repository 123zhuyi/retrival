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
 * \file reduce_var_struct.h
 * \brief ReduceVar/ReduceStdV2 unified TilingData struct (C-style, zero dependency on reduce template)
 * 参考: grouped_bias_add_grad/op_kernel/arch35/grouped_bias_add_grad_struct.h
 */
#ifndef _REDUCE_VAR_STRUCT_H_
#define _REDUCE_VAR_STRUCT_H_

// ========== 独立常量定义（从 reduce 模板提取，避免 include 依赖）==========
constexpr int32_t REDUCE_VAR_MAX_DIM = 9; // 等价于 Ops::Base::ReduceOpTmpl::MAX_DIM
constexpr int32_t MAX_CORE_COUNT = 128;   // 最大核数

struct ReduceVarTilingData {
    // ===== reduce 切分相关字段（原 ReduceVarTilingDataStru）=====
    uint64_t factorACntPerCore;
    uint64_t factorATotalCnt;
    uint64_t ubFactorA;
    uint64_t factorRCntPerCore;
    uint64_t factorRTotalCnt;
    uint64_t ubFactorR;
    uint64_t groupR;
    uint64_t outSize;
    uint64_t basicBlock;
    uint64_t resultBlock;
    int32_t coreNum;
    int32_t useNddma;
    uint8_t isInvert; // 0=no transpose, 1=NDDMA stride-swap transpose
    float meanVar;
    uint64_t shape[REDUCE_VAR_MAX_DIM];
    uint64_t stride[REDUCE_VAR_MAX_DIM];
    uint64_t dstStride[REDUCE_VAR_MAX_DIM];

    // ===== reduce_var 业务字段（原 ReduceVarTilingData 独有）=====
    int64_t correction;
    int64_t correctionInvalid;
    int64_t isMeanOut;
    int64_t workSpaceSize;
    float varFactor;                             // 1/(R-correction)
    float meanFactor;                            // 1/R
    int64_t reduceCntEachGroupR[MAX_CORE_COUNT]; // 每个groupR处理的R的个数
};

#endif
