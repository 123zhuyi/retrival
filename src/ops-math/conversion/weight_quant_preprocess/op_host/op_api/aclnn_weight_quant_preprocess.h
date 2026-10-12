/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_WEIGHT_QUANT_PREPROCESS_H_
#define OP_API_INC_WEIGHT_QUANT_PREPROCESS_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnWeightQuantPreprocess的第一段接口，用于获取计算所需workspace大小以及包含了算子计算流程的执行器。
 * @param [in] weight: 输入的weight张量，ND格式，2-3维。
 * @param [in] weightScale: 权重的反量化scale参数，ND格式，2-3维。
 * @param [in] weightOffsetOptional: 权重的反量化offset参数（可选）。
 * @param [in] biasOptional: 偏置参数（可选）。
 * @param [in] xDtype: Matmul的激活矩阵的数据类型。
 * @param [in] xScaleDtype: 激活的量化scale参数的数据类型（可选）。
 * @param [in] kGroupSize: 权重在per-group量化时K维度的group大小。
 * @param [out] outWeight: 预处理后的目标weight张量，FRACTAL_NZ格式，4-5维。
 * @param [out] outWeightScale: 预处理后的目标weightScale张量。
 * @param [out] outWeightOffsetOptional: 预处理后的目标weightOffset张量（可选）。
 * @param [out] outBiasOptional: 预处理后的目标bias张量（可选）。
 * @param [out] workspaceSize: 需要在Device侧申请的workspace的大小。
 * @param [out] executor: 包含算子计算流程的op执行器。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnWeightQuantPreprocessGetWorkspaceSize(
    const aclTensor* weight, const aclTensor* weightScale, const aclTensor* weightOffsetOptional,
    const aclTensor* biasOptional, const aclDataType xDtype, const aclDataType xScaleDtype, const int64_t kGroupSize,
    aclTensor* outWeight, aclTensor* outWeightScale, aclTensor* outWeightOffsetOptional, aclTensor* outBiasOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnWeightQuantPreprocess的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: 指定执行任务的AscendCL Stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnWeightQuantPreprocess(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_WEIGHT_QUANT_PREPROCESS_H_
