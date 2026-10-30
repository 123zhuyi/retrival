/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_all_to_all_all_gather_batch_matmul.h"
#include <algorithm>
#include "op_mc2.h"
#include "common/op_host/op_api/matmul_util.h"
#include "op_mc2_def.h"
#include "external/aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerAlltoAllAllGatherBatchMatMulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
                                                                          const aclTensor *bias, const char *group_ep,
                                                                          const char *group_tp, int64_t ep_world_size,
                                                                          int64_t tp_world_size, int64_t x_shard_type,
                                                                          int64_t act_type, bool transpose_weight,
                                                                          bool output_y2_flag, bool output_y3_flag,
                                                                          aclTensor *y1, aclTensor *y2, aclTensor *y3,
                                                                          uint64_t *workspaceSize,
                                                                          aclOpExecutor **executor);
extern aclnnStatus aclnnInnerAlltoAllAllGatherBatchMatMul(void *workspace, uint64_t workspaceSize,
                                                          aclOpExecutor *executor, aclrtStream stream);

// check nullptr
static bool CheckNotNull(const aclTensor *x, const aclTensor *weight, const char *groupEp, const char *groupTp,
                         aclTensor* y1Out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(y1Out, return false);
    if ((groupEp == nullptr) || (strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required groupEp name is Empty.");
        return false;
    }
    if ((groupTp == nullptr) || (strnlen(groupTp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required groupTp name is Empty.");
        return false;
    }
    return true;
}

// check input/output dtype
static bool CheckDtypeValid(const aclTensor* x, const aclTensor* weight, const aclTensor* bias, const aclTensor* y1Out,
                            const aclTensor* y2OutOptional, const aclTensor* y3OutOptional)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, MOE_X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(weight, MOE_X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y1Out, MOE_X_DTYPE_SUPPORT_LIST, return false);

    OP_CHECK_DTYPE_NOT_SAME(x, weight, return false);
    OP_CHECK_DTYPE_NOT_SAME(x, y1Out, return false);
    
    if (bias != nullptr) { // x和bias支持同时为fp16，或x为bf16和bias为fp32
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, MOE_BIAS_DTYPE_SUPPORT_LIST, return false);
        if (x->GetDataType() == op::DataType::DT_FLOAT16) {
            OP_CHECK_DTYPE_NOT_SAME(bias, x, return false);
        }
        if (x->GetDataType() == op::DataType::DT_BF16) {
            if (bias->GetDataType() != op::DataType::DT_FLOAT) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected bias dtype to be float32, but got %s.",
                        ToString(bias->GetDataType()).GetString());
                return false;
            }
        }
    }
    if (y2OutOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(y2OutOptional, MOE_X_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SAME(y2OutOptional, x, return false);
    }
    if (y3OutOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(y3OutOptional, MOE_X_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SAME(y3OutOptional, x, return false);
    }
    return true;
}

// check shape dim
static bool CheckIfTensorThreeDim(const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
                                  const aclTensor* y1Out, const aclTensor* y2OutOptional, const aclTensor* y3OutOptional)
{
    OP_CHECK_WRONG_DIMENSION(x, SUPPORTED_DIMENSIONAL, return false);
    OP_CHECK_WRONG_DIMENSION(weight, SUPPORTED_DIMENSIONAL, return false);
    OP_CHECK_WRONG_DIMENSION(y1Out, SUPPORTED_DIMENSIONAL, return false);
    if (bias != nullptr) {
        if ((bias->GetViewShape().GetDimNum() != SUPPORTED_DIMENSIONAL) &&
            (bias->GetViewShape().GetDimNum() != BIAS_SUPPORTED_DIMENSIONAL)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected bias dim to be 2 or 3, but got dim [%zu].",
                    bias->GetViewShape().GetDimNum());
            return false;
        }
    }
    if (y2OutOptional != nullptr) {
        OP_CHECK_WRONG_DIMENSION(y2OutOptional, SUPPORTED_DIMENSIONAL, return false);
    }
    if (y3OutOptional != nullptr) {
        OP_CHECK_WRONG_DIMENSION(y3OutOptional, SUPPORTED_DIMENSIONAL, return false);
    }
    return true;
}

// 维度判断
static bool CheckTensorDimCommonShape(const aclTensor* x, const aclTensor* weight, int64_t epWorldSize, 
                                      const aclTensor* y1Out)
{
    // E = E/ep * ep, x_0 == w_0 * ep
    if ((weight->GetViewShape().GetDim(DIM_0) * epWorldSize) != x->GetViewShape().GetDim(DIM_0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The first dim of weight multi epSize must equal the first dim of x, "
                "but got x_0=[%ld], W_0=[%ld], ep=[%ld].", x->GetViewShape().GetDim(DIM_0),
                weight->GetViewShape().GetDim(DIM_0), epWorldSize);
        return false;
    }
    // y1_0 == w_0
    if (y1Out->GetViewShape().GetDim(DIM_0) != weight->GetViewShape().GetDim(DIM_0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The first dim of y1Out must equal the first dim of weight, "
                "but got y1_0=[%ld], W_0=[%ld].", y1Out->GetViewShape().GetDim(DIM_0),
                weight->GetViewShape().GetDim(DIM_0));
        return false;
    }
    // y1_2 == w_2
    if (y1Out->GetViewShape().GetDim(DIM_2) != weight->GetViewShape().GetDim(DIM_2)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The last dim of y1Out must equal the last dim of weight(without transpose), "
                "but got y1_2=[%ld], W_2=[%ld].", y1Out->GetViewShape().GetDim(DIM_2),
                weight->GetViewShape().GetDim(DIM_2));
        return false;
    }
    return true;
}

static bool CheckTensorDimUniqueShape(const aclTensor* x, const aclTensor* weight, int64_t epWorldSize, 
                                      int64_t tpWorldSize, int64_t xShardType, const aclTensor* y1Out)
{
    if (xShardType == 0) {
        // H = H/tp * tp, w_1 = x_2 * tp
        if ((x->GetViewShape().GetDim(DIM_2) * tpWorldSize) != weight->GetViewShape().GetDim(DIM_1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected H = H/tp * tp, but got W_1=[%ld], x_2=[%ld], tp=[%ld].",
                    weight->GetViewShape().GetDim(DIM_1), x->GetViewShape().GetDim(DIM_2), tpWorldSize);
            return false;
        }
        // y1_1 == x_1 * ep
        if (y1Out->GetViewShape().GetDim(DIM_1) != (x->GetViewShape().GetDim(DIM_1) * epWorldSize)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected y1_1 = x_1 * ep, but got y1_1=[%ld], x_1=[%ld], ep=[%ld].",
                    y1Out->GetViewShape().GetDim(DIM_1), x->GetViewShape().GetDim(DIM_1), epWorldSize);
            return false;
        }
    } else if (xShardType == 1) {
        // w_1 == x_2
        if (x->GetViewShape().GetDim(DIM_2) != weight->GetViewShape().GetDim(DIM_1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The second dim of weight(without transpose) must equal the last dim of x, "
                    "but got x_2=[%ld], W_1=[%ld].", x->GetViewShape().GetDim(DIM_2),
                    weight->GetViewShape().GetDim(DIM_1));
            return false;
        }
        // y1_1 == x_1 * ep * tp
        if (y1Out->GetViewShape().GetDim(DIM_1) != (x->GetViewShape().GetDim(DIM_1) * epWorldSize * tpWorldSize)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The second dim of x multi ep multi tp must equal the second dim of "
                    "y1Out, but got y1_1=[%ld], x_1=[%ld], ep=[%ld], tp=[%ld].", y1Out->GetViewShape().GetDim(DIM_1),
                    x->GetViewShape().GetDim(DIM_1), epWorldSize, tpWorldSize);
            return false;
        }
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "xShardType [%ld] is invalid.", xShardType);
        return false;
    }
    return true;
}

static bool CheckBiasDim(const aclTensor* weight, const aclTensor* bias, const aclTensor* y1Out)
{
    // bias 维度判断
    if (bias != nullptr) {
        if ((bias->GetViewShape().GetDim(DIM_0) != weight->GetViewShape().GetDim(DIM_0)) ||
            ((bias->GetViewShape().GetDimNum() == SUPPORTED_DIMENSIONAL) &&
            (bias->GetViewShape().GetDim(DIM_1) != 1)) ||
            (bias->GetViewShape().GetDim(bias->GetViewShape().GetDimNum() - 1) !=
            y1Out->GetViewShape().GetDim(DIM_2))) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Expected biasOptional shape to be [%ld,%ld] or [%ld,1,%ld], but got [%s].",
                    weight->GetViewShape().GetDim(DIM_0), y1Out->GetViewShape().GetDim(DIM_2),
                    weight->GetViewShape().GetDim(DIM_0), y1Out->GetViewShape().GetDim(DIM_2),
                    ToString(bias->GetViewShape()).GetString());
            return false;
        }
    }
    return true;
}

static bool CheckOutOptionalDim(const aclTensor* weight, const aclTensor* y1Out, const aclTensor* y2OutOptional, 
                                const aclTensor* y3OutOptional)
{
    // y2OutOptional 维度判断
    if (y2OutOptional != nullptr) {
        if ((y2OutOptional->GetViewShape().GetDim(DIM_0) != weight->GetViewShape().GetDim(DIM_0)) ||
            (y2OutOptional->GetViewShape().GetDim(DIM_1) != y1Out->GetViewShape().GetDim(DIM_1)) ||
            (y2OutOptional->GetViewShape().GetDim(DIM_2) != weight->GetViewShape().GetDim(DIM_1))) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected y2OutOptional shape to be [%ld,%ld,%ld], but got [%s].",
                    weight->GetViewShape().GetDim(DIM_0), y1Out->GetViewShape().GetDim(DIM_1),
                    weight->GetViewShape().GetDim(DIM_1), ToString(y2OutOptional->GetViewShape()).GetString());
            return false;
        }
    }
    // y3OutOptional 维度判断
    if (y3OutOptional != nullptr) {
        if ((y3OutOptional->GetViewShape().GetDim(DIM_0) != weight->GetViewShape().GetDim(DIM_0)) ||
            (y3OutOptional->GetViewShape().GetDim(DIM_1) != y1Out->GetViewShape().GetDim(DIM_1)) ||
            (y3OutOptional->GetViewShape().GetDim(DIM_2) != y1Out->GetViewShape().GetDim(DIM_2))) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected y3OutOptional shape to be [%ld,%ld,%ld], but got [%s].",
                    weight->GetViewShape().GetDim(DIM_0), y1Out->GetViewShape().GetDim(DIM_1),
                    y1Out->GetViewShape().GetDim(DIM_2), ToString(y3OutOptional->GetViewShape()).GetString());
            return false;
        }
    }
    return true;
}

static bool CheckTensorDim(const aclTensor* x, const aclTensor* weight, const aclTensor* bias, int64_t epWorldSize,
                           int64_t tpWorldSize, int64_t xShardType, aclTensor* y1Out, aclTensor* y2OutOptional,
                           aclTensor* y3OutOptional)
{
    // 是否为三维判断
    if (!CheckIfTensorThreeDim(x, weight, bias, y1Out, y2OutOptional, y3OutOptional)) {
        return false;
    }

    // 公共shape特性
    if (!CheckTensorDimCommonShape(x, weight, epWorldSize, y1Out)) {
        return false;
    }

    // 非公共shape特性
    if (!CheckTensorDimUniqueShape(x, weight, epWorldSize, tpWorldSize, xShardType, y1Out)) {
        return false;
    }

    // bias维度判断
    if (!CheckBiasDim(weight, bias, y1Out)) {
        return false;
    }

    // OutOptional维度判断
    if (!CheckOutOptionalDim(weight, y1Out, y2OutOptional, y3OutOptional)) {
        return false;
    }
    return true;
}

// 属性校验
static bool CheckAttr(int64_t epWorldSize, int64_t tpWorldSize, int64_t xShardType, int64_t actType)
{
    // xShardType 当前支持 0 或 1
    if ((xShardType != 0) && (xShardType != 1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected xShardType to be 0 or 1, but got [%ld].", xShardType);
        return false;
    }
    // actType 当前支持 NONE/GELU/SILU/FASTGELU/RELU
    if (std::find(ops::ACT_TYPE_SUPPORT_VEC.begin(), ops::ACT_TYPE_SUPPORT_VEC.end(), actType) ==
        ops::ACT_TYPE_SUPPORT_VEC.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unexpected actType [%ld].", actType);
        return false;
    }
    // ep和tp 仅支持2、4、8、16、32
    if ((epWorldSize != WORLD_SIZE_PAIR) && (epWorldSize != WORLD_SIZE_QUAD) && (epWorldSize != WORLD_SIZE_OCTET) && (epWorldSize != WORLD_SIZE_SEXTET) && (epWorldSize != WORLD_SIZE_THIRTYTWO)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected epWorldSize to be 2, 4, 8, 16, or 32, but got [%ld].", epWorldSize);
        return false;
    }
    if ((tpWorldSize != WORLD_SIZE_PAIR) && (tpWorldSize != WORLD_SIZE_QUAD) && (tpWorldSize != WORLD_SIZE_OCTET) && (tpWorldSize != WORLD_SIZE_SEXTET) && (tpWorldSize != WORLD_SIZE_THIRTYTWO)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected tpWorldSize to be 2, 4, 8, 16, or 32, but got [%ld].", tpWorldSize);
        return false;
    }
    return true;
}

// 范围校验
static bool CheckShapeRange(const aclTensor* x, const aclTensor *weight)
{
    // 暂不支持空tensor场景
    // M/tp in [1, 65535], 空tensor N校验
    if ((weight->GetViewShape().GetDim(DIM_2) <= 0) || (weight->GetViewShape().GetDim(DIM_2) > MATMUL_K_LIMIT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected M/tp in range of [1, %ld], but got [%ld].", MATMUL_K_LIMIT,
                weight->GetViewShape().GetDim(DIM_2));
        return false;
    }
    // E/ep in [1, 32], 空tensor E校验
    if ((weight->GetViewShape().GetDim(DIM_0) <= 0) ||
        (weight->GetViewShape().GetDim(DIM_0) > MATMUL_E_OVER_EP_LIMIT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected E/ep in range of [1, %ld], but got [%ld].", MATMUL_E_OVER_EP_LIMIT,
                weight->GetViewShape().GetDim(DIM_0));
        return false;
    }
    // E in [2, 512]
    if ((x->GetViewShape().GetDim(DIM_0) < EXPERT_LOWER_LIMIT) ||
        (x->GetViewShape().GetDim(DIM_0) > EXPERT_UPPER_LIMIT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected E in range of [%ld, %ld], but got [%ld].", EXPERT_LOWER_LIMIT,
                EXPERT_UPPER_LIMIT, x->GetViewShape().GetDim(DIM_0));
        return false;
    }
    // C > 0, C/tp > 0, 空tensor M校验
    if (x->GetViewShape().GetDim(DIM_1) <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected C > 0, but got [%ld].", x->GetViewShape().GetDim(DIM_1));
        return false;
    }
    // H in [1, 65535], 空tensor K校验
    if ((weight->GetViewShape().GetDim(DIM_1) <= 0) || (weight->GetViewShape().GetDim(DIM_1) > MATMUL_K_LIMIT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected H in range of [1, %ld], but got [%ld].", MATMUL_K_LIMIT,
                weight->GetViewShape().GetDim(DIM_1));
        return false;
    }
    return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *weight, const aclTensor *biasOptional,
                               const char *groupEp, const char *groupTp, int64_t epWorldSize, int64_t tpWorldSize,
                               int64_t xShardType, int64_t actType, aclTensor *y1Out, aclTensor *y2OutOptional,
                               aclTensor *y3OutOptional)
{
    CHECK_RET(CheckNotNull(x, weight, groupEp, groupTp, y1Out), ACLNN_ERR_PARAM_NULLPTR);

    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required groupEp name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strnlen(groupTp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required groupTp name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(CheckDtypeValid(x, weight, biasOptional, y1Out, y2OutOptional, y3OutOptional), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckAttr(epWorldSize, tpWorldSize, xShardType, actType), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckTensorDim(x, weight, biasOptional, epWorldSize, tpWorldSize, xShardType, y1Out, y2OutOptional,
              y3OutOptional), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShapeRange(x, weight), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static const aclTensor *TransTensor(const aclTensor *x2)
{
    uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDims(storageDimsNum);
    for (uint64_t i = 0; i < storageDimsNum; i++) {
        storageDims[i] = x2->GetStorageShape().GetDim(i);
    }

    uint64_t viewDimsNum = x2->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDims;
	viewDims.resize(viewDimsNum);
    for (uint64_t i = 0; i < viewDimsNum; i++) {
        viewDims[i] = x2->GetViewShape().GetDim(i);
    }
    viewDims[DIM_0] = x2->GetViewShape().GetDim(DIM_0);
    viewDims[DIM_1] = x2->GetViewShape().GetDim(DIM_2);
    viewDims[DIM_2] = x2->GetViewShape().GetDim(DIM_1);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(x2, &dataType);
    auto stride = x2->GetViewStrides();
    std::vector<int64_t> strideCopy(viewDimsNum);
    strideCopy[DIM_0] = stride[DIM_0];
    strideCopy[DIM_1] = stride[DIM_2];  // transpose the last two dimensions
    strideCopy[DIM_2] = stride[DIM_1];

    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDims.data(), viewDimsNum, dataType, strideCopy.data(), offset, format, storageDims.data(),
                            storageDimsNum, x2->GetTensor()->GetAddr());
}

aclnnStatus aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
                                                              const aclTensor *biasOptional, const char *groupEp,
                                                              const char *groupTp, int64_t epWorldSize,
                                                              int64_t tpWorldSize, int64_t xShardType, int64_t actType,
                                                              aclTensor *y1Out, aclTensor *y2OutOptional,
                                                              aclTensor *y3OutOptional, uint64_t *workspaceSize,
                                                              aclOpExecutor **executor)
{
    auto ret_param = CheckParams(x, weight, biasOptional, groupEp, groupTp, epWorldSize, tpWorldSize, xShardType,
                                 actType, y1Out, y2OutOptional, y3OutOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    bool isY2Out = (y2OutOptional != nullptr);
    bool isY3Out = (y3OutOptional != nullptr);
    bool transposeWeight = Ops::Transformer::IsTransposeLastTwoDims(weight);
    if (isY3Out && (actType ==
        static_cast<int64_t>(ops::AlltoAllAllGatherBatchMatMulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected y3OutOptional and actType not NONE, but got actType [NONE].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if(transposeWeight){
        auto weightTrans = TransTensor(weight);
        aclnnStatus ret = aclnnInnerAlltoAllAllGatherBatchMatMulGetWorkspaceSize(x, weightTrans, biasOptional, groupEp, groupTp,
                                                                             epWorldSize, tpWorldSize, xShardType,
                                                                             actType, transposeWeight, isY2Out, isY3Out,
                                                                             y1Out, y2OutOptional, y3OutOptional,
                                                                             workspaceSize, executor);
        return ret;
    } else {
        aclnnStatus ret = aclnnInnerAlltoAllAllGatherBatchMatMulGetWorkspaceSize(x, weight, biasOptional, groupEp, groupTp,
                                                                             epWorldSize, tpWorldSize, xShardType,
                                                                             actType, transposeWeight, isY2Out, isY3Out,
                                                                             y1Out, y2OutOptional, y3OutOptional,
                                                                             workspaceSize, executor);
        return ret;
    }
}

aclnnStatus aclnnAlltoAllAllGatherBatchMatMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerAlltoAllAllGatherBatchMatMul(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif