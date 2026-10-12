/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_batch_matmul_reduce_scatter_all_to_all.h"
#include "op_mc2.h"
#include "common/op_host/op_api/matmul_util.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(const aclTensor *x,
                                                                              const aclTensor *weight,
                                                                              const aclTensor *bias,
                                                                              const char *group_ep,
                                                                              const char *group_tp,
                                                                              int64_t tp_world_size,
                                                                              int64_t ep_world_size,
                                                                              int64_t y_shard_type,
                                                                              bool transpose_weight, aclTensor *y,
                                                                              uint64_t *workspaceSize,
                                                                              aclOpExecutor **executor);
extern aclnnStatus aclnnInnerBatchMatMulReduceScatterAlltoAll(void *workspace, uint64_t workspaceSize,
                                                              aclOpExecutor *executor, aclrtStream stream);

// check nullptr
static bool CheckNotNull(const aclTensor *x, const aclTensor *weight, const char *groupEp, const char *groupTp,
                         aclTensor* out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(out, return false);
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

// check dtype
static bool CheckDtypeValid(const aclTensor* x, const aclTensor* weight, const aclTensor* bias, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, MOE_X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(weight, MOE_X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, MOE_X_DTYPE_SUPPORT_LIST, return false);

    OP_CHECK_DTYPE_NOT_SAME(x, weight, return false);
    OP_CHECK_DTYPE_NOT_SAME(x, out, return false);

    if (bias != nullptr) { // x和bias支持同时为fp16，或x为bf16和bias为fp32
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, MOE_BIAS_DTYPE_SUPPORT_LIST, return false);
        if (x->GetDataType() == op::DataType::DT_FLOAT16) {
            OP_CHECK_DTYPE_NOT_SAME(bias, x, return false);
        }
        if (x->GetDataType() == op::DataType::DT_BF16) {
            if (bias->GetDataType() != op::DataType::DT_FLOAT) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected bias dtype to be float32, but got %s.",
                        op::ToString(bias->GetDataType()).GetString());
                return false;
            }
        }
    }
    return true;
}

// check shape dim
static bool CheckIfTensorThreeDim(const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
                                  const aclTensor* out)
{
    OP_CHECK_WRONG_DIMENSION(x, SUPPORTED_DIMENSIONAL, return false);
    OP_CHECK_WRONG_DIMENSION(weight, SUPPORTED_DIMENSIONAL, return false);
    OP_CHECK_WRONG_DIMENSION(out, SUPPORTED_DIMENSIONAL, return false);
    if (bias != nullptr) {
        if ((bias->GetViewShape().GetDimNum() != SUPPORTED_DIMENSIONAL) &&
            (bias->GetViewShape().GetDimNum() != BIAS_SUPPORTED_DIMENSIONAL)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected bias dim to be 2 or 3, but got dim [%zu].",
                    bias->GetViewShape().GetDimNum());
            return false;
        }
    }
    return true;
}

// 维度判断
static bool CheckTensorDimCommonShape(const aclTensor* x, const aclTensor* weight, int64_t epWorldSize, 
                                      const aclTensor* out)
{
    // E = E/ep * ep, y_0 == x_0 * ep
    if ((x->GetViewShape().GetDim(DIM_0) * epWorldSize) != out->GetViewShape().GetDim(DIM_0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The first dim of x multi epSize must equal the first dim of out, "
                "but got y_0=[%ld], x_0=[%ld], epSize=[%ld].", out->GetViewShape().GetDim(DIM_0),
                x->GetViewShape().GetDim(DIM_0), epWorldSize);
        return false;
    }
    // x_0 == w_0
    if (x->GetViewShape().GetDim(DIM_0) != weight->GetViewShape().GetDim(DIM_0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The first dim of x must equal the first dim of weight, "
                "but got x_0=[%ld], W_0=[%ld].", x->GetViewShape().GetDim(DIM_0),
                weight->GetViewShape().GetDim(DIM_0));
        return false;
    }
    // x_2 == w_1
    if (x->GetViewShape().GetDim(DIM_2) != weight->GetViewShape().GetDim(DIM_1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The last dim of x must equal the second dim of weight(without transpose), "
                "but got x_2=[%ld], W_1=[%ld].", x->GetViewShape().GetDim(DIM_2),
                weight->GetViewShape().GetDim(DIM_1));
        return false;
    }
    return true;
}

static bool CheckTensorDimUniqueShape(const aclTensor* x, const aclTensor* weight, int64_t epWorldSize, 
                                      int64_t tpWorldSize, int64_t yShardType, const aclTensor* out)
{
    if (yShardType == 0) {
        // H = H/tp * tp, w_2 == y_2 * tp
        if ((out->GetViewShape().GetDim(DIM_2) * tpWorldSize) != weight->GetViewShape().GetDim(DIM_2)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected H = H/tp * tp, but got W_2=[%ld], out_1=[%ld], tp=[%ld].",
                    weight->GetViewShape().GetDim(DIM_2), out->GetViewShape().GetDim(DIM_1), tpWorldSize);
            return false;
        }
        // x_1 == y_1 * ep
        if ((out->GetViewShape().GetDim(DIM_1) * epWorldSize) != x->GetViewShape().GetDim(DIM_1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected x_1 = out_1 * ep, but got x_1=[%ld], out_1=[%ld], ep=[%ld].",
                    x->GetViewShape().GetDim(DIM_1), out->GetViewShape().GetDim(DIM_1), epWorldSize);
            return false;
        }
    } else if (yShardType == 1) {
        // w_2 == y_2
        if (out->GetViewShape().GetDim(DIM_2) != weight->GetViewShape().GetDim(DIM_2)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The last dim of weight(without transpose) must equal the last dim of out, "
                    "but got out_2=[%ld], W_2=[%ld].", out->GetViewShape().GetDim(DIM_2),
                    weight->GetViewShape().GetDim(DIM_2));
            return false;
        }
        // x_1 == y_1 * ep * tp
        if ((out->GetViewShape().GetDim(DIM_1) * epWorldSize * tpWorldSize) != x->GetViewShape().GetDim(DIM_1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The second dim of out multi ep multi tp must equal the second dim of x, "
                    "but got x_1=[%ld], out_1=[%ld], ep=[%ld], tp=[%ld].", x->GetViewShape().GetDim(DIM_1),
                    out->GetViewShape().GetDim(DIM_1), epWorldSize, tpWorldSize);
            return false;
        }
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "yShardType [%ld] is invalid.", yShardType);
        return false;
    }
    return true;
}

static bool CheckBiasDim(const aclTensor* weight, const aclTensor* bias, const aclTensor* out)
{
    if (bias != nullptr) {
        if ((bias->GetViewShape().GetDim(DIM_0) != weight->GetViewShape().GetDim(DIM_0)) ||
            ((bias->GetViewShape().GetDimNum() == SUPPORTED_DIMENSIONAL) &&
            (bias->GetViewShape().GetDim(DIM_1) != 1)) ||
            (bias->GetViewShape().GetDim(bias->GetViewShape().GetDimNum() - 1) != out->GetViewShape().GetDim(DIM_2))) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Expected biasOptional shape to be [%ld,%ld] or [%ld,1,%ld], but got [%s].",
                    weight->GetViewShape().GetDim(DIM_0), out->GetViewShape().GetDim(DIM_2),
                    weight->GetViewShape().GetDim(DIM_0), out->GetViewShape().GetDim(DIM_2),
                    op::ToString(bias->GetViewShape()).GetString());
            return false;
        }
    }
    return true;
}

static bool CheckTensorDim(const aclTensor* x, const aclTensor* weight, const aclTensor* bias, int64_t epWorldSize,
                           int64_t tpWorldSize, int64_t yShardType, const aclTensor* out)
{
    // 是否为三维判断
    if (!CheckIfTensorThreeDim(x, weight, bias, out)) {
        return false;
    }

    // 公共shape特性
    if (!CheckTensorDimCommonShape(x, weight, epWorldSize, out)) {
        return false;
    }

    // 非公共shape特性
    if (!CheckTensorDimUniqueShape(x, weight, epWorldSize, tpWorldSize, yShardType, out)) {
        return false;
    }
    
    // bias 维度判断
    if (!CheckBiasDim(weight, bias, out)) {
        return false;
    }
    
    return true;
}

  // 属性校验
  static bool CheckAttr(int64_t epWorldSize, int64_t tpWorldSize, int64_t yShardType)
  {
    // yShardType 当前支持 0 或 1
    if ((yShardType) != 0 && (yShardType != 1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected yShardType to be 0 or 1, but got [%ld].", yShardType);
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
static bool CheckShapeRange(const aclTensor* x, const aclTensor *weight, const aclTensor *out)
{
    // 暂不支持空tensor场景
    // M/tp in [1, 65535], 空tensor K校验
    if ((x->GetViewShape().GetDim(DIM_2) <= 0) || (x->GetViewShape().GetDim(DIM_2) > MATMUL_K_LIMIT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected M/tp in range of [1, %ld], but got [%ld].", MATMUL_K_LIMIT,
                x->GetViewShape().GetDim(DIM_2));
        return false;
    }
    // E/ep in [1, 32], 空tensor E校验
    if ((x->GetViewShape().GetDim(DIM_0) <= 0) || (x->GetViewShape().GetDim(DIM_0) > MATMUL_E_OVER_EP_LIMIT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected E/ep in range of [1, %ld], but got [%ld].", MATMUL_E_OVER_EP_LIMIT,
                x->GetViewShape().GetDim(DIM_0));
        return false;
    }
    // E in [2, 512]
    if ((out->GetViewShape().GetDim(DIM_0) < EXPERT_LOWER_LIMIT) ||
        (out->GetViewShape().GetDim(DIM_0) > EXPERT_UPPER_LIMIT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected E in range of [%ld, %ld], but got [%ld].", EXPERT_LOWER_LIMIT,
                EXPERT_UPPER_LIMIT, out->GetViewShape().GetDim(DIM_0));
        return false;
    }
    // C > 0, C/tp > 0, 空tensor M校验
    if (out->GetViewShape().GetDim(DIM_1) <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected C > 0, but got [%ld].", out->GetViewShape().GetDim(DIM_1));
        return false;
    }
    // H in [1, 65535], 空tensor N校验
    if ((weight->GetViewShape().GetDim(DIM_2) <= 0) || (weight->GetViewShape().GetDim(DIM_2) > MATMUL_K_LIMIT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected H in range of [1, %ld], but got [%ld].", MATMUL_K_LIMIT,
                weight->GetViewShape().GetDim(DIM_2));
        return false;
    }
    return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *weight, const aclTensor *biasOptional,
                               const char *groupEp, const char *groupTp, int64_t epWorldSize, int64_t tpWorldSize,
                               int64_t yShardType, aclTensor *out)
{
    CHECK_RET(CheckNotNull(x, weight, groupEp, groupTp, out), ACLNN_ERR_PARAM_NULLPTR);

    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required groupEp name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strnlen(groupTp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required groupTp name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(CheckDtypeValid(x, weight, biasOptional, out), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckAttr(epWorldSize, tpWorldSize, yShardType), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckTensorDim(x, weight, biasOptional, epWorldSize, tpWorldSize, yShardType, out),
              ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShapeRange(x, weight, out), ACLNN_ERR_PARAM_INVALID);
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
    viewDims[DIM_1] = x2->GetViewShape().GetDim(DIM_2);  // transpose the last two dimensions
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


aclnnStatus aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
                                                                  const aclTensor *biasOptional, const char *groupEp,
                                                                  const char *groupTp, int64_t epWorldSize,
                                                                  int64_t tpWorldSize, int64_t yShardType,
                                                                  aclTensor *out, uint64_t *workspaceSize,
                                                                  aclOpExecutor **executor)
{
    auto ret_param = CheckParams(x, weight, biasOptional, groupEp, groupTp, epWorldSize, tpWorldSize, yShardType, out);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    bool transposeWeight = Ops::Transformer::IsTransposeLastTwoDims(weight);
    if(transposeWeight){
        if(weight->GetTensor() == nullptr){
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Tensor of weight is null.");
            return ACLNN_ERR_INNER_NULLPTR;
        }
        auto weightTrans = TransTensor(weight);
        aclnnStatus ret = aclnnInnerBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(x, weightTrans, biasOptional, groupEp,
                                                                                 groupTp, epWorldSize, tpWorldSize,
                                                                                 yShardType, transposeWeight, out,
                                                                                 workspaceSize, executor);
        return ret;
    } else {
        aclnnStatus ret = aclnnInnerBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(x, weight, biasOptional, groupEp,
                                                                                 groupTp, epWorldSize, tpWorldSize,
                                                                                 yShardType, transposeWeight, out,
                                                                                 workspaceSize, executor);
        return ret;
    }
}

aclnnStatus aclnnBatchMatMulReduceScatterAlltoAll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerBatchMatMulReduceScatterAlltoAll(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif