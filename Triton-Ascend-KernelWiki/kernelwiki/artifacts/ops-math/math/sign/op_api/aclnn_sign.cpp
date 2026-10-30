/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sign.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "sign.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/op_api_def.h"
#include "op_api/level2_base_caculation.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// 1971中对bf16的支持在下方校验，对bool的支持转成int32计算
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_DOUBLE, DataType::DT_FLOAT,     DataType::DT_FLOAT16,    DataType::DT_INT32,
    DataType::DT_INT64,  DataType::DT_COMPLEX64, DataType::DT_COMPLEX128, DataType::DT_BOOL};

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* result)
{
    if (!CheckType(self->GetDataType(), DTYPE_SUPPORT_LIST)) {
        SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
        auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
        switch (curArch) {
            case NpuArch::DAV_1001: {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "On Ascend910 self dtype [%s] should be in dtype support list [%s].",
                    op::ToString(self->GetDataType()).GetString(), op::ToString(DTYPE_SUPPORT_LIST).GetString());
                return false;
            }
            case NpuArch::DAV_2201:
            case NpuArch::DAV_3510: {
                if (self->GetDataType() != op::DataType::DT_BF16) {
                    OP_LOGE(
                        ACLNN_ERR_PARAM_INVALID,
                        "On %s, self dtype [%s] should be in dtype support list [%s] or be BF16.",
                        op::ToString(socVersion).GetString(), op::ToString(self->GetDataType()).GetString(),
                        op::ToString(DTYPE_SUPPORT_LIST).GetString());
                    return false;
                }
                break;
            }
            default: {
            }
        }
    } else {
        if (self->GetDataType() != result->GetDataType()) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Self dtype should be same as result dtype. self [%s], result [%s]",
                op::ToString(self->GetDataType()).GetString(), op::ToString(result->GetDataType()).GetString());
            return false;
        }
    }
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* result)
{
    // self和out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, result, return false);

    return true;
}

static aclIntArray* GetTensorShapeActivation(const aclTensor* x, aclOpExecutor* executor)
{
    auto shape = x->GetViewShape();
    int64_t dimSize = x->GetViewShape().GetDimNum();

    std::vector<int64_t> valuePerm(dimSize);
    for (int i = 0; i < dimSize; i++) {
        valuePerm[i] = shape[i];
    }
    auto perm = executor->AllocIntArray(valuePerm.data(), dimSize);
    return perm;
}

static const aclTensor* ReshapeLongTensorActivation(
    const aclTensor* x, aclOpExecutor* executor, int originalDimSize, aclIntArray* valuePerm = nullptr)
{
    int64_t dimSize = x->GetViewShape().GetDimNum();
    if (static_cast<int64_t>(originalDimSize) == dimSize && dimSize <= static_cast<int64_t>(MAX_SUPPORT_DIMS_NUMS)) {
        return x;
    }

    auto reshapeSelf = l0op::Reshape(x, valuePerm, executor);
    return reshapeSelf;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* result)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2Tensor(self, result), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, result), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, result), ACLNN_ERR_PARAM_INVALID);

    // 4. ND_element-wise 格式算子，无需判断格式

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSignGetWorkspaceSize(
    const aclTensor* self, aclTensor* result, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSign, DFX_IN(self), DFX_OUT(result));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, result);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (self->IsEmpty() || result->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    int64_t dimSize = self->GetViewShape().GetDimNum();
    auto shapeOriDetail = GetTensorShapeActivation(selfContiguous, uniqueExecutor.get());

    auto selfCast = selfContiguous;
    // 对于bool类型，转换为int32去做
    auto castTypeOrig = selfContiguous->GetDataType();
    if (castTypeOrig == DataType::DT_BOOL) {
        selfCast = l0op::Cast(selfContiguous, DataType::DT_INT32, uniqueExecutor.get());
        CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用l0算子Ceil进行计算
    auto signResult = l0op::Sign(selfCast, uniqueExecutor.get());
    CHECK_RET(signResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto reshapeSignResult = signResult;
    if (static_cast<uint64_t>(dimSize) > MAX_SUPPORT_DIMS_NUMS) {
        reshapeSignResult = ReshapeLongTensorActivation(signResult, uniqueExecutor.get(), dimSize, shapeOriDetail);
    }
    auto reshapeSignResultCast = reshapeSignResult;
    // 对于bool类型，转成bool后输出
    if (castTypeOrig == DataType::DT_BOOL) {
        reshapeSignResultCast = l0op::Cast(reshapeSignResult, DataType::DT_BOOL, uniqueExecutor.get());
        CHECK_RET(reshapeSignResultCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(reshapeSignResultCast, result, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSign(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSign);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif