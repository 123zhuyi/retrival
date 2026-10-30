/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_copy.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "conversion/broadcast_to/op_api/broadcast_to.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/shape_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,        op::DataType::DT_INT16,         op::DataType::DT_INT32,
    op::DataType::DT_INT64,       op::DataType::DT_UINT8,         op::DataType::DT_UINT16,
    op::DataType::DT_UINT32,      op::DataType::DT_UINT64,        op::DataType::DT_FLOAT16,
    op::DataType::DT_FLOAT,       op::DataType::DT_BOOL,          op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64,   op::DataType::DT_COMPLEX128,    op::DataType::DT_BF16,
    op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_HIFLOAT8,
    op::DataType::DT_FLOAT8_E8M0};

static bool CheckNotNull(const aclTensor* self, const aclTensor* src)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(src, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* src)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(src, DTYPE_SUPPORT_LIST, return false);

    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* src)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(src, MAX_SUPPORT_DIMS_NUMS, return false);

    // src是否能广播到self
    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, src, broadcastShape, return false);
    if (broadcastShape != self->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of broadcast result should be %s, but self is %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(self->GetViewShape()).GetString());
        return false;
    }

    return true;
}

static void CheckFormat(const aclTensor* x)
{
    op::Format format = x->GetStorageFormat();
    if (format == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("Format of input gets [%s], this format mat lead to precision failure",
        op::ToString(format).GetString());
    }
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* src)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, src), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, src), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查数据维度和形状是否支持
    CHECK_RET(CheckShape(self, src), ACLNN_ERR_PARAM_INVALID);
    CheckFormat(self);
    return ACLNN_SUCCESS;
}

aclTensor* BraodCastTensor(const op::Shape dstShape, const aclTensor* src, aclOpExecutor* executor)
{
    auto dstTensor = executor->AllocTensor(dstShape, src->GetDataType());
    op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(dstShape);
    auto shape =
        executor->ConvertToTensor(broadcastDims.data(), broadcastDims.size(), static_cast<op::DataType>(ACL_INT64));
    auto result = l0op::BroadcastTo(src, dstTensor, shape, executor);
    return const_cast<aclTensor*>(result);
}

aclnnStatus aclnnInplaceCopyGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* src, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnInplaceCopy, DFX_IN(selfRef, src), DFX_OUT());

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(selfRef, src);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor在kernel中支持, 若self为空，则复制后self还是空
    if (selfRef->IsEmpty() || src->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 若先将src转换成连续的tensor
    auto srcContiguous = l0op::Contiguous(src, uniqueExecutor.get());
    CHECK_RET(srcContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 若数据类型不同，调用 cast， 得到 srcCasted
    auto dstDtype = selfRef->GetDataType();
    auto srcDtype = src->GetDataType();
    auto srcCast = srcContiguous;
    if (dstDtype != srcDtype) {
        srcCast = l0op::Cast(srcContiguous, dstDtype, uniqueExecutor.get());
        CHECK_RET(srcCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 若形状不同，则做broadcast
    auto srcBroadcast = srcCast;
    if (selfRef->GetViewShape() != srcCast->GetViewShape()) {
        srcBroadcast = BraodCastTensor(selfRef->GetViewShape(), srcCast, uniqueExecutor.get());
        CHECK_RET(srcBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 连续-->非连续, 连续-->连续, 都直接 ViewCopy
    auto viewcopyResult = l0op::ViewCopy(srcBroadcast, selfRef, uniqueExecutor.get());
    CHECK_RET(viewcopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceCopy(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnInplaceCopy);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
