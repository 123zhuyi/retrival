/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_linalg_cross.h"
#include "cross.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "conversion/broadcast_to/op_api/broadcast_to.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_DOUBLE,  op::DataType::DT_INT8,
    op::DataType::DT_INT16,      op::DataType::DT_INT32,      op::DataType::DT_INT64,   op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128, op::DataType::DT_UINT8};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_DOUBLE,  op::DataType::DT_INT8,
    op::DataType::DT_INT16,      op::DataType::DT_INT32,      op::DataType::DT_INT64,   op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128, op::DataType::DT_UINT8, op::DataType::DT_BF16};

inline static bool CheckNotNull(const aclTensor *self, const aclTensor *other, const aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *other, const aclTensor *out)
{
    // 检查self的数据类型是否在linalg cross算子的支持列表内
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 ||
        op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_910B, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    }
    // 检查self,other,out的数据类型是否相同
    OP_CHECK_DTYPE_NOT_MATCH(self, other->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);

    return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *other, const aclTensor *out)
{
    // 所有tensor的维度必须小于 MAX_SUPPORT_DIMS_NUMS
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(other, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

static bool CheckBroadcastShape(const aclTensor *self, const aclTensor *other, const aclTensor *out)
{
    // self和other必须符合broadcast关系
    OP_CHECK_BROADCAST(self, other, return false);
    op::Shape broadcastShape;
    BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape);

    // self和other的broadcast shape必须与out的shape一致
    if (broadcastShape != out->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "expected consistent tensor shape for the broadcast shape and out, but got %s and %s respectively.",
            op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
        return false;
    }

    return true;
}

static bool CheckDim(const aclTensor *self, int64_t dim)
{
    // dim的值必须在[-self的维度数量，self的维度数量-1]范围内
    auto dimSize = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    if (dim >= dimSize || dim < -1 * dimSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Dimension out of range(expected to be in range of [%ld, %ld], but got %ld.",
            -1 * dimSize, dimSize - 1, dim);
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *other, int64_t dim,
    const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, other, out), ACLNN_ERR_PARAM_INVALID);

    // 3. ND 算子不检查格式
    // 4. 检查self和out的维度是否在允许范围内
    CHECK_RET(CheckShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查self，other的shape是否符合broadcast关系且与out相同
    CHECK_RET(CheckBroadcastShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查dim是否合法
    CHECK_RET(CheckDim(self, dim), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

inline static aclTensor* BroadcastTensor(const op::Shape dstShape, const aclTensor* src, aclOpExecutor* executor)
{
    auto dstTensor = executor->AllocTensor(dstShape, src->GetDataType());
    op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(dstShape);
    auto shape = executor->ConvertToTensor(broadcastDims.data(), broadcastDims.size(),
        static_cast<op::DataType>(ACL_INT64));
    auto result = l0op::BroadcastTo(src, dstTensor, shape, executor);
    return const_cast<aclTensor*>(result);
}

static aclnnStatus ExecLinalgCrossGetWorkspaceSize(const aclTensor *self, const aclTensor *other,
    int64_t dim, aclTensor *out,  uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, other, dim, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // linalg cross算子的空tensor在kernel中不支持，对标竞品根据算子实际情况补充
    if (out->IsEmpty()) {
        OP_LOGD("empty input tensor");
        // 根据实际支持情况补充
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACL_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
    CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *selfBroadCast = selfContiguous;
    const aclTensor *otherBroadCast = otherContiguous;

    // 声明内部dimInner
    auto dimInner = dim;

    // 若形状不同，则做broadcast
    if (selfContiguous->GetViewShape() != otherContiguous->GetViewShape() && !IsRegBase()) {
        selfBroadCast = BroadcastTensor(out->GetViewShape(), selfContiguous, uniqueExecutor.get());
        CHECK_RET(selfBroadCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        otherBroadCast = BroadcastTensor(out->GetViewShape(), otherContiguous, uniqueExecutor.get());
        CHECK_RET(otherBroadCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // self扩维后且dim<0时需要加上扩充维度的数量
        if (dimInner >= 0 && self->GetViewShape().GetDimNum() < other->GetViewShape().GetDimNum()) {
            dimInner += (other->GetViewShape().GetDimNum() - self->GetViewShape().GetDimNum());
        }
    }

    // 若dim为负数index，将其转为对应正数index
    if (dimInner < 0) {
        dimInner += selfBroadCast->GetViewShape().GetDimNum();
    }

    // 检验selfBroadCast在broadcast后对应dim的维度上的shape是否为3
    if (selfBroadCast->GetViewShape().GetDim(dimInner) != 3) {
        if (dim < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dimension %ld does not have size 3.", dim);
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dimension %ld does not have size 3.", dimInner);
        }
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (self->GetDataType() == op::DataType::DT_BF16 &&
        op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2201 &&
        op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        selfBroadCast = l0op::Cast(selfBroadCast, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(selfBroadCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        otherBroadCast = l0op::Cast(otherBroadCast, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(otherBroadCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用Cross算子Kernel
    auto crossOpOut = l0op::Cross(selfBroadCast, otherBroadCast, dimInner, uniqueExecutor.get());
    CHECK_RET(crossOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (self->GetDataType() == op::DataType::DT_BF16 &&
        op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2201 &&
        op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        crossOpOut = l0op::Cast(crossOpOut, op::DataType::DT_BF16, uniqueExecutor.get());
        CHECK_RET(crossOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(crossOpOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLinalgCrossGetWorkspaceSize(const aclTensor *self, const aclTensor *other, int64_t dim,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    
    L2_DFX_PHASE_1(aclnnLinalgCross, DFX_IN(self, other, dim), DFX_OUT(out));
    return ExecLinalgCrossGetWorkspaceSize(self, other, dim, out, workspaceSize, executor);
}

aclnnStatus aclnnLinalgCross(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnLinalgCross);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif