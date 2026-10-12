/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_add_v3.h"
#include "add.h"
#include "math/axpy/op_api/axpy.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "math/mul/op_api/mul.h"
#include "math/logical_and/op_api/logical_and.h"
#include "math/logical_or/op_api/logical_or.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* Add 算子的完整计算流程如下:
 * self                               other
 *   |                                  |
 *   \                                  /
 * Contiguous(workspace_0)    Contiguous(workspace_2)
 *      \                             /
 *     Cast(workspace_1)     Cast(workspace_3)
 *               \            /
 *             Add(workspace_4)
 *                    |
 *               Cast(workspace_5)
 *                    |
 *                ViewCopy
 *                    |
 *                  result
 */

static constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> AXPY_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ADD_V3_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
    op::DataType::DT_INT8};

static bool CheckPromoteType(
    const op::DataType selfDtype, const op::DataType otherDtype, const aclScalar* alpha, const op::DataType outDtype,
    op::DataType promoteType)
{
    // 检查self和other能否做数据类型推导
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
            op::ToString(selfDtype).GetString(), op::ToString(otherDtype).GetString());
        return false;
    }

    // 检查alpha能否转换为推导后的数据类型
    if (promoteType == op::DataType::DT_BOOL) {
        OP_CHECK(
            IsIntegralType(DataType(alpha->GetDataType()), true),
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Alpha dtype %s can't be cast to the promote dtype %s.",
                op::ToString(DataType(alpha->GetDataType())).GetString(), op::ToString(promoteType).GetString()),
            return false);
    } else if (!CanCast(DataType(alpha->GetDataType()), promoteType)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Alpha dtype %s can't be cast to the promote dtype %s.",
            op::ToString(DataType(alpha->GetDataType())).GetString(), op::ToString(promoteType).GetString());
        return false;
    }
    // 检查推导后的数据类型能否转换为输出的数据类型
    if (!CanCast(promoteType, outDtype)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Promote dtype %s can't be cast to the desired output type %s.",
            op::ToString(promoteType).GetString(), op::ToString(outDtype).GetString());
        return false;
    }
    return true;
}

static bool IsSupportAxpy(const DataType promoteType)
{
    return CheckType(promoteType, AXPY_DTYPE_SUPPORT_LIST);
}

static bool CheckNotNull(const aclScalar* self, const aclTensor* other, const aclScalar* alpha, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    return true;
}

static DataType PromoteTypeScalar(const aclScalar* self, const aclTensor* other, const aclTensor* out)
{
    if (IsComplexType(other->GetDataType()) || IsComplexType(self->GetDataType())) {
        return op::PromoteType(other->GetDataType(), self->GetDataType());
    }
    if (IsFloatingType(other->GetDataType())) {
        return op::DataType::DT_FLOAT;
    }

    if (self->GetDataType() == op::DataType::DT_DOUBLE && out->GetDataType() == op::DataType::DT_FLOAT) {
        return op::DataType::DT_FLOAT;
    }

    if (IsFloatingType(self->GetDataType()) || other->GetDataType() == op::DataType::DT_BOOL) {
        return op::PromoteType(other->GetDataType(), self->GetDataType());
    }

    return other->GetDataType();
}

static aclnnStatus CheckParams(
    const aclScalar* self, const aclTensor* other, const aclScalar* alpha, const aclTensor* y)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, other, alpha, y), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    const std::initializer_list<op::DataType> dtypeSupportList = ADD_V3_DTYPE_SUPPORT_LIST;
    // 检查other的数据类型是否在add算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(other, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和other能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    auto promoteType = PromoteTypeScalar(self, other, y);
    CHECK_RET(
        CheckPromoteType(other->GetDataType(), self->GetDataType(), alpha, y->GetDataType(), promoteType),
        ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape
    CHECK_RET(CheckShape(other, y), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddV3GetWorkspaceSize(
    const aclScalar* self, const aclTensor* other, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAddV3, DFX_IN(self, other, alpha), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, other, alpha, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // add算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (other->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto promoteType = PromoteTypeScalar(self, other, out);

    // 固定写法，将输入other转换成连续的tensor
    auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
    CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入other的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto otherCasted = l0op::Cast(otherContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfTensor = uniqueExecutor.get()->ConvertToTensor(self, promoteType);

    // 进行Add计算
    const aclTensor* addOpOut = nullptr;
    if (!(alpha->ToFloat() > 1 || alpha->ToFloat() < 1)) {
        // alpha为1时不需要Mul
        addOpOut = l0op::Add(selfTensor, otherCasted, uniqueExecutor.get());
    } else if (IsSupportAxpy(promoteType)) {
        addOpOut = l0op::Axpy(selfTensor, otherCasted, alpha->ToFloat(), uniqueExecutor.get());
    } else {
        auto alphaTensor = uniqueExecutor.get()->ConvertToTensor(alpha, promoteType);
        auto otherRes = l0op::Mul(otherCasted, alphaTensor, uniqueExecutor.get());
        CHECK_RET(otherRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        addOpOut = l0op::Add(selfTensor, otherRes, uniqueExecutor.get());
    }
    CHECK_RET(addOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(addOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceAddV3GetWorkspaceSize(
    const aclScalar* selfRef, const aclTensor* other, const aclScalar* alpha, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    auto out = const_cast<aclTensor*>(other);
    return aclnnAddV3GetWorkspaceSize(selfRef, other, alpha, out, workspaceSize, executor);
}

aclnnStatus aclnnAddV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddV3);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceAddV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceAddV3);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif