/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_maxn.h"
#include "maximum.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT8, op::DataType::DT_INT32,
    op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT8,
    op::DataType::DT_INT32,   op::DataType::DT_INT64, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT8, op::DataType::DT_UINT8,
    op::DataType::DT_INT32,   op::DataType::DT_INT64, op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(curArch)) {
        return REGBASE_DTYPE_SUPPORT_LIST;
    }
    if (curArch == NpuArch::DAV_2201) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

// 检查入参是否为nullptr
static bool CheckNotNull(const aclTensorList* tensors, const aclTensor* out, const uint64_t* workspaceSize)
{
    OP_CHECK_NULL(tensors, return false);
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        OP_CHECK_NULL((*tensors)[i], return false);
    }
    OP_CHECK_NULL(out, return false);
    if (workspaceSize == nullptr) {
        return false;
    }
    return true;
}

// 检查输入和输出的数据类型是否在算子的支持列表内
static inline bool CheckDtypeValid(const aclTensorList* tensors, const aclTensor* out)
{
    const auto& DTYPE_SUPPORT_LIST = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        OP_CHECK_DTYPE_NOT_SUPPORT((*tensors)[i], DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_MATCH((*tensors)[i], out->GetDataType(), return false);
    }
    return true;
}

// 检查tensor维度
static inline bool CheckMaxDimension(const aclTensorList* tensors, const aclTensor* out)
{
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        OP_CHECK_MAX_DIM((*tensors)[i], MAX_SUPPORT_DIMS_NUMS, return false);
    }
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

// 检查shape是否满足broadcast
static inline bool CheckInAndOutShape(const aclTensorList* tensors, const aclTensor* out)
{
    if (tensors->Size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The input tensor list should not be empty.");
        return false;
    }

    op::Shape broadcastShape = (*tensors)[0]->GetViewShape();
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        if (!BroadcastInferShape((*tensors)[i]->GetViewShape(), broadcastShape, broadcastShape)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "the size of tensor %s must match the size of tensor %s.",
                op::ToString((*tensors)[i]->GetViewShape()).GetString(), op::ToString(broadcastShape).GetString());
            return false;
        }
    }

    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensorList* tensors, const aclTensor* out, const uint64_t* workspaceSize)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(tensors, out, workspaceSize), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据范围类型之内，需要根据API定义校验
    CHECK_RET(CheckDtypeValid(tensors, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查最大维度
    CHECK_RET(CheckMaxDimension(tensors, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查是否能broadcast，以及计算后的shape和out shape是否一致
    CHECK_RET(CheckInAndOutShape(tensors, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxNGetWorkspaceSize(
    const aclTensorList* tensors, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMaxN, DFX_IN(tensors), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(tensors, out, workspaceSize);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // maxn算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (out->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，输入转换成连续tensor
    auto firstContiguous = l0op::Contiguous((*tensors)[0], uniqueExecutor.get());
    CHECK_RET(firstContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* maxOut = firstContiguous;
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        auto secondContiguous = l0op::Contiguous((*tensors)[i], uniqueExecutor.get());
        CHECK_RET(secondContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        maxOut = l0op::Maximum(firstContiguous, secondContiguous, uniqueExecutor.get());
        CHECK_RET(maxOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        firstContiguous = maxOut;
    }

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(maxOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxN(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnMaxN);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
