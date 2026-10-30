/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_channel_shuffle.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const size_t FIRST_DIM = 0;
static const size_t SECOND_DIM = 1;
static const size_t THIRD_DIM = 2;
static const int64_t MAX_SUPPORT_DIM = 7;
static const int64_t MIN_SUPPORT_DIM = 3;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,     op::DataType::DT_DOUBLE,     op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_INT16,     op::DataType::DT_INT32,      op::DataType::DT_INT64,
    op::DataType::DT_BOOL,    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 检查self、out的数据类型是否一致
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

    return true;
}

static bool CheckAttrValid(const aclTensor* self, const int64_t groups)
{
    // 检查groups值是否小于等于0
    if (groups <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Number of groups to divide channels in must be positive. Value of groups: %ld.", groups);
        return false;
    }

    // 检查groups值是否能被channels整除
    const auto& selfShape = self->GetViewShape();
    auto channels = selfShape.GetDim(SECOND_DIM);
    if (channels % groups != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Number of channels must be divisible by groups.Got %ld channels and %ld groups.", channels, groups);
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIM, return false);
    OP_CHECK_MIN_DIM(self, MIN_SUPPORT_DIM, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, int64_t groups, aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入、输出的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查属性参数是否在支持范围内
    CHECK_RET(CheckAttrValid(self, groups), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入、输出的shape
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessChannelShuffle(const aclTensor* self, const int64_t groups, aclTensor* out,
                                         aclOpExecutor* executor)
{
    // 计算reshape的size
    const auto selfShape = self->GetViewShape();
    int64_t channels = selfShape.GetDim(SECOND_DIM);
    int64_t channelsPerGroup = static_cast<int64_t>(channels / groups);
    size_t tensorSize = selfShape.GetDimNum() + 1;
    std::vector<int64_t> tensorShape(tensorSize);
    for (size_t i = 0; i < tensorSize; i++) {
        if (i == FIRST_DIM) {
            tensorShape[i] = selfShape.GetDim(i);
        } else if (i == SECOND_DIM) {
            tensorShape[i] = groups;
        } else if (i == THIRD_DIM) {
            tensorShape[i] = channelsPerGroup;
        } else {
            tensorShape[i] = selfShape.GetDim(i - 1);
        }
    }
    auto reshapeSize = executor->AllocIntArray(tensorShape.data(), tensorSize);

    // 将输入input转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将self从[*, channels, *...] reshape成[*, groups, channelsPerGroup, *...]
    auto inputReshape = l0op::Reshape(selfContiguous, reshapeSize, executor);
    CHECK_RET(inputReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将reshape后的张量的第二维和第三维进行转置
    size_t newDimNum = inputReshape->GetViewShape().GetDimNum();
    std::vector<int64_t> transPoseArray(newDimNum);
    for (size_t i = 0; i < newDimNum; i++) {
        transPoseArray[i] = i;
    }
    std::swap(transPoseArray[SECOND_DIM], transPoseArray[THIRD_DIM]);
    auto perm = executor->AllocIntArray(transPoseArray.data(), newDimNum);
    auto inputTranspose = l0op::Transpose(inputReshape, perm, executor);
    CHECK_RET(inputTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将transpose后的张量reshape成self的shape
    auto output = l0op::Reshape(inputTranspose, selfShape, executor);
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(output, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChannelShuffleGetWorkspaceSize(const aclTensor* self, int64_t groups, aclTensor* out,
                                                uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnChannelShuffle, DFX_IN(self, groups), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, groups, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // gridsampler2d算子的空tensor在kernel中支持
    if (self->IsEmpty() || out->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 进行计算
    ret = ProcessChannelShuffle(self, groups, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChannelShuffle(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChannelShuffle);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
