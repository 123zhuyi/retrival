/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_global_max_pool.h"
#include "reduce_max.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_errno.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const uint64_t GLOBAL_MAX_POOL_MIN_DIMS_NUMS = 4;
static const uint64_t GLOBAL_MAX_POOL_MAX_DIMS_NUMS = 8;
static const uint64_t DIM_NUMBER_TWO = 2;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::Format> FORMAT_SUPPORT_LIST = {
    op::Format::FORMAT_ND, op::Format::FORMAT_NCHW, op::Format::FORMAT_NCDHW};

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);

    // 检查self的数据类型是否和out一致
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

    return true;
}

static bool CheckFormatValid(const aclTensor* self, const aclTensor* out)
{
    op::Format selfFormat = self->GetStorageFormat();
    op::Format outFormat = out->GetStorageFormat();
    if (selfFormat != outFormat) {
        return false;
    }
    auto findSelfFormat = std::find(FORMAT_SUPPORT_LIST.begin(), FORMAT_SUPPORT_LIST.end(), selfFormat);
    auto findOutFormat = std::find(FORMAT_SUPPORT_LIST.begin(), FORMAT_SUPPORT_LIST.end(), outFormat);
    bool formatValid = findSelfFormat != FORMAT_SUPPORT_LIST.end() && findOutFormat != FORMAT_SUPPORT_LIST.end();
    if (!formatValid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "format invalid.\n");
    }
    return formatValid;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MIN_DIM(self, GLOBAL_MAX_POOL_MIN_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(self, GLOBAL_MAX_POOL_MAX_DIMS_NUMS, return false);
    OP_CHECK_WRONG_DIMENSION(out, self->GetViewShape().GetDimNum(), return false);

    auto selfShape = self->GetViewShape();
    op::Shape outShape;
    if (selfShape.GetDimNum() < DIM_NUMBER_TWO) {
        return false;
    }
    outShape.AppendDim(selfShape.GetDim(0));
    outShape.AppendDim(selfShape.GetDim(1));
    for (size_t i = 2; i < selfShape.GetDimNum(); i++) {
        outShape.AppendDim(1);
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, outShape, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查self、out的数据类型是否合法
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self、out的数据格式是否合法
    CHECK_RET(CheckFormatValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 查输入输出tensor的shape是否为异常
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGlobalMaxPoolGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGlobalMaxPool, DFX_IN(self), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 生成dim
    std::vector<int64_t> dimVector = {};
    int64_t dimNum = selfContiguous->GetViewShape().GetDimNum();
    for (int64_t i = 2; i < dimNum; i++) {
        dimVector.push_back(i);
    }
    const aclIntArray* dim = aclCreateIntArray(dimVector.data(), dimNum - 2);

    // 调用max算子kernel
    auto maxOpOut = l0op::ReduceMax(selfContiguous, dim, true, true, uniqueExecutor.get());
    CHECK_RET(maxOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(maxOpOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGlobalMaxPool(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGlobalMaxPool);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
