/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_global_average_pool.h"
#include "reduce_mean.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_errno.h"
#include "op_api/aclnn_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const uint64_t GLOBAL_AVERAGE_POOL_MIN_DIMS_NUMS = 4;
static const uint64_t GLOBAL_AVERAGE_POOL_MAX_DIMS_NUMS = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> NON_CONTIOUS_SUPPORT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

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
    const auto& supportList = GetDtypeSupportList();
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

    // 检查self的数据类型是否和out一致
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

    return true;
}

static bool CheckFormatValid(const aclTensor* self, const aclTensor* out)
{
    op::Format selfFormat = self->GetStorageFormat();
    op::Format outFormat = out->GetStorageFormat();
    if (selfFormat != outFormat) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The format of self and out must be the same.");
        return false;
    }
    auto findSelfFormat = std::find(FORMAT_SUPPORT_LIST.begin(), FORMAT_SUPPORT_LIST.end(), selfFormat);
    auto findOutFormat = std::find(FORMAT_SUPPORT_LIST.begin(), FORMAT_SUPPORT_LIST.end(), outFormat);
    if (findSelfFormat != FORMAT_SUPPORT_LIST.end() && findOutFormat != FORMAT_SUPPORT_LIST.end()) {
        return true;
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW and NCDHW.");
        return false;
    }
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MIN_DIM(self, GLOBAL_AVERAGE_POOL_MIN_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(self, GLOBAL_AVERAGE_POOL_MAX_DIMS_NUMS, return false);
    OP_CHECK_WRONG_DIMENSION(out, self->GetViewShape().GetDimNum(), return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查self、out的数据类型是否合法
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self、out的数据格式是否合法
    if (!CheckFormatValid(self, out)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 4. 查输入输出tensor的shape是否为异常
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool IsNonContiguousSupport(const aclTensor* self, const aclIntArray* dims)
{
    if (!IsRegBase()) {
        return false;
    }
    if (!CheckType(self->GetDataType(), NON_CONTIOUS_SUPPORT_DTYPE_SUPPORT_LIST)) {
        return false;
    }
    if (!op::IsReduceNonContiguousSupport(self, dims)) {
        return false;
    }
    return true;
}

aclnnStatus aclnnGlobalAveragePoolGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGlobalAveragePool, DFX_IN(self), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 生成dims
    std::vector<int64_t> dimVector = {};
    int64_t dimNum = self->GetViewShape().GetDimNum();
    for (int64_t i = 2; i < dimNum; i++) {
        dimVector.push_back(i);
    }
    const aclIntArray* dims = aclCreateIntArray(dimVector.data(), dimNum - 2);

    if (IsNonContiguousSupport(self, dims)) {
        OP_LOGD("Enter NonContigous");
        auto selfContiguous = uniqueExecutor.get()->CreateView(
            self, self->GetViewShape(), self->GetStorageShape(), self->GetViewStrides(), self->GetViewOffset());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto meanOpOut = l0op::ReduceMean(selfContiguous, dims, true, uniqueExecutor.get());
        CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(CheckShapeAndScalarSame(meanOpOut, out), ACLNN_ERR_PARAM_INVALID);

        auto viewCopyResult = l0op::ViewCopy(meanOpOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        OP_LOGD("Enter Contigous");
        // 固定写法，将输入self转换成连续的tensor
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 调用mean算子kernel
        auto meanOpOut = l0op::ReduceMean(selfContiguous, dims, true, uniqueExecutor.get());
        CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(CheckShapeAndScalarSame(meanOpOut, out), ACLNN_ERR_PARAM_INVALID);

        // 将计算结果拷贝到输出out上，out可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(meanOpOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGlobalAveragePool(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGlobalAveragePool);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
