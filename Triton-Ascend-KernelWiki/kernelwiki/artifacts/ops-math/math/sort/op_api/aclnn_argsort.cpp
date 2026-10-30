/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_argsort.h"
#include <limits.h>
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "sort.h"
#include "aclnn_kernels/transpose.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_INT16,
    op::DataType::DT_INT32, op::DataType::DT_INT64,   op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8,  op::DataType::DT_INT16,
    op::DataType::DT_INT32, op::DataType::DT_INT64,   op::DataType::DT_UINT8, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> ARCH3510_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_BF16,  op::DataType::DT_UINT8,
    op::DataType::DT_INT8,    op::DataType::DT_INT16,  op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_UINT16,  op::DataType::DT_UINT32, op::DataType::DT_UINT64};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else if (IsRegBase()) {
        return ARCH3510_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    const auto& supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    if (op::IsPrivateFormat(self->GetViewFormat()) || op::IsPrivateFormat(out->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Format only support ND,NCL,NCHW,NHWC,HWCN,NDHWC,NCDHW,"
                "input format is [%s] and out format is [%s].",
                ToString(self->GetViewFormat()).GetString(), ToString(out->GetViewFormat()).GetString());
        return false;
    }
    return true;
}

// 将dim从负数转换为正数
static inline int64_t WrapDim(int64_t dimSize, int64_t dim) { return (dim < 0) ? dim + dimSize : dim; }

static bool CheckDimValid(const aclTensor* self, const int64_t dim)
{
    const int64_t dimNum = self->GetViewShape().GetDimNum();
    int64_t minimum = dimNum * (-1);
    int64_t maximum = dimNum - 1;
    if (dimNum == 0) {
        minimum = -1;
        maximum = 0;
    }
    if (dim < minimum || dim > maximum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim must be within the range of [%ld, %ld], but it is %ld.", minimum, maximum,
                dim);
        return false;
    }
    if (dimNum == 0) {
        return true;
    }
    int64_t dimValue = self->GetViewShape().GetDim(WrapDim(dimNum, dim));
    if (dimValue > INT_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the dim being sorted can not have more than INT_MAX elements");
        return false;
    }
    if (!(IsRegBase())) {
        if (1 == dimValue && op::DataType::DT_BF16 == self->GetDataType()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The sort axis value is not support 1 when input type is BF16.");
            return false;
        }
    }
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const int64_t dim, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查dim 是否合法
    CHECK_RET(CheckDimValid(self, dim), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查数据格式是否支持
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查shape是否支持
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 检查tuple <values, indices>里的元素是否为非null。true表示为非null，false表示为null
static bool CheckTupleNullptr(std::tuple<const aclTensor*, const aclTensor*> tensorTuple)
{
    // tensorTuple must be size 2
    if (std::tuple_size<decltype(tensorTuple)>::value != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The length of tuple returned by Sort is not 2.");
        return false;
    }
    return (std::get<0>(tensorTuple) != nullptr) && (std::get<1>(tensorTuple) != nullptr);
}

// 将dim与最后一个dim进行对换
static aclIntArray* GetPermResult(int64_t dim, int64_t dimSize, aclOpExecutor* executor)
{
    std::vector<int64_t> valuePerm(dimSize);
    for (int64_t i = 0; i < dimSize; i++) {
        valuePerm[i] = i;
    }

    std::swap(valuePerm[dim], valuePerm[dimSize - 1]);
    auto perm = executor->AllocIntArray(valuePerm.data(), dimSize);
    return perm;
}

// 如果需要perm， 计算perm  如果不需要，返回nullptr
static aclIntArray* GetPerm(int64_t dim, int64_t dimSize, aclOpExecutor* executor)
{
    // 需要transpose，计算perm。
    if (dim != dimSize - 1) {
        return GetPermResult(dim, dimSize, executor);
    }
    return nullptr;
}

aclnnStatus aclnnArgsortGetWorkspaceSize(const aclTensor* self, int64_t dim, bool descending, aclTensor* out,
                                         uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnArgsort, DFX_IN(self, dim, descending), DFX_OUT(out));

    auto ret = CheckParams(self, dim, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    int64_t dimSize = self->GetViewShape().GetDimNum();
    dimSize = (dimSize < 1) ? 1 : dimSize;

    auto dimOut = WrapDim(dimSize, dim);
    auto perm = GetPerm(dimOut, dimSize, uniqueExecutor.get());
    if (perm != nullptr) {
        selfContiguous = l0op::Transpose(selfContiguous, perm, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto sortOut = l0op::Sort(selfContiguous, -1, descending, false, op::DataType::DT_INT32, uniqueExecutor.get());
    CHECK_RET(CheckTupleNullptr(sortOut), ACLNN_ERR_INNER_NULLPTR);

    auto indicesOut = std::get<1>(sortOut);

    if (perm != nullptr) {
        auto indicesTranspose = l0op::Transpose(indicesOut, perm, uniqueExecutor.get());
        CHECK_RET(indicesTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);
        indicesOut = const_cast<aclTensor*>(indicesTranspose);
    }

    auto indicesCast = l0op::Cast(indicesOut, DataType::DT_INT64, uniqueExecutor.get());
    CHECK_RET(indicesCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(indicesCast, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnArgsort(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnArgsort);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
