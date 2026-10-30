/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_repeat.h"

#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

#include "conversion/broadcast_to/op_api/broadcast_to.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "tile.h"
#include "conversion/unsqueeze/op_host/op_api/unsqueeze.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128, op::DataType::DT_UINT8,   op::DataType::DT_INT8,   op::DataType::DT_INT16,
    op::DataType::DT_INT32,      op::DataType::DT_INT64,   op::DataType::DT_BOOL,   op::DataType::DT_BF16,
    op::DataType::DT_UINT32,     op::DataType::DT_UINT64};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910B310P_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128, op::DataType::DT_UINT8,   op::DataType::DT_INT8,   op::DataType::DT_INT16,
    op::DataType::DT_INT32,      op::DataType::DT_INT64,   op::DataType::DT_BOOL,   op::DataType::DT_BF16,
    op::DataType::DT_UINT32,     op::DataType::DT_UINT64,  op::DataType::DT_UINT16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_950_LIST = {
    op::DataType::DT_FLOAT,       op::DataType::DT_FLOAT16,      op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128,  op::DataType::DT_UINT8,        op::DataType::DT_INT8,   op::DataType::DT_INT16,
    op::DataType::DT_INT32,       op::DataType::DT_INT64,        op::DataType::DT_BOOL,   op::DataType::DT_BF16,
    op::DataType::DT_UINT32,      op::DataType::DT_UINT64,       op::DataType::DT_UINT16, op::DataType::DT_HIFLOAT8,
    op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN};

// 检查tensor是否为nullptr
static inline bool CheckNotNull(const aclTensor* self, const aclIntArray* repeats, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(repeats, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_910B310P_LIST, return false);
    } else if (IsRegBase()) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_950_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    }

    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);
    return true;
}

// tensor维度数不能超过8维
static inline bool CheckTensorDimSize(const aclTensor* self, const aclIntArray* repeats)
{
    int64_t tensorDimSize = self->GetViewShape().GetDimNum();
    int64_t repeatsSize = repeats->Size();
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    if (repeatsSize > static_cast<int64_t>(MAX_SUPPORT_DIMS_NUMS)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "repeats size should not be larger than 8. self %ld, repeats %ld", tensorDimSize,
            repeatsSize);
        return false;
    }
    return true;
}

static inline bool CheckRepeatsSize(const aclTensor* self, const aclIntArray* repeats)
{
    if (self->GetViewShape().GetDimNum() > repeats->Size()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Number of dimensions of repeat dims can not be smaller than number of dimensions of input tensor, self "
            "%lu, repeats %lu.",
            self->GetViewShape().GetDimNum(), repeats->Size());
        return false;
    }
    return true;
}

// 校验repeat中的值均大于等于0
static inline aclnnStatus CheckRepeatsValue(const aclIntArray* repeats)
{
    for (size_t i = 0; i < repeats->Size(); ++i) {
        if ((*repeats)[i] < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "repeats expected %luth value > 0, but get %ld", i + 1, (*repeats)[i]);
            return false;
        }
    }
    return true;
}

// 校验repeat中的值存在0
static inline aclnnStatus CheckRepeatsZero(const aclIntArray* repeats)
{
    for (size_t i = 0; i < repeats->Size(); ++i) {
        if ((*repeats)[i] == 0) {
            return true;
        }
    }
    return false;
}

// 校验repeat算子计算后生成的tensor与out的shape一致
static inline bool CheckRepeatOutShape(const aclTensor* repeatRes, const aclTensor* out)
{
    OP_CHECK_SHAPE_NOT_EQUAL(repeatRes, out, return false);

    return true;
}

// 返回为[1]的intarray shape
static inline aclIntArray* GetBaseShape(aclOpExecutor* executor)
{
    int64_t tensorShape[1] = {};
    tensorShape[0] = 1;
    auto res = executor->AllocIntArray(tensorShape, 1);
    return res;
}

// 如果self为0维tensor，那么转换为1维tensor。其余情况转成连续tensor
static inline const aclTensor* InitializeTensor(const aclTensor* x, aclOpExecutor* executor)
{
    auto xContiguous = l0op::Contiguous(x, executor);
    // 如果tensor为0维，则转换为1维tensor
    if (xContiguous->GetViewShape().GetDimNum() == 0) {
        auto baseShape = GetBaseShape(executor);
        xContiguous = l0op::BroadcastTo(xContiguous, baseShape, executor);
    }
    return xContiguous;
}

static const aclTensor* ViewToRepeatsSize(const aclTensor* self, const aclIntArray* repeats, aclOpExecutor* executor)
{
    auto Dims = (int64_t)repeats->Size() - (int64_t)self->GetViewShape().GetDimNum();
    std::vector<int64_t> unsqueezeDim(Dims);
    for (int64_t idx = 0; idx < Dims; idx++) {
        unsqueezeDim[idx] = idx;
    }
    aclIntArray* dim = executor->AllocIntArray(unsqueezeDim.data(), Dims);
    auto unsqeezeSelf = l0op::UnsqueezeNd(self, dim, executor);
    return unsqeezeSelf;
}
// 校验开始
static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* repeats, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(self, repeats, out), ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckRepeatsSize(self, repeats), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckRepeatsValue(repeats), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckTensorDimSize(self, repeats), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRepeatGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* repeats, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRepeat, DFX_IN(self, repeats), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, repeats, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || out->IsEmpty() || CheckRepeatsZero(repeats)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (self->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGW("Format only support ND");
    }
    if (repeats->Size() == 0) {
        auto viewCopyOut = l0op::ViewCopy(self, out, uniqueExecutor.get());
        CHECK_RET(viewCopyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = InitializeTensor(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfUnsqueeze = selfContiguous;

    if (selfContiguous->GetViewShape().GetDimNum() < repeats->Size()) {
        selfUnsqueeze = ViewToRepeatsSize(selfContiguous, repeats, uniqueExecutor.get());
        CHECK_RET(selfUnsqueeze != nullptr, ACLNN_ERR_PARAM_INVALID);
    }

    auto repeatOut = l0op::Tile(selfUnsqueeze, repeats, uniqueExecutor.get());
    CHECK_RET(repeatOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckRepeatOutShape(repeatOut, out), ACLNN_ERR_PARAM_INVALID);

    auto viewCopyOut = l0op::ViewCopy(repeatOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRepeat(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRepeat);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
