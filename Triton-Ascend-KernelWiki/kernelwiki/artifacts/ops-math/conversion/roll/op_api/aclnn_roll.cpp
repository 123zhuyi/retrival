/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_roll.h"
#include "roll.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "conversion/unsqueeze/op_host/op_api/unsqueeze.h"
#include "conversion/squeeze/op_host/op_api/squeeze.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"
#include "op_api/op_api_def.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_UINT8,
    op::DataType::DT_INT32, op::DataType::DT_UINT32,  op::DataType::DT_BOOL, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT32,   op::DataType::DT_UINT32,
    op::DataType::DT_BOOL,  op::DataType::DT_INT64,   op::DataType::DT_BF16, op::DataType::DT_INT16};

/* *
 * l1: ASCEND910B、ASCEND910_93 或者 ASCEND950 芯片，该算子支持的数据类型列表
 * l2: 其他芯片，该算子支持的数据类型列表
 */
static const std::initializer_list<DataType>& GetDtypeSupportListV1(
    const std::initializer_list<op::DataType>& l1, const std::initializer_list<op::DataType>& l2)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        IsRegBase()) {
        return l1;
    } else {
        return l2;
    }
}

static inline bool CheckNotNull(
    const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims, const aclTensor* out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(shifts, return false);
    OP_CHECK_NULL(dims, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* x, const aclTensor* out)
{
    const auto& supportList =
        GetDtypeSupportListV1(ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_DTYPE_SUPPORT_LIST);
    OP_CHECK_DTYPE_NOT_SUPPORT(x, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
    OP_CHECK_DTYPE_NOT_MATCH(x, out->GetDataType(), return false);
    return true;
}

static inline bool CheckShape(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK_SHAPE_NOT_EQUAL(x, out, return false);
    return true;
}

static inline bool CheckArraySize(const aclIntArray* shifts, const aclIntArray* dims)
{
    if (shifts->Size() != dims->Size() && dims->Size() != 0U) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The size of shifts and dims should be the same when the size of dims is not 0.");
        return false;
    }

    if (dims->Size() == 0U && shifts->Size() != 1U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of shifts must be 1 if the size of dims is 0.");
        return false;
    }
    return true;
}

static bool CheckDimsRange(const aclTensor* x, const aclIntArray* dims)
{
    auto tensorDimSize = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    int64_t dimSize = static_cast<int64_t>(dims->Size());
    for (int64_t i = 0; i < dimSize; i++) {
        int64_t curDim = (*dims)[i];
        auto dimMax = std::max(-1 * tensorDimSize, tensorDimSize - 1);
        auto dimMin = std::min(-1 * tensorDimSize, tensorDimSize - 1);
        if ((curDim > dimMax) || (curDim < dimMin)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The values of dims should be in range [%ld, %ld].", dimMin, dimMax);
            return false;
        }
    }
    return true;
}

static inline bool CheckTensorDimSize(const aclTensor* x)
{
    OP_CHECK_MAX_DIM(x, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, shifts, dims, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(x, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查x和out的shape是否一致
    CHECK_RET(CheckShape(x, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shifts和dims的大小一致情况
    CHECK_RET(CheckArraySize(shifts, dims), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查x不能超过8维
    CHECK_RET(CheckTensorDimSize(x), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 将dim转换为 [0, dimSize-1]
static inline int64_t WrapDim(int64_t dim, uint32_t dimPostExpr)
{
    if (dimPostExpr <= 0)
        dimPostExpr = 1;
    if (dim < 0)
        dim += dimPostExpr;
    return dim;
}

// 新建 aclIntArray: 只有1个元素
static inline aclIntArray* GetIntArray(int64_t x, aclOpExecutor* executor)
{
    int64_t intArray[1] = {x};
    auto res = executor->AllocIntArray(intArray, 1);
    return res;
}

static const aclTensor* roll_transpose(const aclTensor* self, int64_t axis, int64_t shift, aclOpExecutor* executor)
{
    auto selfContiguous = l0op::Contiguous(self, executor);
    auto dimNow = GetIntArray(0, executor);
    auto shiftNow = GetIntArray(shift, executor);

    if (axis == 0) {
        selfContiguous = l0op::Roll(selfContiguous, shiftNow, dimNow, executor);
        return selfContiguous;
    }

    auto dimSize = static_cast<int64_t>(selfContiguous->GetViewShape().GetDimNum());
    std::vector<int64_t> perm(dimSize);
    for (int64_t i = 0; i < dimSize; i++) {
        perm[i] = i;
    }
    std::swap(perm[axis], perm[0]);
    auto valuePerm = executor->AllocIntArray(perm.data(), dimSize);

    selfContiguous = l0op::Transpose(selfContiguous, valuePerm, executor);
    CHECK_RET(selfContiguous != nullptr, nullptr);
    selfContiguous = l0op::Roll(selfContiguous, shiftNow, dimNow, executor);
    CHECK_RET(selfContiguous != nullptr, nullptr);
    selfContiguous = l0op::Transpose(selfContiguous, valuePerm, executor);
    return selfContiguous;
}

// 处理0维tensor场景：dims必须为size 0, shifts必须为size 1。
static aclnnStatus HandleDimZeroTensor(
    const aclTensor* self, const aclIntArray* shifts, const aclIntArray* dims, const aclTensor* out,
    aclOpExecutor* executor)
{
    if (dims->Size() != 0 || shifts->Size() != 1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "When tensor x has no dimensions, shifts should be size 1, dims should be size 0.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto viewCopyRes = l0op::ViewCopy(self, out, executor);
    CHECK_RET(viewCopyRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRollGetWorkspaceSize(
    const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnRoll, DFX_IN(x, shifts, dims), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // dims维度检查放到后面，因为empty和0维tensor需要特殊处理
    auto ret = CheckParams(x, shifts, dims, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // roll支持0维tensor(a), shifts只能有1个元素, dims不能有元素
    if (x->GetViewShape().GetDimNum() == 0) {
        auto res = HandleDimZeroTensor(x, shifts, dims, out, uniqueExecutor.get());
        CHECK_RET(res == ACLNN_SUCCESS, res);

        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // roll逻辑特殊，其余参数场景先校验完，再进行校验。空tensor场景可以无视dim的异常值正常跑。0维tensor场景不允许有dims。
    // 6. 检查dims的大小必须处于 [-N, N-1] 之间。
    CHECK_RET(CheckDimsRange(x, dims), ACLNN_ERR_PARAM_INVALID);

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果是bool, cast成Int8 进行计算。
    bool needCastBool = (x->GetDataType() == DataType::DT_BOOL);
    if (needCastBool) {
        xContiguous = l0op::Cast(xContiguous, DataType::DT_INT8, uniqueExecutor.get());
        CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    int64_t dimSize = static_cast<int64_t>(dims->Size());
    auto tensorDim = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    auto newDims = dims;
    auto newshifts = shifts;

    bool hasDim1 = false;
    int64_t shiftNum = 0;
    for (int64_t i = 0; i < dimSize; i++) {
        if (WrapDim((*dims)[i], tensorDim) == 1) {
            hasDim1 = true;
            shiftNum += (*shifts)[i];
        }
    }
    bool needSqueeze = (x->GetViewShape().GetDimNum() == 2 && x->GetViewShape().GetDim(0) == 1 && hasDim1);
    const int64_t appendDim[] = {0};
    aclIntArray* dimArray = (uniqueExecutor.get())->AllocIntArray(appendDim, 1);
    if (needSqueeze) {
        xContiguous = l0op::SqueezeNd(xContiguous, dimArray, uniqueExecutor.get());
        CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        newDims = GetIntArray(0, uniqueExecutor.get());
        newshifts = GetIntArray(shiftNum, uniqueExecutor.get());
        dimSize = 1;
    }

    // 因为可能有cast的过程，原来的out大小不一致，所以先存到outBase里
    const aclTensor* outBase = nullptr;
    if (IsRegBase() || dimSize == 0) {
        outBase = l0op::Roll(xContiguous, newshifts, newDims, uniqueExecutor.get());
        CHECK_RET(outBase != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        outBase = xContiguous;
        for (int64_t i = 0; i < dimSize; i++) {
            int64_t axis = WrapDim((*newDims)[i], tensorDim);
            int64_t shift = (*newshifts)[i];
            outBase = roll_transpose(outBase, axis, shift, uniqueExecutor.get());
            CHECK_RET(outBase != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    if (needSqueeze) {
        outBase = l0op::UnsqueezeNd(outBase, dimArray, uniqueExecutor.get());
        CHECK_RET(outBase != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 将结果转换为原来的dtype
    if (needCastBool) {
        outBase = l0op::Cast(outBase, DataType::DT_BOOL, uniqueExecutor.get());
        CHECK_RET(outBase != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto viewcopyResult = l0op::ViewCopy(outBase, out, uniqueExecutor.get());
    CHECK_RET(viewcopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRoll(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnRoll);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif