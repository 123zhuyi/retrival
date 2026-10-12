/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_polar.cpp
 * \brief Polar L2 API：Contiguous + numpy 广播 (BroadcastTo) + L0 Polar + ViewCopy。
 *        kernel 假设同 shape elementwise；所有广播/非连续处理在本层完成 ——
 *        因此任意 shape / 任意广播组合（≤8 维）都能正确处理，不会因 outNumel 大或
 *        中间轴广播触发 kernel 内的 fp32 unravel 失精或 UB 容量越界。
 */

#include "aclnn_polar.h"
#include "polar.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "conversion/broadcast_to/op_api/broadcast_to.h" // l0op::BroadcastTo
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"
#include "op_api/op_api_def.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> INPUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {DataType::DT_COMPLEX64};

static bool CheckNotNull(const aclTensor* input, const aclTensor* angle, const aclTensor* out)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(angle, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* input, const aclTensor* angle, const aclTensor* out)
{
    if (!CheckType(input->GetDataType(), INPUT_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input dtype %s not in support list (expect FLOAT).",
                op::ToString(input->GetDataType()).GetString());
        return false;
    }
    if (input->GetDataType() != angle->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input dtype %s != angle dtype %s.",
                op::ToString(input->GetDataType()).GetString(), op::ToString(angle->GetDataType()).GetString());
        return false;
    }
    if (!CheckType(out->GetDataType(), OUTPUT_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out dtype %s not in support list (expect COMPLEX64).",
                op::ToString(out->GetDataType()).GetString());
        return false;
    }
    return true;
}

static bool CheckBroadcastShape(const aclTensor* input, const aclTensor* angle, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(input, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(angle, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);

    OP_CHECK_BROADCAST(input, angle, return false);
    Shape broadcastShape;
    BroadcastInferShape(input->GetViewShape(), angle->GetViewShape(), broadcastShape);

    if (broadcastShape != out->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "broadcast(input, angle) shape %s != out shape %s.",
                op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* input, const aclTensor* angle, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(input, angle, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(input, angle, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckBroadcastShape(input, angle, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclIntArray* GetShapeAsIntArray(const aclTensor* tensor, aclOpExecutor* executor)
{
    int64_t dimNum = static_cast<int64_t>(tensor->GetViewShape().GetDimNum());
    if (dimNum == 0) {
        int64_t shape[1] = {1};
        return executor->AllocIntArray(shape, 1);
    }
    std::vector<int64_t> shape(dimNum);
    for (int64_t i = 0; i < dimNum; i++) {
        shape[i] = tensor->GetViewShape()[i];
    }
    return executor->AllocIntArray(shape.data(), dimNum);
}

aclnnStatus aclnnPolarGetWorkspaceSize(const aclTensor* input, const aclTensor* angle, aclTensor* out,
                                       uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnPolar, DFX_IN(input, angle), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(input, angle, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (input->IsEmpty() || angle->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 1. make contiguous（解决 view/transpose 等非连续输入）
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto angleContiguous = l0op::Contiguous(angle, uniqueExecutor.get());
    CHECK_RET(angleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 2. broadcast 到 output shape。
    //    input 永远 BroadcastTo out（kernel 要求 input/out 同 shape）。
    //    angle：满足 inner-broadcast 资格时**保留原 shape [K]**（kernel bcastMode=1 周期复用，省一次 BroadcastTo
    //           的 HBM 写 + 每 tile angle 搬运 + Sin/Cos 重算）；否则 BroadcastTo 兜底。
    //    资格判定与 op_host/polar_tiling.cpp 严格一致：inN==outN, anN<inN, inN%anN==0, anN<=2048, anN%8==0。
    auto outShape = GetShapeAsIntArray(out, uniqueExecutor.get());
    CHECK_RET(outShape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (inputContiguous->GetViewShape() != out->GetViewShape()) {
        inputContiguous = l0op::BroadcastTo(inputContiguous, outShape, uniqueExecutor.get());
        CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    constexpr int64_t POLAR_TILE_LEN = 2048;
    auto numelOf = [](const op::Shape& s) -> int64_t {
        int64_t n = 1;
        for (size_t i = 0; i < s.GetDimNum(); ++i)
            n *= s[i];
        return n;
    };
    int64_t outN = numelOf(out->GetViewShape());
    int64_t inNn = numelOf(inputContiguous->GetViewShape());
    int64_t anNn = numelOf(angleContiguous->GetViewShape());
    bool innerBcast = (inNn == outN) && (anNn > 0) && (anNn < inNn) && (inNn % anNn == 0) && (anNn <= POLAR_TILE_LEN) &&
                      (anNn % 8 == 0);
    if (!innerBcast && angleContiguous->GetViewShape() != out->GetViewShape()) {
        angleContiguous = l0op::BroadcastTo(angleContiguous, outShape, uniqueExecutor.get());
        CHECK_RET(angleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 3. 调 Polar kernel（同 shape elementwise）
    auto opResult = l0op::Polar(inputContiguous, angleContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 4. 把 kernel 输出 copy 到调用者提供的 out
    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPolar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnPolar);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
