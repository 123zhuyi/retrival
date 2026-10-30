/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_permute.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,       op::DataType::DT_FLOAT16,      op::DataType::DT_INT8,  op::DataType::DT_INT16,
    op::DataType::DT_INT32,       op::DataType::DT_INT64,        op::DataType::DT_UINT8, op::DataType::DT_UINT16,
    op::DataType::DT_UINT32,      op::DataType::DT_UINT64,       op::DataType::DT_BOOL,  op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64,   op::DataType::DT_COMPLEX128,   op::DataType::DT_BF16,  op::DataType::DT_HIFLOAT8,
    op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN};

static inline bool CheckNotNull(const aclTensor* self, const aclIntArray* dims, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(dims, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

static inline bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

inline static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);

    bool bf16flag = CheckSocVersionIsSupportBf16();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (!bf16flag && self->GetDataType() == op::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s is unsupported by the current SOC version [%s].",
                op::ToString(self->GetDataType()).GetString(), op::ToString(socVersion).GetString());
        return false;
    }

    // 检查输入输出的数据类型是否一致
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

    return true;
}

static bool CheckDimsRange(const aclTensor* self, const aclIntArray* dims)
{
    auto tensorDimSize = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    std::vector<bool> checkPerm = {false, false, false, false, false, false, false, false};
    size_t dimSize = dims->Size();
    int64_t curPerm = 0;
    for (size_t i = 0; i < dimSize; i++) {
        int64_t curDim = (*dims)[i];
        auto dimMax = std::max(-1 * tensorDimSize, tensorDimSize - 1);
        auto dimMin = std::min(-1 * tensorDimSize, tensorDimSize - 1);
        if ((curDim > dimMax) || (curDim < dimMin)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The values of dims should be in range [%ld, %ld].", dimMin, dimMax);
            return false;
        }
        curPerm = (*dims)[i] < 0 ? (*dims)[i] + dimSize : (*dims)[i];
        if (checkPerm[curPerm] == false) {
            checkPerm[curPerm] = true;
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Each dimension in perm should not be repeated.");
            return false;
        }
    }
    return true;
}

static bool CheckShapeValid(const aclTensor* self, const aclIntArray* dims, const aclTensor* out)
{
    auto dimSelf = self->GetViewShape().GetDimNum();
    auto dimOut = out->GetViewShape().GetDimNum();
    // 输入输出维度数不同
    if (dimSelf != dimOut) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Permute not support self shape: %s, output shape: %s",
                op::ToString(self->GetViewShape()).GetString(), op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    // dim size 超出 inuput output 维度
    if (dims->Size() != dimSelf) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The dimension of the dims:%ld and the dimension of the input shape:%ld are inconsistent",
                static_cast<int64_t>(dims->Size()), static_cast<int64_t>(dimSelf));
        return false;
    }

    // dims 中包含重复元素
    auto permSize = static_cast<int64_t>(dims->Size());
    std::vector<int64_t> perm(permSize);
    for (int64_t i = 0; i < permSize; i++) {
        auto dimvalue = (*dims)[i];
        perm[i] = dimvalue;
    }

    std::unordered_set<int64_t> uniqueElements;
    for (int64_t num : perm) {
        if (uniqueElements.count(num) != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid dims and it contains duplicate elements");
            return false;
        }
        uniqueElements.insert(num);
    }

    // 超出最大维度
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

    return true;
}

inline static aclnnStatus CheckParam(const aclTensor* self, const aclIntArray* dims, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, dims, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查Shape是否支持
    CHECK_RET(CheckShapeValid(self, dims, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查dims的大小必须处于 [-N, N-1] 之间。
    CHECK_RET(CheckDimsRange(self, dims), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

inline static FVector<int64_t> GetShape(const aclTensor* tensor)
{
    FVector<int64_t> shape;
    if (tensor->GetViewShape().GetDimNum() == 0) {
        shape.emplace_back(1);
    } else {
        size_t dimNum = tensor->GetViewShape().GetDimNum();
        for (size_t idx = 0; idx < dimNum; idx++) {
            auto tmpVal = tensor->GetViewShape().GetDim(idx);
            shape.emplace_back(tmpVal);
        }
    }
    return shape;
}

static const aclTensor* ProcessEmptyTensor(const aclTensor* self, const aclTensor* out, aclOpExecutor* executor)
{
    // 获取shape信息
    op::Shape outShape = out->GetViewShape();
    auto output = executor->AllocTensor(outShape, self->GetDataType());
    if (output->IsEmpty()) {
        OP_LOGI("Returning an empty tensor without actually doing calculation");
        return output;
    } else {
        OP_LOGI("Output should be empty when self is empty");
        return nullptr;
    }
}

static const aclTensor* BuildPermuteGraph(const aclTensor* self, const aclIntArray* dims, const aclTensor* out,
                                          aclOpExecutor* executor)
{
    // 空tensor处理
    if (self->IsEmpty()) {
        auto emptyOut = ProcessEmptyTensor(self, out, executor);
        CHECK_RET(emptyOut != nullptr, nullptr);
        return emptyOut;
    }

    // 转连续
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, nullptr);

    // dims值转换正值
    auto permSize = (int64_t)(dims->Size());
    std::vector<int64_t> perm(permSize);
    for (int64_t i = 0; i < permSize; i++) {
        auto dimvalue = (*dims)[i];
        perm[i] = dimvalue < 0 ? (permSize + dimvalue) : dimvalue;
    }
    aclIntArray* valuePerm = executor->AllocIntArray(perm.data(), permSize);

    // [permute] 完成矩阵的维度交换
    auto selfPermute = l0op::Transpose(selfContiguous, valuePerm, executor);
    CHECK_RET(selfPermute != nullptr, nullptr);

    return selfPermute;
}

aclnnStatus aclnnPermuteGetWorkspaceSize(const aclTensor* self, const aclIntArray* dims, aclTensor* out,
                                         uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnPermute, DFX_IN(self, dims), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto unique_executor = CREATE_EXECUTOR();
    CHECK_RET(unique_executor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 入参检查
    auto checkRet = CheckParam(self, dims, out);
    if (checkRet != ACLNN_SUCCESS) {
        return checkRet;
    }

    // 构建permute计算图
    auto permuteOut = BuildPermuteGraph(self, dims, out, unique_executor.get());
    CHECK_RET(permuteOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckShapeAndScalarSame(permuteOut, out), ACLNN_ERR_PARAM_INVALID);
    if (permuteOut->IsEmpty()) {
        // 当输出为空tensor的场景，空tensor处理
        *workspaceSize = 0;
        unique_executor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto viewCopyResult = l0op::ViewCopy(permuteOut, out, unique_executor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取workspace
    *workspaceSize = unique_executor->GetWorkspaceSize();
    unique_executor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPermute(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnPermute);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
