/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_add_n.h"
#include "add_n.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/contiguous.h"
#include "conversion/broadcast_to/op_api/broadcast_to.h"
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
#include "op_api/aclnn_check.h"
#include "acl/acl_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

#define MAX_SUPPORT_DIMS_NUMS 8

// 根据API定义，列出所有能支持的dtype
static const std::initializer_list<op::DataType> ADD_N_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
    op::DataType::DT_FLOAT};

// 检查入参是否为nullptr
static bool CheckNotNull(const aclTensorList* tensors, const aclTensor* out)
{
    OP_CHECK_NULL(tensors, return false);
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        if ((*tensors)[i] == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Expected a proper Tensor but got null for tensor %lu.", i);
            return false;
        }
    }
    OP_CHECK_NULL(out, return false);
    return true;
}

// 检查参数dtype是否支持
static bool CheckDtypeValid(const aclTensorList* tensors, const aclTensor* out)
{
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        if (!CheckType((*tensors)[i]->GetDataType(), ADD_N_DTYPE_SUPPORT_LIST)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor %lu not implemented for %s, should be in dtype support list [%s].",
                    i, op::ToString((*tensors)[i]->GetDataType()).GetString(),
                    op::ToString(ADD_N_DTYPE_SUPPORT_LIST).GetString());
            return false;
        }
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, ADD_N_DTYPE_SUPPORT_LIST, return false);
    return true;
}

// 检查当前NPU架构是否支持
static bool CheckArch()
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch != NpuArch::DAV_2201 && !IsRegBase(npuArch)) {
        return false;
    }
    return true;
}

// 进行Shape检查
static bool CheckShape(const aclTensorList* tensors, const aclTensor* out)
{
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        auto dimNum = (*tensors)[i]->GetViewShape().GetDimNum();
        if (dimNum > MAX_SUPPORT_DIMS_NUMS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of tensor %lu is %zu, can't be greater than %zu.", i, dimNum,
                    MAX_SUPPORT_DIMS_NUMS);
            return false;
        }
    }
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);

    auto shape = (*tensors)[0]->GetViewShape();
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        if ((*tensors)[i]->GetViewShape() != shape) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensors should have same shape.");
            return false;
        }
    }

    if (shape != out->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensors should have same shape with output.");
        return false;
    }

    return true;
}

// 参数检查
static aclnnStatus CheckParams(const aclTensorList* tensors, const aclTensor* out)
{
    // 固定写法，检查参数是否为空指针
    CHECK_RET(CheckNotNull(tensors, out), ACLNN_ERR_PARAM_NULLPTR);

    // 检查数据类型是否一致
    OP_CHECK_DTYPE_NOT_SAME((*tensors)[0], out, return false);
    if (!CheckArch()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ACLNN only support ASCEND910B(A2), ASCEND910_93(A3) and ASCEND950(A5) series");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 检查输入的数据类型是否为支持的数据类型
    CHECK_RET(CheckDtypeValid(tensors, out), ACLNN_ERR_PARAM_INVALID);

    // 检查输出的shape
    CHECK_RET(CheckShape(tensors, out), ACLNN_ERR_PARAM_INVALID);

    // 检查数据格式是否为ND
    if (IsPrivateFormat((*tensors)[0]->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddNGetWorkspaceSize(const aclTensorList* tensors, aclTensor* out, uint64_t* workspaceSize,
                                      aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAddN, DFX_IN(tensors), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，检查参数
    auto ret = CheckParams(tensors, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensorList处理
    if (tensors->Size() == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 空tensor处理
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        if ((*tensors)[i]->IsEmpty()) {
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }

    aclTensor* addnOut = nullptr;
    if (tensors->Size() == 1) {
        addnOut = const_cast<aclTensor*>((*tensors)[0]);
    } else {
        op::FVector<const aclTensor*> tensorList;
        for (uint64_t i = 0; i < tensors->Size(); i++) {
            auto tensorsContiguous = l0op::Contiguous((*tensors)[i], uniqueExecutor.get());
            CHECK_RET(tensorsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
            tensorList.push_back(tensorsContiguous);
        }

        // 调用AddN算子计算
        const aclTensorList* inputList = uniqueExecutor.get()->AllocTensorList(tensorList.data(), tensorList.size());
        addnOut = const_cast<aclTensor*>(l0op::AddN(inputList, uniqueExecutor.get()));
    }

    CHECK_RET(addnOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上
    auto viewCopyResult = l0op::ViewCopy(addnOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddN(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddN);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
