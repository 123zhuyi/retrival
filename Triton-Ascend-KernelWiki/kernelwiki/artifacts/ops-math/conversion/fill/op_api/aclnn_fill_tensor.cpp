/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "aclnn_fill_tensor.h"
#include "fill.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_910 = {
    DataType::DT_FLOAT, DataType::DT_INT32,     DataType::DT_INT64,     DataType::DT_FLOAT16,
    DataType::DT_INT16, DataType::DT_INT8,      DataType::DT_UINT8,     DataType::DT_DOUBLE,
    DataType::DT_BOOL,  DataType::DT_COMPLEX64, DataType::DT_COMPLEX128};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_910B = {
    DataType::DT_FLOAT, DataType::DT_INT32,     DataType::DT_INT64,      DataType::DT_FLOAT16,
    DataType::DT_INT16, DataType::DT_INT8,      DataType::DT_UINT8,      DataType::DT_DOUBLE,
    DataType::DT_BOOL,  DataType::DT_COMPLEX64, DataType::DT_COMPLEX128, DataType::DT_BF16};

static inline bool CheckNotNull(const aclTensor* self, const aclTensor* value)
{
    // self、value不能为空指针
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(value, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* self)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    bool is910bSocVersion = (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch));
    const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_CURRENT =
        is910bSocVersion ? DTYPE_SUPPORT_LIST_910B : DTYPE_SUPPORT_LIST_910;

    // 检查self的数据类型是否在fill_tensor算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_CURRENT, return false);
    return true;
}

static inline bool CheckShape(const aclTensor* self, const aclTensor* value)
{
    // self的数据维度不能超过8
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);

    // value的数据维度只能是0D或者size=1的1D
    auto& valueViewShape = value->GetViewShape();
    if (valueViewShape.GetDimNum() > 1 || (valueViewShape.GetDimNum() == 1 && valueViewShape.GetDim(0) != 1)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "value shape should be 0D or 1D with size = 1, but got %s.",
            ToString(value->GetViewShape()).GetString());
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* value)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, value), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和out的shape是否一致
    CHECK_RET(CheckShape(self, value), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static inline FVector<int64_t> getShape(const aclTensor* self)
{
    FVector<int64_t> shape;
    size_t dimNum = self->GetViewShape().GetDimNum();
    if (dimNum == 0) {
        shape.push_back(1);
    } else {
        for (size_t idx = 0; idx < dimNum; idx++) {
            int64_t tmpVal = self->GetViewShape().GetDim(idx);
            shape.push_back(tmpVal);
        }
    }
    return shape;
}

aclnnStatus aclnnInplaceFillTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* value, size_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceFillTensor, DFX_IN(selfRef, value), DFX_OUT());

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executorP = uniqueExecutor.get();
    CHECK_RET(executorP != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(selfRef, value);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果selfRef是空tensor，则直接返回
    if (selfRef->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 根据self构造fillShape, 进而生成dims和shapeArray，作为l0op::Fill算子的入参
    FVector<int64_t> fillShape = getShape(selfRef);
    const aclTensor* dims = executorP->ConvertToTensor(fillShape.data(), fillShape.size(), DataType::DT_INT64);
    CHECK_RET(dims != nullptr, ACLNN_ERR_INNER_NULLPTR);
    aclIntArray* shapeArray = executorP->AllocIntArray(fillShape.data(), fillShape.size());
    CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // value如果非连续，需要转连续
    auto contiguousValue = l0op::Contiguous(value, uniqueExecutor.get());
    CHECK_RET(contiguousValue != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入value的数据类型转换成和selfRef一致
    auto castValue = l0op::Cast(contiguousValue, selfRef->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castValue != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto fillOut = l0op::Fill(dims, castValue, shapeArray, executorP);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(fillOut, selfRef, executorP);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceFillTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceFillTensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
