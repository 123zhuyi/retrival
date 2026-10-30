/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_fill_scalar.h"
#include "conversion/fill/op_api/fill.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

/**
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *  A[(selfRef)] --> D([l0op::Fill])
 *  D --> H([l0op::ViewCopy])
 *  H --> I[(selfRef)]
 *  C([value]) --> D
 * ```
 */

namespace op {
inline static bool CheckNotNull(const aclTensor* selfRef, const aclScalar* value)
{
    OP_CHECK_NULL(selfRef, return false);
    OP_CHECK_NULL(value, return false);
    return true;
}

} // namespace op
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,     op::DataType::DT_INT64,     op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,      op::DataType::DT_UINT8,     op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,  op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_GE910B_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,     op::DataType::DT_INT64,      op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,      op::DataType::DT_UINT8,      op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,  op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

// 判断芯片类型是否大于等于910B
static inline bool CheckSocVersionGe910B(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

inline static bool CheckDtypeValid(const aclTensor* self)
{
    bool is910BSocVersion = CheckSocVersionGe910B();
    const std::initializer_list<DataType> CURRENT_DTYPE_SUPPORT_LIST =
        is910BSocVersion ? DTYPE_SUPPORT_GE910B_LIST : DTYPE_SUPPORT_910_LIST;
    // 检查self的数据类型是否在fill_scalar算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, CURRENT_DTYPE_SUPPORT_LIST, return false);
    return true;
}

inline static bool CheckPromoteType(const aclTensor* self, const aclScalar* value)
{
    // 检查value能否转换为推导后的数据类型
    op::DataType promoteType = op::PromoteType(self->GetDataType(), value->GetDataType());
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype %s and value dtype %s can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(value->GetDataType()).GetString());
        return false;
    }
    return true;
}

inline static bool CheckShape(const aclTensor* self)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    return true;
}

inline static aclnnStatus CheckParams(const aclTensor* self, const aclScalar* value)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, value), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    CHECK_RET(CheckPromoteType(self, value), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape
    CHECK_RET(CheckShape(self), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceFillScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* value, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，参数检查
    L2_DFX_PHASE_1(aclnnInplaceFillScalar, DFX_IN(selfRef, value), DFX_OUT());
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(selfRef, value);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (selfRef->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // 固定写法，将输入self转换成连续的tensor
    aclOpExecutor* executorP = uniqueExecutor.get();
    const aclTensor* castTensor = executorP->ConvertToTensor(value, selfRef->GetDataType());
    const aclTensor* dims;
    aclIntArray* shapeArray;
    if (selfRef->GetViewShape().GetDimNum() != 0) {
        size_t dimNum = selfRef->GetViewShape().GetDimNum();
        FVector<int64_t> tmp;
        for (size_t idx = 0; idx < dimNum; idx++) {
            int64_t tmpVal = selfRef->GetViewShape().GetDim(idx);
            tmp.push_back(tmpVal);
        }
        shapeArray = executorP->AllocIntArray(tmp.data(), tmp.size());
        dims = executorP->ConvertToTensor(tmp.data(), tmp.size(), op::ToOpDataType(ACL_INT64));
    } else {
        FVector<int64_t> tmp;
        tmp.push_back(1);
        dims = executorP->ConvertToTensor(tmp.data(), tmp.size(), op::ToOpDataType(ACL_INT64));
        shapeArray = executorP->AllocIntArray(tmp.data(), tmp.size());
    }
    auto fillOut = l0op::Fill(dims, castTensor, shapeArray, executorP);
    auto viewCopyResult = l0op::ViewCopy(fillOut, selfRef, executorP);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceFillScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnInplaceFillScalar);
    OP_LOGI("Entering InplaceFillScalar");
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
