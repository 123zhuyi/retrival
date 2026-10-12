/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_cast.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
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

/* Cast 算子的完整计算流程如下:
 *          self               dtype
 *            \                 /
 *  Contiguous(workspace_0)    /
 *               \            /
 *             Cast(workspace_1)
 *                      |
 *                   ViewCopy
 *                      |
 *                    result
 */

static const size_t MAX_DIM = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16,   op::DataType::DT_FLOAT,     op::DataType::DT_DOUBLE, op::DataType::DT_INT8,
    op::DataType::DT_UINT8,     op::DataType::DT_INT16,     op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_UINT16,    op::DataType::DT_UINT32,    op::DataType::DT_UINT64, op::DataType::DT_BOOL,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_DEFAULT = {
    op::DataType::DT_FLOAT16,   op::DataType::DT_FLOAT,      op::DataType::DT_DOUBLE, op::DataType::DT_INT8,
    op::DataType::DT_UINT8,     op::DataType::DT_INT16,      op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_UINT16,    op::DataType::DT_UINT32,     op::DataType::DT_UINT64, op::DataType::DT_BOOL,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> ASCEND950_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16,   op::DataType::DT_FLOAT,       op::DataType::DT_DOUBLE,
    op::DataType::DT_INT8,      op::DataType::DT_UINT8,       op::DataType::DT_INT16,
    op::DataType::DT_INT32,     op::DataType::DT_INT64,       op::DataType::DT_UINT16,
    op::DataType::DT_UINT32,    op::DataType::DT_UINT64,      op::DataType::DT_BOOL,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128,  op::DataType::DT_BF16,
    op::DataType::DT_HIFLOAT8,  op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN,
    op::DataType::DT_COMPLEX32, op::DataType::DT_FLOAT4_E1M2, op::DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<op::DataType> ASCEND950_SELF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16,   op::DataType::DT_FLOAT,       op::DataType::DT_DOUBLE,
    op::DataType::DT_INT8,      op::DataType::DT_UINT8,       op::DataType::DT_INT16,
    op::DataType::DT_INT32,     op::DataType::DT_INT64,       op::DataType::DT_UINT16,
    op::DataType::DT_UINT32,    op::DataType::DT_UINT64,      op::DataType::DT_BOOL,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128,  op::DataType::DT_BF16,
    op::DataType::DT_HIFLOAT8,  op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN,
    op::DataType::DT_COMPLEX32, op::DataType::DT_FLOAT4_E1M2, op::DataType::DT_FLOAT4_E2M1};

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const DataType dtype)
{
    // 检查self的数据类型是否在算子的支持列表内
    bool isASCEND910B = (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201);
    bool isASCEND910_93 = (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201);
    bool isASCEND950 = (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510);
    bool isAscend910BC = isASCEND910B || isASCEND910_93;

    auto supportList = ASCEND910_DTYPE_SUPPORT_LIST;
    auto selfSupportList = ASCEND910_DTYPE_SUPPORT_LIST;
    if (isAscend910BC) {
        supportList = DTYPE_SUPPORT_LIST_DEFAULT;
        selfSupportList = DTYPE_SUPPORT_LIST_DEFAULT;
    } else if (isASCEND950) {
        supportList = ASCEND950_DTYPE_SUPPORT_LIST;
        selfSupportList = ASCEND950_SELF_DTYPE_SUPPORT_LIST;
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(self, selfSupportList, return false);
    bool isSupport = CheckType(dtype, supportList);
    // 检查参数dtype是否在Cast算子的输出数据类型支持列表内
    if (!isSupport) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The param dtype not implemented for %s, should be in dtype support list %s.",
            op::ToString(dtype).GetString(), op::ToString(supportList).GetString());
        return false;
    }
    return true;
}

static inline bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);
    return true;
}

static inline aclnnStatus CheckParams(const aclTensor* self, const DataType dtype, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, dtype), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入tensor的shape是否为异常，输出和输入的shape是否相同
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCastGetWorkspaceSize(
    const aclTensor* self, const aclDataType dtype, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnCast, DFX_IN(self, dtype), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, op::ToOpDataType(dtype), out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 输入为空tensor时，直接返回dtype类型的空tensor
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用Cast算子kernel,将输入self的数据类型转换成指定的数据类型
    auto castOut = l0op::Cast(selfContiguous, op::ToOpDataType(dtype), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCast(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCast);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
