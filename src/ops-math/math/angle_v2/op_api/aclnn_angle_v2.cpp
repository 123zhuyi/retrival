/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_angle_v2.h"
#include "angle_v2.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<op::DataType> INPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BOOL,  op::DataType::DT_UINT8,    op::DataType::DT_INT8, op::DataType::DT_INT16,
    op::DataType::DT_INT32, op::DataType::DT_INT64,    op::DataType::DT_BF16, op::DataType::DT_FLOAT16,
    op::DataType::DT_FLOAT, op::DataType::DT_COMPLEX64};

static const std::initializer_list<op::DataType> INPUT_DTYPE_910_SUPPORT_LIST = {
    op::DataType::DT_BOOL,    op::DataType::DT_UINT8, op::DataType::DT_INT8,
    op::DataType::DT_INT16,   op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_COMPLEX64};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> OUT_DTYPE_910_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                               op::DataType::DT_FLOAT};

static const inline std::initializer_list<DataType>& GetSupportDtypeList()
{
    static const std::initializer_list<DataType> emptyDtypes = {};
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch)) {
        return INPUT_DTYPE_SUPPORT_LIST;
    } else if (socVersion == SocVersion::ASCEND910) {
        // aicore_910_config：不支持BFLOAT16
        return INPUT_DTYPE_910_SUPPORT_LIST;
    } else {
        return emptyDtypes;
    }
}

static const inline std::initializer_list<DataType>& GetOutSupportDtypeList()
{
    static const std::initializer_list<DataType> emptyDtypes = {};
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch)) {
        return OUT_DTYPE_SUPPORT_LIST;
    } else if (socVersion == SocVersion::ASCEND910) {
        // aicore_910_config：输出不支持BFLOAT16
        return OUT_DTYPE_910_SUPPORT_LIST;
    } else {
        return emptyDtypes;
    }
}

static bool CheckNotNull(const aclTensor* x, const aclTensor* out)
{
    // 检查输入输出是否为空指针
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* out)
{
    static const std::initializer_list<DataType> xDtypeSupportList = GetSupportDtypeList();
    static const std::initializer_list<DataType> outDtypeSupportList = GetOutSupportDtypeList();
    // 检查输入输出的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x, xDtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, outDtypeSupportList, return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* out)
{
    // 输入输出的维度最多支持8维
    OP_CHECK_MAX_DIM(x, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);
    // 检查x和out的形状是否一致
    OP_CHECK_SHAPE_NOT_EQUAL(x, out, return false);
    return true;
}

// 检查数据格式
static bool CheckFormat(const aclTensor* x, const aclTensor* out)
{
    OP_CHECK(!::IsPrivateFormat(x->GetStorageFormat()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AngleV2 does not support private input format on RegBase."),
             return false);
    OP_CHECK(!::IsPrivateFormat(out->GetStorageFormat()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AngleV2 does not support private output format on RegBase."),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入输出的数据类型是否在API支持的数据类型范围之内（根据soc判断）
    CHECK_RET(CheckDtypeValid(x, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入输出的维度（最多8维）以及x和out的形状是否一致
    CHECK_RET(CheckShape(x, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查数据格式是否为ND
    CHECK_RET(CheckFormat(x, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAngleV2GetWorkspaceSize(const aclTensor* x, aclTensor* out, uint64_t* workspaceSize,
                                         aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAngleV2, DFX_IN(x), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto angleOut = l0op::AngleV2(xContiguous, uniqueExecutor.get());
    CHECK_RET(angleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 若AngleV2推导出的输出dtype与用户out的dtype不一致，需要先Cast转换
    const aclTensor* angleCasted = angleOut;
    if (angleOut->GetDataType() != out->GetDataType()) {
        // 将计算结果转换成输出out的数据类型
        angleCasted = l0op::Cast(angleOut, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(angleCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto viewCopyResult = l0op::ViewCopy(angleCasted, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAngleV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAngleV2);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
