/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_reflection_pad1d.h"
#include "reflection_pad_common.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,   op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT8,    op::DataType::DT_UINT8, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,  op::DataType::DT_INT64,     op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_DOUBLE, op::DataType::DT_INT8,      op::DataType::DT_UINT8,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16,   op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> ASCEND950_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_INT32,       op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,       op::DataType::DT_DOUBLE,      op::DataType::DT_INT16,
    op::DataType::DT_INT64,      op::DataType::DT_UINT64,      op::DataType::DT_UINT32,
    op::DataType::DT_UINT16,     op::DataType::DT_UINT8,       op::DataType::DT_BOOL,
    op::DataType::DT_COMPLEX64,  op::DataType::DT_COMPLEX128,  op::DataType::DT_BF16,
    op::DataType::DT_HIFLOAT8,   op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN,
    op::DataType::DT_FLOAT8_E8M0};

inline static bool CheckNotNull(const aclTensor* self, const aclIntArray* padding, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else if (IsRegBase()) {
        return ASCEND950_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

inline static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    auto supportList = GetDtypeSupportList();
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

    // self和out数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    return true;
}

inline static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK(
        self->GetViewFormat() == out->GetViewFormat(),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input and output should be equal, self [%s], gradInoutput [%s].",
                op::ToString(self->GetViewFormat()).GetString(), op::ToString(out->GetViewFormat()).GetString()),
        return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclIntArray* padding, const aclTensor* out)
{
    auto selfDimnum = self->GetViewShape().GetDimNum();
    // self只支持2维和3维
    OP_CHECK_MIN_DIM(self, 2, return false);
    OP_CHECK_MAX_DIM(self, 3, return false);

    // self, out维度需要一致
    OP_CHECK(selfDimnum == out->GetViewShape().GetDimNum(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self, out dim should be same."), return false);

    // padding长度为2
    OP_CHECK(padding->Size() == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding length should be 2, but got %lu.", padding->Size()),
             return false);

    // check padding value. 0, 1 are indexes
    OP_CHECK((*padding)[0] < self->GetViewShape().GetDim(selfDimnum - 1) &&
                 (*padding)[1] < self->GetViewShape().GetDim(selfDimnum - 1),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding size should be less than the corresponding self dimention."),
             return false);

    // check the last dim value of out. 0, 1 are indexes
    OP_CHECK(out->GetViewShape().GetDim(selfDimnum - 1) ==
                 self->GetViewShape().GetDim(selfDimnum - 1) + (*padding)[0] + (*padding)[1],
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "wrong out shape."), return false);
    return true;
}

inline static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* padding, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, padding, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查数据格式是否支持
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, padding, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnReflectionPad1dGetWorkspaceSize(const aclTensor* self, const aclIntArray* padding, aclTensor* out,
                                                 uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnReflectionPad1d, DFX_IN(self, padding), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, padding, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        // 2 is dim number
        if (self->GetViewShape().GetDimNum() == 2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Expected 2D or 3D tensor with possibly 0 batch size and other non-zero dimentions for input.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 3 is dim number
        if (self->GetViewShape().GetDimNum() == 3) {
            // 1, 2 are indexes
            if (self->GetViewShape().GetDim(1) == 0 || self->GetViewShape().GetDim(2) == 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "Expected 2D or 3D tensor with possibly 0 batch size and other non-zero dimentions for input.");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子进行计算
    const aclTensor* padResult = nullptr;
    if (selfContiguous->GetDataType() == op::DataType::DT_COMPLEX64 ||
        selfContiguous->GetDataType() == op::DataType::DT_COMPLEX128) {
        // 2 is limit dimension
        ret = ProcessPadV3(selfContiguous, padding, padResult, 2, uniqueExecutor.get());
    } else {
        ret = ProcessMirrorPad(selfContiguous, padding, padResult, uniqueExecutor.get());
    }
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // output shape check
    CHECK_RET(CheckShapeAndScalarSame(padResult, out), ACLNN_ERR_PARAM_INVALID);
    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(padResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnReflectionPad1d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                 const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnReflectionPad1d);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
