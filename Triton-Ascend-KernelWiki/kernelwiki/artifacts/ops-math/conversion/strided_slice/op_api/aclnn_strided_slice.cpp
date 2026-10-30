/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_strided_slice.h"
#include "strided_slice.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/transdata.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t ARRAY_SIZE = 0;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16,   op::DataType::DT_FLOAT,     op::DataType::DT_INT32,    op::DataType::DT_UINT8,
    op::DataType::DT_BOOL,      op::DataType::DT_INT8,      op::DataType::DT_INT16,    op::DataType::DT_INT64,
    op::DataType::DT_UINT16,    op::DataType::DT_UINT32,    op::DataType::DT_UINT64,   op::DataType::DT_BF16,
    op::DataType::DT_COMPLEX32, op::DataType::DT_COMPLEX64, op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT8_E5M2,
    ge::DT_FLOAT8_E4M3FN};

static inline bool CheckNotNull(
    const aclTensor* self, const aclIntArray* begin, const aclIntArray* end, const aclIntArray* strides, aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(begin, return false);
    OP_CHECK_NULL(end, return false);
    OP_CHECK_NULL(strides, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, aclTensor* out)
{
    if (IsRegBase()) {
        // 检查self的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);

        // 检查out和输入的数据类型是否一致
        OP_CHECK_DTYPE_NOT_SAME(self, out, return false);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnStridedSlice only support ASCEND950.");
        return false;
    }

    return true;
}

static bool CheckInputDims(const aclTensor* self)
{
    // self的数据维度不能超过8
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

    return true;
}

static bool CheckArray(const aclIntArray* begin, const aclIntArray* end, const aclIntArray* strides)
{
    if (begin->Size() != end->Size()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Expected aclnnStridedSlice begin.size() %lu to be equal to end.size() %lu.",
            begin->Size(), end->Size());
        return false;
    }

    if (end->Size() != strides->Size()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Expected aclnnStridedSlice end.size() %lu to be equal to strides.size() %lu.",
            end->Size(), strides->Size());
        return false;
    }

    for (size_t i = 0; i < strides->Size(); i++) {
        if ((*strides)[i] == 0) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Expected strides value must not be zero, but strides No.[%lu] value is zero.",
                i);
            return false;
        }
    }

    return true;
}

static bool CheckInputMask(const aclIntArray* strides, int64_t ellipsisMask, int64_t shrinkAxisMask)
{
    // ellipsisMask 只能有一个bit位为1
    if ((ellipsisMask != 0) && ((ellipsisMask & (ellipsisMask - 1)) != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Multiple ellipses in slice spec not allowed.");
        return false;
    }

    // shrinkAxisMask 中bit位为1的索引，对应的strides需要大于0，即正数
    for (size_t i = 0; i < strides->Size(); i++) {
        if ((shrinkAxisMask >> i) & 1) {
            if ((*strides)[i] <= 0) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "Strides must be positive when shrinkAxisMask has bit set at dimension [%lu].", i);
                return false;
            }
        }
    }

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclIntArray* begin, const aclIntArray* end, const aclIntArray* strides,
    int64_t ellipsisMask, int64_t shrinkAxisMask, aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, begin, end, strides, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入tensor self
    CHECK_RET(CheckInputDims(self), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查数组是否满足要求
    CHECK_RET(CheckArray(begin, end, strides), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查Mask是否满足要求
    CHECK_RET(CheckInputMask(strides, ellipsisMask, shrinkAxisMask), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 第一段接口
aclnnStatus aclnnStridedSliceGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* begin, const aclIntArray* end, const aclIntArray* strides,
    int64_t beginMask, int64_t endMask, int64_t ellipsisMask, int64_t newAxisMask, int64_t shrinkAxisMask,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnStridedSlice,
        DFX_IN(self, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, begin, end, strides, ellipsisMask, shrinkAxisMask, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (begin->Size() == ARRAY_SIZE) {
        auto viewCopyResult = l0op::ViewCopy(selfContiguous, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        auto selfReformat = l0op::ReFormat(selfContiguous, Format::FORMAT_ND, uniqueExecutor.get());
        CHECK_RET(selfReformat != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto beginTensor = uniqueExecutor.get()->ConvertToTensor(begin, op::ToOpDataType(ACL_INT64));
        auto endTensor = uniqueExecutor.get()->ConvertToTensor(end, op::ToOpDataType(ACL_INT64));
        auto stridesTensor = uniqueExecutor.get()->ConvertToTensor(strides, op::ToOpDataType(ACL_INT64));

        auto stridedsliceOut = l0op::StridedSlice(
            selfReformat, beginTensor, endTensor, stridesTensor, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask, uniqueExecutor.get());
        CHECK_RET(stridedsliceOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 检查输出Tensor out
        CHECK_RET(CheckShapeAndScalarSame(stridedsliceOut, out), ACLNN_ERR_PARAM_INVALID);

        // 固定写法，将计算结果拷贝到输出out上，out支持非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(stridedsliceOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnStridedSlice(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnStridedSlice);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
