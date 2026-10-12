/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "view_copy.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ViewCopy);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16,   op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_INT16, op::DataType::DT_UINT16, op::DataType::DT_INT32,
    op::DataType::DT_UINT32,  op::DataType::DT_BOOL,  op::DataType::DT_INT64};

static bool IsAiCoreSupport(const aclTensor* dst, const aclTensor* src, const aclTensor* dstOut)
{
    if (dst == nullptr || src == nullptr || dstOut == nullptr) {
        return false;
    }
    const auto dtype = dst->GetDataType();
    return dtype == src->GetDataType() && dtype == dstOut->GetDataType() &&
           op::CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST);
}

static const aclTensor* ViewCopyAiCore(const aclTensor* dst, const aclTensor* dstSize, const aclTensor* dstStride,
                                       const aclTensor* dstStorageOffset, const aclTensor* src,
                                       const aclTensor* srcSize, const aclTensor* srcStride,
                                       const aclTensor* srcStorageOffset, const aclTensor* dstOut,
                                       aclOpExecutor* executor)
{
    L0_DFX(ViewCopyAiCore, dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset,
           dstOut);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ViewCopy, OP_INPUT(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset),
        OP_OUTPUT(dstOut));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ViewCopy ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return dstOut;
}

const aclTensor* ViewCopy(const aclTensor* dst, const aclTensor* dstSize, const aclTensor* dstStride,
                          const aclTensor* dstStorageOffset, const aclTensor* src, const aclTensor* srcSize,
                          const aclTensor* srcStride, const aclTensor* srcStorageOffset, const aclTensor* dstOut,
                          aclOpExecutor* executor)
{
    if (IsAiCoreSupport(dst, src, dstOut)) {
        return ViewCopyAiCore(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset,
                              dstOut, executor);
    }

    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ViewCopy dtype is unsupported or input/output dtypes are not identical.");
    return nullptr;
}

const aclTensor* ViewCopy(const aclTensor* dst, const op::Shape& dstSize, const op::Strides& dstStride,
                          int64_t dstStorageOffset, const aclTensor* src, const op::Shape& srcSize,
                          const op::Strides& srcStride, int64_t srcStorageOffset, const aclTensor* dstOut,
                          aclOpExecutor* executor)
{
    auto dstSizeVector = op::ToShapeVector(dstSize);
    auto srcSizeVector = op::ToShapeVector(srcSize);
    auto dstSizeTensor = executor->ConvertToTensor(dstSizeVector.data(), dstSizeVector.size(),
                                                   op::ToOpDataType(ACL_INT64));
    auto dstStrideTensor = executor->ConvertToTensor(dstStride.data(), dstStride.size(), op::ToOpDataType(ACL_INT64));
    auto dstOffsetTensor = executor->ConvertToTensor(&dstStorageOffset, 1, op::ToOpDataType(ACL_INT64));
    auto srcSizeTensor = executor->ConvertToTensor(srcSizeVector.data(), srcSizeVector.size(),
                                                   op::ToOpDataType(ACL_INT64));
    auto srcStrideTensor = executor->ConvertToTensor(srcStride.data(), srcStride.size(), op::ToOpDataType(ACL_INT64));
    auto srcOffsetTensor = executor->ConvertToTensor(&srcStorageOffset, 1, op::ToOpDataType(ACL_INT64));

    if (dstSizeTensor == nullptr || dstStrideTensor == nullptr || dstOffsetTensor == nullptr ||
        srcSizeTensor == nullptr || srcStrideTensor == nullptr || srcOffsetTensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ViewCopy metadata tensor creation failed.");
        return nullptr;
    }

    return ViewCopy(dst, dstSizeTensor, dstStrideTensor, dstOffsetTensor, src, srcSizeTensor, srcStrideTensor,
                    srcOffsetTensor, dstOut, executor);
}

const aclTensor* ViewCopy(const aclTensor* dstOut, const aclTensor* dstSize, const aclTensor* dstStride,
                          const aclTensor* dstStorageOffset, const aclTensor* src, const aclTensor* srcSize,
                          const aclTensor* srcStride, const aclTensor* srcStorageOffset, aclOpExecutor* executor)
{
    return ViewCopy(dstOut, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, dstOut,
                    executor);
}

const aclTensor* ViewCopy(const aclTensor* dstOut, const op::Shape& dstSize, const op::Strides& dstStride,
                          int64_t dstStorageOffset, const aclTensor* src, const op::Shape& srcSize,
                          const op::Strides& srcStride, int64_t srcStorageOffset, aclOpExecutor* executor)
{
    return ViewCopy(dstOut, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, dstOut,
                    executor);
}

} // namespace l0op
