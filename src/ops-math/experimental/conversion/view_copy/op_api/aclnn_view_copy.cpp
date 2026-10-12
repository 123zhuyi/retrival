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

#include "aclnn_view_copy.h"

#include "view_copy.h"

#include <vector>

#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace {

constexpr size_t ACLNN_MAX_SHAPE_RANK = 8;

static const std::initializer_list<op::DataType> DATA_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16,   op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_INT16, op::DataType::DT_UINT16, op::DataType::DT_INT32,
    op::DataType::DT_UINT32,  op::DataType::DT_BOOL,  op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> META_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32,
                                                                            op::DataType::DT_INT64};

static bool CheckNotNull(const aclTensor* dst, const aclTensor* dstSize, const aclTensor* dstStride,
                         const aclTensor* dstStorageOffset, const aclTensor* src, const aclTensor* srcSize,
                         const aclTensor* srcStride, const aclTensor* srcStorageOffset, const aclTensor* dstOut)
{
    OP_CHECK_NULL(dst, return false);
    OP_CHECK_NULL(dstSize, return false);
    OP_CHECK_NULL(dstStride, return false);
    OP_CHECK_NULL(dstStorageOffset, return false);
    OP_CHECK_NULL(src, return false);
    OP_CHECK_NULL(srcSize, return false);
    OP_CHECK_NULL(srcStride, return false);
    OP_CHECK_NULL(srcStorageOffset, return false);
    OP_CHECK_NULL(dstOut, return false);
    return true;
}

static bool CheckDtype(const aclTensor* dst, const aclTensor* dstSize, const aclTensor* dstStride,
                       const aclTensor* dstStorageOffset, const aclTensor* src, const aclTensor* srcSize,
                       const aclTensor* srcStride, const aclTensor* srcStorageOffset, const aclTensor* dstOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(dst, DATA_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(src, DATA_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(dstOut, DATA_DTYPE_SUPPORT_LIST, return false);
    if (dst->GetDataType() != src->GetDataType() || dst->GetDataType() != dstOut->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dst, src and dst output must have the same dtype.");
        return false;
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(dstSize, META_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(dstStride, META_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(dstStorageOffset, META_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(srcSize, META_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(srcStride, META_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(srcStorageOffset, META_DTYPE_SUPPORT_LIST, return false);
    if (dstSize->GetDataType() != dstStride->GetDataType() ||
        dstSize->GetDataType() != dstStorageOffset->GetDataType() || dstSize->GetDataType() != srcSize->GetDataType() ||
        dstSize->GetDataType() != srcStride->GetDataType() ||
        dstSize->GetDataType() != srcStorageOffset->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "All ViewCopy metadata tensors must have the same dtype.");
        return false;
    }
    return true;
}

static int64_t ShapeNumel(const op::Shape& shape)
{
    if (shape.GetDimNum() == 0) {
        return 1;
    }

    int64_t numel = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        numel *= shape.GetDim(i);
    }
    return numel;
}

static bool CheckMetaShape(const aclTensor* tensor, int64_t expectedNumel, const char* name)
{
    const int64_t actualNumel = ShapeNumel(tensor->GetViewShape());
    if (actualNumel != expectedNumel) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s numel must be %ld, but got %ld.", name, expectedNumel, actualNumel);
        return false;
    }
    return true;
}

static bool ReadMetaTensorToInt64(const aclTensor* tensor, int64_t expectedNumel, const char* name,
                                  std::vector<int64_t>& values)
{
    values.assign(static_cast<size_t>(expectedNumel), 0);
    const int64_t elementOffset = tensor->GetStorageOffset() + tensor->GetViewOffset();
    const char* storageAddr = static_cast<const char*>(tensor->GetStorageAddr());
    if (storageAddr == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s storage address is null.", name);
        return false;
    }

    if (tensor->GetDataType() == op::DataType::DT_INT32) {
        std::vector<int32_t> hostValues(static_cast<size_t>(expectedNumel), 0);
        const size_t byteSize = static_cast<size_t>(expectedNumel) * sizeof(int32_t);
        const void* deviceAddr = storageAddr + static_cast<size_t>(elementOffset) * sizeof(int32_t);
        auto ret = aclrtMemcpy(hostValues.data(), byteSize, deviceAddr, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER, "aclrtMemcpy failed when reading %s, error code is %d.", name, ret);
            return false;
        }
        for (int64_t i = 0; i < expectedNumel; ++i) {
            values[static_cast<size_t>(i)] = static_cast<int64_t>(hostValues[static_cast<size_t>(i)]);
        }
        return true;
    }

    const size_t byteSize = static_cast<size_t>(expectedNumel) * sizeof(int64_t);
    const void* deviceAddr = storageAddr + static_cast<size_t>(elementOffset) * sizeof(int64_t);
    auto ret = aclrtMemcpy(values.data(), byteSize, deviceAddr, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclrtMemcpy failed when reading %s, error code is %d.", name, ret);
        return false;
    }
    return true;
}

static const aclTensor* ConvertMetaToTensor(const std::vector<int64_t>& values, aclOpExecutor* executor,
                                            const char* name)
{
    auto tensor = executor->ConvertToTensor(values.data(), values.size(), op::DataType::DT_INT64);
    if (tensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to create executor-owned tensor for %s.", name);
    }
    return tensor;
}

static bool MaterializeMetadata(const aclTensor* dstSize, const aclTensor* dstStride, const aclTensor* dstStorageOffset,
                                const aclTensor* srcSize, const aclTensor* srcStride, const aclTensor* srcStorageOffset,
                                aclOpExecutor* executor, const aclTensor*& dstSizeTensor,
                                const aclTensor*& dstStrideTensor, const aclTensor*& dstOffsetTensor,
                                const aclTensor*& srcSizeTensor, const aclTensor*& srcStrideTensor,
                                const aclTensor*& srcOffsetTensor)
{
    const int64_t rank = ShapeNumel(dstSize->GetViewShape());
    std::vector<int64_t> dstSizeValues;
    std::vector<int64_t> dstStrideValues;
    std::vector<int64_t> dstOffsetValues;
    std::vector<int64_t> srcSizeValues;
    std::vector<int64_t> srcStrideValues;
    std::vector<int64_t> srcOffsetValues;
    if (!ReadMetaTensorToInt64(dstSize, rank, "dst_size", dstSizeValues) ||
        !ReadMetaTensorToInt64(dstStride, rank, "dst_stride", dstStrideValues) ||
        !ReadMetaTensorToInt64(dstStorageOffset, 1, "dst_storage_offset", dstOffsetValues) ||
        !ReadMetaTensorToInt64(srcSize, rank, "src_size", srcSizeValues) ||
        !ReadMetaTensorToInt64(srcStride, rank, "src_stride", srcStrideValues) ||
        !ReadMetaTensorToInt64(srcStorageOffset, 1, "src_storage_offset", srcOffsetValues)) {
        return false;
    }

    dstSizeTensor = ConvertMetaToTensor(dstSizeValues, executor, "dst_size");
    dstStrideTensor = ConvertMetaToTensor(dstStrideValues, executor, "dst_stride");
    dstOffsetTensor = ConvertMetaToTensor(dstOffsetValues, executor, "dst_storage_offset");
    srcSizeTensor = ConvertMetaToTensor(srcSizeValues, executor, "src_size");
    srcStrideTensor = ConvertMetaToTensor(srcStrideValues, executor, "src_stride");
    srcOffsetTensor = ConvertMetaToTensor(srcOffsetValues, executor, "src_storage_offset");
    return dstSizeTensor != nullptr && dstStrideTensor != nullptr && dstOffsetTensor != nullptr &&
           srcSizeTensor != nullptr && srcStrideTensor != nullptr && srcOffsetTensor != nullptr;
}

static bool CheckShape(const aclTensor* dst, const aclTensor* dstSize, const aclTensor* dstStride,
                       const aclTensor* dstStorageOffset, const aclTensor* srcSize, const aclTensor* srcStride,
                       const aclTensor* srcStorageOffset, const aclTensor* dstOut)
{
    OP_CHECK_MAX_DIM(dst, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(dstOut, ACLNN_MAX_SHAPE_RANK, return false);
    if (dst->GetViewShape() != dstOut->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dst input and dst output must have the same shape.");
        return false;
    }

    const int64_t rank = ShapeNumel(dstSize->GetViewShape());
    if (rank <= 0 || rank > static_cast<int64_t>(ACLNN_MAX_SHAPE_RANK)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ViewCopy rank must be in [1, 8], but got %ld.", rank);
        return false;
    }

    return CheckMetaShape(dstStride, rank, "dst_stride") && CheckMetaShape(srcSize, rank, "src_size") &&
           CheckMetaShape(srcStride, rank, "src_stride") && CheckMetaShape(dstStorageOffset, 1, "dst_storage_offset") &&
           CheckMetaShape(srcStorageOffset, 1, "src_storage_offset");
}

static aclnnStatus CheckParams(const aclTensor* dst, const aclTensor* dstSize, const aclTensor* dstStride,
                               const aclTensor* dstStorageOffset, const aclTensor* src, const aclTensor* srcSize,
                               const aclTensor* srcStride, const aclTensor* srcStorageOffset, const aclTensor* dstOut)
{
    CHECK_RET(
        CheckNotNull(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, dstOut),
        ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtype(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, dstOut),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(dst, dstSize, dstStride, dstStorageOffset, srcSize, srcStride, srcStorageOffset, dstOut),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static op::Strides BuildContiguousStrides(const op::Shape& shape)
{
    op::Strides strides;
    op::ToContiguousStrides(shape, strides);
    return strides;
}

static const aclTensor* CopyDstToOutput(const aclTensor* dst, const aclTensor* dstOut, aclOpExecutor* executor)
{
    if (dst == dstOut) {
        return dstOut;
    }

    const auto& shape = dstOut->GetViewShape();
    const auto strides = BuildContiguousStrides(shape);
    const auto srcStrides = BuildContiguousStrides(dst->GetViewShape());
    return l0op::ViewCopy(dstOut, shape, strides, 0, dst, dst->GetViewShape(), srcStrides, 0, dstOut, executor);
}

} // namespace

extern "C" aclnnStatus aclnnViewCopyGetWorkspaceSize(const aclTensor* dst, const aclTensor* dstSize,
                                                     const aclTensor* dstStride, const aclTensor* dstStorageOffset,
                                                     const aclTensor* src, const aclTensor* srcSize,
                                                     const aclTensor* srcStride, const aclTensor* srcStorageOffset,
                                                     const aclTensor* dstOut, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnViewCopy,
                   DFX_IN(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset),
                   DFX_OUT(dstOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset,
                           dstOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (dst->IsEmpty() || src->IsEmpty() || dstOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto copyResult = CopyDstToOutput(dst, dstOut, uniqueExecutor.get());
    CHECK_RET(copyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* dstSizeTensor = nullptr;
    const aclTensor* dstStrideTensor = nullptr;
    const aclTensor* dstOffsetTensor = nullptr;
    const aclTensor* srcSizeTensor = nullptr;
    const aclTensor* srcStrideTensor = nullptr;
    const aclTensor* srcOffsetTensor = nullptr;
    CHECK_RET(MaterializeMetadata(dstSize, dstStride, dstStorageOffset, srcSize, srcStride, srcStorageOffset,
                                  uniqueExecutor.get(), dstSizeTensor, dstStrideTensor, dstOffsetTensor, srcSizeTensor,
                                  srcStrideTensor, srcOffsetTensor),
              ACLNN_ERR_INNER);

    auto viewCopyResult = l0op::ViewCopy(dstOut, dstSizeTensor, dstStrideTensor, dstOffsetTensor, src, srcSizeTensor,
                                         srcStrideTensor, srcOffsetTensor, dstOut, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnViewCopy(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnViewCopy);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
