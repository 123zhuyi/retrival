/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file view_copy.cpp
 * \brief
 */
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

static const int64_t DIM_ZERO = 0;
static const int64_t DIM_ONE = 1;
static const int64_t DIM_TWO = 2;
static const int64_t DIM_THREE = 3;
static const int64_t DIM_FOUR = 4;
static const int64_t LAST_DIM_LIMIT = 32;
static const int64_t SMALL_UB_SIZE = 192 * 1024;
static const int64_t ONE_BYTE = 1;
static const int64_t TWO_BYTE = 2;
static const int64_t FOUR_BYTE = 4;
static const int64_t EIGHT_BYTE = 8;
static const int64_t BLOCK_SIZE = 32;
namespace l0op {

OP_TYPE_REGISTER(ViewCopy);
// bfloat16,uint8,int8,bool,float32,int32,uint32,int16,float16,uint16,int64,uint64,hifloat8,float8_e5m2,float8_e4m3fn
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_REGBASE = {
    op::DataType::DT_FLOAT16,  op::DataType::DT_FLOAT,       op::DataType::DT_INT8,         op::DataType::DT_INT16,
    op::DataType::DT_INT32,    op::DataType::DT_INT64,       op::DataType::DT_UINT8,        op::DataType::DT_UINT16,
    op::DataType::DT_UINT32,   op::DataType::DT_UINT64,      op::DataType::DT_BOOL,         op::DataType::DT_BF16,
    op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN};

// 910_93 bfloat16,uint8,int8,bool,float32,int32,uint32,int16,float16,uint16,int64,uint64,
// 910B bfloat16,uint8,int8,bool,float32,int32,uint32,int16,float16,uint16,int64,uint64,
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_INT8,  op::DataType::DT_INT16,
    op::DataType::DT_INT32,   op::DataType::DT_INT64,  op::DataType::DT_UINT8, op::DataType::DT_UINT16,
    op::DataType::DT_UINT32,  op::DataType::DT_UINT64, op::DataType::DT_BOOL,  op::DataType::DT_BF16,
};

// 910  uint8,int8,bool,float32,int32,uint32,int16,float16,uint16,int64,uint64
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_INT8,  op::DataType::DT_INT16,
    op::DataType::DT_INT32,   op::DataType::DT_INT64,  op::DataType::DT_UINT8, op::DataType::DT_UINT16,
    op::DataType::DT_UINT32,  op::DataType::DT_UINT64, op::DataType::DT_BOOL,
};

// DT_DOUBLE,DT_FLOAT,DT_FLOAT16,DT_INT8,DT_UINT8,DT_INT16,DT_INT32,DT_INT64,DT_BOOL,DT_BF16
static const std::initializer_list<op::DataType> AICPU_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_INT8,       op::DataType::DT_INT16,
    op::DataType::DT_INT32,   op::DataType::DT_INT64,  op::DataType::DT_UINT8,      op::DataType::DT_BOOL,
    op::DataType::DT_BF16,    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX64,
};

static const std::initializer_list<op::DataType> ONE_BYTE_DTYPE_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT8_E5M2,
    op::DataType::DT_FLOAT8_E4M3FN};

static const std::initializer_list<op::DataType> TWO_BYTE_DTYPE_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_INT16, op::DataType::DT_UINT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> FOUR_BYTE_DTYPE_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_UINT32};

static const std::initializer_list<op::DataType> EIGHT_BYTE_DTYPE_LIST = {
    op::DataType::DT_INT64, op::DataType::DT_UINT64};

static int64_t GetByteSize(const op::DataType dtype)
{
    if (op::CheckType(dtype, ONE_BYTE_DTYPE_LIST)) {
        return ONE_BYTE;
    }
    if (op::CheckType(dtype, TWO_BYTE_DTYPE_LIST)) {
        return TWO_BYTE;
    }
    if (op::CheckType(dtype, FOUR_BYTE_DTYPE_LIST)) {
        return FOUR_BYTE;
    }
    if (op::CheckType(dtype, EIGHT_BYTE_DTYPE_LIST)) {
        return EIGHT_BYTE;
    }
    return EIGHT_BYTE;
}

static bool IsAiCoreSupport(const op::DataType dataType)
{
    auto curArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201) {
        return op::CheckType(dataType, AICORE_DTYPE_SUPPORT_LIST_910B);
    }
    if (op::IsRegBase(curArch)) {
        return op::CheckType(dataType, AICORE_DTYPE_SUPPORT_LIST_REGBASE);
    }
    return op::CheckType(dataType, AICORE_DTYPE_SUPPORT_LIST);
}
int64_t ReCalSize(int64_t dst_size_len, int64_t& last_stride, const op::Shape& dstSize, const op::Strides& dstStrides)
{
    int64_t isize = 1;
    int64_t istride = 1;
    int64_t new_len = 0;
    for (int64_t i = dst_size_len - 1; i >= 0; --i) {
        if (istride * isize != dstStrides[i]) {
            if (new_len == 0) {
                last_stride = isize;
            }
            if (dstStrides[i] < last_stride) {
                return -1;
            }
            new_len += 1;
            isize = dstSize[i];
            istride = dstStrides[i];
        } else {
            isize *= dstSize[i];
        }
    }
    new_len += 1;
    return new_len;
}

static bool IsAiCoreSupport(
    const op::DataType dataType, const op::Shape& srcSize, const op::Shape& dstSize, const op::Strides& dstStrides)
{
    if (!IsAiCoreSupport(dataType)) {
        return false;
    }
    int64_t byte_per_data = GetByteSize(dataType);
    int64_t data_per_block = BLOCK_SIZE / byte_per_data;
    CHECK_RET(srcSize == dstSize, false);
    // dstSize[x, y, 2] dstStride [z, 1, y]
    if (dstSize.GetDimNum() == DIM_THREE && dstSize[1] == dstStrides[2] && dstStrides[1] == 1 && dstSize[2] == 2 &&
        dstSize[DIM_ONE] * dstSize[DIM_TWO] * DIM_TWO * byte_per_data < SMALL_UB_SIZE &&
        (dstSize[DIM_ONE] * dstSize[DIM_TWO] % data_per_block == 0 ||
         dstStrides[DIM_ZERO] - dstSize[DIM_ONE] * dstSize[DIM_TWO] >= data_per_block)) {
        return true;
    }
    if (dstSize.GetDimNum() == DIM_THREE && dstSize[0] == 1 && dstStrides[0] == 1 && dstStrides[1] == 1 &&
        (dstSize[DIM_ONE] % (data_per_block) == 0 || dstStrides[DIM_TWO] - dstSize[DIM_ONE] >= data_per_block) &&
        dstSize[DIM_ONE] * (dstSize[DIM_TWO] + 1) * byte_per_data < SMALL_UB_SIZE) {
        return true;
    }
    int64_t last_stride = 0;
    int64_t reCalSize = ReCalSize(static_cast<int64_t>(dstStrides.size()), last_stride, dstSize, dstStrides);
    if (reCalSize >= DIM_THREE && reCalSize <= DIM_FOUR && last_stride >= LAST_DIM_LIMIT) {
        return true;
    }
    op::Strides dstSizeStride;
    op::ToContiguousStrides(dstSize, dstSizeStride);
    int64_t diff = -1;
    int64_t diffIndex = -1;
    for (int64_t i = static_cast<int64_t>(dstSizeStride.size()) - 1; i >= 0; i--) {
        if (dstStrides[i] != dstSizeStride[i]) {
            diff = dstStrides[i] - dstSizeStride[i];
            diffIndex = i;
            break;
        }
    }
    if (diff == -1 || reCalSize < 0) {
        return false;
    }
    for (int64_t i = diffIndex - 1; i >= 0; i--) {
        diff = diff * dstSize[i + 1];
        if (diff != dstStrides[i] - dstSizeStride[i]) {
            return false;
        }
    }
    return true;
}

const aclTensor* ViewCopyAiCpu(
    const aclTensor* y, const aclTensor* dstSize, const aclTensor* dstStride, const aclTensor* dstOffset,
    const aclTensor* x, const aclTensor* srcSize, const aclTensor* srcStride, const aclTensor* srcOffset,
    aclOpExecutor* executor)
{
    L0_DFX(ViewCopyAiCpu, y, dstSize, dstStride, dstOffset, x, srcSize, srcStride, srcOffset);
    auto yConst = const_cast<aclTensor*>(y);
    static op::internal::AicpuTaskSpace space("ViewCopy");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        ViewCopy, OP_ATTR_NAMES(), OP_INPUT(y, dstSize, dstStride, dstOffset, x, srcSize, srcStride, srcOffset),
        OP_OUTPUT(yConst));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return y;
}

const aclTensor* ViewCopyAiCore(
    const aclTensor* y, const aclTensor* dstSize, const aclTensor* dstStride, const aclTensor* dstOffset,
    const aclTensor* x, const aclTensor* srcSize, const aclTensor* srcStride, const aclTensor* srcOffset,
    aclOpExecutor* executor)
{
    L0_DFX(ViewCopyAiCore, y, dstSize, dstStride, dstOffset, x, srcSize, srcStride, srcOffset);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ViewCopy, OP_INPUT(y, dstSize, dstStride, dstOffset, x, srcSize, srcStride, srcOffset), OP_OUTPUT(y));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ViewCopyAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return y;
}

const aclTensor* ViewCopy(
    const aclTensor* y, const aclTensor* dstSize, const aclTensor* dstStride, const aclTensor* dstOffset,
    const aclTensor* x, const aclTensor* srcSize, const aclTensor* srcStride, const aclTensor* srcOffset,
    aclOpExecutor* executor)
{
    if (op::CheckType(x->GetDataType(), AICPU_DTYPE_SUPPORT_LIST)) {
        return ViewCopyAiCpu(y, dstSize, dstStride, dstOffset, x, srcSize, srcStride, srcOffset, executor);
    }
    OP_LOGE(
        ACL_ERROR_INVALID_PARAM, "ViewCopy on aicpu do not support dtype: %s.",
        op::ToString(x->GetDataType()).GetString());
    return nullptr;
}

const aclTensor* ViewCopy(
    const aclTensor* y, const op::Shape& dstSize, const op::Strides& dstStride, int64_t dstOffset, const aclTensor* x,
    const op::Shape& srcSize, const op::Strides& srcStride, int64_t srcOffset, aclOpExecutor* executor)
{
    auto srcSizeTensor =
        executor->ConvertToTensor(op::ToShapeVector(srcSize).data(), srcSize.GetDimNum(), op::ToOpDataType(ACL_INT64));
    auto srcStrideTensor = executor->ConvertToTensor(srcStride.data(), srcStride.size(), op::ToOpDataType(ACL_INT64));
    auto srcOffsetTensor = executor->ConvertToTensor(&srcOffset, 1, op::ToOpDataType(ACL_INT64));

    auto dstSizeTensor =
        executor->ConvertToTensor(op::ToShapeVector(dstSize).data(), dstSize.GetDimNum(), op::ToOpDataType(ACL_INT64));
    auto dstStrideTensor = executor->ConvertToTensor(dstStride.data(), dstStride.size(), op::ToOpDataType(ACL_INT64));
    auto dstOffsetTensor = executor->ConvertToTensor(&dstOffset, 1, op::ToOpDataType(ACL_INT64));
    if (op::IsRegBase()) {
        if (IsAiCoreSupport(x->GetDataType())) {
            return ViewCopyAiCore(y, dstSizeTensor, dstStrideTensor, dstOffsetTensor, x, srcSizeTensor, srcStrideTensor,
                srcOffsetTensor, executor);
        }
    } else if (IsAiCoreSupport(x->GetDataType(), srcSize, dstSize, dstStride)) {
        return ViewCopyAiCore(
            y, dstSizeTensor, dstStrideTensor, dstOffsetTensor, x, srcSizeTensor, srcStrideTensor, srcOffsetTensor,
            executor);
    }

    if (op::CheckType(x->GetDataType(), AICPU_DTYPE_SUPPORT_LIST)) {
        return ViewCopyAiCpu(
            y, dstSizeTensor, dstStrideTensor, dstOffsetTensor, x, srcSizeTensor, srcStrideTensor, srcOffsetTensor,
            executor);
    }

    OP_LOGE(
        ACL_ERROR_INVALID_PARAM, "ViewCopy on aicpu do not support dtype: %s.",
        op::ToString(x->GetDataType()).GetString());
    return nullptr;
}

} // namespace l0op
