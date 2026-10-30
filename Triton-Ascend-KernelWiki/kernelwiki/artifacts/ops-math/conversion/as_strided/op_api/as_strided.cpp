/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/view_copy/op_api/view_copy.h"
#include "opdev/op_def.h"
#include "opdev/shape_utils.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AsStrided);
// 910B  float16,float,int8,int16,int32,int64,uint8,uint16,uint32,uint64,bool,bfloat16,complex32,complex64,
// 910_93  float16,float,int8,int16,int32,int64,uint8,uint16,uint32,uint64,bool,bfloat16,complex32,complex64,
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT,     op::DataType::DT_FLOAT16,  op::DataType::DT_INT8,  op::DataType::DT_INT16,
    op::DataType::DT_INT32,     op::DataType::DT_INT64,    op::DataType::DT_UINT8, op::DataType::DT_UINT16,
    op::DataType::DT_UINT32,    op::DataType::DT_UINT64,   op::DataType::DT_BOOL,  op::DataType::DT_BF16,
    op::DataType::DT_COMPLEX32, op::DataType::DT_COMPLEX64};

// 910   float16,float,int8,int16,int32,int64,uint8,uint16,uint32,uint64,bool,complex32,complex64,
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,    op::DataType::DT_FLOAT16, op::DataType::DT_INT8,  op::DataType::DT_INT16,
    op::DataType::DT_INT32,    op::DataType::DT_INT64,   op::DataType::DT_UINT8, op::DataType::DT_UINT16,
    op::DataType::DT_UINT32,   op::DataType::DT_UINT64,  op::DataType::DT_BOOL,  op::DataType::DT_COMPLEX32,
    op::DataType::DT_COMPLEX64};

// regbase   float16,float,int8,int16,int32,int64,uint8,uint16,uint32,uint64,bool,bfloat16,complex32,complex64
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_REGBASE = {
    op::DataType::DT_FLOAT,        op::DataType::DT_FLOAT16,   op::DataType::DT_INT8,     op::DataType::DT_INT16,
    op::DataType::DT_INT32,        op::DataType::DT_INT64,     op::DataType::DT_UINT8,    op::DataType::DT_UINT16,
    op::DataType::DT_UINT32,       op::DataType::DT_UINT64,    op::DataType::DT_BOOL,     op::DataType::DT_BF16,
    op::DataType::DT_COMPLEX32,    op::DataType::DT_COMPLEX64, op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT8_E5M2,
    op::DataType::DT_FLOAT8_E4M3FN};

static bool IsAiCoreSupport(const aclTensor* self)
{
    auto dataType = self->GetDataType();
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201) {
        return op::CheckType(dataType, AICORE_DTYPE_SUPPORT_LIST_910B);
    } else if (IsRegBase()) {
        return op::CheckType(dataType, AICORE_DTYPE_SUPPORT_LIST_REGBASE);
    }
    return op::CheckType(dataType, AICORE_DTYPE_SUPPORT_LIST);
}

const aclTensor* AsStridedAiCore(
    const aclTensor* x, const aclTensor* y, const aclTensor* size, const aclTensor* stride,
    const aclTensor* storageOffset, aclOpExecutor* executor)
{
    L0_DFX(AsStridedAiCore, x, y, size, stride, storageOffset);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AsStrided, OP_INPUT(x, size, stride, storageOffset), OP_OUTPUT(y));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AsStridedAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return y;
}

const aclTensor* AsStrided(
    const aclTensor* x, const aclTensor* y, const aclTensor* size, const aclTensor* stride,
    const aclTensor* storageOffset, aclOpExecutor* executor)
{
    if (IsAiCoreSupport(x)) {
        return AsStridedAiCore(x, y, size, stride, storageOffset, executor);
    }
    // Use ViewCopy
    auto dstShape = y->GetViewShape();
    auto dstStrides = y->GetViewStrides();
    auto dstStorageOffset = y->GetViewOffset();
    auto dstSize = executor->ConvertToTensor(
        op::ToShapeVector(dstShape).data(), dstShape.GetDimNum(), op::ToOpDataType(ACL_INT64));
    auto dstStride = executor->ConvertToTensor(dstStrides.data(), dstStrides.size(), op::ToOpDataType(ACL_INT64));
    auto dstOffset = executor->ConvertToTensor(&dstStorageOffset, 1, op::ToOpDataType(ACL_INT64));
    return ViewCopy(y, dstSize, dstStride, dstOffset, x, size, stride, storageOffset, executor);
}

const aclTensor* AsStrided(
    aclTensor* x, const aclIntArray* size, const aclIntArray* stride, const aclIntArray* storageOffset,
    aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(x->GetDataType(), x->GetStorageFormat(), x->GetOriginalFormat());
    auto sizeTensor = executor->ConvertToTensor(size, op::ToOpDataType(ACL_INT64));
    auto strideTensor = executor->ConvertToTensor(stride, op::ToOpDataType(ACL_INT64));
    auto storageOffsetTensor = executor->ConvertToTensor(storageOffset, op::ToOpDataType(ACL_INT64));
    INFER_SHAPE(AsStrided, OP_INPUT(x, sizeTensor, strideTensor, storageOffsetTensor), OP_OUTPUT(out));
    return AsStrided(x, out, sizeTensor, strideTensor, storageOffsetTensor, executor);
}

} // namespace l0op
