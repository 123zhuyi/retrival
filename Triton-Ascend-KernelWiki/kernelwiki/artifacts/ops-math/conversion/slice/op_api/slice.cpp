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
 * \file slice.cpp
 * \brief
 */
#include "aclnn_kernels/slice.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Slice);
OP_TYPE_REGISTER(SliceV2);

static constexpr size_t MAX_DIM_NUM = 8;
static constexpr int64_t MAX_CORE_NUM = 48;
static constexpr int64_t MAX_UB_SIZE = 192 * 1024;
static constexpr int64_t FP16_BLOCK_NUM = 16;
static constexpr int64_t FP32_BLOCK_NUM = 8;
static constexpr int64_t FP16_BYTE = 2;
static constexpr int64_t FP32_BYTE = 4;
// dsl实现 尾轴切分双对齐模板中,当输出维度积小于该值时,会开启少核优化
static constexpr int64_t DMA_CORE_THRESHOLD = 1048576;  // slice dsl实现中的参数
static constexpr int64_t DMA_CORE_NUMBER = 4;  // slice dsl实现中的参数
static constexpr int64_t BYTE64 = 64;
static constexpr int64_t LAST_DIM_THRESHOLD = 192;  // 尾轴切分双对齐场景尾轴限制参数
static constexpr int64_t LAST_DIM_RATIO = 2;  // 尾轴切分双对齐场景尾轴限制参数


// "float", "float16", "bfloat16"
static const std::initializer_list<op::DataType> SLICEV2_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

// "float", "float16", "int8", "int16", "int32", "int64", "uint8", "uint16",
// "uint32", "uint64", "bool"
static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST_ASCEND910 = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT8,   DataType::DT_INT16,
    DataType::DT_INT32, DataType::DT_UINT8,   DataType::DT_UINT16, DataType::DT_UINT32,
    DataType::DT_INT64, DataType::DT_UINT64,  DataType::DT_BOOL};

// "float", "float16", "int8", "int16", "int32", "int64", "uint8", "uint16",
// "uint32", "uint64", "bool", "bfloat16"
static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST_ASCEND910B = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT8,   DataType::DT_INT16,
    DataType::DT_INT32, DataType::DT_UINT8,   DataType::DT_UINT16, DataType::DT_UINT32,
    DataType::DT_INT64, DataType::DT_UINT64,  DataType::DT_BOOL,   DataType::DT_BF16};

// "float", "float16", "int8", "int16", "int32", "int64", "uint8", "uint16",
// "uint32", "uint64", "bool", "bfloat16", "hifloat8", "float8_e5m2", "float8_e4m3fn"
// "float8_e8m0", "float4_e2m1", "float4_e1m2"
static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST_ASCEND910D = {
    DataType::DT_FLOAT,    DataType::DT_FLOAT16,     DataType::DT_INT8,         DataType::DT_INT16,
    DataType::DT_INT32,    DataType::DT_UINT8,       DataType::DT_UINT16,       DataType::DT_UINT32,
    DataType::DT_INT64,    DataType::DT_UINT64,      DataType::DT_BOOL,         DataType::DT_BF16,
    DataType::DT_HIFLOAT8, DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E8M0,
    DataType::DT_FLOAT4_E2M1, DataType::DT_FLOAT4_E1M2};


static bool IsSliceV2ARAFullLoadSupport(const aclTensor *self, const aclIntArray *offsets,
    const aclIntArray *size)
{
    auto selfDimNum = self->GetViewShape().GetDimNum();
    int64_t curOffset = 0, curSize = 0, curXDim = 0;
    size_t sliceAxis = MAX_DIM_NUM;
    int64_t selfColNum = 1, sizeColNum = 1;
    for (size_t i = 0; i < selfDimNum; ++i) {
        curOffset = (*offsets)[i], curSize = (*size)[i];
        curXDim = static_cast<int64_t>(self->GetViewShape().GetDim(i));
        // 处理offsets和size中负数的情况
        curOffset += curOffset < 0 ? curXDim : 0;
        curSize = curSize == -1 ? curXDim - curOffset : curSize;
        if (curXDim > curSize) {
            sliceAxis = i;
            break;
        }
    }
    for (size_t i = sliceAxis + 1; i < selfDimNum; ++i) {
        curOffset = (*offsets)[i], curSize = (*size)[i];
        curXDim = static_cast<int64_t>(self->GetViewShape().GetDim(i));
        curOffset += curOffset < 0 ? curXDim : 0;
        curSize = curSize == -1 ? curXDim - curOffset : curSize;
        if (curOffset < 0 || curSize < 0 || curOffset + curSize > curXDim) {
            return false;
        }
        selfColNum *= curXDim;
        sizeColNum *= curSize;
    }
    auto dataByte = self->GetDataType() == DataType::DT_FLOAT ? FP32_BYTE : FP16_BYTE;
    auto numInOneBlock = self->GetDataType() == DataType::DT_FLOAT ? FP32_BLOCK_NUM : FP16_BLOCK_NUM;
    int64_t firstXDim = static_cast<int64_t>(self->GetViewShape().GetDim(0));
    // 需要满足ARA单轴切分场景且切分轴大于等于VEC核数,低维A轴32字节对齐,输出R轴和低维A轴足够大才进入该性能优化分支
    // 不支持ARA场景中首维A超过四分之一VEC核数的情况
    if (selfColNum != sizeColNum || sliceAxis == selfDimNum - 1 || sliceAxis == MAX_DIM_NUM ||
        sizeColNum % numInOneBlock != 0 || (*size)[sliceAxis] < MAX_CORE_NUM ||
        (*size)[sliceAxis] * sizeColNum < MAX_UB_SIZE * MAX_CORE_NUM / dataByte ||
        firstXDim > MAX_CORE_NUM / DMA_CORE_NUMBER) {
        return false;
    }
    return true;
}

static bool IsSliceV2BothAlignLastDimSupport(const aclTensor *self, const aclIntArray *offsets,
    const aclIntArray *size)
{
    auto selfDimNum = self->GetViewShape().GetDimNum();
    int64_t selfRowNum = 1, sizeRowNum = 1;
    int64_t curOffset = 0, curSize = 0, curXDim = 0;
    for (size_t i = 0; i < selfDimNum - 1; ++i) {
        curOffset = (*offsets)[i], curSize = (*size)[i];
        curXDim = static_cast<int64_t>(self->GetViewShape().GetDim(i));
        // 处理offsets和size中负数的情况
        curOffset += curOffset < 0 ? curXDim : 0;
        curSize = curSize == -1 ? curXDim - curOffset : curSize;
        selfRowNum *= curXDim;
        sizeRowNum *= curSize;
    }
    auto offsetsLastDim = (*offsets)[selfDimNum - 1];
    auto sizeLastDim = (*size)[selfDimNum - 1];
    int64_t xLastDim = static_cast<int64_t>(self->GetViewShape().GetDim(selfDimNum - 1));
    offsetsLastDim += offsetsLastDim < 0 ? xLastDim : 0;
    sizeLastDim = sizeLastDim == -1 ? xLastDim - offsetsLastDim : sizeLastDim;
    auto dataByte = self->GetDataType() == DataType::DT_FLOAT ? FP32_BYTE : FP16_BYTE;
    auto numInOneBlock = self->GetDataType() == DataType::DT_FLOAT ? FP32_BLOCK_NUM : FP16_BLOCK_NUM;
    // 满足尾轴切分双对齐场景且输出维度积大于1048576,尾轴offset非64Btye对齐,输入尾轴小于等于192
    if (selfRowNum == sizeRowNum && sizeLastDim % numInOneBlock == 0 && xLastDim % numInOneBlock == 0 &&
        sizeRowNum * sizeLastDim > DMA_CORE_THRESHOLD && (offsetsLastDim * dataByte) % BYTE64 != 0 &&
        xLastDim <= LAST_DIM_THRESHOLD) {
        return true;
    }
    return false;
}

static bool IsSliceV2AiCoreSupport(const aclTensor *self, const aclIntArray *offsets, const aclIntArray *size)
{
    bool IsSupport = false;
    auto selfDimNum = self->GetViewShape().GetDimNum();
    auto offsetsLastDim = (*offsets)[selfDimNum - 1];
    auto sizeLastDim = (*size)[selfDimNum - 1];
    int64_t xLastDim = static_cast<int64_t>(self->GetViewShape().GetDim(selfDimNum - 1));
    offsetsLastDim += offsetsLastDim < 0 ? xLastDim : 0;
    sizeLastDim = sizeLastDim == -1 ? xLastDim - offsetsLastDim : sizeLastDim;
    if (offsetsLastDim < 0 || sizeLastDim < 0 || offsetsLastDim + sizeLastDim > xLastDim) {
        return false;
    }
    if (sizeLastDim < xLastDim) { // 尾轴切分看是否满足SliceV2BothAlignLastDim条件
        IsSupport = IsSliceV2BothAlignLastDimSupport(self, offsets, size);
    } else { // 非尾轴切分看是否满足SliceV2ARAFullLoad条件
        IsSupport = IsSliceV2ARAFullLoadSupport(self, offsets, size);
    }
    if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B ||
        op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910_93) {
        return (IsSupport && op::CheckType(self->GetDataType(), SLICEV2_AICORE_DTYPE_SUPPORT_LIST));
    }
    return false;
}

const aclTensor *SliceV2AiCore(const aclTensor *x, const aclTensor *y, const aclTensor *offsets,
    const aclTensor *size, aclOpExecutor *executor)
{
    L0_DFX(SliceV2AiCore, x, y, offsets, size);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(SliceV2, OP_INPUT(x, offsets, size), OP_OUTPUT(y));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
       "SliceV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return y;
}

static bool IsAiCoreSupport(const aclTensor* self)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_ASCEND910B);
    } else if (IsRegBase()) {
        return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_ASCEND910D);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_ASCEND910);
}

const aclTensor* SliceAiCore(
    const aclTensor* x, const aclTensor* y, const aclTensor* offset, const aclTensor* size, aclOpExecutor* executor)
{
    L0_DFX(SliceAiCore, x, y, offset, size);
    ADD_TO_LAUNCHER_LIST_AICORE(Slice, OP_INPUT(x, offset, size), OP_OUTPUT(y));
    return y;
}

const aclTensor* SliceAiCpu(
    const aclTensor* x, const aclTensor* y, const aclTensor* offset, const aclTensor* size, aclOpExecutor* executor)
{
    L0_DFX(SliceAiCpu, x, y, offset, size);
    static internal::AicpuTaskSpace space("Slice", ge::DEPEND_CONST_VALUE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        Slice, OP_ATTR_NAMES({"T", "Index"}), OP_INPUT(x, offset, size), OP_OUTPUT(y),
        OP_ATTR(x->GetDataType(), size->GetDataType()));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return y;
}

const aclTensor* Slice(
    const aclTensor* x, const aclTensor* y, const aclTensor* offset, const aclTensor* size, aclOpExecutor* executor)
{
    if (IsAiCoreSupport(x)) {
        return SliceAiCore(x, y, offset, size, executor);
    }
    return SliceAiCpu(x, y, offset, size, executor);
}

const aclTensor* Slice(const aclTensor* x, const aclIntArray* offsets, const aclIntArray* size, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(x->GetDataType(), x->GetStorageFormat(), x->GetOriginalFormat());
    auto offsetTensor = executor->ConvertToTensor(offsets, ToOpDataType(ACL_INT64));
    auto sizeTensor = executor->ConvertToTensor(size, ToOpDataType(ACL_INT64));
    INFER_SHAPE(Slice, OP_INPUT(x, offsetTensor, sizeTensor), OP_OUTPUT(out), OP_EMPTY_ARG);
    if (IsSliceV2AiCoreSupport(x, offsets, size)) {
        return SliceV2AiCore(x, out, offsetTensor, sizeTensor, executor);
    }
    return Slice(x, out, offsetTensor, sizeTensor, executor);
}

} // namespace l0op
