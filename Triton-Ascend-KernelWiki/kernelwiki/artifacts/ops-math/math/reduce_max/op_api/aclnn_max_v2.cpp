/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_max_v2.h"
#include "reduce_max.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_errno.h"
#include <bitset>

using namespace op;
using std::bitset;

#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_MASK_LEN = 64;

// 算子支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_UINT8,
    op::DataType::DT_INT8,  op::DataType::DT_INT16,   op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_DOUBLE,  op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_UINT8, op::DataType::DT_INT8,
    op::DataType::DT_INT16, op::DataType::DT_INT32,   op::DataType::DT_INT64, op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static inline bool CheckNotNull(const aclTensor* self, const aclIntArray* dims, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(dims, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList()
{
    if ((GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
         GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) ||
        IsRegBase()) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    auto supportList = GetDtypeSupportList();
    // 检查self与out的数据类型是否一致
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    // 检查self的数据类型是否支持, out与self一致，不需要额外检查
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    return true;
}

static inline uint64_t GetPosDim(int64_t dim, int64_t dimNum)
{
    if (dimNum <= 0) {
        dimNum = 1;
    }
    return dim >= 0 ? dim : dim + dimNum;
}

static inline const aclIntArray* GetAllDims(
    const aclTensor* self, const bool noopWithEmptyDims, aclOpExecutor* executor)
{
    auto inputShape = self->GetViewShape();
    size_t inputDimNum = inputShape.GetDimNum();
    FVector<int64_t> dims;
    if (!noopWithEmptyDims) {
        for (size_t idx = 0; idx < inputDimNum; idx++) {
            dims.emplace_back(idx);
        }
    }
    return executor->AllocIntArray(dims.data(), dims.size());
}

static void reduce_maxInferShape(
    const op::Shape& selfShape, const aclIntArray* dims, bool keepDims, const bool noopWithEmptyDims,
    op::Shape& reduceShape)
{
    bitset<MAX_MASK_LEN> dimMask = bitset<MAX_MASK_LEN>();

    if (dims->Size() == 0) {
        if (noopWithEmptyDims) {
            for (size_t i = 0; i < selfShape.GetDimNum(); ++i) {
                reduceShape.AppendDim(selfShape.GetDim(i));
            }
            return;
        } else {
            if (keepDims) {
                for (size_t i = 0; i < selfShape.GetDimNum(); i++) {
                    reduceShape.AppendDim(1);
                }
            }
            return;
        }
    }

    for (size_t i = 0; i < dims->Size(); i++) {
        int64_t index = GetPosDim(dims->operator[](i), selfShape.GetDimNum());
        // 前序已检查， 此处如果dim不会重复
        dimMask.set(index);
    }

    for (size_t i = 0; i < selfShape.GetDimNum(); i++) {
        if (!dimMask[i]) {
            reduceShape.AppendDim(selfShape.GetDim(i));
        } else if (keepDims) {
            reduceShape.AppendDim(1);
        }
    }
}

static bool CheckDimValid(const aclTensor* self, const aclIntArray* dims)
{
    auto selfViewShape = self->GetViewShape();
    auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
    bool isScalar = false;
    // self为标量时，dim range [-1, 0]
    if (selfDimNum <= 0) {
        selfDimNum = 1;
        isScalar = true;
    }
    // dim为负时需要转正校验
    bitset<MAX_MASK_LEN> dimMask = bitset<MAX_MASK_LEN>();

    for (size_t i = 0; i < dims->Size(); i++) {
        if (dims->operator[](i) >= selfDimNum || dims->operator[](i) < (-selfDimNum)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Provided dims %ld must be in the range of [%ld, %ld].", dims->operator[](i),
                -selfDimNum, selfDimNum - 1);
            return false;
        }
        uint64_t index = GetPosDim(dims->operator[](i), selfDimNum);
        // 非标量reduce的dims不能为0
        if (!isScalar && selfViewShape.GetDim(index) == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected reducution dims %lu to have non-zero size.", index);
            return false;
        }
        // dims重复
        if (dimMask[index]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim %lu appears multiple times in the list of dims.", index);
            return false;
        }

        dimMask.set(index);
    }

    return true;
}

static bool CheckShape(
    const aclTensor* self, const aclIntArray* dims, const bool keepDims, const bool noopWithEmptyDims,
    const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
    op::Shape reduceShape;
    reduce_maxInferShape(self->GetViewShape(), dims, keepDims, noopWithEmptyDims, reduceShape);

    // out的shape必须满足Infer shape
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, reduceShape, return false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclIntArray* dims, const bool keepDims, const bool noopWithEmptyDims,
    const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, dims, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查reduce的轴是否超出self维度范围或者重复
    CHECK_RET(CheckDimValid(self, dims), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查out的shape是否满足reduce推导
    CHECK_RET(CheckShape(self, dims, keepDims, noopWithEmptyDims, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxV2GetWorkspaceSize(
    const aclTensor* self, const aclIntArray* dims, const bool keepDims, bool noopWithEmptyDims, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMaxV2, DFX_IN(self, dims, keepDims, noopWithEmptyDims), DFX_OUT(out));
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(self, dims, keepDims, noopWithEmptyDims, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 算子的空tensor处理
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 当输入tensor是0维时，直接将输入tensor作为输出返回
    if (self->GetViewShape().GetDimNum() == 0) {
        auto viewCopyResult = l0op::ViewCopy(self, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 空dims处理
    if (dims->Size() == 0) {
        dims = GetAllDims(self, noopWithEmptyDims, uniqueExecutor.get());
        CHECK_RET(dims != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入self的数据类型转换成目标数据类型, bool 转为float, 其余保持原类型
    op::DataType selfCastType =
        (self->GetDataType() == op::DataType::DT_BOOL) ? op::DataType::DT_FLOAT : self->GetDataType();

    auto selfCasted = l0op::Cast(selfContiguous, selfCastType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用max算子kernel
    auto maxResult = l0op::ReduceMax(selfCasted, dims, keepDims, noopWithEmptyDims, uniqueExecutor.get());
    CHECK_RET(maxResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将max算子的输出转换成目标数据类型，
    auto castMaxOut = l0op::Cast(maxResult, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castMaxOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castMaxOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMaxV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
