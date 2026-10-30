/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_mean.h"
#include "reduce_mean.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_errno.h"
#include "conversion/fill/op_api/fill.h"
#include "op_api/aclnn_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

/* Mean算子完整计算流程如下：
 *
 * self ——> Contiguous(workspace_0) ——> Cast(workspace_1) ——>
 * Mean(workspace2) ——> Cast(workspace_3) ——> ViewCopy ——> Out
 *
 */

static bool CheckNotNull(const aclTensor* self, const aclIntArray* dim, aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(dim, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

// 算子支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32,     op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,     op::DataType::DT_INT8,      op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,     op::DataType::DT_INT64,     op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,      op::DataType::DT_UINT8,     op::DataType::DT_DOUBLE,
    op::DataType::DT_BF16,  op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> EMPTY_INPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_BF16,  op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> NON_CONTIOUS_SUPPORT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* self, aclDataType dtype, aclTensor* out)
{
    // 检查self的数据类型是否支持
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 检查dtype指定的数据类型是否支持
    if (!CheckType(op::ToOpDataType(dtype), supportList)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "type %s should be in dtype support list [%s].",
            op::ToString(op::ToOpDataType(dtype)).GetString(), op::ToString(supportList).GetString());
        return false;
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

    return true;
}

static bool CheckDtypeValidONNX(const aclTensor* self, aclTensor* out)
{
    // 检查self的数据类型是否支持
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

    return true;
}

static bool CheckPromoteType(const aclTensor* self, aclDataType dtype, aclTensor* out)
{
    // 检查self能否转换为指定的dtype
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), op::ToOpDataType(dtype), return false);
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(op::ToOpDataType(dtype), out->GetDataType(), return false);

    return true;
}

static bool CheckDimValid(const aclTensor* self, const aclIntArray* dim)
{
    auto selfViewShape = self->GetViewShape();
    auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
    // self为标量时，dim range [-1, 0]
    if (selfDimNum <= 0) {
        selfDimNum = 1;
    }
    // 获取dim元素
    for (size_t i = 0; i < dim->Size(); i++) {
        if (dim->operator[](i) >= selfDimNum || dim->operator[](i) < (-selfDimNum)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "provided dim %ld must be in the range of [%ld, %ld].", dim->operator[](i),
                -selfDimNum, selfDimNum - 1);
            return false;
        }
    }
    // dim值不能重复
    uint64_t dimMask[64] = {0};
    for (size_t i = 0; i < dim->Size(); i++) {
        int64_t realDim = dim->operator[](i);
        if (realDim < 0) {
            realDim = selfDimNum + realDim;
        }
        if (dimMask[realDim] == 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "provided dim %ld is repeated.", dim->operator[](i));
            return false;
        } else {
            dimMask[realDim] = 1;
        }
    }
    return true;
}

constexpr size_t MAX_DIM_LEN = 8;

static bool CheckShape(const aclTensor* self, aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* dim, aclDataType dtype, aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, dim, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, dtype, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self能否做dtype指定的数据类型推导以及推导的数据类型能否转换为输出数据类型
    CHECK_RET(CheckPromoteType(self, dtype, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查reduce的轴是否超出self维度范围
    CHECK_RET(CheckDimValid(self, dim), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查out的shape是否满足reduce推导  是否考虑兼容：shape不满足也可计算
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamsONNX(const aclTensor* self, const aclIntArray* dim, aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, dim, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValidONNX(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查reduce的轴是否超出self维度范围
    CHECK_RET(CheckDimValid(self, dim), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查out的shape是否满足reduce推导  是否考虑兼容：shape不满足也可计算
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus FillScalar(aclTensor* out, float val, aclOpExecutor* executor)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(out, EMPTY_INPUT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    FVector<int64_t> shape;
    size_t dimNum = out->GetViewShape().GetDimNum();
    for (size_t idx = 0; idx < dimNum; idx++) {
        int64_t tmpVal = out->GetViewShape().GetDim(idx);
        shape.push_back(tmpVal);
    }
    auto dims = executor->ConvertToTensor(shape.data(), shape.size(), DataType::DT_INT64);
    CHECK_RET(dims != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto shapeArray = executor->AllocIntArray(shape.data(), shape.size());
    CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

    FVector<float> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
    CHECK_RET(valTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static bool IsNonContiguousSupport(const aclTensor* self, DataType dtype, const aclIntArray* dim)
{
    if (!IsRegBase()) {
        return false;
    }
    if (!CheckType(self->GetDataType(), NON_CONTIOUS_SUPPORT_DTYPE_SUPPORT_LIST) || self->GetDataType() != dtype) {
        return false;
    }
    if (!op::IsReduceNonContiguousSupport(self, dim)) {
        return false;
    }
    return true;
}

static void CheckFormat(const aclTensor* self) {
    ge::Format selfStorageFormat = self->GetStorageFormat();
    if (selfStorageFormat == ge::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("aclnnMean doesn't support format NZ.");
    }
}

aclnnStatus aclnnMeanGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* dim, bool keepDim, aclDataType dtype, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMean, DFX_IN(self, dim, keepDim, dtype), DFX_OUT(out));
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(self, dim, dtype, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查self的format是否支持
    CheckFormat(self);
        
    // 算子的空tensor处理
    if (self->IsEmpty()) {
        // 空tensor填充NAN
        ret = FillScalar(out, NAN, uniqueExecutor.get());
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ret;
    }

    if (IsNonContiguousSupport(self, op::ToOpDataType(dtype), dim)) {
        OP_LOGD("Enter NonContigous");
        auto selfContiguous = uniqueExecutor.get()->CreateView(
            self, self->GetViewShape(), self->GetStorageShape(), self->GetViewStrides(), self->GetViewOffset());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto meanOpOut = l0op::ReduceMean(selfContiguous, dim, keepDim, uniqueExecutor.get());
        CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(CheckShapeAndScalarSame(meanOpOut, out), ACLNN_ERR_PARAM_INVALID);
        auto castMeanOut = l0op::Cast(meanOpOut, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castMeanOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(castMeanOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // 将输入self转换成连续的tensor
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将输入self的数据类型转换成目标数据类型,
        auto selfCasted = l0op::Cast(selfContiguous, op::ToOpDataType(dtype), uniqueExecutor.get());
        CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 调用mean算子kernel
        auto meanOpOut = l0op::ReduceMean(selfCasted, dim, keepDim, uniqueExecutor.get());
        CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(CheckShapeAndScalarSame(meanOpOut, out), ACLNN_ERR_PARAM_INVALID);

        // 将mean算子的输出转换成目标数据类型，
        auto castMeanOut = l0op::Cast(meanOpOut, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castMeanOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将计算结果拷贝到输出out上，out可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(castMeanOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMean(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMean);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnMeanV2GetWorkspaceSize(
    const aclTensor* self, const aclIntArray* dim, bool keepDim, bool noopWithEmptyAxes, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMeanV2, DFX_IN(self, dim, keepDim, noopWithEmptyAxes), DFX_OUT(out));
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParamsONNX(self, dim, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
        
    // 检查self的format是否支持
    CheckFormat(self);
        
    // 算子的空tensor处理
    if (self->IsEmpty()) {
        // 空tensor填充NAN
        ret = FillScalar(out, NAN, uniqueExecutor.get());
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ret;
    }

    uint64_t dimSize = dim->Size();
    if (dimSize == 0 && noopWithEmptyAxes == true) {
        CHECK_RET(CheckShapeAndScalarSame(self, out), ACLNN_ERR_PARAM_INVALID);
        auto result = l0op::ViewCopy(self, out, uniqueExecutor.get());
        CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    if (IsNonContiguousSupport(self, out->GetDataType(), dim)) {
        OP_LOGD("Enter NonContigous");
        auto selfContiguous = uniqueExecutor.get()->CreateView(
            self, self->GetViewShape(), self->GetStorageShape(), self->GetViewStrides(), self->GetViewOffset());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto meanOpOut = l0op::ReduceMean(selfContiguous, dim, keepDim, noopWithEmptyAxes, uniqueExecutor.get());
        CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(CheckShapeAndScalarSame(meanOpOut, out), ACLNN_ERR_PARAM_INVALID);
        auto castMeanOut = l0op::Cast(meanOpOut, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castMeanOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(castMeanOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        OP_LOGD("Enter Contigous");
        // 将输入self转换成连续的tensor
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 调用mean算子kernel
        auto meanOpOut = l0op::ReduceMean(selfContiguous, dim, keepDim, noopWithEmptyAxes, uniqueExecutor.get());
        CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(CheckShapeAndScalarSame(meanOpOut, out), ACLNN_ERR_PARAM_INVALID);

        // 将mean算子的输出转换成目标数据类型，
        auto castMeanOut = l0op::Cast(meanOpOut, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castMeanOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将计算结果拷贝到输出out上，out可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(castMeanOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMeanV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMeanV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif