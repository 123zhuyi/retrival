/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <bitset>
#include <vector>

#include "aclnn_all.h"
#include "reduce_all.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/reshape.h"
#include "conversion/fill/op_api/fill.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;
using std::bitset;

static const aclIntArray* getAllDims(const aclTensor* self, aclOpExecutor* executor)
{
    auto input_shape = self->GetViewShape();
    const size_t input_dim_num = input_shape.GetDimNum();
    std::vector<int64_t> dims(input_dim_num);
    for (size_t idx = 0; idx < input_dim_num; idx++) {
        dims[idx] = idx;
    }
    return executor->AllocIntArray(dims.data(), input_dim_num);
}

static const std::initializer_list<DataType> Ascend910_self_dtype_support_list = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_UINT8,
    op::DataType::DT_INT8,  op::DataType::DT_INT16,   op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_BOOL,    op::DataType::DT_DOUBLE};

static const std::initializer_list<DataType> Ascend910B_self_dtype_support_list = {
    op::DataType::DT_FLOAT,  op::DataType::DT_FLOAT16, op::DataType::DT_UINT8, op::DataType::DT_INT8,
    op::DataType::DT_INT16,  op::DataType::DT_INT32,   op::DataType::DT_INT64, op::DataType::DT_BOOL,
    op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetSelfDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 || IsRegBase()) {
        return Ascend910B_self_dtype_support_list;
    }
    return Ascend910_self_dtype_support_list;
}

static const std::initializer_list<DataType> out_dtype_support_list = {op::DataType::DT_UINT8, op::DataType::DT_BOOL};

static inline uint64_t GetPosDim(int64_t dim, int64_t dimNum)
{
    if (dimNum <= 0) {
        dimNum = 1;
    }
    return dim >= 0 ? dim : dim + dimNum;
}

int64_t make_wrap_dim(int64_t dim, int64_t dim_post_expr)
{
    // this will make range [-1, 0]
    if (dim_post_expr <= 0) {
        dim_post_expr = 1;
    }
    if (dim < 0) {
        dim += dim_post_expr;
    }
    return dim;
}

static bitset<DIM_BITS_LEN> make_dim_mask(const op::FVector<int64_t, op::MAX_DIM_NUM>& dims, int64_t ndim)
{
    bitset<DIM_BITS_LEN> mask = bitset<DIM_BITS_LEN>();
    if (dims.empty()) {
        mask.flip();
    } else {
        for (int64_t dim : dims) {
            mask.set(make_wrap_dim(dim, ndim));
        }
    }
    return mask;
}

static op::FVector<int64_t, op::MAX_DIM_NUM> reduce_ops_npu_output_size(
    const aclTensor* self, const op::FVector<int64_t, op::MAX_DIM_NUM>& dim, bool keepdim)
{
    auto self_shape = self->GetViewShape();
    int64_t ndim = self->GetViewShape().GetDimNum();
    bitset<DIM_BITS_LEN> mask = make_dim_mask(dim, ndim);
    op::FVector<int64_t, op::MAX_DIM_NUM> shape;
    for (size_t i = 0; i < self_shape.GetDimNum(); i++) {
        shape.push_back(self_shape.GetDim(i));
    }

    for (int idx = shape.size() - 1; idx >= 0; idx--) {
        if (mask[idx]) {
            if (keepdim) {
                shape[idx] = 1;
            } else {
                shape.erase(shape.begin() + idx);
            }
        }
    }

    return shape;
}

/**
 * 计算输出tensor的shape信息
 *
 * @param self 输入tensor
 * @param dim 需要reduce的维度
 * @param keepdim 输出维度是否与输入维度保持一致
 * @return 输出tensor的shape
 */
static op::Shape output_shape(const aclTensor* self, const op::FVector<int64_t, op::MAX_DIM_NUM>& dim, bool keepdim)
{
    op::Shape outShape;
    auto dims = reduce_ops_npu_output_size(self, dim, keepdim);
    outShape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
        outShape.SetDim(i, dims.at(i));
    }
    return outShape;
}

static bool CheckNotNull(const aclTensor* self, const aclIntArray* dim, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(dim, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, GetSelfDtypeSupportList(), return false);

    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, out_dtype_support_list, return false);
    return true;
}

static bool CheckFormat(const aclTensor* self)
{
    if (op::IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW, self [%s]",
            ToString(self->GetStorageFormat()).GetString());
        return false;
    }

    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out, const aclIntArray* dim, bool keepdim)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);

    op::FVector<int64_t, op::MAX_DIM_NUM> dims;
    for (size_t idx = 0; idx < dim->Size(); idx++) {
        dims.emplace_back((*dim)[idx]);
    }
    auto outShape = output_shape(self, dims, keepdim);
    if (outShape == out->GetViewShape()) {
        return true;
    }
    OP_LOGE(
        ACLNN_ERR_PARAM_INVALID, "Expect out shape [%s], but got: [%s].", op::ToString(outShape).GetString(),
        op::ToString(out->GetViewShape()).GetString());
    return false;
}

/**
 * 检查dim参数是否超出self的维度范围
 *
 * @param self
 * @param dim
 * @return
 */
static bool CheckDim(const aclTensor* self, const aclIntArray* dim)
{
    auto input_shape = self->GetViewShape();
    int64_t input_dim_num = input_shape.GetDimNum();
    if (input_dim_num == 0) {
        input_dim_num = 1;
    }
    bitset<DIM_BITS_LEN> dimMask = bitset<DIM_BITS_LEN>();
    for (size_t idx = 0; idx < dim->Size(); idx++) {
        if ((*dim)[idx] < -(input_dim_num) || (*dim)[idx] >= input_dim_num) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Dimension out of range (expected to be in range of [-%ld, %ld], but got %ld)",
                input_dim_num, input_dim_num - 1, (*dim)[idx]);
            return false;
        }
        uint64_t index = GetPosDim((*dim)[idx], input_dim_num);
        if (dimMask[index]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim %lu appears multiple times in the list of dims.", index);
            return false;
        }

        dimMask.set(index);
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* dim, bool keepdim, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, dim, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查数据格式是否支持
    CHECK_RET(CheckFormat(self), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查dim参数是否在范围内
    CHECK_RET(CheckDim(self, dim), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, out, dim, keepdim), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static const aclTensor* GetTensorWithValueTrue(aclTensor* out, aclOpExecutor* executor)
{
    if (out->IsEmpty()) {
        return out;
    }
    aclScalar* scalar = executor->AllocScalar(1);
    auto valueTensor = executor->ConvertToTensor(scalar, out->GetDataType());
    op::FVector<int64_t, MAX_DIM_NUM> outputDims = op::ToShapeVector(out->GetViewShape());
    aclIntArray* dimArray = executor->AllocIntArray(outputDims.data(), outputDims.size());
    auto dimTensor = executor->ConvertToTensor(dimArray, op::DataType::DT_INT64);
    auto falseTensor = l0op::Fill(dimTensor, valueTensor, dimArray, executor);
    if (falseTensor == nullptr) {
        return nullptr;
    }
    auto viewCopyResult = l0op::ViewCopy(falseTensor, out, executor);
    return viewCopyResult;
}

aclnnStatus aclnnAllGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* dim, bool keepdim, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAll, DFX_IN(self, dim, keepdim), DFX_OUT(out));

    // 固定写法，参数检查
    auto ret = CheckParams(self, dim, keepdim, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空dim处理
    if (dim->Size() == 0) {
        dim = getAllDims(self, uniqueExecutor.get());
        CHECK_RET(dim != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 空Tensor处理
    if (self->IsEmpty()) {
        auto trueTensor = GetTensorWithValueTrue(out, uniqueExecutor.get());
        CHECK_RET(trueTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCasted = selfContiguous;
    if (!((selfContiguous->GetDataType() == op::DataType::DT_FLOAT16 || selfContiguous->GetDataType() == op::DataType::DT_BF16 || selfContiguous->GetDataType() == op::DataType::DT_FLOAT) &&
            (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B || GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 || GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510))) {
        // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
        selfCasted = l0op::Cast(selfContiguous, DataType::DT_BOOL, uniqueExecutor.get());
        CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用l0算子All进行计算
    auto allResult = l0op::ReduceAll(selfCasted, dim, keepdim, uniqueExecutor.get());
    CHECK_RET(allResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto allResultCasted = l0op::Cast(allResult, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(allResultCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(allResultCasted, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAll(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAll);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
