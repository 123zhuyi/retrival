/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "concat_v2_aicpu.h"

#include <complex>
#include <unordered_map>

using namespace std;

namespace {
const char* const ConcatV2 = "ConcatV2";
}

namespace aicpu {
uint32_t ConcatV2CpuKernel::CheckConcatV2Params(const CpuKernelContext& ctx)
{
    AttrValue* n_ptr = ctx.GetAttr("N");
    KERNEL_CHECK_NULLPTR(n_ptr, KERNEL_STATUS_PARAM_INVALID, "Get attr N failed.");
    n_ = n_ptr->GetInt();
    KERNEL_CHECK_FALSE(
        (n_ >= 1), KERNEL_STATUS_PARAM_INVALID, "Attr N must >= 1, but got attr N[%ld]", n_);
    uint32_t input_num = ctx.GetInputsSize();
    KERNEL_CHECK_FALSE(
        (static_cast<int64_t>(input_num) - 1 == n_), KERNEL_STATUS_PARAM_INVALID,
        "Input num must equal attr N[%ld] + 1, but got input num[%u]", n_, input_num);
    return KERNEL_STATUS_OK;
}

uint32_t ConcatV2CpuKernel::ParseConcatDim(const CpuKernelContext& ctx, int64_t& concat_dim) const
{
    auto concat_dim_ptr = ctx.Input(static_cast<uint32_t>(n_));
    KERNEL_CHECK_NULLPTR(concat_dim_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input concat_dim failed.");
    auto shape_ptr = concat_dim_ptr->GetTensorShape();
    KERNEL_CHECK_NULLPTR(shape_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input concat_dim shape failed.");
    int32_t dims = shape_ptr->GetDims();
    KERNEL_CHECK_FALSE(
        (dims == 0) || ((dims == 1) && (shape_ptr->NumElements() == 1)),
        KERNEL_STATUS_PARAM_INVALID, "Input concat_dim should be a scalar integer, but got rank[%d].", dims);
    auto dtype = concat_dim_ptr->GetDataType();
    auto data_raw = concat_dim_ptr->GetData();
    KERNEL_CHECK_NULLPTR(data_raw, KERNEL_STATUS_PARAM_INVALID, "Get input concat_dim data failed.");
    if (dtype == DT_INT32) {
        int32_t tmp = 0;
        KERNEL_CHECK_FALSE(
            memcpy_s(&tmp, sizeof(tmp), data_raw, sizeof(tmp)) == EOK,
            KERNEL_STATUS_PARAM_INVALID, "memcpy concat_dim(int32) failed.");
        concat_dim = static_cast<int64_t>(tmp);
        return KERNEL_STATUS_OK;
    }
    if (dtype == DT_INT64) {
        KERNEL_CHECK_FALSE(
            memcpy_s(&concat_dim, sizeof(concat_dim), data_raw, sizeof(concat_dim)) == EOK,
            KERNEL_STATUS_PARAM_INVALID, "memcpy concat_dim(int64) failed.");
        return KERNEL_STATUS_OK;
    }
    KERNEL_LOG_ERROR("Unsupported concat_dim data type: %d", dtype);
    return KERNEL_STATUS_PARAM_INVALID;
}

uint32_t ConcatV2CpuKernel::InitConcatV2Params(const CpuKernelContext& ctx)
{
    int64_t concat_dim = 0;
    uint32_t ret = ParseConcatDim(ctx, concat_dim);
    KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), ret, "ParseConcatDim failed.");
    auto input0_ptr = ctx.Input(0);
    KERNEL_CHECK_NULLPTR(input0_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input x0 failed.");
    auto input0_shape_ptr = input0_ptr->GetTensorShape();
    KERNEL_CHECK_NULLPTR(input0_shape_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input x0 shape failed.");
    input_dims_ = input0_shape_ptr->GetDims();
    data_type_ = input0_ptr->GetDataType();
    KERNEL_LOG_INFO("ConcatV2 init: data_type=%d, input_dims=%d.", data_type_, input_dims_);
    axis_ = concat_dim < 0 ? concat_dim + input_dims_ : concat_dim;
    KERNEL_CHECK_FALSE(
        (axis_ >= 0 && axis_ < input_dims_), KERNEL_STATUS_PARAM_INVALID,
        "Input concat_dim need in range[%d, %d), but got %ld.", -input_dims_, input_dims_, concat_dim);
    inputs_flat_dim0_ = 1;
    for (int32_t d = 0; d < axis_; ++d) {
        inputs_flat_dim0_ *= input0_shape_ptr->GetDimSize(d);
    }
    return KERNEL_STATUS_OK;
}

uint32_t ConcatV2CpuKernel::ValidateInputShape(
    Tensor* input_i_ptr, TensorShape* input0_shape_ptr, int64_t i) const
{
    auto shape_ptr = input_i_ptr->GetTensorShape();
    KERNEL_CHECK_NULLPTR(shape_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input x[%ld] shape failed.", i);
    int32_t dims = shape_ptr->GetDims();
    KERNEL_CHECK_FALSE(
        (dims == input_dims_), KERNEL_STATUS_PARAM_INVALID,
        "Ranks of inputs should match: shape[0]=%d vs. shape[%ld]=%d", input_dims_, i, dims);
    for (int32_t j = 0; j < input_dims_; ++j) {
        if (j == axis_) {
            continue;
        }
        int64_t dim_0j = input0_shape_ptr->GetDimSize(j);
        int64_t dim_ij = shape_ptr->GetDimSize(j);
        KERNEL_CHECK_FALSE(
            (dim_0j == dim_ij), KERNEL_STATUS_PARAM_INVALID,
            "Dim mismatch at axis %d: shape[0]=%ld vs. shape[%ld]=%ld", j, dim_0j, i, dim_ij);
    }
    return KERNEL_STATUS_OK;
}

uint32_t ConcatV2CpuKernel::CheckAndInitParams(const CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(this->CheckConcatV2Params(ctx), "CheckConcatV2Params failed.");
    KERNEL_HANDLE_ERROR(this->InitConcatV2Params(ctx), "InitConcatV2Params failed.");
    return KERNEL_STATUS_OK;
}

uint32_t ConcatV2CpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_LOG_INFO("%s start.", ConcatV2);
    uint32_t ret = CheckAndInitParams(ctx);
    KERNEL_CHECK_FALSE(
        (ret == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID, "CheckAndInitParams failed, ret=[%u].", ret);
    using ComputeFunc = uint32_t (ConcatV2CpuKernel::*)(CpuKernelContext&);
    static const std::unordered_map<DataType, ComputeFunc> calls_map = {
        {DT_FLOAT16, &ConcatV2CpuKernel::DoCompute<Eigen::half>},
        {DT_FLOAT, &ConcatV2CpuKernel::DoCompute<float>},
        {DT_INT8, &ConcatV2CpuKernel::DoCompute<int8_t>},
        {DT_INT16, &ConcatV2CpuKernel::DoCompute<int16_t>},
        {DT_INT32, &ConcatV2CpuKernel::DoCompute<int32_t>},
        {DT_INT64, &ConcatV2CpuKernel::DoCompute<int64_t>},
        {DT_UINT8, &ConcatV2CpuKernel::DoCompute<uint8_t>},
        {DT_UINT16, &ConcatV2CpuKernel::DoCompute<uint16_t>},
        {DT_UINT32, &ConcatV2CpuKernel::DoCompute<uint32_t>},
        {DT_UINT64, &ConcatV2CpuKernel::DoCompute<uint64_t>},
        {DT_BOOL, &ConcatV2CpuKernel::DoCompute<bool>},
        {DT_DOUBLE, &ConcatV2CpuKernel::DoCompute<double>},
        {DT_COMPLEX64, &ConcatV2CpuKernel::DoCompute<std::complex<float>>},
        {DT_COMPLEX128, &ConcatV2CpuKernel::DoCompute<std::complex<double>>},
        {DT_BFLOAT16, &ConcatV2CpuKernel::DoCompute<Eigen::bfloat16>},
    };
    auto iter = calls_map.find(data_type_);
    if (iter == calls_map.end()) {
        KERNEL_LOG_ERROR("Unsupported datatype[%d]", data_type_);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    ComputeFunc fn = iter->second;
    uint32_t result = (this->*fn)(ctx);
    if (result == KERNEL_STATUS_OK) {
        KERNEL_LOG_INFO("%s success.", ConcatV2);
    } else {
        KERNEL_LOG_ERROR("%s failed, result=[%u].", ConcatV2, result);
    }
    return result;
}

REGISTER_CPU_KERNEL(ConcatV2, ConcatV2CpuKernel);
} // namespace aicpu
