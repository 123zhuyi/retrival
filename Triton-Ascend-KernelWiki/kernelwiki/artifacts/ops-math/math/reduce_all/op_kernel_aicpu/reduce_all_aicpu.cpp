/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_all_aicpu.h"

#include <map>
#include <vector>

#include "cpu_kernel_utils.h"
#include "log.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kReduceAllInputNum = 1;
constexpr uint32_t kReduceAllOutputNum = 1;
const char *const kReduceAll = "ReduceAll";
}

namespace aicpu {
uint32_t ReduceAllCpuKernel::GenDataNoAxis(const CpuKernelContext &ctx) const
{
    auto x_data = reinterpret_cast<bool *>(ctx.Input(kFirstInputIndex)->GetData());
    auto y_data = reinterpret_cast<bool *>(ctx.Output(kFirstOutputIndex)->GetData());
    int64_t input_data_size = ctx.Input(kFirstInputIndex)->NumElements();
    bool output_y = true;
    for (int64_t i = 0; i < input_data_size; ++i) {
        output_y = output_y && x_data[i];
    }
    y_data[0] = output_y;
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ReduceAllCpuKernel::AxisCal(
    T axis, const std::vector<int64_t> &data_dims, int64_t &head_dim, int64_t &end_dim) const
{
    bool axis_appear = false;
    size_t data_dims_size = data_dims.size();
    for (size_t i = 0; i < data_dims_size; i++) {
        if (static_cast<T>(i) == axis) {
            axis_appear = true;
            continue;
        }
        if (axis_appear) {
            if (data_dims[i] != 0 && end_dim > (INT64_MAX / data_dims[i])) {
                KERNEL_LOG_ERROR("Product is overflow. multiplier 1: %ld. multiplier 2: %ld.", end_dim, data_dims[i]);
                return KERNEL_STATUS_PARAM_INVALID;
            }
            end_dim *= data_dims[i];
        } else {
            if (data_dims[i] != 0 && head_dim > (INT64_MAX / data_dims[i])) {
                KERNEL_LOG_ERROR(
                    "Product is overflow. multiplier 1: %ld. multiplier 2: %ld.", head_dim, data_dims[i]);
                return KERNEL_STATUS_PARAM_INVALID;
            }
            head_dim *= data_dims[i];
        }
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
std::vector<int64_t> ReduceAllCpuKernel::GetOutputShape(const std::vector<int64_t> &input_shape, const T &axis)
{
    std::vector<int64_t> output_shape;
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (static_cast<T>(i) == axis) {
            if (keep_dims_) {
                output_shape.push_back(1);
            }
            continue;
        }
        output_shape.push_back(input_shape[i]);
    }
    return output_shape;
}

template <typename T>
uint32_t ReduceAllCpuKernel::AxesRankCheckAndReverse(
    const CpuKernelContext &ctx, const T *axis_data, const int64_t &axes_num, std::map<T, int64_t> &axis_map,
    int32_t &rank)
{
    T axis_temp = 0;
    rank = static_cast<T>(rank);
    for (int64_t i = 0; i < axes_num; i++) {
        if (axis_data[i] < -rank || axis_data[i] > (rank - 1)) {
            KERNEL_LOG_ERROR(
                "[%s] the value of axes should be in [-%d, %d], axes is %ld", ctx.GetOpType().c_str(), rank, rank,
                static_cast<int64_t>(axis_data[i]));
            return KERNEL_STATUS_PARAM_INVALID;
        }
        if (axis_data[i] < 0) {
            axis_temp = axis_data[i] + rank;
        } else {
            axis_temp = axis_data[i];
        }
        if (axis_map.find(axis_temp) != axis_map.end()) {
            KERNEL_LOG_ERROR(
                "[%s] invalid reduction arguments: axes contains duplicate dimension: %ld", ctx.GetOpType().c_str(),
                static_cast<int64_t>(axis_temp));
            return KERNEL_STATUS_PARAM_INVALID;
        }
        axis_map.emplace(std::pair<T, int64_t>(axis_temp, i));
    }
    return KERNEL_STATUS_OK;
}

template <typename T, typename T2>
uint32_t ReduceAllCpuKernel::ReduceAllOneAxes(
    const T *input_data, std::vector<int64_t> &input_dims, T *output_data, const int64_t &output_num,
    std::vector<T2> &axes)
{
    if (axes_idx_ >= axes.size()) {
        for (int64_t i = 0; i < output_num; i++) {
            output_data[i] = input_data[i];
        }
        return KERNEL_STATUS_OK;
    }
    int64_t head_dim = 1;
    int64_t end_dim = 1;
    uint32_t ret = AxisCal<T2>(axes[axes_idx_], input_dims, head_dim, end_dim);
    if (ret != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    auto *output_data_temp = new (std::nothrow) T[head_dim * end_dim];
    KERNEL_CHECK_NULLPTR(output_data_temp, KERNEL_STATUS_INNER_ERROR, "apply memory failed.");
    bool tmp_x = true;
    bool tmp_y = true;
    auto axis_dim = input_dims[axes[axes_idx_]];
    for (int64_t i = 0; i < head_dim; ++i) {
        for (int64_t j = 0; j < end_dim; ++j) {
            tmp_x = input_data[i * end_dim * axis_dim + j];
            for (int64_t k = 1; k < axis_dim; ++k) {
                tmp_y = input_data[i * end_dim * axis_dim + j + k * end_dim];
                tmp_x = tmp_x && tmp_y;
            }
            output_data_temp[i * end_dim + j] = tmp_x;
        }
    }
    input_dims = GetOutputShape<T2>(input_dims, axes[axes_idx_]);
    ++axes_idx_;
    uint32_t result = ReduceAllOneAxes<T, T2>(output_data_temp, input_dims, output_data, output_num, axes);
    delete[] output_data_temp;
    return result;
}

template <typename T, typename T2>
uint32_t ReduceAllCpuKernel::ReduceAllCompute(const CpuKernelContext &ctx)
{
    axes_idx_ = 0;
    Tensor *x = ctx.Input(kFirstInputIndex);
    Tensor *axes = ctx.Input(kSecondInputIndex);
    Tensor *y = ctx.Output(kFirstInputIndex);

    auto *output_data = reinterpret_cast<T *>(y->GetData());
    auto *keep_dims = ctx.GetAttr("keep_dims");
    KERNEL_CHECK_NULLPTR(keep_dims, KERNEL_STATUS_PARAM_INVALID, "Get attr [keep_dims] failed.");
    keep_dims_ = keep_dims->GetBool();
    int64_t output_num = y->NumElements();

    if (x->GetDataSize() == 0) {
        KERNEL_LOG_INFO("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
        if (output_num > 0) {
            for (int64_t i = 0; i < output_num; ++i) {
                output_data[i] = true;
            }
        }
        return KERNEL_STATUS_OK;
    }

    if (axes == nullptr || axes->GetDataSize() == 0) {
        return GenDataNoAxis(ctx);
    }

    auto *input_data = reinterpret_cast<T *>(x->GetData());
    auto *axis_data = reinterpret_cast<T2 *>(axes->GetData());
    int64_t axes_num = axes->GetTensorShape()->NumElements();
    std::vector<int64_t> input_dims = x->GetTensorShape()->GetDimSizes();
    int32_t rank = x->GetTensorShape()->GetDims();
    std::map<T2, int64_t> axis_map;

    if (AxesRankCheckAndReverse<T2>(ctx, axis_data, axes_num, axis_map, rank) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    std::vector<T2> axes_data;
    for (auto iter = axis_map.rbegin(); iter != axis_map.rend(); iter++) {
        axes_data.push_back((*iter).first);
    }

    uint32_t res = ReduceAllOneAxes<T, T2>(input_data, input_dims, output_data, output_num, axes_data);
    if (res != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t ReduceAllCpuKernel::ReduceAllCheck(const CpuKernelContext &ctx) const
{
    auto *x = ctx.Input(kFirstInputIndex);
    auto *axes = ctx.Input(kSecondInputIndex);
    if (x != nullptr && x->GetData() != nullptr) {
        KERNEL_CHECK_FALSE(
            (x->GetDataType() == DT_BOOL), KERNEL_STATUS_PARAM_INVALID,
            "Data type of x is not support, x data type is [%u].", static_cast<uint32_t>(x->GetDataType()));
    }
    if (axes != nullptr && axes->GetData() != nullptr) {
        KERNEL_CHECK_FALSE(
            (axes->GetDataType() == DT_INT32 || axes->GetDataType() == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
            "Data type of axis is not support, axis data type is [%u].",
            static_cast<uint32_t>(axes->GetDataType()));
    }
    return KERNEL_STATUS_OK;
}

uint32_t ReduceAllCpuKernel::Compute(CpuKernelContext &ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kReduceAllInputNum, kReduceAllOutputNum),
        "[%s] check input and output failed.", kReduceAll);
    KERNEL_HANDLE_ERROR(ReduceAllCheck(ctx), "[%s] check params failed.", kReduceAll);

    Tensor *axes = ctx.Input(kSecondInputIndex);
    if (axes == nullptr || axes->GetDataSize() == 0) {
        return ReduceAllCompute<bool, int32_t>(ctx);
    }

    auto axes_data_type = axes->GetDataType();
    uint32_t ret = KERNEL_STATUS_PARAM_INVALID;
    switch (axes_data_type) {
        case DT_INT32:
            ret = ReduceAllCompute<bool, int32_t>(ctx);
            break;
        case DT_INT64:
            ret = ReduceAllCompute<bool, int64_t>(ctx);
            break;
        default:
            KERNEL_LOG_ERROR("Data type not support[%s].", DTypeStr(axes_data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }

    if (ret != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return ret;
}

REGISTER_CPU_KERNEL(kReduceAll, ReduceAllCpuKernel);
}  // namespace aicpu