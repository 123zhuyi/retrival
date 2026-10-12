/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matrix_diag_part_v3_aicpu.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <tuple>
#include <vector>

#include "cpu_kernel_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *const kMatrixDiagPartV3 = "MatrixDiagPartV3";
constexpr int64_t kZero = 0;
constexpr int64_t kDimSize = 2;
constexpr int64_t kInputDimSize = 2;
constexpr int64_t kInputNum = 3;
constexpr int64_t kOutputNum = 1;
constexpr int64_t kDiagIndexInputIndex = 1;
constexpr int64_t kPaddingValueInputIndex = 2;
constexpr int64_t kTwoNum = 2;
} // namespace

namespace aicpu {
std::pair<int64_t, int64_t> MatrixDiagPartV3CpuKernel::ComputeDiagLenAndContentOffset(int64_t diag_index,
                                                                                       int64_t max_diag_len) const
{
    const bool left_align = (diag_index >= 0 && left_align_superdiagonal_) ||
                            (diag_index <= 0 && left_align_subdiagonal_);
    const int64_t diag_len = std::min(num_rows_ + std::min(kZero, diag_index), num_cols_ - std::max(kZero, diag_index));
    const int64_t content_offset = left_align ? 0 : (max_diag_len - diag_len);
    return {diag_len, content_offset};
}

template <typename T>
void MatrixDiagPartV3CpuKernel::PadOutput(T *output_data, int64_t output_base_index, int64_t content_offset,
                                          int64_t diag_len, int64_t max_diag_len, T padding_value) const
{
    const bool left_align = (content_offset == 0);
    const int64_t padding_start = left_align ? diag_len : 0;
    const int64_t padding_end = left_align ? max_diag_len : content_offset;
    for (int64_t n = padding_start; n < padding_end; ++n) {
        output_data[output_base_index + n] = padding_value;
    }
}

template <typename T>
void MatrixDiagPartV3CpuKernel::ExtractDiagonal(T *output_data, int64_t output_base_index, const T *input_data,
                                                int64_t batch, int64_t diag_index, int64_t max_diag_len,
                                                T padding_value) const
{
    int64_t content_offset = 0;
    int64_t diag_len = 0;
    const int64_t y_offset = std::max(kZero, -diag_index);
    const int64_t x_offset = std::max(kZero, diag_index);
    std::tie(diag_len, content_offset) = ComputeDiagLenAndContentOffset(diag_index, max_diag_len);
    const size_t batch_base_index = static_cast<size_t>(batch) * static_cast<size_t>(num_rows_ * num_cols_);
    for (int64_t n = 0; n < diag_len; ++n) {
        output_data[output_base_index + content_offset + n] =
            input_data[batch_base_index + static_cast<size_t>((n + y_offset) * num_cols_ + n + x_offset)];
    }
    PadOutput(output_data, output_base_index, content_offset, diag_len, max_diag_len, padding_value);
}

KernelStatus MatrixDiagPartV3CpuKernel::CheckParam(const CpuKernelContext &ctx)
{
    Tensor *input_tensor = ctx.Input(0);
    Tensor *k = ctx.Input(kDiagIndexInputIndex);
    Tensor *padding_value = ctx.Input(kPaddingValueInputIndex);

    KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "[%s] get output data failed.",
                         kMatrixDiagPartV3);
    KERNEL_CHECK_NULLPTR(padding_value->GetData(), KERNEL_STATUS_PARAM_INVALID,
                         "[%s] get padding_value data failed.", kMatrixDiagPartV3);
    KERNEL_CHECK_NULLPTR(k, KERNEL_STATUS_PARAM_INVALID, "[%s] get k data failed.", kMatrixDiagPartV3);

    auto k_shape = k->GetTensorShape();
    std::vector<int64_t> k_dims = k_shape->GetDimSizes();
    KERNEL_CHECK_FALSE(k_dims.size() <= kDimSize, KERNEL_STATUS_PARAM_INVALID, "%s dims size [%zu] must <= 2.",
                       kMatrixDiagPartV3, k_dims.size());

    std::string align = "RIGHT_LEFT";
    AttrValue *attr_align = ctx.GetAttr("align");
    if (attr_align != nullptr) {
        align = attr_align->GetString();
    }
    left_align_superdiagonal_ = (align == "LEFT_LEFT" || align == "LEFT_RIGHT");
    left_align_subdiagonal_ = (align == "LEFT_LEFT" || align == "RIGHT_LEFT");
    KERNEL_CHECK_FALSE((align == "RIGHT_LEFT" || align == "LEFT_LEFT" || align == "RIGHT_RIGHT" ||
                        align == "LEFT_RIGHT"),
                       KERNEL_STATUS_PARAM_INVALID,
                       "align must be one of RIGHT_LEFT,LEFT_LEFT,RIGHT_RIGHT,LEFT_RIGHT.");

    auto input_shape = input_tensor->GetTensorShape();
    std::vector<int64_t> input_dims = input_shape->GetDimSizes();
    KERNEL_CHECK_FALSE(input_dims.size() >= kInputDimSize, KERNEL_STATUS_PARAM_INVALID,
                       "input dims must >=2 while %zu", input_dims.size());
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MatrixDiagPartV3CpuKernel::MultiProcessFunc(const CpuKernelContext &ctx, int64_t upper_diag_index,
                                                     int64_t max_diag_len)
{
    auto input_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
    auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
    auto padding_value = reinterpret_cast<T *>(ctx.Input(kPaddingValueInputIndex)->GetData());
    const int64_t output_elements_in_batch = num_diags_ * max_diag_len;
    uint32_t min_core_num = 1;
    int64_t max_core_num = static_cast<int64_t>(std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx)));
    if (max_core_num > num_array_) {
        max_core_num = num_array_;
    }

    auto shard = [&](size_t start, size_t end) {
        int64_t output_base_index = static_cast<int64_t>(start) * output_elements_in_batch;
        for (size_t batch = start; batch < end; ++batch) {
            for (int64_t m = 0; m < num_diags_; ++m) {
                const int64_t diag_index = upper_diag_index - m;
                ExtractDiagonal(output_data, output_base_index, input_data, static_cast<int64_t>(batch), diag_index,
                                max_diag_len, padding_value[0]);
                output_base_index += max_diag_len;
            }
        }
    };

    if (max_core_num == 0) {
        KERNEL_LOG_ERROR("max_core_num could not be 0.");
        return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }
    return CpuKernelUtils::ParallelFor(ctx, num_array_, num_array_ / max_core_num, shard);
}

template <typename T>
uint32_t MatrixDiagPartV3CpuKernel::SingleProcessFunc(const CpuKernelContext &ctx, int64_t upper_diag_index,
                                                      int64_t max_diag_len)
{
    auto input_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
    auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
    auto padding_value = reinterpret_cast<T *>(ctx.Input(kPaddingValueInputIndex)->GetData());
    int64_t output_base_index = 0;
    for (int64_t batch = 0; batch < num_array_; ++batch) {
        for (int64_t m = 0; m < num_diags_; ++m) {
            const int64_t diag_index = upper_diag_index - m;
            ExtractDiagonal(output_data, output_base_index, input_data, batch, diag_index, max_diag_len,
                            padding_value[0]);
            output_base_index += max_diag_len;
        }
    }
    return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

template <typename T>
uint32_t MatrixDiagPartV3CpuKernel::DoCompute(const CpuKernelContext &ctx)
{
    Tensor *input = ctx.Input(0);
    Tensor *k = ctx.Input(kDiagIndexInputIndex);
    auto k_data = reinterpret_cast<int32_t *>(k->GetData());

    int64_t lower_diag_index = k_data[0];
    int64_t upper_diag_index = k_data[0];
    if (k->NumElements() == kTwoNum) {
        upper_diag_index = k_data[1];
    }
    KERNEL_CHECK_FALSE(lower_diag_index <= upper_diag_index, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID),
                       "k[0] must not be larger than k[1].");

    auto input_shape = input->GetTensorShape();
    const int64_t rank = input_shape->GetDims();
    if (rank < kTwoNum) {
        KERNEL_LOG_ERROR("input dims must >=2");
        return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }
    num_rows_ = input_shape->GetDimSize(static_cast<int32_t>(rank - kTwoNum));
    num_cols_ = input_shape->GetDimSize(static_cast<int32_t>(rank - 1));
    num_array_ = input_shape->NumElements() / (num_rows_ * num_cols_);
    num_diags_ = (upper_diag_index - lower_diag_index) + 1;
    const int64_t max_diag_len =
        std::min(num_rows_ + std::min(upper_diag_index, kZero), num_cols_ - std::max(lower_diag_index, kZero));
    const int64_t output_elements_in_batch = num_diags_ * max_diag_len;
    const int64_t data_num = num_array_ * output_elements_in_batch;
    const int64_t kParallelArrayNumSameShape = 2048;
    if (data_num >= kParallelArrayNumSameShape) {
        return MultiProcessFunc<T>(ctx, upper_diag_index, max_diag_len);
    }
    return SingleProcessFunc<T>(ctx, upper_diag_index, max_diag_len);
}

uint32_t MatrixDiagPartV3CpuKernel::Compute(CpuKernelContext &ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                        "MatrixDiagPartV3 check input and output number failed.");
    KERNEL_CHECK_FALSE(CheckParam(ctx) == KERNEL_STATUS_OK, static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID),
                       "CheckParam failed.");

    switch (static_cast<DataType>(ctx.Input(0)->GetDataType())) {
        case DT_INT8:
            return DoCompute<int8_t>(ctx);
        case DT_INT16:
            return DoCompute<int16_t>(ctx);
        case DT_INT32:
            return DoCompute<int32_t>(ctx);
        case DT_INT64:
            return DoCompute<int64_t>(ctx);
        case DT_UINT8:
            return DoCompute<uint8_t>(ctx);
        case DT_FLOAT:
            return DoCompute<std::float_t>(ctx);
        case DT_DOUBLE:
            return DoCompute<std::double_t>(ctx);
        case DT_UINT16:
            return DoCompute<uint16_t>(ctx);
        case DT_UINT32:
            return DoCompute<uint32_t>(ctx);
        case DT_UINT64:
            return DoCompute<uint64_t>(ctx);
        case DT_COMPLEX64:
            return DoCompute<std::complex<std::float_t>>(ctx);
        case DT_COMPLEX128:
            return DoCompute<std::complex<std::double_t>>(ctx);
        case DT_FLOAT16:
            return DoCompute<Eigen::half>(ctx);
        default:
            KERNEL_LOG_ERROR("MatrixDiagPartV3 does not support data type [%s].",
                             DTypeStr(static_cast<DataType>(ctx.Input(0)->GetDataType())).c_str());
            return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }
}

REGISTER_CPU_KERNEL(kMatrixDiagPartV3, MatrixDiagPartV3CpuKernel);
} // namespace aicpu
