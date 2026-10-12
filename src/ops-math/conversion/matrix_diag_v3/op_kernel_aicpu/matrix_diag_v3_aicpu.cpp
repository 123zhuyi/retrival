/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matrix_diag_v3_aicpu.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <tuple>

#include "cpu_kernel_utils.h"
#include "log.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "utils/status.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 5;
const uint32_t kDiagIndexInputIndex = 1;
const uint32_t kNumRowsInputIndex = 2;
const uint32_t kNumColsInputIndex = 3;
const uint32_t kPaddingValueInputIndex = 4;
const int64_t kConstTwo = 2;
const char *const kMatrixDiagV3 = "MatrixDiagV3";

#define MATRIX_DIAG_V3_COMPUTE_CASE(DTYPE, TYPE, CTX)                     \
    case (DTYPE): {                                                       \
        KernelStatus result = DoCompute<TYPE>(CTX);                       \
        if (result != KERNEL_STATUS_OK) {                                 \
            KERNEL_LOG_ERROR("MatrixDiagV3 kernel compute failed.");      \
            return static_cast<uint32_t>(result);                         \
        }                                                                 \
        break;                                                            \
    }
} // namespace

namespace aicpu {
KernelStatus MatrixDiagV3CpuKernel::CheckParam(const CpuKernelContext &ctx)
{
    std::string align = "RIGHT_LEFT";
    AttrValue *attr_align = ctx.GetAttr("align");
    if (attr_align != nullptr) {
        align = attr_align->GetString();
    }

    KERNEL_CHECK_FALSE((align == "RIGHT_LEFT" || align == "RIGHT_RIGHT" || align == "LEFT_LEFT" ||
                        align == "LEFT_RIGHT"),
                       KERNEL_STATUS_PARAM_INVALID,
                       "Attr 'align' of 'MatrixDiagV3' is not in: 'LEFT_RIGHT', 'RIGHT_LEFT', 'LEFT_LEFT', "
                       "'RIGHT_RIGHT'.");
    left_align_superdiagonal_ = align == "LEFT_LEFT" || align == "LEFT_RIGHT";
    left_align_subdiagonal_ = align == "LEFT_LEFT" || align == "RIGHT_LEFT";

    const auto diagonal_data_type = ctx.Input(0)->GetDataType();
    const auto output_data_type = ctx.Output(0)->GetDataType();
    KERNEL_CHECK_FALSE(diagonal_data_type == output_data_type, KERNEL_STATUS_PARAM_INVALID,
                       "The data type of input0 [%s] need be same with output0 [%s].",
                       DTypeStr(diagonal_data_type).c_str(), DTypeStr(output_data_type).c_str());

    KERNEL_CHECK_FALSE(GetDiagIndex(ctx) == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID, "GetDiagIndex failed.");
    const int64_t padding_value_num = ctx.Input(kPaddingValueInputIndex)->NumElements();
    KERNEL_CHECK_FALSE(padding_value_num == 1, KERNEL_STATUS_PARAM_INVALID,
                       "padding_value must have only one element, received [%ld] elements.", padding_value_num);

    const int32_t diag_rank = ctx.Input(0)->GetTensorShape()->GetDims();
    max_diag_len_ = ctx.Input(0)->GetTensorShape()->GetDimSize(diag_rank - 1);
    const int64_t expected_num_diags = static_cast<int64_t>(upper_diag_index_ - lower_diag_index_) + 1;
    if (expected_num_diags > 1 && diag_rank < kConstTwo) {
        KERNEL_LOG_ERROR("diagonal rank is 1 but k implies [%ld] diagonals.", expected_num_diags);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (expected_num_diags > 1) {
        const int64_t actual_num_diags = ctx.Input(0)->GetTensorShape()->GetDimSize(diag_rank - kConstTwo);
        KERNEL_CHECK_FALSE(expected_num_diags == actual_num_diags, KERNEL_STATUS_PARAM_INVALID,
                           "k parameter implies [%ld] diagonals, but diagonal data contains [%ld] diagonals.",
                           expected_num_diags, actual_num_diags);
    }
    min_num_rows_ = static_cast<int32_t>(max_diag_len_ - std::min(upper_diag_index_, 0));
    min_num_cols_ = static_cast<int32_t>(max_diag_len_ + std::max(lower_diag_index_, 0));
    if (num_rows_ != -1 && num_rows_ < min_num_rows_) {
        KERNEL_LOG_ERROR("The number of rows is too small.");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (num_cols_ != -1 && num_cols_ < min_num_cols_) {
        KERNEL_LOG_ERROR("The number of columns is too small.");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    KERNEL_CHECK_FALSE(AdjustRowsAndCols() == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID,
                       "AdjustRowsAndCols failed.");
    return KERNEL_STATUS_OK;
}

uint32_t MatrixDiagV3CpuKernel::Compute(CpuKernelContext &ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                        "MatrixDiagV3 check input and output number failed.");
    KERNEL_CHECK_FALSE(CheckParam(ctx) == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID, "CheckParam failed.");
    switch (ctx.Input(0)->GetDataType()) {
        MATRIX_DIAG_V3_COMPUTE_CASE(DT_BOOL, bool, ctx)
        MATRIX_DIAG_V3_COMPUTE_CASE(DT_INT8, int8_t, ctx)
        MATRIX_DIAG_V3_COMPUTE_CASE(DT_INT16, int16_t, ctx)
        MATRIX_DIAG_V3_COMPUTE_CASE(DT_INT32, int32_t, ctx)
        MATRIX_DIAG_V3_COMPUTE_CASE(DT_INT64, int64_t, ctx)
        MATRIX_DIAG_V3_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
        MATRIX_DIAG_V3_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
        MATRIX_DIAG_V3_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
        MATRIX_DIAG_V3_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
        MATRIX_DIAG_V3_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
        MATRIX_DIAG_V3_COMPUTE_CASE(DT_FLOAT, std::float_t, ctx)
        MATRIX_DIAG_V3_COMPUTE_CASE(DT_DOUBLE, std::double_t, ctx)
        MATRIX_DIAG_V3_COMPUTE_CASE(DT_COMPLEX128, std::complex<std::double_t>, ctx)
        MATRIX_DIAG_V3_COMPUTE_CASE(DT_COMPLEX64, std::complex<std::float_t>, ctx)
        default:
            KERNEL_LOG_ERROR("MatrixDiagV3 kernel data type [%s] not support.",
                             DTypeStr(ctx.Input(0)->GetDataType()).c_str());
            return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }
    return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

std::pair<int, int> MatrixDiagV3CpuKernel::ComputeDiagLenAndContentOffset(int diag_index) const
{
    const bool left_align = (diag_index >= 0 && left_align_superdiagonal_) ||
                            (diag_index <= 0 && left_align_subdiagonal_);
    const int diag_len = std::min(num_rows_ + std::min(0, diag_index), num_cols_ - std::max(0, diag_index));
    const int content_offset = left_align ? 0 : (static_cast<int>(max_diag_len_) - diag_len);
    return {diag_len, content_offset};
}

KernelStatus MatrixDiagV3CpuKernel::GetDiagIndex(const CpuKernelContext &ctx)
{
    auto *k_tensor = ctx.Input(kDiagIndexInputIndex);
    auto *num_rows_tensor = ctx.Input(kNumRowsInputIndex);
    auto *num_cols_tensor = ctx.Input(kNumColsInputIndex);

    auto *k_data = reinterpret_cast<int32_t *>(k_tensor->GetData());
    lower_diag_index_ = k_data[0];
    upper_diag_index_ = lower_diag_index_;
    const int64_t k_num = k_tensor->NumElements();
    if (k_num <= 0 || k_num > kConstTwo) {
        KERNEL_LOG_ERROR("k must have only one or two elements, received [%ld] elements.", k_num);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (k_num == kConstTwo) {
        upper_diag_index_ = k_data[1];
    }
    KERNEL_CHECK_FALSE(lower_diag_index_ <= upper_diag_index_, KERNEL_STATUS_PARAM_INVALID,
                       "lower_diag_index must be smaller than upper_diag_index, received [%d] is larger than [%d].",
                       lower_diag_index_, upper_diag_index_);

    KERNEL_CHECK_FALSE(num_rows_tensor->NumElements() == 1, KERNEL_STATUS_PARAM_INVALID,
                       "num_rows must have only one element, received [%ld] elements.", num_rows_tensor->NumElements());
    KERNEL_CHECK_FALSE(num_cols_tensor->NumElements() == 1, KERNEL_STATUS_PARAM_INVALID,
                       "num_cols must have only one element, received [%ld] elements.", num_cols_tensor->NumElements());
    num_rows_ = reinterpret_cast<int32_t *>(num_rows_tensor->GetData())[0];
    num_cols_ = reinterpret_cast<int32_t *>(num_cols_tensor->GetData())[0];
    return KERNEL_STATUS_OK;
}

KernelStatus MatrixDiagV3CpuKernel::AdjustRowsAndCols()
{
    if (num_rows_ == -1 && num_cols_ == -1) {
        num_rows_ = std::max(min_num_rows_, min_num_cols_);
        num_cols_ = num_rows_;
    } else if (num_rows_ == -1) {
        num_rows_ = min_num_rows_;
    } else if (num_cols_ == -1) {
        num_cols_ = min_num_cols_;
    }
    if (num_rows_ != min_num_rows_ && num_cols_ != min_num_cols_) {
        KERNEL_LOG_ERROR("The number of rows or columns is not consistent with the specified d_lower, d_upper, and diagonal.");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
KernelStatus MatrixDiagV3CpuKernel::DoCompute(const CpuKernelContext &ctx)
{
    auto *padding_value_data = reinterpret_cast<T *>(ctx.Input(kPaddingValueInputIndex)->GetData());
    const T padding_value = padding_value_data[0];
    const int64_t num_diags = static_cast<int64_t>(upper_diag_index_ - lower_diag_index_) + 1;
    const int64_t diag_elements_in_batch = num_diags * max_diag_len_;
    int64_t diag_batch_base_index = 0;
    const int64_t num_elements = ctx.Output(0)->NumElements();
    const int64_t num_batches = num_elements / static_cast<int64_t>(num_rows_ * num_cols_);
    for (int64_t batch = 0; batch < num_batches; ++batch) {
        SetResult<T>(ctx, padding_value, diag_batch_base_index);
        diag_batch_base_index += diag_elements_in_batch;
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
void MatrixDiagV3CpuKernel::SetResult(const CpuKernelContext &ctx, T padding_value, int64_t diag_batch_base_index)
{
    auto *output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
    auto *diagonal_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
    for (int64_t i = 0; i < num_rows_; ++i) {
        for (int64_t j = 0; j < num_cols_; ++j) {
            const int diag_index = static_cast<int>(j - i);
            const int diag_index_in_input = upper_diag_index_ - diag_index;
            int diag_len = 0;
            int content_offset = 0;
            std::tie(diag_len, content_offset) = ComputeDiagLenAndContentOffset(diag_index);
            const int index_in_the_diagonal = (j - std::max<int64_t>(diag_index, 0)) + content_offset;
            if (lower_diag_index_ <= diag_index && diag_index <= upper_diag_index_) {
                output_data[elem_] =
                    diagonal_data[diag_batch_base_index + diag_index_in_input * max_diag_len_ + index_in_the_diagonal];
            } else {
                output_data[elem_] = padding_value;
            }
            ++elem_;
        }
    }
}

REGISTER_CPU_KERNEL(kMatrixDiagV3, MatrixDiagV3CpuKernel);
} // namespace aicpu
