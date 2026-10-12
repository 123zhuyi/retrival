/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATRIX_DIAG_PART_V3_AICPU_H_
#define MATRIX_DIAG_PART_V3_AICPU_H_

#include "cpu_kernel.h"
#include "utils/status.h"

namespace aicpu {
class MatrixDiagPartV3CpuKernel : public CpuKernel {
public:
    MatrixDiagPartV3CpuKernel() = default;
    ~MatrixDiagPartV3CpuKernel() override = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    KernelStatus CheckParam(const CpuKernelContext &ctx);
    template <typename T>
    void PadOutput(T *output_data, int64_t output_base_index, int64_t content_offset, int64_t diag_len,
                   int64_t max_diag_len, T padding_value) const;
    template <typename T>
    void ExtractDiagonal(T *output_data, int64_t output_base_index, const T *input_data, int64_t batch,
                         int64_t diag_index, int64_t max_diag_len, T padding_value) const;
    template <typename T>
    uint32_t DoCompute(const CpuKernelContext &ctx);
    template <typename T>
    uint32_t MultiProcessFunc(const CpuKernelContext &ctx, int64_t upper_diag_index, int64_t max_diag_len);
    template <typename T>
    uint32_t SingleProcessFunc(const CpuKernelContext &ctx, int64_t upper_diag_index, int64_t max_diag_len);
    std::pair<int64_t, int64_t> ComputeDiagLenAndContentOffset(int64_t diag_index, int64_t max_diag_len) const;

    bool left_align_superdiagonal_ = true;
    bool left_align_subdiagonal_ = true;
    int64_t num_rows_ = 0;
    int64_t num_cols_ = 0;
    int64_t num_array_ = 0;
    int64_t num_diags_ = 0;
};
} // namespace aicpu

#endif // MATRIX_DIAG_PART_V3_AICPU_H_
