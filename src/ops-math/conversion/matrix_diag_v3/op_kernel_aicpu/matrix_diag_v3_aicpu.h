/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATRIX_DIAG_V3_AICPU_H_
#define MATRIX_DIAG_V3_AICPU_H_

#include "cpu_kernel.h"
#include "utils/status.h"

namespace aicpu {
class MatrixDiagV3CpuKernel : public CpuKernel {
public:
    ~MatrixDiagV3CpuKernel() override = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    KernelStatus CheckParam(const CpuKernelContext &ctx);
    std::pair<int, int> ComputeDiagLenAndContentOffset(int diag_index) const;
    template <typename T>
    KernelStatus DoCompute(const CpuKernelContext &ctx);
    KernelStatus AdjustRowsAndCols();
    KernelStatus GetDiagIndex(const CpuKernelContext &ctx);
    template <typename T>
    void SetResult(const CpuKernelContext &ctx, T padding_value, int64_t diag_batch_base_index);

    bool left_align_superdiagonal_ = true;
    bool left_align_subdiagonal_ = true;
    int32_t num_rows_ = -1;
    int32_t num_cols_ = -1;
    int64_t max_diag_len_ = 0;
    int32_t min_num_rows_ = 0;
    int32_t min_num_cols_ = 0;
    int32_t lower_diag_index_ = 0;
    int32_t upper_diag_index_ = 0;
    int64_t elem_ = 0;
};
} // namespace aicpu

#endif // MATRIX_DIAG_V3_AICPU_H_