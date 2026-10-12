/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_CONCATV2_H_
#define AICPU_KERNELS_CONCATV2_H_

#include <atomic>
#include <cstring>
#include <vector>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"

namespace aicpu {
// Flattened input descriptor: raw base pointer + per-row element count (dim1).
// Avoids Eigen::TensorMap / shared_ptr indirections on the hot path.
template <typename T>
struct ConcatInputDesc {
    const T* base = nullptr;
    int64_t dim1 = 0;
};

class ConcatV2CpuKernel : public CpuKernel {
public:
    ConcatV2CpuKernel() = default;

    ~ConcatV2CpuKernel() override = default;

    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    uint32_t CheckConcatV2Params(const CpuKernelContext& ctx);

    uint32_t InitConcatV2Params(const CpuKernelContext& ctx);

    uint32_t ParseConcatDim(const CpuKernelContext& ctx, int64_t& concat_dim) const;

    uint32_t CheckAndInitParams(const CpuKernelContext& ctx);

    template <typename T>
    uint32_t PrepareInputs(
        const CpuKernelContext& ctx, std::vector<ConcatInputDesc<T>>& inputs, int64_t& row_size)
    {
        inputs.reserve(static_cast<size_t>(n_));
        row_size = 0;
        auto input0_shape_ptr = ctx.Input(0)->GetTensorShape();
        KERNEL_CHECK_NULLPTR(input0_shape_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input x0 shape failed.");
        for (int64_t i = 0; i < n_; ++i) {
            Tensor* input_i_ptr = ctx.Input(static_cast<uint32_t>(i));
            KERNEL_CHECK_NULLPTR(input_i_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input x[%ld] failed.", i);
            int64_t input_i_num = input_i_ptr->NumElements();
            if (input_i_num == 0) {
                continue;
            }
            uint32_t status = ValidateInputShape(input_i_ptr, input0_shape_ptr.get(), i);
            if (status != KERNEL_STATUS_OK) {
                return status;
            }
            auto* data_raw = input_i_ptr->GetData();
            KERNEL_CHECK_NULLPTR(data_raw, KERNEL_STATUS_PARAM_INVALID, "Get input x[%ld] data failed.", i);
            ConcatInputDesc<T> desc;
            desc.base = PtrToPtr<void, const T>(data_raw);
            desc.dim1 = (inputs_flat_dim0_ == 0) ? 0 : (input_i_num / inputs_flat_dim0_);
            row_size += desc.dim1;
            inputs.emplace_back(desc);
        }
        return KERNEL_STATUS_OK;
    }

    uint32_t ValidateInputShape(Tensor* input_i_ptr, TensorShape* input0_shape_ptr, int64_t i) const;

    template <typename T>
    uint32_t PrepareOutput(CpuKernelContext& ctx, T*& out_base, int64_t& output_num) const
    {
        Tensor* output_ptr = ctx.Output(0);
        KERNEL_CHECK_NULLPTR(output_ptr, KERNEL_STATUS_PARAM_INVALID, "Get output failed.");
        auto output_data_ptr = output_ptr->GetData();
        KERNEL_CHECK_NULLPTR(output_data_ptr, KERNEL_STATUS_PARAM_INVALID, "Get output data failed.");
        output_num = output_ptr->NumElements();
        out_base = PtrToPtr<void, T>(output_data_ptr);
        return KERNEL_STATUS_OK;
    }

    // Copy one row of concat data [inputs[*]] -> [out_row], returns EOK on success.
    template <typename T>
    static errno_t CopyOneRow(
        T* out_row, int64_t row_index, const std::vector<ConcatInputDesc<T>>& inputs)
    {
        T* dst = out_row;
        for (const auto& desc : inputs) {
            const size_t bytes = static_cast<size_t>(desc.dim1) * sizeof(T);
            if (bytes == 0) {
                continue;
            }
            const T* src = desc.base + row_index * desc.dim1;
            errno_t e = memcpy_s(dst, bytes, src, bytes);
            if (e != EOK) {
                return e;
            }
            dst += desc.dim1;
        }
        return EOK;
    }

    template <typename T>
    uint32_t RunParallelByRow(
        const CpuKernelContext& ctx, T* out_base, int64_t row_size,
        const std::vector<ConcatInputDesc<T>>& inputs) const
    {
        std::atomic<uint32_t> shard_err(KERNEL_STATUS_OK);
        const int64_t total_rows = inputs_flat_dim0_;
        auto work = [&shard_err, &inputs, out_base, row_size](int64_t start_row, int64_t end_row) {
            for (int64_t r = start_row; r < end_row; ++r) {
                T* out_row = out_base + r * row_size;
                errno_t e = CopyOneRow<T>(out_row, r, inputs);
                if (e != EOK) {
                    KERNEL_LOG_ERROR("CopyOneRow failed at row[%ld], errno=%d.", r, static_cast<int>(e));
                    shard_err.store(KERNEL_STATUS_INNER_ERROR, std::memory_order_relaxed);
                    return;
                }
            }
        };
        const int64_t per_unit_bytes = row_size * static_cast<int64_t>(sizeof(T));
        uint32_t rc = CpuKernelUtils::ParallelFor(ctx, total_rows, per_unit_bytes, work);
        KERNEL_CHECK_FALSE(
            (rc == KERNEL_STATUS_OK), KERNEL_STATUS_INNER_ERROR, "ParallelFor exec failed, rc=[%u].", rc);
        uint32_t err = shard_err.load(std::memory_order_relaxed);
        KERNEL_CHECK_FALSE(
            (err == KERNEL_STATUS_OK), err, "ConcatV2 shard reported error[%u].", err);
        return KERNEL_STATUS_OK;
    }

    template <typename T>
    uint32_t RunParallelByInput(
        const CpuKernelContext& ctx, T* out_base,
        const std::vector<ConcatInputDesc<T>>& inputs) const
    {
        // axis=0 / single outer row: copy inputs sequentially with internal parallelism per block.
        std::atomic<uint32_t> shard_err(KERNEL_STATUS_OK);
        // Compute prefix offsets so shards can write independently.
        const size_t num = inputs.size();
        std::vector<int64_t> offsets(num + 1, 0);
        for (size_t k = 0; k < num; ++k) {
            offsets[k + 1] = offsets[k] + inputs[k].dim1;
        }
        const int64_t total = offsets[num];
        if (total == 0) {
            return KERNEL_STATUS_OK;
        }
        auto work = [&shard_err, &inputs, &offsets, out_base, num](int64_t start, int64_t end) {
            // Find first input whose range intersects [start, end).
            for (size_t k = 0; k < num; ++k) {
                int64_t seg_beg = offsets[k];
                int64_t seg_end = offsets[k + 1];
                int64_t beg = std::max(seg_beg, start);
                int64_t fin = std::min(seg_end, end);
                if (beg >= fin) {
                    continue;
                }
                const int64_t local = beg - seg_beg;
                const int64_t len = fin - beg;
                const size_t bytes = static_cast<size_t>(len) * sizeof(T);
                errno_t e = memcpy_s(out_base + beg, bytes, inputs[k].base + local, bytes);
                if (e != EOK) {
                    KERNEL_LOG_ERROR("memcpy_s failed in ByInput shard, errno=%d.", static_cast<int>(e));
                    shard_err.store(KERNEL_STATUS_INNER_ERROR, std::memory_order_relaxed);
                    return;
                }
            }
        };
        uint32_t rc = CpuKernelUtils::ParallelFor(ctx, total, static_cast<int64_t>(sizeof(T)), work);
        KERNEL_CHECK_FALSE(
            (rc == KERNEL_STATUS_OK), KERNEL_STATUS_INNER_ERROR, "ParallelFor exec failed, rc=[%u].", rc);
        uint32_t err = shard_err.load(std::memory_order_relaxed);
        KERNEL_CHECK_FALSE(
            (err == KERNEL_STATUS_OK), err, "ConcatV2 shard reported error[%u].", err);
        return KERNEL_STATUS_OK;
    }

    template <typename T>
    uint32_t DoCompute(CpuKernelContext& ctx)
    {
        std::vector<ConcatInputDesc<T>> inputs;
        int64_t row_size = 0;
        uint32_t rc = PrepareInputs<T>(ctx, inputs, row_size);
        KERNEL_CHECK_FALSE((rc == KERNEL_STATUS_OK), rc, "PrepareInputs failed.");
        T* out_base = nullptr;
        int64_t output_num = 0;
        rc = PrepareOutput<T>(ctx, out_base, output_num);
        KERNEL_CHECK_FALSE((rc == KERNEL_STATUS_OK), rc, "PrepareOutput failed.");
        if (inputs.empty() || row_size == 0 || inputs_flat_dim0_ == 0) {
            KERNEL_LOG_INFO("ConcatV2 early-return: no data to copy.");
            return KERNEL_STATUS_OK;
        }
        KERNEL_LOG_INFO(
            "ConcatV2 dispatch: flat_dim0=%ld, row_size=%ld, num_inputs=%zu.",
            inputs_flat_dim0_, row_size, inputs.size());
        if (inputs_flat_dim0_ == 1) {
            return RunParallelByInput<T>(ctx, out_base, inputs);
        }
        return RunParallelByRow<T>(ctx, out_base, row_size, inputs);
    }

    DataType data_type_ = DT_DOUBLE;
    int32_t input_dims_ = 0;
    int64_t n_ = 0;
    int64_t axis_ = 0;
    int64_t inputs_flat_dim0_ = 0;
};
} // namespace aicpu
#endif // AICPU_KERNELS_CONCATV2_H_
