/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_ADD_H
#define AICPU_KERNELS_ADD_H
#define EIGEN_USE_THREADS
#define EIGEN_USE_SIMPLE_THREAD_POOL

#include "cpu_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
class AddCpuKernel : public CpuKernel {
 public:
  AddCpuKernel() = default;
  ~AddCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  /**
   * @brief top-level dispatch for one dtype; chooses a fast-path
   *        (same-shape / scalar-bcast) or falls back to generic Eigen bcast.
   */
  template <typename T>
  uint32_t AddCompute(const CpuKernelContext &ctx) const;

  /**
   * @brief Run a per-shard body either serially (small workloads) or via
   *        ParallelFor (large workloads). Centralizes the threshold check,
   *        per-unit sizing, dispatch and error logging shared by every
   *        fast-path variant. `elem_bytes` is sizeof(T) for the worker.
   */
  template <typename Body>
  uint32_t RunMaybeParallel(const CpuKernelContext &ctx, const char *tag,
                            int64_t total, int64_t elem_bytes,
                            const Body &body) const;

  /**
   * @brief Validate raw shapes / declared output shape and populate
   *        broadcast info. On success `calc_info` contains a fully derived
   *        broadcast plan. Centralizes the rank/shape checks so AddCompute
   *        stays under the per-method line limit.
   */
  uint32_t ValidateAndBroadcast(const CpuKernelContext &ctx,
                                BCalcInfo &calc_info) const;

  /**
   * @brief fast path: element-wise add with identical shapes.
   *        Parallelized when output-bytes exceed threshold.
   */
  template <typename T>
  uint32_t AddSameShape(const CpuKernelContext &ctx, const T *x0, const T *x1,
                        T *y, int64_t total) const;

  /**
   * @brief fast path: one operand is a scalar (single element) broadcast
   *        over the other. Parallelized when output-bytes exceed threshold.
   * @param scalar_side 0 -> x0 is scalar, 1 -> x1 is scalar
   */
  template <typename T>
  uint32_t AddScalarBcast(const CpuKernelContext &ctx, const T *vec,
                          T scalar_val, T *y, int64_t total) const;

  /**
   * @brief generic broadcast path (keeps original Eigen tensor semantics).
   */
  template <typename T>
  uint32_t AddGenericBcast(const CpuKernelContext &ctx,
                           BCalcInfo &calc_info) const;

  /**
   * @brief check if input&output addr is aligned
   */
  bool AlignedCheck(const BCalcInfo &calc_info) const;

  template <typename T>
  uint32_t AddCalculateWithRankCheck(const CpuKernelContext &ctx,
                                     BCalcInfo &calc_info) const;

  template <int32_t RANK, typename T>
  uint32_t AddCalculateWithAlignedCheck(const CpuKernelContext &ctx,
                                        BCalcInfo &calc_info) const;

  /**
   * @brief Eigen calculate for all types (fallback generic broadcast).
   */
  template <int32_t RANK, typename T, int32_t OPTION>
  uint32_t AddCalculate(BCalcInfo &calc_info) const;
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_ADD_H_
