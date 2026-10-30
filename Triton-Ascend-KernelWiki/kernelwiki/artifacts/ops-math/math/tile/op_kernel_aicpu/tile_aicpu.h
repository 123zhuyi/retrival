/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_DEVICE_TILE_H
#define AICPU_KERNELS_DEVICE_TILE_H
#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#include "cpu_kernel.h"
#include "cpu_types.h"
#include "utils/bcast.h"
#include "driver/ascend_hal.h"

namespace aicpu {
class TileCpuKernel : public CpuKernel {
 public:
  TileCpuKernel() = default;
  ~TileCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  bool is_empty_tensor_;
  std::vector<int64_t> multiples_;
  uint32_t TileComputeUsingMemcpy(void *dst_addr, void *src_addr, size_t copy_len) const;
  uint32_t TileComputeUsingSdma(void *dst_addr, void *src_addr, size_t copy_len) const;
  void SetCopyHook(const bool condition) {
    if (condition) {
      copy_hook_ = &TileCpuKernel::TileComputeUsingSdma;
    } else {
      copy_hook_ = &TileCpuKernel::TileComputeUsingMemcpy;
    }
  }
  uint32_t CallCopyHook(void *dst, void *src, size_t copy_len) {
    return (this->*copy_hook_)(dst, src, copy_len);
  }
    uint32_t (TileCpuKernel::*copy_hook_)(void*, void*, size_t) const;
  template <typename T>
  uint32_t TileComputeWith2DNotUsingEigen(const CpuKernelContext &ctx);
  template <typename T>
  uint32_t TileComputeWith3DNotUsingEigen(const CpuKernelContext &ctx);
  template <typename T>
  uint32_t TileCompute3DSharderFirst(const CpuKernelContext &ctx, T *input_x_data, T *output_data,
                                      int64_t x_first_dim, int64_t x_second_dim, int64_t x_third_dim,
                                      int64_t last_axes_dims, int64_t second_axes_dims);
  template <typename T>
  uint32_t TileCompute3DSharderSecond(const CpuKernelContext &ctx, T *output_data,
                                       int64_t x_first_dim, int64_t x_second_dim, int64_t x_third_dim,
                                       int64_t mul_third_dim, int64_t last_axes_dims, int64_t last_two_axes_dims);
  template <typename T>
  uint32_t TileCompute3DSharderThird(const CpuKernelContext &ctx, T *output_data,
                                      int64_t x_first_dim, int64_t x_second_dim, int64_t mul_second_dim,
                                      int64_t last_axes_dims, int64_t last_two_axes_dims);
  template <typename T>
  uint32_t TileCompute3DSharderFourth(const CpuKernelContext &ctx, T *output_data,
                                       int64_t mul_first_dim, int64_t last_two_axes_dims, int64_t x_first_dim);
  template <typename T>
  uint32_t TileComputeWith1D(T *input_x_data, T *output_data, int64_t x_dim, int64_t mul_dim);
  template <typename T>
  uint32_t TileCheckCopySupported(const CpuKernelContext &ctx);
  template <typename T>
  uint32_t TileKernelCompute(const CpuKernelContext &ctx);
  uint32_t TileParamCheck(const CpuKernelContext &ctx);
  uint32_t GetMultiplesValue(Tensor *tensor, std::vector<int64_t> &mtp_value);
  template <typename T>
  uint32_t TileCompute(const CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_DEVICE_TILE_H
