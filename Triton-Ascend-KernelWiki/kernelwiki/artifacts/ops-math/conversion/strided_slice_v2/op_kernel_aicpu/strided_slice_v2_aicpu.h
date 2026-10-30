/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_KERNELS_NORMALIZED_STRIDED_SLICE_V2_H_
#define AICPU_KERNELS_NORMALIZED_STRIDED_SLICE_V2_H_

#include "cpu_kernel.h"
#include "log.h"
#include "status.h"

namespace aicpu {
class StridedSliceV2CpuKernel : public CpuKernel {
 public:
  ~StridedSliceV2CpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t CheckParam(const Tensor *begin, const Tensor *end,
                      const Tensor *axes, const Tensor *strides);

  uint32_t CheckBeginEndDataType(const Tensor *begin, const Tensor *end);

  uint32_t CheckBeginEndShape(const Tensor *begin, const Tensor *end);

  uint32_t CheckStridesDataTypeAndShape(const Tensor *begin,
                                          const Tensor *strides,
                                          const std::shared_ptr<TensorShape> &begin_shape);

  uint32_t CheckAxesDataType(const Tensor *begin, const Tensor *axes);

  uint32_t GetInputTensors(CpuKernelContext &ctx, Tensor *&x, Tensor *&begin,
                            Tensor *&end, Tensor *&axes, Tensor *&strides);

  uint32_t GetMaskAttrs(CpuKernelContext &ctx, int64_t &begin_mask_value,
                         int64_t &end_mask_value, int64_t &ellipsis_mask_value,
                         int64_t &new_axis_mask_value, int64_t &shrink_axis_mask_value);

  uint32_t CheckAndBuildParamsByType(const Tensor *x, const Tensor *begin,
                                      const Tensor *end, const Tensor *axes,
                                      const Tensor *strides,
                                      std::vector<int64_t> &begin_vec,
                                      std::vector<int64_t> &end_vec,
                                      std::vector<int64_t> &strides_vec);

  template <typename T>
  uint32_t BuildBeginParam(const std::shared_ptr<TensorShape> &x_shape,
                           const Tensor *begin,
                           std::vector<int64_t> &begin_vec);

  template <typename T>
  uint32_t BuildEndParam(const std::shared_ptr<TensorShape> &x_shape,
                         const Tensor *end, std::vector<int64_t> &end_vec);

  template <typename T>
  uint32_t BuildStridesParam(const std::shared_ptr<TensorShape> &x_shape,
                             const Tensor *strides,
                             std::vector<int64_t> &strides_vec);

  template <typename T>
  uint32_t BuildAxesParam(const std::shared_ptr<TensorShape> &x_shape,
                          const Tensor *axes, std::vector<int64_t> &axes_vec);

  template <typename T>
  uint32_t BuildParam(const Tensor *x, const Tensor *begin, const Tensor *end,
                      const Tensor *axes, const Tensor *strides,
                      std::vector<int64_t> &begin_vec,
                      std::vector<int64_t> &end_vec,
                      std::vector<int64_t> &strides_vec);

  template <typename T>
  uint32_t CheckAndBuildParam(const Tensor *x, const Tensor *begin,
                              const Tensor *end, const Tensor *axes,
                              const Tensor *strides,
                              std::vector<int64_t> &begin_vec,
                              std::vector<int64_t> &end_vec,
                              std::vector<int64_t> &strides_vec);

  uint32_t DoStridedSliceV2(const CpuKernelContext &ctx,
                            const std::vector<int64_t> &begin_vec,
                            const std::vector<int64_t> &end_vec,
                            const std::vector<int64_t> &strides_vec);
};
}  // namespace aicpu
#endif
