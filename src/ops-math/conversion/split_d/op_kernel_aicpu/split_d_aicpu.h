/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_KERNELS_NORMALIZED_SPLIT_D_H_
#define AICPU_KERNELS_NORMALIZED_SPLIT_D_H_

#include <memory>
#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"
#include "securec.h"

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "log.h"
#include "status.h"

namespace aicpu {
class SplitDCpuKernel : public CpuKernel {
 public:
  SplitDCpuKernel() : data_type_(DT_DOUBLE),
                      split_dim_(0),
                      num_split_(0),
                      value_num_(0),
                      size_splits_(0),
                      value_data_ptr_(nullptr) {
    output_ptr_vec_.clear();
    value_shape_vec_.clear();
  }

  ~SplitDCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t CheckAndInitParams(const CpuKernelContext &ctx);
  
  template <typename T>
  uint32_t SplitDWithOneOutput(const T *input_data_ptr,
                               std::vector<T *> output_data_vec);

  template <typename T>
  uint32_t SplitDWithDimZero(T *input_data_ptr,
                             std::vector<T *> output_data_vec);

  template <typename T>
  uint32_t SplitDCompute(T *input_data_ptr,
                         std::vector<T *> output_data_vec);

  template <typename T>
  uint32_t DoCompute();

  DataType data_type_;
  int32_t split_dim_;
  int64_t num_split_;
  int64_t value_num_;
  int64_t size_splits_;
  void *value_data_ptr_;
  std::vector<void *> output_ptr_vec_;
  std::vector<int64_t> value_shape_vec_;
};
}  // namespace aicpu
#endif