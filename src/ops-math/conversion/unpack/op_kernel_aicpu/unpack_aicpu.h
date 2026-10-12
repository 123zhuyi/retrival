/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file unpack_aicpu.h
 * \brief
 */
#ifndef AICPU_KERNELS_NORMALIZED_UNPACK_H_
#define AICPU_KERNELS_NORMALIZED_UNPACK_H_

#include <memory>
#include <vector>
#include "cpu_types.h"
#include "utils/bcast.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "securec.h"
#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "log.h"
#include "status.h"

namespace aicpu {
class UnpackCpuKernel : public CpuKernel {
 public:
  UnpackCpuKernel()
      : data_type(DT_DOUBLE), unpack_axis(0), unpack_num(0), value_num(0) {
    output_ptr_vec.clear();
    value_shape_vec.clear();
  }
  ~UnpackCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t CheckAndInitParams(CpuKernelContext &ctx);

  template <typename T>
  uint32_t UnpackWithOneOutput(T *input_data_ptr,
                               std::vector<T *> output_data_vec);

  template <typename T>
  uint32_t UnpackWithDimZero(T *input_data_ptr,
                             std::vector<T *> output_data_vec);

  template <typename T>
  uint32_t UnpackCompute(T *input_data_ptr, std::vector<T *> output_data_vec,
                         CpuKernelContext &ctx);

  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

 private:
  DataType data_type;
  uint64_t unpack_axis;
  int64_t unpack_num;
  int64_t value_num;
  void *value_data_ptr;
  std::vector<void *> output_ptr_vec;
  std::vector<int64_t> value_shape_vec;
};
}  // namespace aicpu
#endif