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
 * \file split_v_aicpu.h
 * \brief SplitV AICPU kernel implementation
 */
#ifndef AICPU_KERNELS_NORMALIZED_SPLIT_V_H_
#define AICPU_KERNELS_NORMALIZED_SPLIT_V_H_

#include <memory>
#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"
#include "securec.h"

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "log.h"
#include "status.h"

namespace aicpu {
class SplitVCpuKernel : public CpuKernel {
 public:
  SplitVCpuKernel() : data_type_(DT_DOUBLE),
                      split_dim_(0),
                      num_split_(0),
                      value_num_(0),
                      value_data_ptr_(nullptr) {
    size_splits_.clear();
    output_ptr_vec_.clear();
    value_shape_vec_.clear();
  }

  ~SplitVCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  /**
   * @brief Init params
   * @param ctx cpu kernel context
   * @return status if success
   */
  uint32_t CheckAndInitParams(const CpuKernelContext &ctx);

  /**
   * @brief Validate and get num_split attribute
   * @param ctx cpu kernel context
   * @return status if success
   */
  uint32_t ValidateAndGetNumSplit(const CpuKernelContext &ctx);

  /**
   * @brief Validate and get split_dim input
   * @param ctx cpu kernel context
   * @return status if success
   */
  uint32_t ValidateAndGetSplitDim(const CpuKernelContext &ctx);

  /**
   * @brief Validate and get value input
   * @param ctx cpu kernel context
   * @param real_dim output the size of split dimension
   * @return status if success
   */
  uint32_t ValidateAndGetValue(const CpuKernelContext &ctx, int64_t &real_dim);

  /**
   * @brief Validate and get size_splits input
   * @param ctx cpu kernel context
   * @param real_dim total size of dim which be split
   * @return status if success
   */
  uint32_t ValidateAndGetSizeSplits(const CpuKernelContext &ctx, int64_t real_dim);

  /**
   * @brief Validate and get all outputs
   * @param ctx cpu kernel context
   * @return status if success
   */
  uint32_t ValidateAndGetOutputs(const CpuKernelContext &ctx);

  /**
   * @brief get size of each split
   * @param size_splits_data_ptr data store split size
   * @param real_dim total size of dim which be split
   * @return status if success
   */
  template <typename T>
  uint32_t GetSizeSplits(void *size_splits_data_ptr, int64_t real_dim);

  /**
   * @brief split data when split num is 1
   * @param input_data_ptr ptr which store input data
   * @param output_data_vec vector which store all output data ptr
   * @return status if success
   */
  template <typename T>
  uint32_t SplitVWithOneOutput(const T *input_data_ptr,
                               std::vector<T *> output_data_vec);

  /**
   * @brief split data when split dim is 0
   * @param input_data_ptr ptr which store input data
   * @param output_data_vec vector which store all output data ptr
   * @return status if success
   */
  template <typename T>
  uint32_t SplitVWithDimZero(T *input_data_ptr,
                             std::vector<T *> output_data_vec);

  /**
   * @brief split data
   * @param input_data_ptr ptr which store input data
   * @param output_data_vec vector which store all output data ptr
   * @return status if success
   */
  template <typename T>
  uint32_t SplitVCompute(T *input_data_ptr,
                         std::vector<T *> output_data_vec);

  template <typename T>
  uint32_t DoCompute();

  DataType data_type_;
  int32_t split_dim_;
  int64_t num_split_;
  int64_t value_num_;
  void *value_data_ptr_;
  std::vector<void *> output_ptr_vec_;
  std::vector<int64_t> size_splits_;
  std::vector<int64_t> value_shape_vec_;
};
}  // namespace aicpu
#endif
