/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_STRIDED_SLICE_H_
#define AICPU_KERNELS_NORMALIZED_STRIDED_SLICE_H_

#include <vector>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace aicpu {
class StridedSliceCpuKernel : public CpuKernel {
 public:
  ~StridedSliceCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

  /**
   * @brief init strided slice params with masks
   * @param x_shape StridedSlice input [x]'s shape
   * @param begin_mask begin mask
   * @param end_mask end mask
   * @param ellipsis_mask ellipsis mask
   * @param new_axis_mask new axis mask
   * @param shrink_axis_mask shrink axis mask
   * @param begin StridedSlice param begin
   * @param end StridedSlice param end
   * @param strides StridedSlice param strides
   * @return status code
   */
  static uint32_t InitParamsWithMasks(const std::vector<int64_t> &x_shape,
                                      int64_t begin_mask, int64_t end_mask,
                                      int64_t ellipsis_mask,
                                      int64_t new_axis_mask,
                                      int64_t shrink_axis_mask,
                                      std::vector<int64_t> &begin,
                                      std::vector<int64_t> &end,
                                      std::vector<int64_t> &strides);

  /**
   * @brief check strided slice parameters and get data
   * @param x_tensor StridedSlice input [x]
   * @param y_tensor StridedSlice output [y]
   * @param strides StridedSlice param strides
   * @param x_data output x data pointer
   * @param y_data output y data pointer
   * @param x_size output x size
   * @param y_size output y size
   * @param x_shape output x shape
   * @return status code
   */
  template <typename T>
  static uint32_t CheckCalStridedSliceParams(const Tensor *x_tensor, Tensor *y_tensor,
                                            const std::vector<int64_t> &strides,
                                            T* &x_data, T* &y_data,
                                            int64_t &x_size, int64_t &y_size,
                                            std::vector<int64_t> &x_shape) {
    KERNEL_CHECK_NULLPTR(x_tensor, KERNEL_STATUS_INNER_ERROR,
                        "[CalStridedSlice] check x_tensor is [nullptr].");
    KERNEL_CHECK_NULLPTR(y_tensor, KERNEL_STATUS_INNER_ERROR,
                        "[CalStridedSlice] check y_tensor is [nullptr].");
    x_data = static_cast<T *>(x_tensor->GetData());
    if (x_data == nullptr) {
        KERNEL_LOG_WARN("where x_data is a nullptr");
        y_tensor->SetData(nullptr);
        y_tensor->SetTensorShape(nullptr);
        return KERNEL_STATUS_OK;
    }
    y_size = y_tensor->NumElements();
    if (y_size == 0) {
      return KERNEL_STATUS_OK;
    }
    y_data = static_cast<T *>(y_tensor->GetData());
    KERNEL_CHECK_NULLPTR(y_data, KERNEL_STATUS_INNER_ERROR,
                        "[CalStridedSlice] check y_data is [nullptr].");
    x_size = x_tensor->NumElements();
    auto x_tensor_shape = x_tensor->GetTensorShape();
    KERNEL_CHECK_NULLPTR(x_tensor_shape, KERNEL_STATUS_INNER_ERROR,
                        "[CalStridedSlice] check x_tensor_shape is [nullptr].");
    x_shape = x_tensor_shape->GetDimSizes();

    for (size_t i = 0; i < strides.size(); ++i) {
      KERNEL_CHECK_FALSE((strides[i] != 0), KERNEL_STATUS_PARAM_INVALID,
          "[CalStridedSlice] strides[%zu] must be non-zero.", i);
    }
    return KERNEL_STATUS_OK;
  }

  /**
   * @brief calculate y shape temp and validate
   * @param x_shape StridedSlice input [x]'s shape
   * @param strides StridedSlice param strides
   * @param begin_tmp StridedSlice begin temp (modified in-place)
   * @param end_tmp StridedSlice end temp (modified in-place)
   * @param y_shape_tmp output y shape temp
   * @return status code
   */
  static uint32_t CalculateYShapeTmpAndValidate(const std::vector<int64_t> &x_shape,
                                                  const std::vector<int64_t> &strides,
                                                  std::vector<int64_t> &begin_tmp,
                                                  std::vector<int64_t> &end_tmp,
                                                  std::vector<int64_t> &y_shape_tmp) {
    y_shape_tmp = CalYShapeTmp(x_shape, strides, begin_tmp, end_tmp);
    for (size_t i = 0; i < y_shape_tmp.size(); ++i) {
      KERNEL_CHECK_FALSE((y_shape_tmp[i] != 0), KERNEL_STATUS_INNER_ERROR,
          "[CalStridedSlice] y_shape_tmp[%zu] must be non-zero.", i);
    }
    return KERNEL_STATUS_OK;
  }

  /**
   * @brief calculate y shape temp
   * @param x_shape StridedSlice input [x]'s shape
   * @param strides StridedSlice param strides
   * @param begin_tmp StridedSlice begin temp (modified in-place)
   * @param end_tmp StridedSlice end temp (modified in-place)
   * @param y_shape_tmp output y shape temp
   * @return status code
   */
  static uint32_t CalculateYShapeTmp(const std::vector<int64_t> &x_shape,
                                         const std::vector<int64_t> &strides,
                                         std::vector<int64_t> &begin_tmp,
                                         std::vector<int64_t> &end_tmp,
                                         std::vector<int64_t> &y_shape_tmp) {
    y_shape_tmp = CalYShapeTmp(x_shape, strides, begin_tmp, end_tmp);
    return KERNEL_STATUS_OK;
  }

  /**
   * @brief process turbo shard
   * @param x_data x data pointer
   * @param y_data y data pointer
   * @param x_size x size
   * @param x_shape x shape
   * @param y_shape_tmp y shape temp
   * @param begin_tmp begin temp
   * @param start start index
   * @param endTmp end index
   */
  template <typename T>
  static void ProcessTurboShard(T* x_data, T* y_data, int64_t x_size,
                                  const std::vector<int64_t> &x_shape,
                                  const std::vector<int64_t> &y_shape_tmp,
                                  const std::vector<int64_t> &begin_tmp,
                                  int64_t start, int64_t endTmp) {
    int64_t factor = x_shape.back() / y_shape_tmp.back();
    int64_t offset = begin_tmp.back();
    for (int64_t y_idx = start; y_idx < endTmp; ++y_idx) {
      int64_t x_idx = y_idx * factor + offset;
      KERNEL_CHECK_FALSE_VOID((x_idx < x_size), "[CalStridedSlice] x_idx [%ld] overflow x_size [%ld].",
          x_idx, x_size);
      y_data[y_idx] = x_data[x_idx];
    }
  }

  /**
   * @brief process shard
   * @param x_data x data pointer
   * @param y_data y data pointer
   * @param x_size x size
   * @param x_shape x shape
   * @param y_shape_tmp y shape temp
   * @param begin_tmp begin temp
   * @param strides strides
   * @param block block
   * @param start start index
   * @param endTmp end index
   */
  template <typename T>
  static void ProcessShard(T* x_data, T* y_data, int64_t x_size,
                            const std::vector<int64_t> &x_shape,
                            const std::vector<int64_t> &y_shape_tmp,
                            const std::vector<int64_t> &begin_tmp,
                            const std::vector<int64_t> &strides,
                            const std::vector<int64_t> &block,
                            int64_t start, int64_t endTmp) {
    for (int64_t y_idx = start; y_idx < endTmp; ++y_idx) {
      int64_t x_idx = 0;
      int64_t y_idx_tmp = y_idx;
 
      size_t i = x_shape.size() - 1;
      for (; i > 3; i -= 4) {  // 每轮循环处理4个维度，提升性能，i > 3 确保可以安全访问 i-3，避免数组越界
          int64_t idx_in_dim_0 = y_idx_tmp % y_shape_tmp[i];
          x_idx += (begin_tmp[i] + idx_in_dim_0 * strides[i]) * block[i];
          y_idx_tmp = y_idx_tmp / y_shape_tmp[i];
          
          int64_t idx_in_dim_1 = y_idx_tmp % y_shape_tmp[i - 1];
          x_idx += (begin_tmp[i - 1] + idx_in_dim_1 * strides[i - 1]) * block[i - 1];
          y_idx_tmp = y_idx_tmp / y_shape_tmp[i - 1];

          int64_t idx_in_dim_2 = y_idx_tmp % y_shape_tmp[i - 2];
          x_idx += (begin_tmp[i - 2] + idx_in_dim_2 * strides[i - 2]) * block[i - 2];  // 处理第3个维度(i-2)
          y_idx_tmp = y_idx_tmp / y_shape_tmp[i - 2]; // 处理第3个维度(i-2)

          int64_t idx_in_dim_3 = y_idx_tmp % y_shape_tmp[i - 3];
          x_idx += (begin_tmp[i - 3] + idx_in_dim_3 * strides[i - 3]) * block[i - 3];  // 处理第4个维度(i-3)
          y_idx_tmp = y_idx_tmp / y_shape_tmp[i - 3]; // 处理第4个维度(i-3)
      }
 
      for (; i > 0; --i) {
          int64_t idx_in_dim = y_idx_tmp % y_shape_tmp[i];
          x_idx += (begin_tmp[i] + idx_in_dim * strides[i]) * block[i];
          y_idx_tmp = y_idx_tmp / y_shape_tmp[i];
      }
 
      x_idx += (begin_tmp[0] + y_idx_tmp * strides[0]) * block[0];
      KERNEL_CHECK_FALSE_VOID((x_idx < x_size), "[CalStridedSlice] x_idx [%ld] overflow x_size [%ld].",
          x_idx, x_size);
      y_data[y_idx] = x_data[x_idx];
    }
  }

  /**
   * @brief calculate strided slice
   * @param ctx op context
   * @param begin StridedSlice param begin
   * @param end StridedSlice param end
   * @param strides StridedSlice param strides
   * @param x_tensor StridedSlice input [x]
   * @param y_tensor StridedSlice output [y]
   * @return status code
   */
  template <typename T>
  static uint32_t CalStridedSlice(const CpuKernelContext &ctx,
                                  const std::vector<int64_t> &begin,
                                  const std::vector<int64_t> &end,
                                  const std::vector<int64_t> &strides,
                                  const Tensor *x_tensor, Tensor *y_tensor) {
    T* x_data = nullptr;
    T* y_data = nullptr;
    int64_t x_size = 0;
    int64_t y_size = 0;
    std::vector<int64_t> x_shape;
    uint32_t ret = CheckCalStridedSliceParams<T>(x_tensor, y_tensor, strides,
                                                  x_data, y_data, x_size, y_size, x_shape);
    if (ret != KERNEL_STATUS_OK) {
      return ret;
    }

    if (y_size == 0) {
      return KERNEL_STATUS_OK;
    }

    std::vector<int64_t> begin_tmp = begin;
    std::vector<int64_t> end_tmp = end;
    std::vector<int64_t> y_shape_tmp;
    ret = CalculateYShapeTmpAndValidate(x_shape, strides, begin_tmp, end_tmp, y_shape_tmp);
    if (ret != KERNEL_STATUS_OK) {
      return ret;
    }

    std::vector<int64_t> block = CalBlocks(x_shape);
    if (IsTurbo(x_shape, y_shape_tmp)) {
      auto turboShardCall = [&](int64_t start, int64_t endTmp)->void {
        ProcessTurboShard<T>(x_data, y_data, x_size, x_shape, y_shape_tmp, begin_tmp, start, endTmp);
      };
      return CpuKernelUtils::ParallelFor(ctx, y_size, 1, turboShardCall);
    } else {
      auto shardCall = [&](int64_t start, int64_t endTmp)->void {
        ProcessShard<T>(x_data, y_data, x_size, x_shape, y_shape_tmp, begin_tmp, strides, block, start, endTmp);
      };
      return CpuKernelUtils::ParallelFor(ctx, y_size, 1, shardCall);
    }
  }

 private:
  /**
   * @brief parse kernel parms
   * @param ctx op context
   * @return status code
   */
  uint32_t ParseKernelParams(const CpuKernelContext &ctx);

  uint32_t ParseIndexInput(const CpuKernelContext &ctx, uint32_t index,
                           std::vector<int64_t> &vec);
  uint32_t GetMaskAttr(const CpuKernelContext &ctx, const std::string attr, int64_t &mask) const;

  /**
   * @brief convert negative idx to positive
   *        calculate y_shape temp with [begin_tmp, end_tmp, strides]
   * @param x_shape StridedSlice input [x]'s shape
   * @param strides StridedSlice param strides
   * @param begin_tmp StridedSlice begin temp
   * @param end_tmp StridedSlice end temp
   * @return y_shape temp
   */
  static std::vector<int64_t> CalYShapeTmp(
      const std::vector<int64_t> &x_shape,
      const std::vector<int64_t> &strides,
      std::vector<int64_t> &begin_tmp,
      std::vector<int64_t> &end_tmp) {
    std::vector<int64_t> y_shape_tmp(x_shape.size());
    for (size_t i = 0; i < begin_tmp.size(); ++i) {
      if (begin_tmp[i] < 0) {
        begin_tmp[i] += x_shape[i];
      }
      begin_tmp[i] = std::max(begin_tmp[i], int64_t(0));
      begin_tmp[i] = std::min(begin_tmp[i], x_shape[i] - 1);
      if (end_tmp[i] < 0) {
        end_tmp[i] += x_shape[i];
      }
      end_tmp[i] = std::max(end_tmp[i], int64_t(-1));
      end_tmp[i] = std::min(end_tmp[i], x_shape[i]);
      int64_t y_range = end_tmp[i] - begin_tmp[i];
      y_shape_tmp[i] = y_range / strides[i];
      if ((y_range % strides[i]) != 0) {
        y_shape_tmp[i] += 1;
      }
    }
    return y_shape_tmp;
  }

  /**
   * @brief calculate blocks for x
   * @param x_shape StridedSlice input [x]'s shape
   * @return blocks for x
   */
  static std::vector<int64_t> CalBlocks(const std::vector<int64_t> &x_shape) {
    std::vector<int64_t> block(x_shape.size());
    int64_t block_tmp = 1;
    for (size_t i = x_shape.size() - 1; i > 0; --i) {
      block[i] = block_tmp;
      block_tmp *= x_shape[i];
    }
    block[0] = block_tmp;
    return block;
  }

  /**
   * @brief check if StridedSlice can be accelerated
   * @param x_shape StridedSlice input [x]'s shape
   * @param y_shape_tmp StridedSlice input [x]'s shape
   * @return it can be accelerated or not
   */
  static bool IsTurbo(const std::vector<int64_t> &x_shape,
                      const std::vector<int64_t> &y_shape_tmp) {
    size_t size = y_shape_tmp.size();
    if (size == 0) {
      return false;
    }

    for (size_t i = 0; i < size - 1; ++i) {
      if (y_shape_tmp[i] != x_shape[i]) {
        return false;
      }
    }
    return (y_shape_tmp[size - 1] == 1);
  }

  std::vector<int64_t> begin_;
  std::vector<int64_t> end_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> x_shape_;
  int64_t begin_mask_ = 0;
  int64_t end_mask_ = 0;
  int64_t ellipsis_mask_ = 0;
  int64_t new_axis_mask_ = 0;
  int64_t shrink_axis_mask_ = 0;
};
}  // namespace aicpu
#endif
