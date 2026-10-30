/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tile_aicpu.h"

#include <complex>
#include <iostream>

#include "Eigen/Core"
#include "aicpu/math_aicpu_register.h"
#include "cpu_kernel_utils.h"
#include "securec.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *const kTile = "Tile";
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
constexpr int32_t kIndexZero = 0;
constexpr int32_t kIndexOne = 1;
constexpr int32_t kIndexTwo = 2;
constexpr int32_t kDim0 = 0;
constexpr int32_t kDim1 = 1;
constexpr int32_t kDim2 = 2;
constexpr int32_t kDim3 = 3;
constexpr int32_t kDim4 = 4;
constexpr int32_t kDim5 = 5;
constexpr int32_t kDim6 = 6;
constexpr int32_t kDim7 = 7;
constexpr int32_t kDim8 = 8;
constexpr int32_t kParallelShapeSize = 5 * 1024;

#define TILE_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                      \
    if (TileCompute<TYPE>(CTX) != KERNEL_STATUS_OK) {  \
      KERNEL_LOG_ERROR("Tile kernel compute failed."); \
      return KERNEL_STATUS_INNER_ERROR;                \
    }                                                  \
    break;                                             \
  }

#define TILE_COMPUTE_GET_MULTIPLE(input_dims, mul_value) \
  for (int32_t i = 0; i < (input_dims); ++i) {           \
    (mul_value)[i] = multiples_.at(i);                   \
  }

#define TILE_COMPUTE_DIM2(input_x_data, shape_x, output_data, shape_output) \
  do {                                                                      \
    typedef Eigen::TensorMap<Eigen::Tensor<T, kDim2, Eigen::RowMajor>,      \
                             Eigen::Aligned>                                \
        EigenTensorNd;                                                    \
    EigenTensorNd input_nd((input_x_data), (shape_x).at(0),               \
                             (shape_x).at(1));                              \
    EigenTensorNd output_nd((output_data), (shape_output).at(0),          \
                              (shape_output).at(1));                        \
    Eigen::array<Eigen::DenseIndex, kDim2> mtp_2d;                          \
    TILE_COMPUTE_GET_MULTIPLE(kDim2, mtp_2d)                                \
    output_nd = input_nd.broadcast(mtp_2d);                                 \
  } while (0)

#define TILE_COMPUTE_DIM3(input_x_data, shape_x, output_data, shape_output)    \
  do {                                                                         \
    typedef Eigen::TensorMap<Eigen::Tensor<T, kDim3, Eigen::RowMajor>,         \
                             Eigen::Aligned>                                   \
        EigenTensorNd;                                                       \
    EigenTensorNd input_nd((input_x_data), (shape_x).at(0), (shape_x).at(1), \
                             (shape_x).at(2));                                 \
    EigenTensorNd output_nd((output_data), (shape_output).at(0),             \
                              (shape_output).at(1), (shape_output).at(2));     \
    Eigen::array<Eigen::DenseIndex, kDim3> mtp_3d;                             \
    TILE_COMPUTE_GET_MULTIPLE(kDim3, mtp_3d)                                   \
    output_nd = input_nd.broadcast(mtp_3d);                                    \
  } while (0)

#define TILE_COMPUTE_DIM4(input_x_data, shape_x, output_data, shape_output)    \
  do {                                                                         \
    typedef Eigen::TensorMap<Eigen::Tensor<T, kDim4, Eigen::RowMajor>,         \
                             Eigen::Aligned>                                   \
        EigenTensorNd;                                                       \
    EigenTensorNd input_nd((input_x_data), (shape_x).at(0), (shape_x).at(1), \
                             (shape_x).at(kDim2), (shape_x).at(kDim3));        \
    EigenTensorNd output_nd((output_data), (shape_output).at(0),             \
                              (shape_output).at(1), (shape_output).at(kDim2),  \
                              (shape_output).at(kDim3));                       \
    Eigen::array<Eigen::DenseIndex, kDim4> mtp_4d;                             \
    TILE_COMPUTE_GET_MULTIPLE(kDim4, mtp_4d)                                   \
    output_nd = input_nd.broadcast(mtp_4d);                                    \
  } while (0)

#define TILE_COMPUTE_DIM5(input_x_data, shape_x, output_data, shape_output)    \
  do {                                                                         \
    typedef Eigen::TensorMap<Eigen::Tensor<T, kDim5, Eigen::RowMajor>,         \
                             Eigen::Aligned>                                   \
        EigenTensorNd;                                                       \
    EigenTensorNd input_nd((input_x_data), (shape_x).at(0), (shape_x).at(1), \
                             (shape_x).at(kDim2), (shape_x).at(kDim3),         \
                             (shape_x).at(kDim4));                             \
    EigenTensorNd output_nd((output_data), (shape_output).at(0),             \
                              (shape_output).at(1), (shape_output).at(kDim2),  \
                              (shape_output).at(kDim3),                        \
                              (shape_output).at(kDim4));                       \
    Eigen::array<Eigen::DenseIndex, kDim5> mtp_5d;                             \
    TILE_COMPUTE_GET_MULTIPLE(kDim5, mtp_5d)                                   \
    output_nd = input_nd.broadcast(mtp_5d);                                    \
  } while (0)

#define TILE_COMPUTE_DIM6(input_x_data, shape_x, output_data, shape_output)    \
  do {                                                                         \
    typedef Eigen::TensorMap<Eigen::Tensor<T, kDim6, Eigen::RowMajor>,         \
                             Eigen::Aligned>                                   \
        EigenTensorNd;                                                       \
    EigenTensorNd input_nd((input_x_data), (shape_x).at(0), (shape_x).at(1), \
                             (shape_x).at(kDim2), (shape_x).at(kDim3),         \
                             (shape_x).at(kDim4), (shape_x).at(kDim5));        \
    EigenTensorNd output_nd(                                                 \
        (output_data), (shape_output).at(0), (shape_output).at(1),             \
        (shape_output).at(kDim2), (shape_output).at(kDim3),                    \
        (shape_output).at(kDim4), (shape_output).at(kDim5));                   \
    Eigen::array<Eigen::DenseIndex, kDim6> mtp_6d;                             \
    TILE_COMPUTE_GET_MULTIPLE(kDim6, mtp_6d)                                   \
    output_nd = input_nd.broadcast(mtp_6d);                                    \
  } while (0)

#define TILE_COMPUTE_DIM7(input_x_data, shape_x, output_data, shape_output)    \
  do {                                                                         \
    typedef Eigen::TensorMap<Eigen::Tensor<T, kDim7, Eigen::RowMajor>,         \
                             Eigen::Aligned>                                   \
        EigenTensorNd;                                                       \
    EigenTensorNd input_nd((input_x_data), (shape_x).at(0), (shape_x).at(1), \
                             (shape_x).at(kDim2), (shape_x).at(kDim3),         \
                             (shape_x).at(kDim4), (shape_x).at(kDim5),         \
                             (shape_x).at(kDim6));                             \
    EigenTensorNd output_nd(                                                 \
        (output_data), (shape_output).at(0), (shape_output).at(1),             \
        (shape_output).at(kDim2), (shape_output).at(kDim3),                    \
        (shape_output).at(kDim4), (shape_output).at(kDim5),                    \
        (shape_output).at(kDim6));                                             \
    Eigen::array<Eigen::DenseIndex, kDim7> mtp_7d;                             \
    TILE_COMPUTE_GET_MULTIPLE(kDim7, mtp_7d)                                   \
    output_nd = input_nd.broadcast(mtp_7d);                                    \
  } while (0)

#define TILE_COMPUTE_DIM8(input_x_data, shape_x, output_data, shape_output)    \
  do {                                                                         \
    typedef Eigen::TensorMap<Eigen::Tensor<T, kDim8, Eigen::RowMajor>,         \
                             Eigen::Aligned>                                   \
        EigenTensorNd;                                                       \
    EigenTensorNd input_nd((input_x_data), (shape_x).at(0), (shape_x).at(1), \
                             (shape_x).at(kDim2), (shape_x).at(kDim3),         \
                             (shape_x).at(kDim4), (shape_x).at(kDim5),         \
                             (shape_x).at(kDim6), (shape_x).at(kDim7));        \
    EigenTensorNd output_nd(                                                 \
        (output_data), (shape_output).at(0), (shape_output).at(1),             \
        (shape_output).at(kDim2), (shape_output).at(kDim3),                    \
        (shape_output).at(kDim4), (shape_output).at(kDim5),                    \
        (shape_output).at(kDim6), (shape_output).at(kDim7));                   \
    Eigen::array<Eigen::DenseIndex, kDim8> mtp_8d;                             \
    TILE_COMPUTE_GET_MULTIPLE(kDim8, mtp_8d)                                   \
    output_nd = input_nd.broadcast(mtp_8d);                                    \
  } while (0)
}  // namespace

namespace aicpu {
uint32_t TileCpuKernel::TileComputeUsingMemcpy(
    void *dst_addr, void *src_addr, size_t copy_len) const
{
    auto ret = memcpy_s(dst_addr, copy_len, src_addr, copy_len);
    if (ret != 0) {
      KERNEL_LOG_ERROR("failed to call memcpy_s copy len %zu.", copy_len);
      return KERNEL_STATUS_INNER_ERROR;
    }
    return KERNEL_STATUS_OK;
}

uint32_t TileCpuKernel::TileComputeUsingSdma(
    void *dst_addr, void *src_addr, size_t copy_len) const
{
    auto ret = halSdmaCopy(reinterpret_cast<DVdeviceptr>(dst_addr), copy_len,
                          reinterpret_cast<DVdeviceptr>(src_addr), copy_len);
    if (ret != DRV_ERROR_NONE) {
      KERNEL_LOG_ERROR("failed to call halSdmaCopy copy len %zu.", copy_len);
      return KERNEL_STATUS_INNER_ERROR;
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TileCpuKernel::TileComputeWith2DNotUsingEigen(const CpuKernelContext &ctx) {
  Tensor *input_x = ctx.Input(kFirstInputIndex);
  Tensor *output = ctx.Output(kFirstOutputIndex);
  auto input_x_data = static_cast<T *>(input_x->GetData());
  auto output_data = static_cast<T *>(output->GetData());
  const std::vector<int64_t> input_x_dims = input_x->GetTensorShape()->GetDimSizes();
  const int64_t x_first_dim = input_x_dims[kIndexZero];
  const int64_t mul_first_dim = multiples_[kIndexZero];
  const int64_t x_second_dim = input_x_dims[kIndexOne];
  const int64_t mul_second_dim = multiples_[kIndexOne];
  KERNEL_CHECK_FALSE(CheckInt64MulOverflow(x_second_dim, mul_second_dim),
                     KERNEL_STATUS_INNER_ERROR, "int64 mul over flow");
  const int64_t last_axes_dims = x_second_dim * mul_second_dim;
  const uint64_t output_data_size = output->GetDataSize();
  KERNEL_CHECK_FALSE((output_data_size >= static_cast<uint64_t>(x_second_dim * sizeof(T))),
                     KERNEL_STATUS_INNER_ERROR, "memcpy size=[%ld] should less or equal to output data size=[%lu]",
                     x_second_dim * sizeof(T), output_data_size);
  uint32_t result = KERNEL_STATUS_OK;
  auto first_sharder = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      result = CallCopyHook(output_data + i * mul_second_dim * x_second_dim,
                            input_x_data + i * x_second_dim, x_second_dim * sizeof(T));
    }
    return result;
  };
  (void)CpuKernelUtils::ParallelFor(ctx, x_first_dim, 1, first_sharder);
  auto second_sharder = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      for (int64_t j = 1; j < mul_second_dim; j++) {
        result = CallCopyHook(output_data + i * last_axes_dims + j * x_second_dim,
                              output_data + i * last_axes_dims, x_second_dim * sizeof(T));
      }
    }
    return result;
  };
  (void)CpuKernelUtils::ParallelFor(ctx, x_first_dim, 1, second_sharder);
  KERNEL_CHECK_FALSE((output_data_size >= static_cast<uint64_t>(x_first_dim * last_axes_dims * sizeof(T))),
                     KERNEL_STATUS_INNER_ERROR, "memcpy size=[%ld] should less or equal to output data size=[%lu]",
                     x_first_dim * last_axes_dims * sizeof(T), output_data_size);
  auto third_sharder = [&](int64_t start, int64_t end) {
    int64_t cpy_size = x_first_dim * last_axes_dims;
    for (int64_t i = start; i < end; i++) {
      if (i != 0) {
        result = CallCopyHook(output_data + i * cpy_size, output_data, cpy_size * sizeof(T));
      }
    }
    return result;
  };
  (void)CpuKernelUtils::ParallelFor(ctx, mul_first_dim, 1, third_sharder);
  return result;
}

template <typename T>
uint32_t TileCpuKernel::TileComputeWith3DNotUsingEigen(const CpuKernelContext &ctx) {
  Tensor *input_x = ctx.Input(kFirstInputIndex);
  Tensor *output = ctx.Output(kFirstOutputIndex);
  auto input_x_data = static_cast<T *>(input_x->GetData());
  auto output_data = static_cast<T *>(output->GetData());
  const std::vector<int64_t> input_x_dims = input_x->GetTensorShape()->GetDimSizes();
  const int64_t x_first_dim = input_x_dims[kIndexZero];
  const int64_t x_second_dim = input_x_dims[kIndexOne];
  const int64_t x_third_dim = input_x_dims[kIndexTwo];
  const int64_t mul_first_dim = multiples_[kIndexZero];
  const int64_t mul_second_dim = multiples_[kIndexOne];
  const int64_t mul_third_dim = multiples_[kIndexTwo];
  const int64_t last_axes_dims = x_third_dim * mul_third_dim;
  KERNEL_CHECK_FALSE(CheckInt64MulOverflow(x_third_dim, mul_third_dim),
                     KERNEL_STATUS_INNER_ERROR, "int64 mul over flow");
  const int64_t second_axes_dims = x_second_dim * mul_second_dim;
  KERNEL_CHECK_FALSE(CheckInt64MulOverflow(last_axes_dims, second_axes_dims),
                     KERNEL_STATUS_INNER_ERROR, "int64 mul over flow");
  const int64_t last_two_axes_dims = last_axes_dims * second_axes_dims;
  const uint64_t output_data_size = output->GetDataSize();
  KERNEL_CHECK_FALSE((output_data_size >= static_cast<uint64_t>(x_third_dim * sizeof(T))),
                     KERNEL_STATUS_INNER_ERROR,
                     "memcpy size=[%ld] should less or equal to output data size=[%lu]",
                     x_third_dim * sizeof(T), output_data_size);
  KERNEL_HANDLE_ERROR(TileCompute3DSharderFirst<T>(ctx, input_x_data, output_data,
                                                   x_first_dim, x_second_dim, x_third_dim,
                                                   last_axes_dims, second_axes_dims));

  KERNEL_HANDLE_ERROR(TileCompute3DSharderSecond<T>(ctx, output_data, x_first_dim, x_second_dim, x_third_dim,
                                                    mul_third_dim, last_axes_dims, last_two_axes_dims));

  KERNEL_CHECK_FALSE((output_data_size >= static_cast<uint64_t>(last_axes_dims * x_second_dim * sizeof(T))),
                     KERNEL_STATUS_INNER_ERROR,
                     "memcpy size=[%ld] should less or equal to output data size=[%lu]",
                     last_axes_dims * x_second_dim * sizeof(T), output_data_size);
  KERNEL_HANDLE_ERROR(TileCompute3DSharderThird<T>(ctx, output_data, x_first_dim, x_second_dim, mul_second_dim,
                                                   last_axes_dims, last_two_axes_dims));

  KERNEL_CHECK_FALSE((output_data_size >= static_cast<uint64_t>(last_two_axes_dims * x_first_dim * sizeof(T))),
                     KERNEL_STATUS_INNER_ERROR,
                     "memcpy size=[%ld] should less or equal to output data size=[%lu]",
                     last_two_axes_dims * x_first_dim * sizeof(T), output_data_size);
  KERNEL_HANDLE_ERROR(TileCompute3DSharderFourth<T>(ctx, output_data, mul_first_dim, last_two_axes_dims, x_first_dim));

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TileCpuKernel::TileKernelCompute(const CpuKernelContext &ctx) {
  Tensor *input_x = ctx.Input(kFirstInputIndex), *output = ctx.Output(kFirstOutputIndex);
  auto x_shape = input_x->GetTensorShape();
  KERNEL_CHECK_NULLPTR(x_shape, KERNEL_STATUS_PARAM_INVALID, "x shape ptr is nullptr.");
  auto output_shape = output->GetTensorShape();
  KERNEL_CHECK_NULLPTR(output_shape, KERNEL_STATUS_PARAM_INVALID, "output shape ptr is nullptr.");
  const int32_t input_rank = x_shape->GetDims();
  const int64_t output_rank = output_shape->GetDims();
  KERNEL_CHECK_FALSE((input_rank == output_rank), KERNEL_STATUS_PARAM_INVALID,
                     "output rank must be equal to input rank, current input "
                     "rand [%d], output rank [%ld]", input_rank, output_rank);
  const std::vector<int64_t> input_x_dims = x_shape->GetDimSizes(), output_dims = output_shape->GetDimSizes();
  auto input_x_data = static_cast<T *>(input_x->GetData());
  auto output_data = static_cast<T *>(output->GetData());
  bool use_eigen = (std::count(multiples_.begin(), multiples_.end(), 0) > 0) ||
                     (input_x->NumElements() < kParallelShapeSize);
  switch (input_rank) {
    case kDim0:
      *output_data = *input_x_data;
      break;
    case kDim1:
      KERNEL_HANDLE_ERROR(
          TileComputeWith1D<T>(input_x_data, output_data, input_x_dims[kIndexZero], multiples_[kIndexZero]),
          "Tile 1D handle failed.");
      break;
    case kDim2:
      if (use_eigen) {
        TILE_COMPUTE_DIM2(input_x_data, input_x_dims, output_data, output_dims);
      } else {
        KERNEL_HANDLE_ERROR(TileComputeWith2DNotUsingEigen<T>(ctx), "Tile 2D handle failed.");
      }
      break;
    case kDim3:
      if (use_eigen) {
        TILE_COMPUTE_DIM3(input_x_data, input_x_dims, output_data, output_dims);
      } else {
        KERNEL_HANDLE_ERROR(TileComputeWith3DNotUsingEigen<T>(ctx), "Tile 3D handle failed.");
      }
      break;
    case kDim4: TILE_COMPUTE_DIM4(input_x_data, input_x_dims, output_data, output_dims); break;
    case kDim5: TILE_COMPUTE_DIM5(input_x_data, input_x_dims, output_data, output_dims); break;
    case kDim6: TILE_COMPUTE_DIM6(input_x_data, input_x_dims, output_data, output_dims); break;
    case kDim7: TILE_COMPUTE_DIM7(input_x_data, input_x_dims, output_data, output_dims); break;
    case kDim8: TILE_COMPUTE_DIM8(input_x_data, input_x_dims, output_data, output_dims); break;
    default:
      KERNEL_LOG_ERROR("Tile : Unhandled input dimensions [%d].", input_rank);
      return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TileCpuKernel::TileCheckCopySupported(const CpuKernelContext &ctx) {
#if defined(RUN_ON_HOST) || defined(OPS_MATH_AICPU_HOST_KERNEL)
  SetCopyHook(false);
#else
  if (&halSdmaCopy != nullptr) {
    // device and host
    auto input_x_data = reinterpret_cast<T *>(ctx.Input(kFirstInputIndex)->GetData());
    auto output_data = reinterpret_cast<T *>(ctx.Output(kFirstOutputIndex)->GetData());
    auto ret = halSdmaCopy(reinterpret_cast<DVdeviceptr>(output_data), sizeof(T),
                           reinterpret_cast<DVdeviceptr>(input_x_data), sizeof(T));
    if (ret == DRV_ERROR_NOT_SUPPORT) {
      // host use memcpy
      SetCopyHook(false);
    } else if (ret == DRV_ERROR_NONE) {
      // device use sdma
      SetCopyHook(true);
    } else {
      KERNEL_LOG_ERROR("failed to call halSdmaCopy.");
      return KERNEL_STATUS_INNER_ERROR;
    }
  } else {
    SetCopyHook(false);
  }
#endif
return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TileCpuKernel::TileCompute(const CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(TileCheckCopySupported<T>(ctx), "check copy supported failed");
  return TileKernelCompute<T>(ctx);
}

uint32_t TileCpuKernel::GetMultiplesValue(Tensor *tensor,
                                          std::vector<int64_t> &mtp_value) {
  auto type = tensor->GetDataType();
  if (type == DT_INT32) {
    auto data = static_cast<int32_t *>(tensor->GetData());
    for (int64_t i = 0; i < tensor->NumElements(); i++) {
      mtp_value.emplace_back(static_cast<int64_t>(*(data + i)));
    }
  } else if (type == DT_INT64) {
    auto data = static_cast<int64_t *>(tensor->GetData());
    for (int64_t i = 0; i < tensor->NumElements(); i++) {
      mtp_value.emplace_back(*(data + i));
    }
  } else {
    KERNEL_LOG_ERROR("unsupported multiples dtype");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t TileCpuKernel::TileParamCheck(const CpuKernelContext &ctx) {
  is_empty_tensor_ = false;
  auto x_tensor = ctx.Input(kFirstInputIndex);
  const std::vector<int64_t> input_x_dims = x_tensor->GetTensorShape()->GetDimSizes();
  if (IsScalar(input_x_dims)) {
    return KERNEL_STATUS_OK;
  }
  auto multiples_tensor = ctx.Input(kSecondInputIndex);
  const std::vector<int64_t> shape_multiples = multiples_tensor->GetTensorShape()->GetDimSizes();
  auto output_tensor = ctx.Output(kFirstOutputIndex);
  DataType x_dtype = x_tensor->GetDataType();
  DataType out_dtype = output_tensor->GetDataType();
  KERNEL_CHECK_FALSE((x_dtype == out_dtype), KERNEL_STATUS_PARAM_INVALID,
                     "output type [%d] must be same as input dtype [%d].", out_dtype, x_dtype)
  KERNEL_CHECK_FALSE((shape_multiples.size() == 1), KERNEL_STATUS_PARAM_INVALID,
                     "Expected multiples to "
                     "be 1-D tensors , but got [%zu]-D tensors.", input_x_dims.size())
  KERNEL_CHECK_FALSE(
      (multiples_tensor->NumElements() == (unsigned int)input_x_dims.size()),
      KERNEL_STATUS_PARAM_INVALID,
      "Expected the size of multiples to be [%zu], but "
      "got [%ld].", input_x_dims.size(), multiples_tensor->NumElements())
  KERNEL_CHECK_FALSE((input_x_dims.size() >= 1), KERNEL_STATUS_PARAM_INVALID,
                     "Expected the dimension of x to be equal or greater than "
                     "1-D, but got [%zu].", input_x_dims.size())
  KERNEL_CHECK_FALSE(
      (GetMultiplesValue(multiples_tensor, multiples_) == KERNEL_STATUS_OK),
      KERNEL_STATUS_PARAM_INVALID,
      "multiples must be either int32 or int64, "
      "but got [%s].", DTypeStr(multiples_tensor->GetDataType()).c_str())
  std::vector<int64_t> shape_output(input_x_dims.size());
  for (int64_t i = 0; i < (unsigned int)input_x_dims.size(); ++i) {
    int64_t multiple_value = multiples_.at(i);
    if (input_x_dims.at(i) == 0) {
      is_empty_tensor_ = true;
      return KERNEL_STATUS_OK;
    }
    KERNEL_CHECK_FALSE(
        (0 <= multiple_value), KERNEL_STATUS_PARAM_INVALID, "Expected mtp[%ld] shoule be greater than or equal to 0 but got [%ld].", i, multiple_value)
    shape_output[i] = input_x_dims.at(i) * multiple_value;
  }
  std::shared_ptr<TensorShape> output_shape = output_tensor->GetTensorShape();
  output_shape->SetDimSizes(shape_output);
  if (!output_tensor->SetTensorShape(output_shape.get())) {
    KERNEL_LOG_ERROR("Set output shape failed");
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

uint32_t TileCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Tile NormalCheck fail.");
  KERNEL_HANDLE_ERROR(TileParamCheck(ctx), "Tile check params failed.");

  // if input empty tensor, return a empty output
  if (is_empty_tensor_) {
    return KERNEL_STATUS_OK;
  }

  auto x_dtype = ctx.Input(kFirstInputIndex)->GetDataType();
  switch (x_dtype) {
    TILE_COMPUTE_CASE(DT_BOOL, bool, ctx)
    TILE_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    TILE_COMPUTE_CASE(DT_QINT8, int8_t, ctx)
    TILE_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    TILE_COMPUTE_CASE(DT_QUINT8, uint8_t, ctx)
    TILE_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    TILE_COMPUTE_CASE(DT_QINT16, int16_t, ctx)
    TILE_COMPUTE_CASE(DT_QUINT16, uint16_t, ctx)
    TILE_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    TILE_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    TILE_COMPUTE_CASE(DT_QINT32, int32_t, ctx)
    TILE_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    TILE_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    TILE_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    TILE_COMPUTE_CASE(DT_FLOAT, float, ctx)
    TILE_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    TILE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    TILE_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    TILE_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("Tile kernel data type [%u] not support.", x_dtype);
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

OPS_MATH_REGISTER_CPU_KERNELV2(kTile, TileCpuKernel);

template <typename T>
uint32_t TileCpuKernel::TileCompute3DSharderFirst(const CpuKernelContext &ctx, T *input_x_data, T *output_data,
                                                   int64_t x_first_dim, int64_t x_second_dim, int64_t x_third_dim,
                                                   int64_t last_axes_dims, int64_t second_axes_dims) {
  uint32_t result = KERNEL_STATUS_OK;
  auto sharder = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      for (int64_t j = 0; j < x_second_dim; j++) {
        result = CallCopyHook(output_data + i * last_axes_dims * second_axes_dims + j * last_axes_dims,
                              input_x_data + i * x_third_dim * x_second_dim + j * x_third_dim,
                              x_third_dim * sizeof(T));
      }
    }
    return result;
  };
  (void)CpuKernelUtils::ParallelFor(ctx, x_first_dim, 1, sharder);
  return result;
}

template <typename T>
uint32_t TileCpuKernel::TileCompute3DSharderSecond(const CpuKernelContext &ctx, T *output_data,
                                                    int64_t x_first_dim, int64_t x_second_dim, int64_t x_third_dim,
                                                    int64_t mul_third_dim, int64_t last_axes_dims,
                                                    int64_t last_two_axes_dims) {
  uint32_t result = KERNEL_STATUS_OK;
  auto sharder = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      for (int64_t j = 0; j < x_second_dim; j++) {
        for (int64_t k = 1; k < mul_third_dim; k++) {
          result = CallCopyHook(output_data + i * last_two_axes_dims + j * last_axes_dims + k * x_third_dim,
                                output_data + i * last_two_axes_dims + j * last_axes_dims,
                                x_third_dim * sizeof(T));
        }
      }
    }
    return result;
  };
  (void)CpuKernelUtils::ParallelFor(ctx, x_first_dim, 1, sharder);
  return result;
}

template <typename T>
uint32_t TileCpuKernel::TileCompute3DSharderThird(const CpuKernelContext &ctx, T *output_data,
                                                   int64_t x_first_dim, int64_t x_second_dim,
                                                   int64_t mul_second_dim, int64_t last_axes_dims,
                                                   int64_t last_two_axes_dims) {
  uint32_t result = KERNEL_STATUS_OK;
  auto sharder = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      for (int64_t j = 1; j < mul_second_dim; j++) {
        result = CallCopyHook(output_data + i * last_two_axes_dims + j * last_axes_dims * x_second_dim,
                              output_data + i * last_two_axes_dims, last_axes_dims * x_second_dim * sizeof(T));
      }
    }
    return result;
  };
  (void)CpuKernelUtils::ParallelFor(ctx, x_first_dim, 1, sharder);
  return result;
}

template <typename T>
uint32_t TileCpuKernel::TileCompute3DSharderFourth(const CpuKernelContext &ctx, T *output_data,
                                                    int64_t mul_first_dim, int64_t last_two_axes_dims,
                                                    int64_t x_first_dim) {
  uint32_t result = KERNEL_STATUS_OK;
  auto sharder = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      if (i != 0) {
        result = CallCopyHook(output_data + i * last_two_axes_dims * x_first_dim,
                              output_data, last_two_axes_dims * x_first_dim * sizeof(T));
      }
    }
    return result;
  };
  (void)CpuKernelUtils::ParallelFor(ctx, mul_first_dim, 1, sharder);
  return result;
}

template <typename T>
uint32_t TileCpuKernel::TileComputeWith1D(T *input_x_data, T *output_data, int64_t x_dim, int64_t mul_dim) {
  for (int64_t i = 0; i < mul_dim; i++) {
    KERNEL_HANDLE_ERROR(CallCopyHook(output_data + i * x_dim, input_x_data, x_dim * sizeof(T)));
  }
  if (mul_dim == 0) {
    KERNEL_HANDLE_ERROR(CallCopyHook(output_data, input_x_data, x_dim * sizeof(T)));
  }
  return KERNEL_STATUS_OK;
}

}  // namespace aicpu
