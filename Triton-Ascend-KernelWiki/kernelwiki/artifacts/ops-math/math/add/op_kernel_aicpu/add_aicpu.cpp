/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "add_aicpu.h"

#include <algorithm>
#include <complex>

#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"

namespace {
const char *const kAdd = "Add";
constexpr int64_t kParallelBytesThresh = 192LL * 1024LL;
// per-shard target ~256 KiB of output. Sized large enough to amortize
constexpr int64_t kBytesPerShard = 256LL * 1024LL;
}  // namespace

namespace aicpu {
uint32_t CheckPermissionType(const CpuKernelContext &ctx) {
  const DataType input0_data_type = ctx.Input(kFirstInputIndex)->GetDataType();
  const DataType input1_data_type = ctx.Input(kSecondInputIndex)->GetDataType();
  const DataType output_data_type = ctx.Output(kFirstOutputIndex)->GetDataType();
  if (input0_data_type != output_data_type) {
    KERNEL_LOG_ERROR(
        "[Add] Output must have the same dtype as input, but got "
        "input0[%s] input1[%s] output[%s].",
        DTypeStr(input0_data_type).c_str(),
        DTypeStr(input1_data_type).c_str(),
        DTypeStr(output_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t AddCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalMathCheck(ctx), "Normal match check params failed.");
  KERNEL_HANDLE_ERROR(CheckPermissionType(ctx), "Type permission check failed.");
  Tensor *input0 = ctx.Input(kFirstInputIndex);
  Tensor *input1 = ctx.Input(kSecondInputIndex);
  if ((input0->GetDataSize() == 0) || (input1->GetDataSize() == 0)) {
    KERNEL_LOG_INFO("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_OK;
  }

  const DataType input0_data_type = input0->GetDataType();
  KERNEL_LOG_INFO("[%s] Compute begin, dtype=%d, in0_bytes=%lu, in1_bytes=%lu, out_bytes=%lu.",
                  ctx.GetOpType().c_str(), static_cast<int>(input0_data_type),
                  input0->GetDataSize(), input1->GetDataSize(),
                  ctx.Output(kFirstOutputIndex)->GetDataSize());

  // choose compute function depend on dataType
  switch (input0_data_type) {
    case DT_FLOAT16:
      return AddCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return AddCompute<float>(ctx);
    case DT_DOUBLE:
      return AddCompute<double>(ctx);
    case DT_INT8:
      return AddCompute<int8_t>(ctx);
    case DT_INT16:
      return AddCompute<int16_t>(ctx);
    case DT_INT32:
      return AddCompute<int32_t>(ctx);
    case DT_INT64:
      return AddCompute<int64_t>(ctx);
    case DT_UINT8:
      return AddCompute<uint8_t>(ctx);
    case DT_UINT16:
      return AddCompute<uint16_t>(ctx);
    case DT_UINT32:
      return AddCompute<uint32_t>(ctx);
    case DT_UINT64:
      return AddCompute<uint64_t>(ctx);
    case DT_COMPLEX64:
      return AddCompute<std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return AddCompute<std::complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR(
          "[%s] Data type of input is not supported, got dtype=[%s]. "
          "Expect one of {FP16,FP32,FP64,INT8..INT64,UINT8..UINT64,CPX64,CPX128}.",
          ctx.GetOpType().c_str(), DTypeStr(input0_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename Body>
uint32_t AddCpuKernel::RunMaybeParallel(const CpuKernelContext &ctx,
                                        const char *tag, int64_t total,
                                        int64_t elem_bytes,
                                        const Body &body) const {
  const int64_t total_bytes = total * elem_bytes;
  if (total_bytes < kParallelBytesThresh) {
    KERNEL_LOG_INFO("[%s] %s serial path, total_bytes=%ld.",
                    ctx.GetOpType().c_str(), tag, total_bytes);
    body(0, total);
    return KERNEL_STATUS_OK;
  }
  // Guard against zero/negative elem_bytes. All current callers pass
  // sizeof(T) which is always >= 1, but this keeps the divisor safe for
  // any future caller and silences static analyzers.
  if (__builtin_expect(elem_bytes <= 0, 0)) {
    KERNEL_LOG_ERROR("[%s] %s invalid elem_bytes=%ld, fallback to serial.",
                     ctx.GetOpType().c_str(), tag, elem_bytes);
    body(0, total);
    return KERNEL_STATUS_OK;
  }
  const int64_t per_unit = std::max<int64_t>(1, kBytesPerShard / elem_bytes);
  KERNEL_LOG_INFO("[%s] %s parallel path, total=%ld, per_unit=%ld.",
                  ctx.GetOpType().c_str(), tag, total, per_unit);
  const uint32_t rc = CpuKernelUtils::ParallelFor(ctx, total, per_unit, body);
  if (__builtin_expect(rc != KERNEL_STATUS_OK, 0)) {
    KERNEL_LOG_ERROR("[%s] %s ParallelFor failed, rc=%u, total=%ld, per_unit=%ld.",
                     ctx.GetOpType().c_str(), tag, rc, total, per_unit);
    return rc;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t AddCpuKernel::AddSameShape(const CpuKernelContext &ctx, const T *x0,
                                    const T *x1, T *y, int64_t total) const {
  auto body = [x0, x1, y](int64_t beg, int64_t end) {
    const T *__restrict__ a = x0 + beg;
    const T *__restrict__ b = x1 + beg;
    T *__restrict__ o = y + beg;
    const int64_t n = end - beg;
    for (int64_t i = 0; i < n; ++i) {
      o[i] = a[i] + b[i];
    }
  };
  return RunMaybeParallel(ctx, "same-shape", total,
                          static_cast<int64_t>(sizeof(T)), body);
}

template <typename T>
uint32_t AddCpuKernel::AddScalarBcast(const CpuKernelContext &ctx, const T *vec,
                                      T scalar_val, T *y,
                                      int64_t total) const {
  auto body = [vec, scalar_val, y](int64_t beg, int64_t end) {
    const T *__restrict__ a = vec + beg;
    T *__restrict__ o = y + beg;
    const T s = scalar_val;
    const int64_t n = end - beg;
    for (int64_t i = 0; i < n; ++i) {
      o[i] = a[i] + s;
    }
  };
  return RunMaybeParallel(ctx, "scalar-bcast", total,
                          static_cast<int64_t>(sizeof(T)), body);
}

template <typename T>
uint32_t AddCpuKernel::AddGenericBcast(const CpuKernelContext &ctx,
                                       BCalcInfo &calc_info) const {
  (void)ctx;
  return AddCalculateWithRankCheck<T>(ctx, calc_info);
}

uint32_t AddCpuKernel::ValidateAndBroadcast(const CpuKernelContext &ctx,
                                            BCalcInfo &calc_info) const {
  // Raw-shape validation (must match the original kernel's failure modes):
  //   1. Any raw tensor rank > 8 is rejected even if size-1 dims could be
  //      squeezed away by Bcast (keeps parity with the original Eigen path
  //      that switches on calc_info.shape_out.size() up to kRank8).
  //   2. Both inputs must broadcast to the declared output shape; otherwise
  //      reject. Bcast internally only derives a compatible broadcast shape
  //      from the two inputs and never cross-checks the output tensor.
  const auto &raw0 = calc_info.input_0->GetTensorShape()->GetDimSizes();
  const auto &raw1 = calc_info.input_1->GetTensorShape()->GetDimSizes();
  const auto &raw_out = calc_info.output->GetTensorShape()->GetDimSizes();
  if (raw0.size() > static_cast<size_t>(kRank8) ||
      raw1.size() > static_cast<size_t>(kRank8) ||
      raw_out.size() > static_cast<size_t>(kRank8)) {
    KERNEL_LOG_ERROR(
        "[%s] Rank of input/output must be <= 8, got in0_rank=%zu, "
        "in1_rank=%zu, out_rank=%zu.",
        ctx.GetOpType().c_str(), raw0.size(), raw1.size(), raw_out.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  Bcast bcast;
  if (bcast.GenerateBcastInfo(calc_info) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("[%s] Generate broadcast info failed.",
                     ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  bcast.GetBcastVec(calc_info);

  // Validate declared output shape against the broadcast result. The rank
  // must match and every dim must equal the broadcast dim.
  if (raw_out.size() != calc_info.shape_out.size()) {
    KERNEL_LOG_ERROR(
        "[%s] Output rank [%zu] does not match broadcast rank [%zu].",
        ctx.GetOpType().c_str(), raw_out.size(), calc_info.shape_out.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t i = 0; i < raw_out.size(); ++i) {
    if (raw_out[i] != calc_info.shape_out[i]) {
      KERNEL_LOG_ERROR(
          "[%s] Output dim[%zu]=%ld mismatches broadcast dim=%ld.",
          ctx.GetOpType().c_str(), i, raw_out[i], calc_info.shape_out[i]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t AddCpuKernel::AddCompute(const CpuKernelContext &ctx) const {
  BCalcInfo calc_info;
  calc_info.input_0 = ctx.Input(kFirstInputIndex);
  calc_info.input_1 = ctx.Input(kSecondInputIndex);
  calc_info.output = ctx.Output(kFirstOutputIndex);

  KERNEL_CHECK_NULLPTR(calc_info.input_0->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "[%s] Get input[0] data failed",
                       ctx.GetOpType().c_str())
  KERNEL_CHECK_NULLPTR(calc_info.input_1->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "[%s] Get input[1] data failed",
                       ctx.GetOpType().c_str())
  KERNEL_CHECK_NULLPTR(calc_info.output->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "[%s] Get output data failed", ctx.GetOpType().c_str())

  T *const x0_ptr = PtrToPtr<void, T>(calc_info.input_0->GetData());
  T *const x1_ptr = PtrToPtr<void, T>(calc_info.input_1->GetData());
  T *const y_ptr = PtrToPtr<void, T>(calc_info.output->GetData());

  const int64_t n0 = calc_info.input_0->NumElements();
  const int64_t n1 = calc_info.input_1->NumElements();
  const int64_t ny = calc_info.output->NumElements();

  KERNEL_LOG_INFO("[%s] Input[0] size=%lu, Input[1] size=%lu, Output size=%lu.",
                  ctx.GetOpType().c_str(), calc_info.input_0->GetDataSize(),
                  calc_info.input_1->GetDataSize(),
                  calc_info.output->GetDataSize());

  const uint32_t vrc = ValidateAndBroadcast(ctx, calc_info);
  if (vrc != KERNEL_STATUS_OK) {
    return vrc;
  }

  const auto &raw0 = calc_info.input_0->GetTensorShape()->GetDimSizes();
  const auto &raw1 = calc_info.input_1->GetTensorShape()->GetDimSizes();
  const auto &raw_out = calc_info.output->GetTensorShape()->GetDimSizes();

  if (raw0 == raw1 && raw0 == raw_out && n0 == n1 && n0 == ny) {
    KERNEL_LOG_INFO("[%s] same-shape branch selected, elems=%ld.",
                    ctx.GetOpType().c_str(), ny);
    return AddSameShape<T>(ctx, x0_ptr, x1_ptr, y_ptr, ny);
  }
  if (n0 == 1 && n1 == ny && raw1 == raw_out) {
    KERNEL_LOG_INFO("[%s] x0-scalar bcast branch selected, elems=%ld.",
                    ctx.GetOpType().c_str(), ny);
    return AddScalarBcast<T>(ctx, x1_ptr, *x0_ptr, y_ptr, ny);
  }
  if (n1 == 1 && n0 == ny && raw0 == raw_out) {
    KERNEL_LOG_INFO("[%s] x1-scalar bcast branch selected, elems=%ld.",
                    ctx.GetOpType().c_str(), ny);
    return AddScalarBcast<T>(ctx, x0_ptr, *x1_ptr, y_ptr, ny);
  }
  return AddGenericBcast<T>(ctx, calc_info);
}

template <typename T>
uint32_t AddCpuKernel::AddCalculateWithRankCheck(const CpuKernelContext &ctx,
                                                 BCalcInfo &calc_info) const {
  const int32_t rank = static_cast<int32_t>(calc_info.shape_out.size());
  switch (rank) {
    case 0: {
      // Rank-0 already handled by same-shape fast path, but keep for safety.
      const T v0 = *PtrToPtr<void, const T>(calc_info.input_0->GetData());
      const T v1 = *PtrToPtr<void, const T>(calc_info.input_1->GetData());
      T *value_out = PtrToPtr<void, T>(calc_info.output->GetData());
      *value_out = v0 + v1;
      return KERNEL_STATUS_OK;
    }
    case kRank1:
      return AddCalculateWithAlignedCheck<kRank1, T>(ctx, calc_info);
    case kRank2:
      return AddCalculateWithAlignedCheck<kRank2, T>(ctx, calc_info);
    case kRank3:
      return AddCalculateWithAlignedCheck<kRank3, T>(ctx, calc_info);
    case kRank4:
      return AddCalculateWithAlignedCheck<kRank4, T>(ctx, calc_info);
    case kRank5:
      return AddCalculateWithAlignedCheck<kRank5, T>(ctx, calc_info);
    case kRank6:
      return AddCalculateWithAlignedCheck<kRank6, T>(ctx, calc_info);
    case kRank7:
      return AddCalculateWithAlignedCheck<kRank7, T>(ctx, calc_info);
    case kRank8:
      return AddCalculateWithAlignedCheck<kRank8, T>(ctx, calc_info);
    default:
      KERNEL_LOG_ERROR(
          "[%s] Rank of output must be in [0,8], got rank=%zu.",
          ctx.GetOpType().c_str(), calc_info.shape_out.size());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <int32_t RANK, typename T>
uint32_t AddCpuKernel::AddCalculateWithAlignedCheck(
    const CpuKernelContext &ctx, BCalcInfo &calc_info) const {
  (void)ctx;
  if (AlignedCheck(calc_info)) {
    return AddCalculate<RANK, T, Eigen::Aligned>(calc_info);
  }
  return AddCalculate<RANK, T, Eigen::Unaligned>(calc_info);
}

bool AddCpuKernel::AlignedCheck(const BCalcInfo &calc_info) const {
  return AddrAlignedCheck(calc_info.input_0->GetData()) &&
         AddrAlignedCheck(calc_info.input_1->GetData()) &&
         AddrAlignedCheck(calc_info.output->GetData());
}

template <int32_t RANK, typename T, int32_t OPTION>
uint32_t AddCpuKernel::AddCalculate(BCalcInfo &calc_info) const {
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input0(
      PtrToPtr<void, T>(calc_info.input_0->GetData()),
      calc_info.input_0->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input1(
      PtrToPtr<void, T>(calc_info.input_1->GetData()),
      calc_info.input_1->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> output(
      PtrToPtr<void, T>(calc_info.output->GetData()),
      calc_info.output->GetTensorShape()->NumElements());
  const auto &input_shape_0 = calc_info.input_0->GetTensorShape()->GetDimSizes();
  const auto &input_shape_1 = calc_info.input_1->GetTensorShape()->GetDimSizes();
  if (input_shape_0.empty()) {
    const T v0 = *PtrToPtr<void, const T>(calc_info.input_0->GetData());
    output = v0 + input1;
    return KERNEL_STATUS_OK;
  }

  if (input_shape_1.empty()) {
    const T v1 = *PtrToPtr<void, const T>(calc_info.input_1->GetData());
    output = input0 + v1;
    return KERNEL_STATUS_OK;
  }

  Eigen::DSizes<Eigen::DenseIndex, RANK> reshape0;
  Eigen::DSizes<Eigen::DenseIndex, RANK> reshape1;
  Eigen::DSizes<Eigen::DenseIndex, RANK> shape_out;
  Eigen::array<Eigen::DenseIndex, RANK> bcast0;
  Eigen::array<Eigen::DenseIndex, RANK> bcast1;

  for (int32_t i = 0; i < RANK; i++) {
    reshape0[(RANK - i) - 1] = calc_info.reshape_0[i];
    reshape1[(RANK - i) - 1] = calc_info.reshape_1[i];
    shape_out[(RANK - i) - 1] = calc_info.shape_out[i];
    bcast0[(RANK - i) - 1] = calc_info.bcast_0[i];
    bcast1[(RANK - i) - 1] = calc_info.bcast_1[i];
  }
  output.reshape(shape_out) = input0.reshape(reshape0).broadcast(bcast0) +
                              input1.reshape(reshape1).broadcast(bcast1);
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kAdd, AddCpuKernel);
}  // namespace aicpu
