/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log_aicpu.h"
#include <cfloat>
#include "aicpu/math_aicpu_register.h"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"
#include "utils/eigen_tensor.h"

namespace {
constexpr uint32_t kLogInputNum = 1U;
constexpr uint32_t kLogOutputNum = 1U;
const char* const kLog = "Log";
constexpr int64_t kDataDefaultParallelNum = 60 * 1024;
constexpr int64_t kFloat16AndIntParallelNum = 35 * 1024;
constexpr int64_t kCompAndBf16ParallelNum = 18 * 1024;
constexpr float kLog2Base = 2.0F;
constexpr float kLog10Base = 10.0F;
} // namespace

namespace aicpu {
template <typename T,
          typename std::enable_if<!std::is_same<T, std::complex<float>>::value &&
                                      !std::is_same<T, std::complex<double>>::value &&
                                      !std::is_same<T, Eigen::bfloat16>::value && !std::is_same<T, Eigen::half>::value,
                                  T>::type* = nullptr>
inline void Log2Compute(T* input, T* output, int64_t start, int64_t end)
{
    for (int64_t index = start; index < end; index++) {
        output[index] = std::log2(input[index]);
    }
}

template <typename T,
          typename std::enable_if<std::is_same<T, Eigen::bfloat16>::value || std::is_same<T, Eigen::half>::value,
                                  T>::type* = nullptr>
inline void Log2Compute(T* input, T* output, int64_t start, int64_t end)
{
    for (int64_t index = start; index < end; index++) {
        output[index] = (T)(std::log2(input[index]));
    }
}

template <typename T, typename std::enable_if<std::is_same<T, std::complex<float>>::value ||
                                                  std::is_same<T, std::complex<double>>::value,
                                              T>::type* = nullptr>
inline void Log2Compute(T* input, T* output, int64_t start, int64_t end)
{
    T log_2 = T(std::log(kLog2Base));
    for (int64_t index = start; index < end; index++) {
        output[index] = std::log(input[index]) / log_2;
    }
}

template <typename T,
          typename std::enable_if<std::is_same<T, Eigen::bfloat16>::value || std::is_same<T, Eigen::half>::value,
                                  T>::type* = nullptr>
inline void Log10Compute(T* input, T* output, int64_t start, int64_t end)
{
    for (int64_t index = start; index < end; index++) {
        output[index] = (T)(std::log10(input[index]));
    }
}

template <typename T,
          typename std::enable_if<!std::is_same<T, Eigen::bfloat16>::value && !std::is_same<T, Eigen::half>::value,
                                  T>::type* = nullptr>
inline void Log10Compute(T* input, T* output, int64_t start, int64_t end)
{
    for (int64_t index = start; index < end; index++) {
        output[index] = std::log10(input[index]);
    }
}

template <typename Tin, typename Tout>
inline void ComputeNaturalLog(Tin* input, Tout* output, int64_t start, int64_t end)
{
    for (int64_t index = start; index < end; index++) {
        output[index] = static_cast<Tout>(std::log(input[index]));
    }
}

template <typename Tin, typename Tout>
inline void ComputeShiftLog(Tin* input, Tout* output, float shift, int64_t start, int64_t end)
{
    Tout shift_value = static_cast<Tout>(shift);
    for (int64_t index = start; index < end; index++) {
        output[index] = Eigen::numext::log(shift_value + input[index]);
    }
}

bool NeedParallel(int64_t data_num, DataType dtype)
{
    return (data_num > kDataDefaultParallelNum) || (data_num > kFloat16AndIntParallelNum && dtype == DT_FLOAT16) ||
           (data_num > kCompAndBf16ParallelNum &&
            (dtype == DT_BFLOAT16 || dtype == DT_COMPLEX64 || dtype == DT_COMPLEX128));
}

bool NaturalLogCondition(float base, float shift, float scale)
{
    return (std::fabs(base - (-1.0F)) < FLT_EPSILON) && (std::fabs(shift) < FLT_EPSILON) &&
           (std::fabs(scale - 1.0F) < FLT_EPSILON);
}

bool Log2Condition(float base, float shift, float scale)
{
    return (std::fabs(base - kLog2Base) < FLT_EPSILON) && (std::fabs(shift) < FLT_EPSILON) &&
           (std::fabs(scale - 1.0F) < FLT_EPSILON);
}

bool Log10Condition(float base, float shift, float scale)
{
    return (std::fabs(base - kLog10Base) < FLT_EPSILON) && (std::fabs(shift) < FLT_EPSILON) &&
           (std::fabs(scale - 1.0F) < FLT_EPSILON);
}

template <typename Tin, typename Tout>
uint32_t LogCompute(CpuKernelContext& ctx, float base, float shift, float scale)
{
    Tin* input = reinterpret_cast<Tin*>(ctx.Input(0)->GetData());
    Tout* output = reinterpret_cast<Tout*>(ctx.Output(0)->GetData());

    auto shard_log = [&](int64_t start, int64_t end) {
        if (NaturalLogCondition(base, shift, scale)) {
            ComputeNaturalLog(input, output, start, end);
        } else if (Log2Condition(base, shift, scale)) {
            Log2Compute(input, output, start, end);
        } else if (Log10Condition(base, shift, scale)) {
            Log10Compute(input, output, start, end);
        } else if ((std::fabs(base - (-1.0F)) < FLT_EPSILON) && (std::fabs(shift) >= FLT_EPSILON) &&
                   (std::fabs(scale - 1.0F) < FLT_EPSILON)) {
            ComputeShiftLog(input, output, shift, start, end);
        } else {
            Tout shift_value = static_cast<Tout>(shift);
            Tout scale_value = static_cast<Tout>(scale);
            Tout log_base = (std::fabs(base - (-1.0F)) < FLT_EPSILON) ? static_cast<Tout>(1.0) :
                                                                        static_cast<Tout>(Eigen::numext::log(base));

            for (int64_t index = start; index < end; index++) {
                output[index] = Eigen::numext::log(shift_value + scale_value * input[index]) / log_base;
            }
        }
    };

    int64_t data_num = ctx.Input(0)->NumElements();
    DataType dtype = ctx.Input(0)->GetDataType();
    if (NeedParallel(data_num, dtype)) {
        uint32_t max_core_num = std::max(1U, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, CeilMultiple(data_num, max_core_num), shard_log),
                            "Log Compute failed.")
    } else {
        shard_log(0, data_num);
    }

    return KERNEL_STATUS_OK;
}

template <typename Tin, typename Tout>
uint32_t LogComputeIntegr(CpuKernelContext& ctx, float base, float shift, float scale)
{
    Tin* input = reinterpret_cast<Tin*>(ctx.Input(0)->GetData());
    Tout* output = reinterpret_cast<Tout*>(ctx.Output(0)->GetData());

    auto shard_log = [&](int64_t start, int64_t end) {
        if (NaturalLogCondition(base, shift, scale)) {
            ComputeNaturalLog(input, output, start, end);
        } else if (Log2Condition(base, shift, scale)) {
            for (int64_t index = start; index < end; index++) {
                output[index] = static_cast<Tout>(std::log2(input[index]));
            }
        } else if (Log10Condition(base, shift, scale)) {
            for (int64_t index = start; index < end; index++) {
                output[index] = static_cast<Tout>(std::log10(input[index]));
            }
        } else if ((std::fabs(base - (-1.0F)) < FLT_EPSILON) && (std::fabs(shift) >= FLT_EPSILON) &&
                   (std::fabs(scale - 1.0F) < FLT_EPSILON)) {
            Tout shift_value = static_cast<Tout>(shift);
            for (int64_t index = start; index < end; index++) {
                output[index] = std::log(shift_value + input[index]);
            }
        } else {
            Tout shift_value = static_cast<Tout>(shift);
            Tout scale_value = static_cast<Tout>(scale);
            Tout log_base = (std::fabs(base - (-1.0F)) < FLT_EPSILON) ? log_base = static_cast<Tout>(1.0) :
                                                                        static_cast<Tout>(Eigen::numext::log(base));

            for (int64_t index = start; index < end; index++) {
                output[index] = std::log(shift_value + scale_value * input[index]) / log_base;
            }
        }
    };

    int64_t data_num = ctx.Input(0)->NumElements();
    if (data_num > kFloat16AndIntParallelNum) {
        uint32_t max_core_num = std::max(1U, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, CeilMultiple(data_num, max_core_num), shard_log),
                            "Log Compute failed.")
    } else {
        shard_log(0, data_num);
    }

    return KERNEL_STATUS_OK;
}

static const std::map<
    uint32_t,
    std::map<uint32_t, std::function<uint32_t(CpuKernelContext& ctx, float base_, float shift_, float scale_)>>>
    log_calls = {{DT_UINT8, {{DT_FLOAT, LogComputeIntegr<uint8_t, float>}}},
                 {DT_INT8, {{DT_FLOAT, LogComputeIntegr<int8_t, float>}}},
                 {DT_INT16, {{DT_FLOAT, LogComputeIntegr<int16_t, float>}}},
                 {DT_INT32, {{DT_FLOAT, LogComputeIntegr<int32_t, float>}}},
                 {DT_INT64, {{DT_FLOAT, LogComputeIntegr<int64_t, float>}}},
                 {DT_FLOAT, {{DT_FLOAT, LogCompute<float, float>}}},
                 {DT_DOUBLE, {{DT_DOUBLE, LogCompute<double, double>}}},
                 {DT_FLOAT16, {{DT_FLOAT16, LogCompute<Eigen::half, Eigen::half>}}},
                 {DT_BFLOAT16, {{DT_BFLOAT16, LogCompute<Eigen::bfloat16, Eigen::bfloat16>}}},
                 {DT_BOOL, {{DT_FLOAT, LogComputeIntegr<bool, float>}}},
                 {DT_COMPLEX64, {{DT_COMPLEX64, LogCompute<std::complex<float>, std::complex<float>>}}},
                 {DT_COMPLEX128, {{DT_COMPLEX128, LogCompute<std::complex<double>, std::complex<double>>}}}};

uint32_t LogCpuKernel::LogOpCheck(CpuKernelContext& ctx)
{
    auto input = ctx.Input(0);
    auto output = ctx.Output(0);
    DataType input_type = input->GetDataType();
    DataType output_type = output->GetDataType();

    if (input_type == DT_UINT8 || input_type == DT_INT8 || input_type == DT_INT16 || input_type == DT_INT32 ||
        input_type == DT_INT64 || input_type == DT_BOOL) {
        KERNEL_CHECK_FALSE((output_type == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID,
                           "When the log op input data is [%s], the output type"
                           " must be [DT_FLOAT32]. Current output type is [%s].",
                           DTypeStr(input_type).c_str(), DTypeStr(output_type).c_str())
    } else {
        KERNEL_CHECK_FALSE((input_type == output_type), KERNEL_STATUS_PARAM_INVALID,
                           "The data type of output [%s] need to be the same as input [%s].",
                           DTypeStr(output_type).c_str(), DTypeStr(input_type).c_str())
    }

    AttrValue* base_attr = ctx.GetAttr("base");
    base_ = (base_attr == nullptr) ? -1.0f : (base_attr->GetFloat());
    KERNEL_CHECK_FALSE((IsValueEqual<float>(base_, -1.0f) || (base_ > 0 && !IsValueEqual<float>(base_, 1.0f))),
                       KERNEL_STATUS_PARAM_INVALID,
                       "The Log op base attribute "
                       "[%f] is invaild, please use -1.0, (0, 1) and (1, inf).",
                       base_)

    AttrValue* shift_attr = ctx.GetAttr("shift");
    shift_ = (shift_attr == nullptr) ? 0.0f : (shift_attr->GetFloat());

    AttrValue* scale_attr = ctx.GetAttr("scale");
    scale_ = (scale_attr == nullptr) ? 1.0f : (scale_attr->GetFloat());

    KERNEL_LOG_DEBUG("Log op check params succeeded. base = %f, shift = %f, scale = %f.", base_, shift_, scale_);
    return KERNEL_STATUS_OK;
}

std::uint32_t LogCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kLogInputNum, kLogOutputNum), "Check Log op input and output failed.");
    KERNEL_HANDLE_ERROR(LogOpCheck(ctx), "Check Log op params failed.");

    // empty tensor
    if (ctx.Output(0)->NumElements() == 0) {
        KERNEL_LOG_DEBUG("log op output shape element number is zero.");
        return KERNEL_STATUS_OK;
    }

    auto input_type = ctx.Input(0)->GetDataType();
    auto output_type = ctx.Output(0)->GetDataType();
    uint32_t ret;
    try {
        ret = log_calls.at(input_type).at(output_type)(ctx, base_, shift_, scale_);
    } catch (const std::out_of_range& e) {
        KERNEL_LOG_ERROR("Log op params input x and output y doesn't support tensor "
                         "types: [%s][%s], e.what:[%s]",
                         DTypeStr(input_type).c_str(), DTypeStr(output_type).c_str(), e.what());
        ret = static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }

    return ret;
}

OPS_MATH_REGISTER_CPU_KERNELV2(kLog, LogCpuKernel);
} // namespace aicpu
