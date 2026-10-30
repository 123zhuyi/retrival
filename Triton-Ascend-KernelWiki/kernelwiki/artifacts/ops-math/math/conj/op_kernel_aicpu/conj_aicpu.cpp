/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "conj_aicpu.h"

#include <complex>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char* const kConj = "Conj";
constexpr int64_t kParallelDataNums = 512 * 1024;

#define CONJ_COMPUTE_CASE(DTYPE, TYPE, CTX)                  \
    case (DTYPE): {                                          \
        uint32_t result = ConjCompute<TYPE>(CTX);            \
        if (result != KERNEL_STATUS_OK) {                    \
            KERNEL_LOG_ERROR("Conj kernel compute failed."); \
            return result;                                   \
        }                                                    \
        break;                                               \
    }
} // namespace

namespace aicpu {
uint32_t Conj::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kConj);
    KERNEL_HANDLE_ERROR(ConjCheck(ctx), "[%s] check params failed.", kConj);
    DataType data_type = ctx.Input(0)->GetDataType();
    switch (data_type) {
        CONJ_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
        CONJ_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
        default:
            KERNEL_LOG_ERROR("Conj kernel data type [%s] not support.", DTypeStr(data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t Conj::ConjCheck(const CpuKernelContext& ctx) const
{
    auto input = ctx.Input(0);
    auto output = ctx.Output(0);
    KERNEL_CHECK_NULLPTR(input->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.")
    KERNEL_CHECK_NULLPTR(output->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t Conj::ConjCompute(const CpuKernelContext& ctx) const
{
    auto input_x = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto output_y = reinterpret_cast<T*>(ctx.Output(0)->GetData());
    int64_t data_num = ctx.Input(0)->NumElements();
    int64_t data_size = data_num * static_cast<int64_t>(sizeof(T));
    if (data_size <= kParallelDataNums) {
        for (int64_t i = 0; i < data_num; i++) {
            *(output_y + i) = std::conj(*(input_x + i));
        }
    } else {
        uint32_t min_core_num = 1;
        int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
        if (max_core_num > data_num) {
            max_core_num = data_num;
        }
        auto shard_conj = [&input_x, &output_y](size_t start, size_t end) {
            for (size_t i = start; i < end; i++) {
                *(output_y + i) = std::conj(*(input_x + i));
            }
        };
        KERNEL_CHECK_FALSE((max_core_num != 0), KERNEL_STATUS_PARAM_INVALID,
                           "The max core num is zero, please check input 0 elements num.");
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_conj),
                            "Conj Compute failed.")
    }
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kConj, Conj);
} // namespace aicpu
