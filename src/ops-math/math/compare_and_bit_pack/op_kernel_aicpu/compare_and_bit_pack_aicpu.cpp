/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compare_and_bit_pack_aicpu.h"

#include <iostream>

#include "Eigen/Dense"
#include "cpu_kernel_utils.h"
#include "securec.h"
#include "utils/kernel_util.h"

namespace {
const char* const kCompareAndBitpack = "CompareAndBitpack";
}

namespace aicpu {
uint32_t CompareAndBitpackCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(ParaCheck(ctx), "CompareAndBitpack ParaCheck fail.");
    auto data_type = ctx.Input(0)->GetDataType();
    uint32_t res = KERNEL_STATUS_OK;
    switch (data_type) {
        case DT_FLOAT16:
            res = CompareAndBitpackCompute<Eigen::half>(ctx);
            break;
        case DT_FLOAT:
            res = CompareAndBitpackCompute<float>(ctx);
            break;
        case DT_DOUBLE:
            res = CompareAndBitpackCompute<double>(ctx);
            break;
        case DT_INT8:
            res = CompareAndBitpackCompute<int8_t>(ctx);
            break;
        case DT_INT16:
            res = CompareAndBitpackCompute<int16_t>(ctx);
            break;
        case DT_INT32:
            res = CompareAndBitpackCompute<int32_t>(ctx);
            break;
        case DT_INT64:
            res = CompareAndBitpackCompute<int64_t>(ctx);
            break;
        case DT_BOOL:
            res = CompareAndBitpackCompute<bool>(ctx);
            break;
        default:
            KERNEL_LOG_ERROR("CompareAndBitpack invalid input type [%s]", DTypeStr(data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
    if (res != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_INNER_ERROR;
    }
    return KERNEL_STATUS_OK;
}

uint32_t CompareAndBitpackCpuKernel::ParaCheck(const CpuKernelContext& ctx) const
{
    Tensor* input0 = ctx.Input(kFirstInputIndex);
    Tensor* input1 = ctx.Input(kSecondInputIndex);
    Tensor* output = ctx.Output(kFirstOutputIndex);

    KERNEL_CHECK_NULLPTR(input0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input0 data failed.")
    KERNEL_CHECK_NULLPTR(input1->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input1 data failed.")
    KERNEL_CHECK_NULLPTR(output->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
    KERNEL_CHECK_NULLPTR(input0->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get input0 shape failed")
    KERNEL_CHECK_NULLPTR(input1->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get input1 shape failed")
    int32_t last_dim_index = input0->GetTensorShape()->GetDims() - 1;
    KERNEL_CHECK_FALSE((IsScalar(input1->GetTensorShape()->GetDimSizes())), KERNEL_STATUS_PARAM_INVALID,
                       "Input1[threshold] must be a scalar");
    KERNEL_CHECK_FALSE((IsVectorOrHigher(input0->GetTensorShape()->GetDimSizes())), KERNEL_STATUS_PARAM_INVALID,
                       "Input0 should be at least a vector, but saw a scalar");
    KERNEL_CHECK_FALSE((((input0->GetTensorShape()->GetDimSize(last_dim_index)) % 8) == 0), KERNEL_STATUS_PARAM_INVALID,
                       "Inner dimension of input0 should be divisible by 8");
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t CompareAndBitpackCpuKernel::CompareAndBitpackCompute(const CpuKernelContext& ctx)
{
    T* input0 = reinterpret_cast<T*>(ctx.Input(kFirstInputIndex)->GetData());
    T* input1 = reinterpret_cast<T*>(ctx.Input(kSecondInputIndex)->GetData());
    uint8_t* output = reinterpret_cast<uint8_t*>(ctx.Output(kFirstOutputIndex)->GetData());
    int64_t data_num = ctx.Output(kFirstOutputIndex)->NumElements();
    T thresh = *input1;
    if (data_num <= kParallelDataNums) {
        for (int64_t i = 0; i < data_num; ++i) {
            uint8_t* out = output + i;
            const T* input = input0 + 8 * i;
            *out = ((((input[0] > thresh) << 7)) | (((input[1] > thresh) << 6)) | (((input[2] > thresh) << 5)) |
                    (((input[3] > thresh) << 4)) | (((input[4] > thresh) << 3)) | (((input[5] > thresh) << 2)) |
                    (((input[6] > thresh) << 1)) | (((input[7] > thresh))));
        }
    } else {
        uint32_t min_core_num = 1;
        int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
        if (max_core_num > data_num) {
            max_core_num = data_num;
        }
        auto shard = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; i++) {
                uint8_t* out = output + i;
                const T* input = input0 + 8 * i;
                *out = ((((input[0] > thresh) << 7)) | (((input[1] > thresh) << 6)) | (((input[2] > thresh) << 5)) |
                        (((input[3] > thresh) << 4)) | (((input[4] > thresh) << 3)) | (((input[5] > thresh) << 2)) |
                        (((input[6] > thresh) << 1)) | (((input[7] > thresh))));
            }
        };
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard),
                            "CompareAndBitpack Compute failed.")
    }
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kCompareAndBitpack, CompareAndBitpackCpuKernel);
} // namespace aicpu
