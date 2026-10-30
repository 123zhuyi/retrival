/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reshape_aicpu.h"

#include "log.h"
#include "securec.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kReshapeInputNum = 2;
constexpr uint32_t kReshapeOutputNum = 1;
const char* const kReshape = "Reshape";
} // namespace

namespace aicpu {
uint32_t ReshapeCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kReshapeInputNum, kReshapeOutputNum), "[%s] check params failed.", kReshape);

    void* input_data = ctx.Input(0)->GetData();
    KERNEL_CHECK_NULLPTR(input_data, KERNEL_STATUS_PARAM_INVALID, "[%s] get input_data[0] failed.", kReshape);
    void* output_data = ctx.Output(0)->GetData();
    KERNEL_CHECK_NULLPTR(output_data, KERNEL_STATUS_PARAM_INVALID, "[%s] get output_data[0] failed.", kReshape);

    int64_t input_size = static_cast<int64_t>(ctx.Input(0)->GetDataSize());
    int64_t output_size = static_cast<int64_t>(ctx.Output(0)->GetDataSize());
    if (output_size != input_size) {
        KERNEL_LOG_WARN("[%s] output size [%ld] not match input size [%ld].", kReshape, output_size, input_size);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (output_data != input_data) {
        auto cpret = BiggerMemCpy(output_data, output_size, input_data, input_size);
        KERNEL_CHECK_FALSE(
            cpret, KERNEL_STATUS_INNER_ERROR, "[%s] memcpy_s to output failed, destMax [%ld], count [%ld].", kReshape,
            output_size, input_size);
    }
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kReshape, ReshapeCpuKernel);
} // namespace aicpu