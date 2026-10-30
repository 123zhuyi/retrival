/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coordinates_1d_to_2d_aicpu.h"

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const char* const kCoordinates1DTo2D = "Coordinates1DTo2D";
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 3;
const uint32_t kShapeNum = 4;
} // namespace

namespace aicpu {
template <typename T>
uint32_t Coordinates1DTo2DCompute(Tensor* x, Tensor* shape, Tensor* output_row,
                                 Tensor* output_col, Tensor* output_n)
{
    const T* x_data = static_cast<const T*>(x->GetData());
    const T* shape_data = static_cast<const T*>(shape->GetData());
    T* row_data = static_cast<T*>(output_row->GetData());
    T* col_data = static_cast<T*>(output_col->GetData());
    T* n_data = static_cast<T*>(output_n->GetData());

    KERNEL_CHECK_FALSE((shape_data[3] != 0), KERNEL_STATUS_PARAM_INVALID,
                       "Input[shape] element[3] must not be zero, but got[%ld].",
                       static_cast<int64_t>(shape_data[3]));

    T val = *x_data;
    T col_num = shape_data[3];
    *row_data = val / col_num;
    *col_data = val % col_num;
    *n_data = col_num;
    KERNEL_LOG_INFO(
        "Input x[%ld], shape row[%ld], shape col[%ld], "
        "output row index[%ld], output col index[%ld], output n[%ld].",
        static_cast<int64_t>(val), static_cast<int64_t>(shape_data[2]),
        static_cast<int64_t>(col_num), static_cast<int64_t>(*row_data),
        static_cast<int64_t>(*col_data), static_cast<int64_t>(*n_data));
    return KERNEL_STATUS_OK;
}

uint32_t Coordinates1DTo2DCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                        "Check Coordinates1DTo2D params failed.");

    Tensor* x = ctx.Input(0);
    Tensor* shape = ctx.Input(1);
    Tensor* output_row = ctx.Output(0);
    Tensor* output_col = ctx.Output(1);
    Tensor* output_n = ctx.Output(2);

    DataType x_dt = x->GetDataType();
    KERNEL_CHECK_FALSE((x_dt == shape->GetDataType()), KERNEL_STATUS_INNER_ERROR,
                       "Input[x] data type[%s] and input[shape] data type[%s] must be same.",
                       DTypeStr(x_dt).c_str(), DTypeStr(shape->GetDataType()).c_str());

    KERNEL_CHECK_FALSE((shape->NumElements() == kShapeNum), KERNEL_STATUS_INNER_ERROR,
                       "Input[shape] element number must be equal to 4, but got[%ld].",
                       shape->NumElements());

    switch (x_dt) {
        case DT_INT32:
            return Coordinates1DTo2DCompute<int32_t>(x, shape, output_row, output_col, output_n);
        case DT_INT64:
            return Coordinates1DTo2DCompute<int64_t>(x, shape, output_row, output_col, output_n);
        case DT_UINT64:
            return Coordinates1DTo2DCompute<uint64_t>(x, shape, output_row, output_col, output_n);
        default:
            KERNEL_LOG_ERROR("Unsupported input[x] data type[%s]", DTypeStr(x_dt).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

REGISTER_CPU_KERNEL(kCoordinates1DTo2D, Coordinates1DTo2DCpuKernel);
} // namespace aicpu