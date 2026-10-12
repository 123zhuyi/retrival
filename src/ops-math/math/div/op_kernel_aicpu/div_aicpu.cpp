/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "div_aicpu.h"

#include <complex>
#include <limits>
#include <type_traits>

#include "cmath"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char* kDiv = "Div";
constexpr int64_t kParallelDataNum = 2 * 1024;
constexpr int64_t kParallelDataNumMid = 16 * 1024;
constexpr int64_t kParallelDataNumSameShape = 7 * 1024;
constexpr int64_t kParallelDataNumSameShapeMid = 35 * 1024;

template <typename T>
typename std::enable_if<std::is_signed<T>::value, bool>::type IsDivOverflow(T lhs, T rhs)
{
    return lhs == std::numeric_limits<T>::min() && rhs == static_cast<T>(-1);
}

template <typename T>
typename std::enable_if<!std::is_signed<T>::value, bool>::type IsDivOverflow(T, T)
{
    return false;
}

template <typename T>
typename std::enable_if<std::is_signed<T>::value, bool>::type NeedFloorAdjust(T lhs, T rhs, T mod)
{
    return mod != static_cast<T>(0) && ((lhs < static_cast<T>(0)) != (rhs < static_cast<T>(0)));
}

template <typename T>
typename std::enable_if<!std::is_signed<T>::value, bool>::type NeedFloorAdjust(T, T, T)
{
    return false;
}

template <typename T>
uint32_t CheckDivOverflow(T lhs, T rhs)
{
    if (IsDivOverflow(lhs, rhs)) {
        KERNEL_LOG_ERROR("Invalid argument: Integer division overflow.");
        return aicpu::KERNEL_STATUS_INNER_ERROR;
    }
    return aicpu::KERNEL_STATUS_OK;
}

template <typename T>
uint32_t CheckNoBcastDivOverflow(aicpu::CpuKernelContext& ctx)
{
    if (!std::is_signed<T>::value) {
        return aicpu::KERNEL_STATUS_OK;
    }

    auto input0 = reinterpret_cast<const T*>(ctx.Input(0)->GetData());
    auto input1 = reinterpret_cast<const T*>(ctx.Input(1)->GetData());
    int64_t input0_elements_nums = ctx.Input(0)->NumElements();
    int64_t input1_elements_nums = ctx.Input(1)->NumElements();
    int64_t data_num = ctx.Output(0)->NumElements();
    aicpu::BcastShapeType type = input0_elements_nums == input1_elements_nums ?
                                     aicpu::BcastShapeType::SAME_SHAPE :
                                     (input0_elements_nums == 1 ? aicpu::BcastShapeType::X_ONE_ELEMENT :
                                                                  aicpu::BcastShapeType::Y_ONE_ELEMENT);

    switch (type) {
        case aicpu::BcastShapeType::SAME_SHAPE:
            for (int64_t i = 0; i < data_num; ++i) {
                uint32_t result = CheckDivOverflow(*(input0 + i), *(input1 + i));
                if (result != aicpu::KERNEL_STATUS_OK) {
                    return result;
                }
            }
            break;
        case aicpu::BcastShapeType::X_ONE_ELEMENT:
            for (int64_t i = 0; i < data_num; ++i) {
                uint32_t result = CheckDivOverflow(*input0, *(input1 + i));
                if (result != aicpu::KERNEL_STATUS_OK) {
                    return result;
                }
            }
            break;
        case aicpu::BcastShapeType::Y_ONE_ELEMENT:
            for (int64_t i = 0; i < data_num; ++i) {
                uint32_t result = CheckDivOverflow(*(input0 + i), *input1);
                if (result != aicpu::KERNEL_STATUS_OK) {
                    return result;
                }
            }
            break;
        default:
            KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
            break;
    }
    return aicpu::KERNEL_STATUS_OK;
}

template <typename T>
uint32_t CheckBcastDivOverflow(aicpu::CpuKernelContext& ctx, aicpu::Bcast& bcast)
{
    if (!std::is_signed<T>::value) {
        return aicpu::KERNEL_STATUS_OK;
    }

    auto input0 = reinterpret_cast<const T*>(ctx.Input(0)->GetData());
    auto input1 = reinterpret_cast<const T*>(ctx.Input(1)->GetData());
    int64_t data_num = ctx.Output(0)->NumElements();
    for (int64_t i = 0; i < data_num; ++i) {
        T lhs = *(input0 + bcast.GetBroadcastXIndex(i));
        T rhs = *(input1 + bcast.GetBroadcastYIndex(i));
        uint32_t result = CheckDivOverflow(lhs, rhs);
        if (result != aicpu::KERNEL_STATUS_OK) {
            return result;
        }
    }
    return aicpu::KERNEL_STATUS_OK;
}

inline aicpu::BcastShapeType GetNoBcastShapeType(int64_t input0_elements_nums, int64_t input1_elements_nums)
{
    return input0_elements_nums == input1_elements_nums ?
               aicpu::BcastShapeType::SAME_SHAPE :
               (input0_elements_nums == 1 ? aicpu::BcastShapeType::X_ONE_ELEMENT :
                                           aicpu::BcastShapeType::Y_ONE_ELEMENT);
}

inline uint32_t GetDivParallelCoreNum(
    const aicpu::CpuKernelContext& ctx, int64_t data_num, int64_t parallel_data_num_mid)
{
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (data_num <= parallel_data_num_mid) {
        max_core_num = std::min(max_core_num, 4U);
    }
    if (max_core_num > data_num) {
        max_core_num = data_num;
    }
    return max_core_num;
}

template <typename ComputeFn>
uint32_t RunDivRangeCompute(const aicpu::CpuKernelContext& ctx, int64_t data_num, int64_t parallel_data_num,
    int64_t parallel_data_num_mid, const ComputeFn& compute)
{
    if (data_num < parallel_data_num) {
        return compute(0, data_num);
    }

    uint32_t max_core_num = GetDivParallelCoreNum(ctx, data_num, parallel_data_num_mid);
    auto sharder_div = [&](int64_t start, int64_t end) { (void)compute(start, end); };
    return aicpu::CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_div);
}

template <typename T>
uint32_t ComputeIntDivValue(T lhs, T rhs, T* output)
{
    if (rhs == static_cast<T>(0)) {
        KERNEL_LOG_ERROR("Invalid argument: Division by zero.");
        return aicpu::KERNEL_STATUS_INNER_ERROR;
    }

    T mod = lhs % rhs;
    if (NeedFloorAdjust(lhs, rhs, mod)) {
        *output = lhs / rhs - static_cast<T>(1);
    } else {
        *output = lhs / rhs;
    }
    return aicpu::KERNEL_STATUS_OK;
}

template <typename T, typename LhsGetter, typename RhsGetter>
uint32_t ComputeIntDivRange(
    int64_t start, int64_t end, T* output, const LhsGetter& lhs_getter, const RhsGetter& rhs_getter)
{
    for (int64_t i = start; i < end; ++i) {
        uint32_t result = ComputeIntDivValue(lhs_getter(i), rhs_getter(i), output + i);
        if (result != aicpu::KERNEL_STATUS_OK) {
            return result;
        }
    }
    return aicpu::KERNEL_STATUS_OK;
}

template <typename T, typename LhsGetter, typename RhsGetter>
uint32_t ComputeDivRange(
    int64_t start, int64_t end, T* output, const LhsGetter& lhs_getter, const RhsGetter& rhs_getter)
{
    for (int64_t i = start; i < end; ++i) {
        *(output + i) = lhs_getter(i) / rhs_getter(i);
    }
    return aicpu::KERNEL_STATUS_OK;
}

#define DIV_COMPUTE_CASE_INT(DTYPE, TYPE, CTX)              \
    case (DTYPE): {                                         \
        uint32_t result = DivComputeInt<TYPE>(CTX);         \
        if (result != KERNEL_STATUS_OK) {                   \
            KERNEL_LOG_ERROR("Div kernel compute failed."); \
            return result;                                  \
        }                                                   \
        break;                                              \
    }

#define DIV_COMPUTE_CASE(DTYPE, TYPE, CTX)                  \
    case (DTYPE): {                                         \
        uint32_t result = DivCompute<TYPE>(CTX);            \
        if (result != KERNEL_STATUS_OK) {                   \
            KERNEL_LOG_ERROR("Div kernel compute failed."); \
            return result;                                  \
        }                                                   \
        break;                                              \
    }
} // namespace

namespace aicpu {
uint32_t DivCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kDiv);
    KERNEL_HANDLE_ERROR(DivParamCheck(ctx), "Div check params failed.");

    auto data_type = ctx.Input(0)->GetDataType();
    switch (data_type) {
        DIV_COMPUTE_CASE_INT(DT_INT8, int8_t, ctx)
        DIV_COMPUTE_CASE_INT(DT_INT16, int16_t, ctx)
        DIV_COMPUTE_CASE_INT(DT_INT32, int32_t, ctx)
        DIV_COMPUTE_CASE_INT(DT_INT64, int64_t, ctx)
        DIV_COMPUTE_CASE_INT(DT_UINT8, uint8_t, ctx)
        DIV_COMPUTE_CASE_INT(DT_UINT16, uint16_t, ctx)
        DIV_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
        DIV_COMPUTE_CASE(DT_FLOAT, float, ctx)
        DIV_COMPUTE_CASE(DT_DOUBLE, double, ctx)
        DIV_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
        DIV_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
        default:
            KERNEL_LOG_ERROR("Div kernel data type [%s] not support.", DTypeStr(data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t DivCpuKernel::DivParamCheck(CpuKernelContext& ctx)
{
    Tensor* input_0 = ctx.Input(0);
    Tensor* input_1 = ctx.Input(1);
    Tensor* output = ctx.Output(0);
    KERNEL_CHECK_NULLPTR(input_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 0 data failed.")
    KERNEL_CHECK_NULLPTR(input_1->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 1 data failed.")
    KERNEL_CHECK_NULLPTR(output->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
    DataType input0_type = input_0->GetDataType();
    DataType input1_type = input_1->GetDataType();
    KERNEL_CHECK_FALSE(
        (input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
        "The data type of input0 [%s] need be same with "
        "input1 [%s].",
        DTypeStr(input0_type).c_str(), DTypeStr(input1_type).c_str())
    KERNEL_LOG_DEBUG(
        "DivCpuKernel[%s], input0: size[%llu];"
        "input1: size[%llu], output: size[%llu].",
        ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(), output->GetDataSize());

    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t DivCpuKernel::DivParamCheckZero(CpuKernelContext& ctx)
{
    auto input1 = reinterpret_cast<T*>(ctx.Input(1)->GetData());
    int64_t input1_elements_nums = ctx.Input(1)->NumElements();
    for (int64_t i = 0; i < input1_elements_nums; i++) {
        if (static_cast<double>(*(input1 + i)) == 0) {
            KERNEL_LOG_ERROR("Invalid argument: Division by zero.");
            return KERNEL_STATUS_INNER_ERROR;
        }
    }
    return KERNEL_STATUS_OK;
}

/**
special compute is used in the following situations.
1. the shapes of input1 and input2 are the same
2. input1 is a 1D tensor with only one element or input1 is scalar
3. input2 is a 1D tensor with only one element or input2 is scalar
4. the shapes of input1 and input2 are different
*/
template <typename T>
uint32_t DivCpuKernel::SpecialComputeInt(
    BcastShapeType type, int64_t start, int64_t end, const T* input1, const T* input2, T* output)
{
    switch (type) {
        case BcastShapeType::SAME_SHAPE:
            return ComputeIntDivRange<T>(
                start, end, output, [&](int64_t i) { return *(input1 + i); }, [&](int64_t i) { return *(input2 + i); });
        case BcastShapeType::X_ONE_ELEMENT:
            return ComputeIntDivRange<T>(
                start, end, output, [&](int64_t) { return *input1; }, [&](int64_t i) { return *(input2 + i); });
        case BcastShapeType::Y_ONE_ELEMENT:
            return ComputeIntDivRange<T>(
                start, end, output, [&](int64_t i) { return *(input1 + i); }, [&](int64_t) { return *input2; });
        default:
            KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
            break;
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t DivCpuKernel::SpecialCompute(
    BcastShapeType type, int64_t start, int64_t end, const T* input1, const T* input2, T* output)
{
    switch (type) {
        case BcastShapeType::SAME_SHAPE:
            return ComputeDivRange<T>(
                start, end, output, [&](int64_t i) { return *(input1 + i); }, [&](int64_t i) { return *(input2 + i); });
        case BcastShapeType::X_ONE_ELEMENT:
            return ComputeDivRange<T>(
                start, end, output, [&](int64_t) { return *input1; }, [&](int64_t i) { return *(input2 + i); });
        case BcastShapeType::Y_ONE_ELEMENT:
            return ComputeDivRange<T>(
                start, end, output, [&](int64_t i) { return *(input1 + i); }, [&](int64_t) { return *input2; });
        default:
            KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
            break;
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t DivCpuKernel::NoBcastComputeInt(CpuKernelContext& ctx)
{
    auto in0 = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto in1 = reinterpret_cast<T*>(ctx.Input(1)->GetData());
    auto out = reinterpret_cast<T*>(ctx.Output(0)->GetData());
    int64_t in0_elements_nums = ctx.Input(0)->NumElements();
    int64_t in1_elements_nums = ctx.Input(1)->NumElements();
    int64_t data_num = ctx.Output(0)->NumElements();
    BcastShapeType type = GetNoBcastShapeType(in0_elements_nums, in1_elements_nums);
    return RunDivRangeCompute(ctx, data_num, kParallelDataNumSameShape, kParallelDataNumSameShapeMid,
        [&](int64_t start, int64_t end) { return SpecialComputeInt<T>(type, start, end, in0, in1, out); });
}

template <typename T>
uint32_t DivCpuKernel::NoBcastCompute(CpuKernelContext& ctx)
{
    auto in0 = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto in1 = reinterpret_cast<T*>(ctx.Input(1)->GetData());
    auto out = reinterpret_cast<T*>(ctx.Output(0)->GetData());
    int64_t in0_elements_nums = ctx.Input(0)->NumElements();
    int64_t in1_elements_nums = ctx.Input(1)->NumElements();
    int64_t data_num = ctx.Output(0)->NumElements();
    BcastShapeType type = GetNoBcastShapeType(in0_elements_nums, in1_elements_nums);
    return RunDivRangeCompute(ctx, data_num, kParallelDataNumSameShape, kParallelDataNumSameShapeMid,
        [&](int64_t start, int64_t end) { return SpecialCompute<T>(type, start, end, in0, in1, out); });
}

template <typename T>
uint32_t DivCpuKernel::BcastComputeInt(CpuKernelContext& ctx, Bcast& bcast)
{
    auto in0 = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto in1 = reinterpret_cast<T*>(ctx.Input(1)->GetData());
    auto out = reinterpret_cast<T*>(ctx.Output(0)->GetData());
    int64_t data_num = ctx.Output(0)->NumElements();
    return RunDivRangeCompute(ctx, data_num, kParallelDataNum, kParallelDataNumMid,
        [&](int64_t start, int64_t end) {
            return ComputeIntDivRange<T>(start, end, out, [&](int64_t i) { return *(in0 + bcast.GetBroadcastXIndex(i)); },
                [&](int64_t i) { return *(in1 + bcast.GetBroadcastYIndex(i)); });
        });
}

template <typename T>
uint32_t DivCpuKernel::BcastCompute(CpuKernelContext& ctx, Bcast& bcast)
{
    auto in0 = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto in1 = reinterpret_cast<T*>(ctx.Input(1)->GetData());
    auto out = reinterpret_cast<T*>(ctx.Output(0)->GetData());
    int64_t data_num = ctx.Output(0)->NumElements();
    return RunDivRangeCompute(ctx, data_num, kParallelDataNum, kParallelDataNumMid,
        [&](int64_t start, int64_t end) {
            return ComputeDivRange<T>(start, end, out, [&](int64_t i) { return *(in0 + bcast.GetBroadcastXIndex(i)); },
                [&](int64_t i) { return *(in1 + bcast.GetBroadcastYIndex(i)); });
        });
}

template <typename T>
uint32_t DivCpuKernel::DivComputeInt(CpuKernelContext& ctx)
{
    Tensor* input0_tensor = ctx.Input(0);
    auto input0_shape = input0_tensor->GetTensorShape()->GetDimSizes();
    int64_t input0_elements_nums = input0_tensor->NumElements();
    Tensor* input1_tensor = ctx.Input(1);
    auto input1_shape = input1_tensor->GetTensorShape()->GetDimSizes();
    int64_t input1_elements_nums = input1_tensor->NumElements();
    bool is_need_bcast = (input0_shape == input1_shape) || (input0_elements_nums == 1) || (input1_elements_nums == 1);
    uint32_t result = DivParamCheckZero<T>(ctx);
    if (result != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Invalid argument: Division by zero.");
        return result;
    }

    if (is_need_bcast) {
        result = CheckNoBcastDivOverflow<T>(ctx);
        if (result != KERNEL_STATUS_OK) {
            return result;
        }
        return NoBcastComputeInt<T>(ctx);
    }

    Bcast bcast(input0_shape, input1_shape);
    if (!bcast.IsValid()) {
        KERNEL_LOG_ERROR("[%s] broadcast failed.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    result = CheckBcastDivOverflow<T>(ctx, bcast);
    if (result != KERNEL_STATUS_OK) {
        return result;
    }
    return BcastComputeInt<T>(ctx, bcast);
}

template <typename T>
uint32_t DivCpuKernel::DivCompute(CpuKernelContext& ctx)
{
    Tensor* input0_tensor = ctx.Input(0);
    auto input0_shape = input0_tensor->GetTensorShape()->GetDimSizes();
    int64_t input0_elements_nums = input0_tensor->NumElements();
    Tensor* input1_tensor = ctx.Input(1);
    auto input1_shape = input1_tensor->GetTensorShape()->GetDimSizes();
    int64_t input1_elements_nums = input1_tensor->NumElements();
    bool is_need_bcast = (input0_shape == input1_shape) || (input0_elements_nums == 1) || (input1_elements_nums == 1);
    if (is_need_bcast) {
        return NoBcastCompute<T>(ctx);
    }

    Bcast bcast(input0_shape, input1_shape);
    if (!bcast.IsValid()) {
        KERNEL_LOG_ERROR("[%s] broadcast failed.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return BcastCompute<T>(ctx, bcast);
}

REGISTER_CPU_KERNEL(kDiv, DivCpuKernel);
} // namespace aicpu