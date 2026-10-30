/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mul_aicpu.h"

#include <algorithm>
#include <unordered_map>
#include <functional>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char* const kMul = "Mul";
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
constexpr int64_t kParallelDataNum = 6 * 1024;
constexpr int64_t kParallelDataNumMid = 33 * 1024;
constexpr int64_t kParallelDataNumSameShape = 7 * 1024;
} // namespace

namespace aicpu {
uint32_t MulCpuKernel::MulSameTypeCompute(const CpuKernelContext& ctx)
{
    auto data_type = static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
    switch (data_type) {
        case DT_FLOAT16:
            return MulCompute<Eigen::half>(ctx);
        case DT_BFLOAT16:
            return MulCompute<Eigen::bfloat16>(ctx);
        case DT_FLOAT:
            return MulCompute<float>(ctx);
        case DT_DOUBLE:
            return MulCompute<double>(ctx);
        case DT_INT8:
            return MulCompute<int8_t>(ctx);
        case DT_INT16:
            return MulCompute<int16_t>(ctx);
        case DT_INT32:
            return MulCompute<int32_t>(ctx);
        case DT_INT64:
            return MulCompute<int64_t>(ctx);
        case DT_UINT8:
            return MulCompute<uint8_t>(ctx);
        case DT_UINT16:
            return MulCompute<uint16_t>(ctx);
        case DT_UINT32:
            return MulCompute<uint32_t>(ctx);
        case DT_UINT64:
            return MulCompute<uint64_t>(ctx);
        case DT_COMPLEX64:
            return MulCompute<std::complex<float>>(ctx);
        case DT_COMPLEX128:
            return MulCompute<std::complex<double>>(ctx);
        default:
            KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].",
                             ctx.GetOpType().c_str(), DTypeStr(data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

template <typename T>
uint32_t MulCpuKernel::MulCompute(const CpuKernelContext& ctx)
{
    BCalcInfo calc_info;
    calc_info.input_0 = ctx.Input(kFirstInputIndex);
    calc_info.input_1 = ctx.Input(kSecondInputIndex);
    calc_info.output = ctx.Output(kFirstOutputIndex);
    KERNEL_CHECK_NULLPTR(calc_info.input_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "[%s] Get input 0 data failed",
                         ctx.GetOpType().c_str())
    KERNEL_CHECK_NULLPTR(calc_info.input_1->GetData(), KERNEL_STATUS_PARAM_INVALID, "[%s] Get input 1 data failed",
                         ctx.GetOpType().c_str())
    KERNEL_CHECK_NULLPTR(calc_info.output->GetData(), KERNEL_STATUS_PARAM_INVALID, "[%s] Get output data failed",
                         ctx.GetOpType().c_str())
    KERNEL_LOG_INFO("[%s] Input[0] data size is [%lu], input[1] data size is [%lu], output data size is [%lu].",
                    ctx.GetOpType().c_str(), calc_info.input_0->GetDataSize(), calc_info.input_1->GetDataSize(),
                    calc_info.output->GetDataSize());

    Bcast bcast;
    if (bcast.GenerateBcastInfo(calc_info) != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("[%s] Generate broadcast info failed.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    bcast.GetBcastVec(calc_info);
    return MulDispatch<T>(calc_info);
}

template <typename T>
uint32_t MulCpuKernel::MulDispatch(BCalcInfo& calc_info)
{
    int32_t rank = static_cast<int32_t>(calc_info.shape_out.size());
    switch (rank) {
        case 0: {
            T v0 = *(reinterpret_cast<const T*>(calc_info.input_0->GetData()));
            T v1 = *(reinterpret_cast<const T*>(calc_info.input_1->GetData()));
            T* value_out = reinterpret_cast<T*>(calc_info.output->GetData());
            *(value_out) = v0 * v1;
            return KERNEL_STATUS_OK;
        }
        case kRank1:
            return MulCalculateWithAlignedCheck<kRank1, T>(calc_info);
        case kRank2:
            return MulCalculateWithAlignedCheck<kRank2, T>(calc_info);
        case kRank3:
            return MulCalculateWithAlignedCheck<kRank3, T>(calc_info);
        case kRank4:
            return MulCalculateWithAlignedCheck<kRank4, T>(calc_info);
        case kRank5:
            return MulCalculateWithAlignedCheck<kRank5, T>(calc_info);
        case kRank6:
            return MulCalculateWithAlignedCheck<kRank6, T>(calc_info);
        case kRank7:
            return MulCalculateWithAlignedCheck<kRank7, T>(calc_info);
        case kRank8:
            return MulCalculateWithAlignedCheck<kRank8, T>(calc_info);
        default:
            KERNEL_LOG_ERROR("Rank of output should less than 8 but get [%zu].", calc_info.shape_out.size());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

template <int32_t RANK, typename T>
uint32_t MulCpuKernel::MulCalculateWithAlignedCheck(BCalcInfo& calc_info)
{
    if (AlignedCheck(calc_info)) {
        return MulCalculate<RANK, T, Eigen::Aligned>(calc_info);
    }
    return MulCalculate<RANK, T, Eigen::Unaligned>(calc_info);
}

bool MulCpuKernel::AlignedCheck(const BCalcInfo& calc_info) const
{
    return AddrAlignedCheck(calc_info.input_0->GetData()) && AddrAlignedCheck(calc_info.input_1->GetData()) &&
           AddrAlignedCheck(calc_info.output->GetData());
}

template <int32_t RANK, typename T, int32_t OPTION>
uint32_t MulCpuKernel::MulCalculate(BCalcInfo& calc_info)
{
    Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input0(static_cast<T*>(calc_info.input_0->GetData()),
                                                         calc_info.input_0->GetTensorShape()->NumElements());
    Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input1(static_cast<T*>(calc_info.input_1->GetData()),
                                                         calc_info.input_1->GetTensorShape()->NumElements());
    Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> output(static_cast<T*>(calc_info.output->GetData()),
                                                         calc_info.output->GetTensorShape()->NumElements());
    auto input_shape_0 = calc_info.input_0->GetTensorShape()->GetDimSizes();
    auto input_shape_1 = calc_info.input_1->GetTensorShape()->GetDimSizes();
    if (input_shape_0.empty()) {
        T v0 = *(reinterpret_cast<const T*>(calc_info.input_0->GetData()));
        output = v0 * input1;
        return KERNEL_STATUS_OK;
    }
    if (input_shape_1.empty()) {
        T v1 = *(reinterpret_cast<const T*>(calc_info.input_1->GetData()));
        output = input0 * v1;
        return KERNEL_STATUS_OK;
    }

    Eigen::DSizes<Eigen::DenseIndex, RANK> reshape_0;
    Eigen::DSizes<Eigen::DenseIndex, RANK> reshape_1;
    Eigen::DSizes<Eigen::DenseIndex, RANK> shape_out;
    Eigen::array<Eigen::DenseIndex, RANK> bcast_0;
    Eigen::array<Eigen::DenseIndex, RANK> bcast_1;
    for (int32_t i = 0; i < RANK; i++) {
        reshape_0[(RANK - i) - 1] = calc_info.reshape_0[i];
        reshape_1[(RANK - i) - 1] = calc_info.reshape_1[i];
        shape_out[(RANK - i) - 1] = calc_info.shape_out[i];
        bcast_0[(RANK - i) - 1] = calc_info.bcast_0[i];
        bcast_1[(RANK - i) - 1] = calc_info.bcast_1[i];
    }
    if (input_shape_0 == input_shape_1) {
        output.reshape(shape_out) = input0.reshape(reshape_0) * input1.reshape(reshape_1);
    } else {
        output.reshape(shape_out) = input0.reshape(reshape_0).broadcast(bcast_0) *
                                    input1.reshape(reshape_1).broadcast(bcast_1);
    }
    return KERNEL_STATUS_OK;
}

int64_t GetMulParallelCoreNum(const CpuKernelContext& ctx, int64_t data_num)
{
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(static_cast<int64_t>(min_core_num),
                                    static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx)) - kResvCpuNum);
    if (data_num <= kParallelDataNumMid) {
        max_core_num = std::min(max_core_num, static_cast<int64_t>(4));
    }
    if (max_core_num > data_num) {
        max_core_num = data_num;
    }
    if (max_core_num < 1) {
        max_core_num = 1;
    }
    return max_core_num;
}

template <typename TIn1, typename TIn2, typename TOut>
typename std::enable_if<std::is_same<TIn1, TOut>::value, void>::type inline MulImpl(TIn1 a, TIn2 b, TOut& output)
{
    output = a * static_cast<TIn1>(b);
}

template <typename TIn1, typename TIn2, typename TOut>
typename std::enable_if<std::is_same<TIn2, TOut>::value, void>::type inline MulImpl(TIn1 a, TIn2 b, TOut& output)
{
    output = static_cast<TIn2>(a) * b;
}

template <typename TIn1, typename TIn2, typename TOut>
typename std::enable_if<!std::is_same<TIn1, TOut>::value && !std::is_same<TIn2, TOut>::value,
                        void>::type inline MulImpl(TIn1 a, TIn2 b, TOut& output)
{
    output = static_cast<TOut>(a) * static_cast<TOut>(b);
}

template <typename TIn1, typename TIn2, typename TOut>
uint32_t BcastCompute(const CpuKernelContext& ctx, const Bcast& bcast)
{
    auto in0 = reinterpret_cast<TIn1*>(ctx.Input(0)->GetData());
    auto in1 = reinterpret_cast<TIn2*>(ctx.Input(1)->GetData());
    auto out = reinterpret_cast<TOut*>(ctx.Output(0)->GetData());
    int64_t data_num = ctx.Output(0)->NumElements();
    if (data_num >= kParallelDataNum) {
        int64_t max_core_num = GetMulParallelCoreNum(ctx, data_num);
        if (max_core_num == 0) {
            KERNEL_LOG_ERROR("Mul max_core_num is zero, division by zero.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        auto sharder_mul = [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                MulImpl(*(in0 + bcast.GetBroadcastXIndex(i)), *(in1 + bcast.GetBroadcastYIndex(i)), *(out + i));
            }
        };
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_mul),
                            "Mul Compute failed.")
    } else {
        for (int64_t i = 0; i < data_num; ++i) {
            MulImpl(*(in0 + bcast.GetBroadcastXIndex(i)), *(in1 + bcast.GetBroadcastYIndex(i)), *(out + i));
        }
    }
    return KERNEL_STATUS_OK;
}

template <typename TIn1, typename TIn2, typename TOut>
void SpecialCompute(BcastShapeType type, int64_t start, int64_t end, CpuKernelContext& ctx)
{
    auto in1 = reinterpret_cast<TIn1*>(ctx.Input(0)->GetData());
    auto in2 = reinterpret_cast<TIn2*>(ctx.Input(1)->GetData());
    auto output = reinterpret_cast<TOut*>(ctx.Output(0)->GetData());
    switch (type) {
        case BcastShapeType::SAME_SHAPE:
            for (int64_t i = start; i < end; ++i) {
                MulImpl(*(in1 + i), *(in2 + i), *(output + i));
            }
            break;
        case BcastShapeType::X_ONE_ELEMENT:
            for (int64_t i = start; i < end; ++i) {
                MulImpl(*in1, *(in2 + i), *(output + i));
            }
            break;
        case BcastShapeType::Y_ONE_ELEMENT:
            for (int64_t i = start; i < end; ++i) {
                MulImpl(*(in1 + i), *in2, *(output + i));
            }
            break;
        default:
            KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
            break;
    }
}

template <typename TIn1, typename TIn2, typename TOut>
uint32_t NoBcastCompute(CpuKernelContext& ctx)
{
    int64_t element_num_in0 = ctx.Input(0)->NumElements();
    int64_t element_num_in1 = ctx.Input(1)->NumElements();
    int64_t data_num = ctx.Output(0)->NumElements();
    BcastShapeType type = (element_num_in0 == element_num_in1 ?
                               BcastShapeType::SAME_SHAPE :
                               (element_num_in0 == 1 ? BcastShapeType::X_ONE_ELEMENT : BcastShapeType::Y_ONE_ELEMENT));
    if (data_num >= kParallelDataNumSameShape) {
        int64_t max_core_num = GetMulParallelCoreNum(ctx, data_num);
        if (max_core_num == 0) {
            KERNEL_LOG_ERROR("Mul max_core_num is zero, division by zero.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        auto sharder_mul = [&](int64_t start, int64_t end) { SpecialCompute<TIn1, TIn2, TOut>(type, start, end, ctx); };
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_mul),
                            "Mul Compute failed.")
    } else {
        SpecialCompute<TIn1, TIn2, TOut>(type, 0, data_num, ctx);
    }
    return KERNEL_STATUS_OK;
}

template <typename TIn1, typename TIn2, typename TOut>
uint32_t MulDiffTypeCompute(CpuKernelContext& ctx)
{
    Tensor* tensor_in0 = ctx.Input(0);
    auto shape_in0 = tensor_in0->GetTensorShape()->GetDimSizes();
    Tensor* tensor_in1 = ctx.Input(1);
    auto shape_in1 = tensor_in1->GetTensorShape()->GetDimSizes();

    bool no_need_bcast = (shape_in0 == shape_in1) || (tensor_in0->NumElements() == 1) ||
                         (tensor_in1->NumElements() == 1);
    if (no_need_bcast) {
        return NoBcastCompute<TIn1, TIn2, TOut>(ctx);
    }

    Bcast bcast(shape_in0, shape_in1);
    if (!bcast.IsValid()) {
        KERNEL_LOG_ERROR("[%s] broadcast failed.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return BcastCompute<TIn1, TIn2, TOut>(ctx, bcast);
}

static const std::unordered_map<int32_t, std::unordered_map<int32_t, std::function<uint32_t(CpuKernelContext&)>>>&
GetMulDiffTypeCalls()
{
    static const std::unordered_map<int32_t, std::unordered_map<int32_t, std::function<uint32_t(CpuKernelContext&)>>>
        kcalls = {
            {DT_UINT8,
             {{DT_INT8, MulDiffTypeCompute<uint8_t, int8_t, int16_t>},
              {DT_INT16, MulDiffTypeCompute<uint8_t, int16_t, int16_t>},
              {DT_INT32, MulDiffTypeCompute<uint8_t, int32_t, int32_t>},
              {DT_INT64, MulDiffTypeCompute<uint8_t, int64_t, int64_t>},
              {DT_BFLOAT16, MulDiffTypeCompute<uint8_t, Eigen::bfloat16, Eigen::bfloat16>},
              {DT_FLOAT16, MulDiffTypeCompute<uint8_t, Eigen::half, Eigen::half>},
              {DT_FLOAT, MulDiffTypeCompute<uint8_t, float, float>},
              {DT_DOUBLE, MulDiffTypeCompute<uint8_t, double, double>},
              {DT_COMPLEX64, MulDiffTypeCompute<uint8_t, std::complex<float>, std::complex<float>>},
              {DT_COMPLEX128, MulDiffTypeCompute<uint8_t, std::complex<double>, std::complex<double>>}}},
            {DT_INT8,
             {{DT_INT16, MulDiffTypeCompute<int8_t, int16_t, int16_t>},
              {DT_INT32, MulDiffTypeCompute<int8_t, int32_t, int32_t>},
              {DT_INT64, MulDiffTypeCompute<int8_t, int64_t, int64_t>},
              {DT_BFLOAT16, MulDiffTypeCompute<int8_t, Eigen::bfloat16, Eigen::bfloat16>},
              {DT_FLOAT16, MulDiffTypeCompute<int8_t, Eigen::half, Eigen::half>},
              {DT_FLOAT, MulDiffTypeCompute<int8_t, float, float>},
              {DT_DOUBLE, MulDiffTypeCompute<int8_t, double, double>},
              {DT_UINT8, MulDiffTypeCompute<int8_t, uint8_t, int16_t>},
              {DT_COMPLEX64, MulDiffTypeCompute<int8_t, std::complex<float>, std::complex<float>>},
              {DT_COMPLEX128, MulDiffTypeCompute<int8_t, std::complex<double>, std::complex<double>>}}},
            {DT_INT16,
             {{DT_INT8, MulDiffTypeCompute<int16_t, int8_t, int16_t>},
              {DT_INT32, MulDiffTypeCompute<int16_t, int32_t, int32_t>},
              {DT_INT64, MulDiffTypeCompute<int16_t, int64_t, int64_t>},
              {DT_BFLOAT16, MulDiffTypeCompute<int16_t, Eigen::bfloat16, Eigen::bfloat16>},
              {DT_FLOAT16, MulDiffTypeCompute<int16_t, Eigen::half, Eigen::half>},
              {DT_FLOAT, MulDiffTypeCompute<int16_t, float, float>},
              {DT_DOUBLE, MulDiffTypeCompute<int16_t, double, double>},
              {DT_UINT8, MulDiffTypeCompute<int16_t, uint8_t, int16_t>},
              {DT_COMPLEX64, MulDiffTypeCompute<int16_t, std::complex<float>, std::complex<float>>},
              {DT_COMPLEX128, MulDiffTypeCompute<int16_t, std::complex<double>, std::complex<double>>}}},
            {DT_INT32,
             {{DT_INT8, MulDiffTypeCompute<int32_t, int8_t, int32_t>},
              {DT_INT16, MulDiffTypeCompute<int32_t, int16_t, int32_t>},
              {DT_INT64, MulDiffTypeCompute<int32_t, int64_t, int64_t>},
              {DT_BFLOAT16, MulDiffTypeCompute<int32_t, Eigen::bfloat16, Eigen::bfloat16>},
              {DT_FLOAT16, MulDiffTypeCompute<int32_t, Eigen::half, Eigen::half>},
              {DT_FLOAT, MulDiffTypeCompute<int32_t, float, float>},
              {DT_DOUBLE, MulDiffTypeCompute<int32_t, double, double>},
              {DT_UINT8, MulDiffTypeCompute<int32_t, uint8_t, int32_t>},
              {DT_COMPLEX64, MulDiffTypeCompute<int32_t, std::complex<float>, std::complex<float>>},
              {DT_COMPLEX128, MulDiffTypeCompute<int32_t, std::complex<double>, std::complex<double>>}}},
            {DT_INT64,
             {{DT_INT8, MulDiffTypeCompute<int64_t, int8_t, int64_t>},
              {DT_INT16, MulDiffTypeCompute<int64_t, int16_t, int64_t>},
              {DT_INT32, MulDiffTypeCompute<int64_t, int32_t, int64_t>},
              {DT_BFLOAT16, MulDiffTypeCompute<int64_t, Eigen::bfloat16, Eigen::bfloat16>},
              {DT_FLOAT16, MulDiffTypeCompute<int64_t, Eigen::half, Eigen::half>},
              {DT_FLOAT, MulDiffTypeCompute<int64_t, float, float>},
              {DT_DOUBLE, MulDiffTypeCompute<int64_t, double, double>},
              {DT_UINT8, MulDiffTypeCompute<int64_t, uint8_t, int64_t>},
              {DT_COMPLEX64, MulDiffTypeCompute<int64_t, std::complex<float>, std::complex<float>>},
              {DT_COMPLEX128, MulDiffTypeCompute<int64_t, std::complex<double>, std::complex<double>>}}},
            {DT_BFLOAT16,
             {{DT_INT8, MulDiffTypeCompute<Eigen::bfloat16, int8_t, Eigen::bfloat16>},
              {DT_INT16, MulDiffTypeCompute<Eigen::bfloat16, int16_t, Eigen::bfloat16>},
              {DT_INT32, MulDiffTypeCompute<Eigen::bfloat16, int32_t, Eigen::bfloat16>},
              {DT_INT64, MulDiffTypeCompute<Eigen::bfloat16, int64_t, Eigen::bfloat16>},
              {DT_FLOAT16, MulDiffTypeCompute<Eigen::bfloat16, Eigen::half, float>},
              {DT_FLOAT, MulDiffTypeCompute<Eigen::bfloat16, float, float>},
              {DT_DOUBLE, MulDiffTypeCompute<Eigen::bfloat16, double, double>},
              {DT_UINT8, MulDiffTypeCompute<Eigen::bfloat16, uint8_t, Eigen::bfloat16>},
              {DT_COMPLEX64, MulDiffTypeCompute<Eigen::bfloat16, std::complex<float>, std::complex<float>>},
              {DT_COMPLEX128, MulDiffTypeCompute<Eigen::bfloat16, std::complex<double>, std::complex<double>>}}},
            {DT_FLOAT16,
             {{DT_INT8, MulDiffTypeCompute<Eigen::half, int8_t, Eigen::half>},
              {DT_INT16, MulDiffTypeCompute<Eigen::half, int16_t, Eigen::half>},
              {DT_INT32, MulDiffTypeCompute<Eigen::half, int32_t, Eigen::half>},
              {DT_INT64, MulDiffTypeCompute<Eigen::half, int64_t, Eigen::half>},
              {DT_FLOAT, MulDiffTypeCompute<Eigen::half, float, float>},
              {DT_BFLOAT16, MulDiffTypeCompute<Eigen::half, Eigen::bfloat16, float>},
              {DT_DOUBLE, MulDiffTypeCompute<Eigen::half, double, double>},
              {DT_UINT8, MulDiffTypeCompute<Eigen::half, uint8_t, Eigen::half>},
              {DT_COMPLEX64, MulDiffTypeCompute<Eigen::half, std::complex<float>, std::complex<float>>},
              {DT_COMPLEX128, MulDiffTypeCompute<Eigen::half, std::complex<double>, std::complex<double>>}}},
            {DT_FLOAT,
             {{DT_INT8, MulDiffTypeCompute<float, int8_t, float>},
              {DT_INT16, MulDiffTypeCompute<float, int16_t, float>},
              {DT_INT32, MulDiffTypeCompute<float, int32_t, float>},
              {DT_INT64, MulDiffTypeCompute<float, int64_t, float>},
              {DT_BFLOAT16, MulDiffTypeCompute<float, Eigen::bfloat16, float>},
              {DT_FLOAT16, MulDiffTypeCompute<float, Eigen::half, float>},
              {DT_DOUBLE, MulDiffTypeCompute<float, double, double>},
              {DT_UINT8, MulDiffTypeCompute<float, uint8_t, float>},
              {DT_COMPLEX64, MulDiffTypeCompute<float, std::complex<float>, std::complex<float>>},
              {DT_COMPLEX128, MulDiffTypeCompute<float, std::complex<double>, std::complex<double>>}}},
            {DT_DOUBLE,
             {{DT_INT8, MulDiffTypeCompute<double, int8_t, double>},
              {DT_INT16, MulDiffTypeCompute<double, int16_t, double>},
              {DT_INT32, MulDiffTypeCompute<double, int32_t, double>},
              {DT_INT64, MulDiffTypeCompute<double, int64_t, double>},
              {DT_BFLOAT16, MulDiffTypeCompute<double, Eigen::bfloat16, double>},
              {DT_FLOAT16, MulDiffTypeCompute<double, Eigen::half, double>},
              {DT_FLOAT, MulDiffTypeCompute<double, float, double>},
              {DT_UINT8, MulDiffTypeCompute<double, uint8_t, double>},
              {DT_COMPLEX64, MulDiffTypeCompute<double, std::complex<float>, std::complex<double>>},
              {DT_COMPLEX128, MulDiffTypeCompute<double, std::complex<double>, std::complex<double>>}}},
            {DT_COMPLEX64,
             {{DT_INT8, MulDiffTypeCompute<std::complex<float>, int8_t, std::complex<float>>},
              {DT_INT16, MulDiffTypeCompute<std::complex<float>, int16_t, std::complex<float>>},
              {DT_INT32, MulDiffTypeCompute<std::complex<float>, int32_t, std::complex<float>>},
              {DT_INT64, MulDiffTypeCompute<std::complex<float>, int64_t, std::complex<float>>},
              {DT_BFLOAT16, MulDiffTypeCompute<std::complex<float>, Eigen::bfloat16, std::complex<float>>},
              {DT_FLOAT16, MulDiffTypeCompute<std::complex<float>, Eigen::half, std::complex<float>>},
              {DT_FLOAT, MulDiffTypeCompute<std::complex<float>, float, std::complex<float>>},
              {DT_DOUBLE, MulDiffTypeCompute<std::complex<float>, double, std::complex<double>>},
              {DT_UINT8, MulDiffTypeCompute<std::complex<float>, uint8_t, std::complex<float>>},
              {DT_COMPLEX128, MulDiffTypeCompute<std::complex<float>, std::complex<double>, std::complex<double>>}}},
            {DT_COMPLEX128,
             {{DT_INT8, MulDiffTypeCompute<std::complex<double>, int8_t, std::complex<double>>},
              {DT_INT16, MulDiffTypeCompute<std::complex<double>, int16_t, std::complex<double>>},
              {DT_INT32, MulDiffTypeCompute<std::complex<double>, int32_t, std::complex<double>>},
              {DT_INT64, MulDiffTypeCompute<std::complex<double>, int64_t, std::complex<double>>},
              {DT_BFLOAT16, MulDiffTypeCompute<std::complex<double>, Eigen::bfloat16, std::complex<double>>},
              {DT_FLOAT16, MulDiffTypeCompute<std::complex<double>, Eigen::half, std::complex<double>>},
              {DT_FLOAT, MulDiffTypeCompute<std::complex<double>, float, std::complex<double>>},
              {DT_DOUBLE, MulDiffTypeCompute<std::complex<double>, double, std::complex<double>>},
              {DT_UINT8, MulDiffTypeCompute<std::complex<double>, uint8_t, std::complex<double>>},
              {DT_COMPLEX64, MulDiffTypeCompute<std::complex<double>, std::complex<float>, std::complex<double>>}}}};
    return kcalls;
}

uint32_t MulCpuKernel::Compute(CpuKernelContext& ctx)
{
    if (NormalCheck(ctx, kInputNum, kOutputNum) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    Tensor* input0 = ctx.Input(kFirstInputIndex);
    Tensor* input1 = ctx.Input(kSecondInputIndex);
    if ((input0->GetDataSize() == 0) || (input1->GetDataSize() == 0)) {
        KERNEL_LOG_INFO("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_OK;
    }

    auto dtype_in1 = ctx.Input(kFirstInputIndex)->GetDataType();
    auto dtype_in2 = ctx.Input(kSecondInputIndex)->GetDataType();
    auto dtype_out = ctx.Output(kFirstOutputIndex)->GetDataType();
    KERNEL_LOG_DEBUG("Mul kernel get input1 dtype[%s], input2 dtype[%s], output dtype[%s].",
                     DTypeStr(dtype_in1).c_str(), DTypeStr(dtype_in2).c_str(), DTypeStr(dtype_out).c_str());
    if (dtype_in1 == dtype_in2) {
        return MulSameTypeCompute(ctx);
    }

    const auto& func_map = GetMulDiffTypeCalls().find(dtype_in1);
    if (func_map != GetMulDiffTypeCalls().end()) {
        const auto& funcs = func_map->second.find(dtype_in2);
        if (funcs != func_map->second.end()) {
            return (funcs->second)(ctx);
        }
    }
    return KERNEL_STATUS_PARAM_INVALID;
}

REGISTER_CPU_KERNEL(kMul, MulCpuKernel);
} // namespace aicpu
