
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_REDUCE_OP_H
#define AICPU_KERNELS_NORMALIZED_REDUCE_OP_H
#include <unsupported/Eigen/CXX11/Tensor>
#include <securec.h>

#include "Eigen/Dense"
#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "log.h"
#include "utils/kernel_util.h"
#include "exe_graph/runtime/shape.h"

namespace aicpu {
using QuickVector = gert::Shape;
constexpr size_t kMaxDimNum = 8U;
constexpr size_t kXIdx = 0U;
constexpr size_t kYIdx = 0U;

constexpr int32_t DIM2 = 2;
constexpr int32_t DIM3 = 3;
constexpr int32_t DIM4 = 4;
constexpr int32_t DIM5 = 5;
constexpr int32_t DIM6 = 6;
constexpr int32_t DIM7 = 7;
constexpr int32_t DIM8 = 8;

constexpr Eigen::array<int32_t, 1> kReductionDimsZero{0};
constexpr Eigen::array<int32_t, 1> kReductionDimsOne{1};
constexpr Eigen::array<int32_t, 2> kReductionDimsZeroTwo{0, 2};                 // 2 is for size
constexpr Eigen::array<int32_t, 2> kReductionDimsOneThree{1, 3};                // 2 is for size
constexpr Eigen::array<int32_t, 3> kReductionDimsZeroTwoForth{0, 2, 4};         // 3 is for size
constexpr Eigen::array<int32_t, 3> kReductionDimsOneThreeFive{1, 3, 5};         // 3 is for size
constexpr Eigen::array<int32_t, 4> kReductionDimsZeroTwoForthSix{0, 2, 4, 6};   // 4 is for size
constexpr Eigen::array<int32_t, 4> kReductionDimsOneThreeFiveSeven{1, 3, 5, 7}; // 4 is for size

struct ComplexFloatMeanReducer : Eigen::internal::MeanReducer<std::complex<float>> {
    inline std::complex<float> Finalize(const std::complex<float> accum) const
    {
        return accum / static_cast<std::complex<float>>(scalarCount_);
    }

    template <typename Packet>
    inline std::complex<float> FinalizeBoth(std::complex<float> saccum, const Packet& vaccum) const
    {
        Eigen::internal::scalar_sum_op<std::complex<float>> sum_op;
        return sum_op(saccum, predux(vaccum)) /
               static_cast<std::complex<float>>(
                   scalarCount_ + packetCount_ * Eigen::internal::unpacket_traits<Packet>::size);
    }
};

struct ComplexDoubleMeanReducer : Eigen::internal::MeanReducer<std::complex<double>> {
    inline std::complex<double> Finalize(const std::complex<double> accum) const
    {
        return accum / static_cast<std::complex<double>>(scalarCount_);
    }

    template <typename Packet>
    inline std::complex<double> FinalizeBoth(std::complex<double> saccum, const Packet& vaccum) const
    {
        Eigen::internal::scalar_sum_op<std::complex<double>> sum_op;
        return sum_op(saccum, predux(vaccum)) /
               static_cast<std::complex<double>>(
                   scalarCount_ + packetCount_ * Eigen::internal::unpacket_traits<Packet>::size);
    }
};

uint32_t CheckAxes(const CpuKernelContext* context, const QuickVector& axes, bool (&bitmap)[kMaxDimNum])
{
    auto input_dim_num = static_cast<int64_t>(context->Input(kXIdx)->GetTensorShape()->GetDims());
    for (size_t i = 0U; i < axes.GetDimNum(); i++) {
        auto index = axes[i];
        if (index < -input_dim_num || index >= input_dim_num) {
            KERNEL_LOG_ERROR(
                "%s kernel reduction dimension axes[%zu]=%ld is invalid, input dimNum is %ld",
                context->GetOpType().c_str(), i, index, input_dim_num);
            return KERNEL_STATUS_PARAM_INVALID;
        }
        index = (index + input_dim_num) % input_dim_num;
        bitmap[index] = true;
    }
    return KERNEL_STATUS_OK;
}

uint32_t ReductionHelper(
    CpuKernelContext* context, bool& reduce_first_axis, const QuickVector& axes, QuickVector& input_reshape,
    QuickVector& out_reshape)
{
    bool bitmap[kMaxDimNum] = {false};
    auto ret = CheckAxes(context, axes, bitmap);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    // 从第一个不为1的维度开始
    size_t dim_index = 0;
    auto x_shape = context->Input(kXIdx)->GetTensorShape()->GetDimSizes();
    for (; dim_index < x_shape.size(); ++dim_index) {
        if (x_shape[dim_index] != 1)
            break;
    }
    if (dim_index >= x_shape.size()) {
        // input 所有维度都为1
        reduce_first_axis = true;
    } else {
        reduce_first_axis = bitmap[dim_index];
        input_reshape.AppendDim(x_shape[dim_index]);
        ++dim_index;
        for (; dim_index < x_shape.size(); ++dim_index) {
            const auto size = x_shape[dim_index];
            if (size == 1) {
                bitmap[dim_index] = bitmap[dim_index - 1];
            }

            if (bitmap[dim_index - 1] != bitmap[dim_index]) {
                input_reshape.AppendDim(size);
            } else {
                input_reshape[input_reshape.GetDimNum() - 1] *= size;
            }
        }
    }
    for (size_t i = reduce_first_axis ? 1 : 0; i < input_reshape.GetDimNum();
         i += 2) { // 2 input_reshape在reduce轴和非reudce轴之间切换
        out_reshape.AppendDim(input_reshape[i]);
    }
    return KERNEL_STATUS_OK;
}

template <typename T, typename Reducer>
void ReduceScalar(T* input_data, const QuickVector& input_reshape, T* out_data)
{
    Eigen::TensorMap<Eigen::Tensor<T, 1>> input_eigen(input_data, input_reshape.GetShapeSize());
    Eigen::TensorMap<Eigen::Tensor<T, 0>> out_eigen(out_data);
    Reducer reducer;
    out_eigen = input_eigen.reduce(kReductionDimsZero, reducer);
}

template <typename T, typename Reducer, typename ReductionAxes, int32_t DIMS, int32_t UNREDUCE_DIMS>
void Reduce(
    const QuickVector& input_reshape, const QuickVector& out_reshape, const ReductionAxes& axis_eigen, T* input_data,
    T* out_data)
{
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> input_eigen(input_data, input_reshape.GetShapeSize());
    Eigen::DSizes<Eigen::DenseIndex, DIMS> input_eigen_reshape;
    for (int32_t i = 0; i < DIMS; i++) {
        input_eigen_reshape[i] = input_reshape[i];
    }
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> out_eigen(out_data, out_reshape.GetShapeSize());

    Eigen::DSizes<Eigen::DenseIndex, UNREDUCE_DIMS> out_eigen_reshape;
    for (int32_t i = 0; i < UNREDUCE_DIMS; i++) {
        out_eigen_reshape[i] = out_reshape[i];
    }
    Reducer reducer;
    out_eigen.reshape(out_eigen_reshape) = input_eigen.reshape(input_eigen_reshape).reduce(axis_eigen, reducer);
}

template <typename T, typename Reducer>
void ReduceBigger(
    const QuickVector& input_reshape, const QuickVector& out_reshape, const bool reduce_first_axis, T* input_data,
    T* out_data)
{
    if (input_reshape.GetDimNum() == DIM5 && reduce_first_axis) { // input_reshape dims = 5
        Reduce<T, Reducer, Eigen::array<int32_t, DIM3>, DIM5, DIM2>( // input_reshape dims = 5, reduce dims = 2, unreduce dims = 3
            input_reshape, out_reshape, kReductionDimsZeroTwoForth, input_data, out_data);
    } else if (input_reshape.GetDimNum() == DIM5 && !reduce_first_axis) { // input_reshape dims = 5
        Reduce<T, Reducer, Eigen::array<int32_t, DIM2>, DIM5, DIM3>( // input_reshape dims = 5, reduce dims = 2, unreduce dims = 3
            input_reshape, out_reshape, kReductionDimsOneThree, input_data, out_data);
    } else if (input_reshape.GetDimNum() == DIM6 && reduce_first_axis) { // input_reshape dims = 6
        Reduce<T, Reducer, Eigen::array<int32_t, DIM3>, DIM6, DIM3>( // input_reshape dims = 6, reduce dims = 3, unreduce dims = 3
            input_reshape, out_reshape, kReductionDimsZeroTwoForth, input_data, out_data);
    } else if (input_reshape.GetDimNum() == DIM6 && !reduce_first_axis) { // input_reshape dims = 6
        Reduce<T, Reducer, Eigen::array<int32_t, DIM3>, DIM6, DIM3>( // input_reshape dims = 6, reduce dims = 3, unreduce dims = 3
            input_reshape, out_reshape, kReductionDimsOneThreeFive, input_data, out_data);
    } else if (input_reshape.GetDimNum() == DIM7 && reduce_first_axis) { // input_reshape dims = 7
        Reduce<T, Reducer, Eigen::array<int32_t, DIM4>, DIM7, DIM3>( // input_reshape dims = 7, reduce dims = 4, unreduce dims = 3
            input_reshape, out_reshape, kReductionDimsZeroTwoForthSix, input_data, out_data);
    } else if (input_reshape.GetDimNum() == DIM7 && !reduce_first_axis) { // input_reshape dims = DIM7
        Reduce<T, Reducer, Eigen::array<int32_t, DIM3>, DIM7, DIM4>( // input_reshape dims = 7, reduce dims = 3, unreduce dims = 4
            input_reshape, out_reshape, kReductionDimsOneThreeFive, input_data, out_data);
    } else if (input_reshape.GetDimNum() == DIM8 && reduce_first_axis) { // input_reshape dims = 8
        Reduce<T, Reducer, Eigen::array<int32_t, DIM4>, DIM8, DIM4>( // input_reshape dims = 8, reduce dims = 4, unreduce dims = 4
            input_reshape, out_reshape, kReductionDimsZeroTwoForthSix, input_data, out_data);
    } else if (input_reshape.GetDimNum() == DIM8 && !reduce_first_axis) { // input_reshape dims = 8
        Reduce<T, Reducer, Eigen::array<int32_t, DIM4>, DIM8, DIM4>( // input_reshape dims = 8, reduce dims = 4, unreduce dims = 4
            input_reshape, out_reshape, kReductionDimsOneThreeFiveSeven, input_data, out_data);
    } else {
        KERNEL_LOG_ERROR(
            "reductionOp kernel input_reshape dims should be less or equal than 8, but now is [%zu]",
            input_reshape.GetDimNum());
    }
}

template <typename T, typename Reducer>
void ReduceDim3AndDim4(
    const QuickVector& input_reshape, const QuickVector& out_reshape, const bool reduce_first_axis, T* input_data,
    T* out_data)
{
    if (input_reshape.GetDimNum() == DIM3 && reduce_first_axis) { // input_reshape dims = 3
        Reduce<T, Reducer, Eigen::array<int32_t, DIM2>, DIM3, 1>(           // input_reshape dims = 3, reduce dims = 2
            input_reshape, out_reshape, kReductionDimsZeroTwo, input_data, out_data);
    } else if (input_reshape.GetDimNum() == DIM3 && !reduce_first_axis) { // input_reshape dims = 3
        Reduce<T, Reducer, Eigen::array<int32_t, 1>, DIM3, DIM2>(            // input_reshape dims = 3, unreduce dims = 2
            input_reshape, out_reshape, kReductionDimsOne, input_data, out_data);
    } else if (input_reshape.GetDimNum() == DIM4 && reduce_first_axis) { // input_reshape dims = 4
        Reduce<T, Reducer, Eigen::array<int32_t, DIM2>, DIM4, DIM2>( // input_reshape dims = 4, reduce/unreduce dims = 2
            input_reshape, out_reshape, kReductionDimsZeroTwo, input_data, out_data);
    } else if (input_reshape.GetDimNum() == DIM4 && !reduce_first_axis) { // input_reshape dims = 4
        Reduce<T, Reducer, Eigen::array<int32_t, DIM2>, DIM4, DIM2>( // input_reshape dims = 4, reduce/unreduce dims = 2
            input_reshape, out_reshape, kReductionDimsOneThree, input_data, out_data);
    } else {
        KERNEL_LOG_ERROR(
            "reductionOp kernel input_reshape dims should be less or equal than 8, but now is [%zu]",
            input_reshape.GetDimNum());
    }
}


template <typename T>
uint32_t ProcessingEmptyTensor(CpuKernelContext* context)
{
    auto input_type = context->Input(kXIdx)->GetDataType();
    auto out_tensor = context->Output(kYIdx)->GetData();
    auto output_num = context->Output(kYIdx)->NumElements();
    if (output_num < 1) {
        KERNEL_LOG_INFO("Output num less than 1, output num is [%ld]", output_num);
        return KERNEL_STATUS_OK;
    }
    KERNEL_CHECK_NULLPTR(
        out_tensor, KERNEL_STATUS_PARAM_INVALID, "%s get out_tensor failed.", context->GetOpType().c_str());
    auto out_data = static_cast<T*>(out_tensor);
    KERNEL_CHECK_NULLPTR(
        out_data, KERNEL_STATUS_PARAM_INVALID, "%s get out_data failed.", context->GetOpType().c_str());
    static const std::vector<DataType> OutputZeroValue = {DT_INT8,  DT_INT16,  DT_INT32,      DT_INT64,  DT_UINT8,
                                                          DT_UINT16, DT_UINT32, DT_UINT64, DT_COMPLEX64, DT_COMPLEX128};
    static const std::vector<DataType> OutputNanValue = {DT_FLOAT16, DT_FLOAT, DT_DOUBLE};
    T output_data{};
    if (find(OutputZeroValue.begin(), OutputZeroValue.end(), input_type) != OutputZeroValue.end()) {
        output_data = static_cast<T>(0);
    } else if (find(OutputNanValue.begin(), OutputNanValue.end(), input_type) != OutputNanValue.end()) {
        output_data = static_cast<T>(std::numeric_limits<T>::quiet_NaN());
    } else {
        KERNEL_LOG_WARN(
            "Empty tensor type not support for input tensor, output num is [%ld], "
            "data type is [%s]",
            output_num, DTypeStr(input_type).c_str());
        return KERNEL_STATUS_OK;
    }
    for (int64_t output_index = 0; output_index < output_num; output_index++) {
        *(out_data + output_index) = output_data;
    }
    return KERNEL_STATUS_OK;
}

// 超大函数 重构
template <typename T, typename Reducer>
uint32_t ReduceDispatch(const QuickVector& input_reshape, const QuickVector& out_reshape, bool reduce_first_axis,
    T* input_data, T* out_data)
{
    // 因为tensor的维度最高为8，因此input_reshape的size最高为4
    if (input_reshape.GetDimNum() == 1 && reduce_first_axis) {  // reduce成一个标量
        ReduceScalar<T, Reducer>(input_data, input_reshape, out_data);
    } else if (input_reshape.GetDimNum() == DIM2 && reduce_first_axis) { // input_reshape dims = 2
        Reduce<T, Reducer, Eigen::array<int32_t, 1>, DIM2, 1>(           // input_reshape dims = 2
            input_reshape, out_reshape, kReductionDimsZero, input_data, out_data);
    } else if (input_reshape.GetDimNum() == DIM2 && !reduce_first_axis) { // input_reshape dims = 2
        Reduce<T, Reducer, Eigen::array<int32_t, 1>, DIM2, 1>(            // input_reshape dims = 2
            input_reshape, out_reshape, kReductionDimsOne, input_data, out_data);
    } else if (input_reshape.GetDimNum() == DIM3 || input_reshape.GetDimNum() == DIM4) { // input_reshape dims = 4 or =3
        ReduceDim3AndDim4<T, Reducer>(input_reshape, out_reshape, reduce_first_axis, input_data, out_data);
    } else if (input_reshape.GetDimNum() > DIM4 && input_reshape.GetDimNum() <= DIM8) { // input_reshape dims > 4 and <= 8
        ReduceBigger<T, Reducer>(input_reshape, out_reshape, reduce_first_axis, input_data, out_data);
    } else {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t CheckInputs(CpuKernelContext* context) {
    auto input = context->Input(kXIdx);
    if (input == nullptr) {
        KERNEL_LOG_ERROR("%s get input failed.", context->GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID; 
    }
    auto input_shape = input->GetTensorShape();
    if (input_shape == nullptr) {
        KERNEL_LOG_ERROR("%s get input_shape failed.", context->GetOpType().c_str()); 
        return KERNEL_STATUS_PARAM_INVALID; 
    }
    auto output = context->Output(kYIdx);
    if (output == nullptr) {
        KERNEL_LOG_ERROR("%s get output failed.", context->GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID; 
    }
    auto input_tensor = input->GetData();
    if (input_tensor == nullptr) {
        KERNEL_LOG_ERROR("%s get input_tensor failed.", context->GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID; 
    }
    auto out_tensor = output->GetData();
    if (out_tensor == nullptr) {
        KERNEL_LOG_ERROR("%s get out_tensor failed.", context->GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID; 
    }    
    auto out_data = static_cast<T*>(out_tensor);
    if (out_data == nullptr) {
        KERNEL_LOG_ERROR("%s get out_data failed.", context->GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID; 
    }
    auto out_shape = output->GetTensorShape(); 
    if (out_shape == nullptr) {
        KERNEL_LOG_ERROR("%s get out_shape failed.", context->GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID; 
    }
    return KERNEL_STATUS_OK;
}

template <typename T, typename Reducer>
uint32_t ReductionOp(CpuKernelContext* context, const QuickVector& axes)
{
    if (CheckInputs<T>(context) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID; 
    } 
    
    // empty tensor
    auto input = context->Input(kXIdx);
    auto data_num = input->NumElements();
    if (data_num == 0) {
        KERNEL_LOG_DEBUG("input x elements number is zero");
        auto ret = ProcessingEmptyTensor<T>(context);
        if (ret != KERNEL_STATUS_OK) {
            return ret;
        }
        return KERNEL_STATUS_OK;
    }
    bool reduce_first_axis = false;
    QuickVector input_reshape;
    QuickVector out_reshape;
    auto ret = ReductionHelper(context, reduce_first_axis, axes, input_reshape, out_reshape);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }    
    
    auto input_tensor = input->GetData();
    auto input_data = static_cast<T*>(input_tensor);
    auto output = context->Output(kYIdx);
    auto out_tensor = output->GetData();
    auto out_data = static_cast<T*>(out_tensor);

    if ((input_reshape.GetDimNum() == 0) || (input_reshape.GetDimNum() == 1 && !reduce_first_axis)) {
        // 并没有进行reduce，直接进行内存copy
        auto memcpy_ret = BiggerMemCpy(out_data, output->NumElements() * sizeof(T), input_data, data_num * sizeof(T));
        if (!memcpy_ret) {
            KERNEL_LOG_ERROR("%s memcpy_s failed", context->GetOpType().c_str());
            return KERNEL_STATUS_PARAM_INVALID;
        }
        return KERNEL_STATUS_OK;
    } else {
        if (ReduceDispatch<T, Reducer>(input_reshape, out_reshape, reduce_first_axis, input_data, out_data) ==
            KERNEL_STATUS_PARAM_INVALID) {
            KERNEL_LOG_ERROR("%s kernel input_reshape dims should be less or equal than 8, but now is [%zu]",
                context->GetOpType().c_str(), input_reshape.GetDimNum());

            return KERNEL_STATUS_PARAM_INVALID;
        } else {
            return KERNEL_STATUS_OK;
        }
    }
}
} // namespace aicpu
#endif
