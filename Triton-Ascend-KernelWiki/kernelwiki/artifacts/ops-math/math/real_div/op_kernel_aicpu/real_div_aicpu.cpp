/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "real_div_aicpu.h"

#include <stdint.h>
#include <algorithm>
#include <vector>

#include "Eigen/Dense"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"

namespace aicpu {
namespace {
const char* const kDIV = "Div";
const char* const kRealDiv = "RealDiv";
constexpr int64_t kParallelDataNum = 2 * 1024;
constexpr int64_t kParallelDataNumMid = 16 * 1024;
constexpr int64_t kParallelDataNumSameShape = 7 * 1024;
constexpr int64_t kParallelDataNumSameShapeMid = 35 * 1024;
constexpr int32_t kMaxBcastDims = 8;
constexpr uint32_t kReserveCpuNum = 2U;

// BcastDivInfo: pre-computed stride info for stride-based broadcast iteration.
// Replaces per-element GetBroadcastXIndex/YIndex (which does O(ndims) integer
// divisions per element) with an O(1)-amortized carry-propagation scheme.
struct BcastDivInfo {
    int32_t ndims;
    int64_t out_shape[kMaxBcastDims];
    int64_t x_strides[kMaxBcastDims];   // effective input-x stride; 0 for broadcast dims
    int64_t y_strides[kMaxBcastDims];   // effective input-y stride; 0 for broadcast dims
    int64_t out_strides[kMaxBcastDims]; // row-major strides of the output shape
    int64_t total_elements;
};

// Compute BcastDivInfo from the two input shapes.
// Steps as follows
//   1. Pad to equal rank, validate broadcast, compute output shape.
//   2. Compute effective strides (natural stride for non-broadcast dims, 0 otherwise).
//   3. Remove output-size-1 dimensions (no iteration needed).
//   4. Collapse adjacent contiguous dimensions to reduce loop depth.
bool ComputeBcastDivInfo(const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape, BcastDivInfo& info)
{
    const int32_t x_rank = static_cast<int32_t>(x_shape.size());
    const int32_t y_rank = static_cast<int32_t>(y_shape.size());
    const int32_t max_rank = std::max(x_rank, y_rank);
    if (max_rank == 0 || max_rank > kMaxBcastDims) {
        return false;
    }

    // Step 1: Pad shorter shape with leading 1s & validate.
    int64_t xp[kMaxBcastDims], yp[kMaxBcastDims], out[kMaxBcastDims];
    for (int32_t i = 0; i < max_rank; ++i) {
        xp[i] = (i >= max_rank - x_rank) ? x_shape[i - (max_rank - x_rank)] : 1;
        yp[i] = (i >= max_rank - y_rank) ? y_shape[i - (max_rank - y_rank)] : 1;
    }
    for (int32_t i = 0; i < max_rank; ++i) {
        if (xp[i] == yp[i]) {
            out[i] = xp[i];
        } else if (xp[i] == 1) {
            out[i] = yp[i];
        } else if (yp[i] == 1) {
            out[i] = xp[i];
        } else {
            return false;
        }
    }

    // Step 2: Natural strides from each input's own shape.
    int64_t xn[kMaxBcastDims], yn[kMaxBcastDims];
    xn[max_rank - 1] = 1;
    yn[max_rank - 1] = 1;
    for (int32_t d = max_rank - 2; d >= 0; --d) {
        xn[d] = xn[d + 1] * xp[d + 1];
        yn[d] = yn[d + 1] * yp[d + 1];
    }

    // Effective stride = natural stride if dim matches output, else 0 (broadcast).
    int64_t xe[kMaxBcastDims], ye[kMaxBcastDims];
    for (int32_t d = 0; d < max_rank; ++d) {
        xe[d] = (xp[d] == out[d]) ? xn[d] : 0;
        ye[d] = (yp[d] == out[d]) ? yn[d] : 0;
    }

    // Step 3: Remove output dims of size 1 (no work to iterate).
    int64_t to[kMaxBcastDims], tx[kMaxBcastDims], ty[kMaxBcastDims];
    int32_t ndims = 0;
    for (int32_t d = 0; d < max_rank; ++d) {
        if (out[d] != 1) {
            to[ndims] = out[d];
            tx[ndims] = xe[d];
            ty[ndims] = ye[d];
            ndims++;
        }
    }
    if (ndims == 0) {
        info.ndims = 1;
        info.out_shape[0] = 1;
        info.x_strides[0] = 1;
        info.y_strides[0] = 1;
        info.out_strides[0] = 1;
        info.total_elements = 1;
        return true;
    }

    // Step 4: Collapse contiguous dims.
    // Two adjacent dims (prev, cur) can be merged if for BOTH x and y:
    //   prev_stride == cur_stride * cur_size  (or both strides are 0).
    info.out_shape[0] = to[0];
    info.x_strides[0] = tx[0];
    info.y_strides[0] = ty[0];
    int32_t collapsed = 1;

    for (int32_t d = 1; d < ndims; ++d) {
        bool x_ok =
            (info.x_strides[collapsed - 1] == tx[d] * to[d]) || (info.x_strides[collapsed - 1] == 0 && tx[d] == 0);
        bool y_ok =
            (info.y_strides[collapsed - 1] == ty[d] * to[d]) || (info.y_strides[collapsed - 1] == 0 && ty[d] == 0);

        if (x_ok && y_ok) {
            info.out_shape[collapsed - 1] *= to[d];
            info.x_strides[collapsed - 1] = tx[d];
            info.y_strides[collapsed - 1] = ty[d];
        } else {
            info.out_shape[collapsed] = to[d];
            info.x_strides[collapsed] = tx[d];
            info.y_strides[collapsed] = ty[d];
            collapsed++;
        }
    }

    info.ndims = collapsed;
    info.out_strides[collapsed - 1] = 1;
    for (int32_t d = collapsed - 2; d >= 0; --d) {
        info.out_strides[d] = info.out_strides[d + 1] * info.out_shape[d + 1];
    }
    info.total_elements = info.out_strides[0] * info.out_shape[0];
    return true;
}

template <typename T>
void SpecialComputeImpl(BcastShapeType type, int64_t start, int64_t end, const T* in0, const T* in1, T* out)
{
    switch (type) {
        case BcastShapeType::SAME_SHAPE:
            for (int64_t i = start; i < end; ++i) {
                out[i] = in0[i] / in1[i];
            }
            break;
        case BcastShapeType::X_ONE_ELEMENT:
            for (int64_t i = start; i < end; ++i) {
                out[i] = in0[0] / in1[i];
            }
            break;
        case BcastShapeType::Y_ONE_ELEMENT:
            for (int64_t i = start; i < end; ++i) {
                out[i] = in0[i] / in1[0];
            }
            break;
        default:
            break;
    }
}

template <typename T>
uint32_t NoBcastComputeImpl(const CpuKernelContext& ctx)
{
    auto in0 = reinterpret_cast<T*>(ctx.Input(kFirstInputIndex)->GetData());
    auto in1 = reinterpret_cast<T*>(ctx.Input(kSecondInputIndex)->GetData());
    auto out = reinterpret_cast<T*>(ctx.Output(kFirstOutputIndex)->GetData());
    int64_t in0_num = ctx.Input(kFirstInputIndex)->NumElements();
    int64_t in1_num = ctx.Input(kSecondInputIndex)->NumElements();
    int64_t data_num = ctx.Output(kFirstOutputIndex)->NumElements();

    BcastShapeType type = (in0_num == in1_num) ? BcastShapeType::SAME_SHAPE :
                          (in0_num == 1)       ? BcastShapeType::X_ONE_ELEMENT :
                                                 BcastShapeType::Y_ONE_ELEMENT;

    if (data_num >= kParallelDataNumSameShape) {
        uint32_t min_core_num = 1U;
        uint32_t max_core_num = std::max(min_core_num, std::max(CpuKernelUtils::GetCPUNum(ctx), kReserveCpuNum) - kReserveCpuNum);
        if (data_num <= kParallelDataNumSameShapeMid) {
            max_core_num = std::min(max_core_num, 4U);
        }
        if (static_cast<int64_t>(max_core_num) > data_num) {
            max_core_num = static_cast<uint32_t>(data_num);
        }
        auto sharder = [&type, &in0, &in1, &out](int64_t start, int64_t end) {
            SpecialComputeImpl<T>(type, start, end, in0, in1, out);
        };
        KERNEL_HANDLE_ERROR(
            CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder), "RealDiv Compute failed.")
    } else {
        SpecialComputeImpl<T>(type, 0, data_num, in0, in1, out);
    }
    return KERNEL_STATUS_OK;
}

// BcastComputeImpl: stride-based broadcast division.
//   1. Decompose the shard-start index into multi-dim coordinates (O(ndims)
//      divisions done ONCE per shard).
//   2. Iterate the innermost dimension in a tight loop with constant strides
//      (zero divisions, compiler auto-vectorisable on ARM NEON).
//   3. On innermost-dim boundary, propagate a carry to outer dims (amortised
//      O(1) additions per element).
//
// This reduces index-computation cost from ~2*ndims divisions per element
// to essentially zero in the hot path.
template <typename T>
uint32_t BcastComputeImpl(const CpuKernelContext& ctx, const BcastDivInfo& info)
{
    auto in0 = reinterpret_cast<const T*>(ctx.Input(kFirstInputIndex)->GetData());
    auto in1 = reinterpret_cast<const T*>(ctx.Input(kSecondInputIndex)->GetData());
    auto out = reinterpret_cast<T*>(ctx.Output(kFirstOutputIndex)->GetData());
    const int64_t data_num = info.total_elements;
    const int32_t ndims = info.ndims;
    const int64_t x_inner = info.x_strides[ndims - 1];
    const int64_t y_inner = info.y_strides[ndims - 1];

    auto sharder = [&in0, &in1, &out, &ndims, &x_inner, &y_inner, &info](int64_t start, int64_t end) {
        // Decompose start into multi-dim coordinates & offsets.
        int64_t coords[kMaxBcastDims] = {0};
        int64_t x_off = 0;
        int64_t y_off = 0;

        if (start > 0) {
            int64_t rem = start;
            for (int32_t d = 0; d < ndims; ++d) {
                coords[d] = rem / info.out_strides[d];
                rem -= coords[d] * info.out_strides[d];
                x_off += coords[d] * info.x_strides[d];
                y_off += coords[d] * info.y_strides[d];
            }
        }

        // Main loop: process one innermost-dim strip per iteration.
        int64_t idx = start;
        while (idx < end) {
            const int64_t inner_remain = info.out_shape[ndims - 1] - coords[ndims - 1];
            const int64_t chunk = std::min(inner_remain, end - idx);

            // Inner loop dispatch based on innermost stride pattern.
            // The branch is loop-invariant (same path every iteration) so the
            // CPU branch predictor handles it perfectly.
            if (x_inner == 1 && y_inner == 1) {
                // Both inputs contiguous in innermost dim — best case.
                // Compiler will auto-vectorise with NEON vdivq on aarch64.
                const T* xp = in0 + x_off;
                const T* yp = in1 + y_off;
                T* op = out + idx;
                for (int64_t i = 0; i < chunk; ++i) {
                    op[i] = xp[i] / yp[i];
                }
            } else if (y_inner == 0) {
                // Y broadcast in innermost dim — hoist scalar load.
                const T y_val = in1[y_off];
                for (int64_t i = 0; i < chunk; ++i) {
                    out[idx + i] = in0[x_off + i * x_inner] / y_val;
                }
            } else if (x_inner == 0) {
                // X broadcast in innermost dim — hoist scalar load.
                const T x_val = in0[x_off];
                for (int64_t i = 0; i < chunk; ++i) {
                    out[idx + i] = x_val / in1[y_off + i * y_inner];
                }
            } else {
                // General strided case.
                for (int64_t i = 0; i < chunk; ++i) {
                    out[idx + i] = in0[x_off + i * x_inner] / in1[y_off + i * y_inner];
                }
            }

            idx += chunk;
            x_off += chunk * x_inner;
            y_off += chunk * y_inner;
            coords[ndims - 1] += chunk;

            // Carry propagation: reset overflowed dims, increment parent.
            for (int32_t d = ndims - 1; d >= 0; --d) {
                if (coords[d] < info.out_shape[d])
                    break;
                x_off -= coords[d] * info.x_strides[d];
                y_off -= coords[d] * info.y_strides[d];
                coords[d] = 0;
                if (d > 0) {
                    coords[d - 1]++;
                    x_off += info.x_strides[d - 1];
                    y_off += info.y_strides[d - 1];
                }
            }
        }
    };

    if (data_num >= kParallelDataNum) {
        uint32_t min_core_num = 1U;
        uint32_t max_core_num = std::max(min_core_num, std::max(CpuKernelUtils::GetCPUNum(ctx), kReserveCpuNum) - kReserveCpuNum);
        if (data_num <= kParallelDataNumMid) {
            max_core_num = std::min(max_core_num, 4U);
        }
        if (static_cast<int64_t>(max_core_num) > data_num) {
            max_core_num = static_cast<uint32_t>(data_num);
        }
        int64_t shard_size = data_num / max_core_num;
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, shard_size, sharder), "RealDiv Compute failed.")
    } else {
        sharder(0, data_num);
    }
    return KERNEL_STATUS_OK;
}

} // anonymous namespace

uint32_t RealDivKernel::RealDivSameTypeCompute(const CpuKernelContext& ctx, DataType data_type)
{
    switch (data_type) {
        case DT_FLOAT16:
            return RealDivCompute<Eigen::half>(ctx, false);
        case DT_FLOAT:
            return RealDivCompute<float>(ctx, false);
        case DT_DOUBLE:
            return RealDivCompute<double>(ctx, false);
        case DT_INT8:
            return RealDivCompute<int8_t>(ctx);
        case DT_INT16:
            return RealDivCompute<int16_t>(ctx);
        case DT_INT32:
            return RealDivCompute<int32_t>(ctx);
        case DT_INT64:
            return RealDivCompute<int64_t>(ctx);
        case DT_UINT8:
            return RealDivCompute<uint8_t>(ctx);
        case DT_UINT16:
            return RealDivCompute<uint16_t>(ctx);
        case DT_UINT32:
            return RealDivCompute<uint32_t>(ctx);
        case DT_UINT64:
            return RealDivCompute<uint64_t>(ctx);
        case DT_COMPLEX64:
            return RealDivCompute<std::complex<float>>(ctx, false);
        case DT_COMPLEX128:
            return RealDivCompute<std::complex<double>>(ctx, false);
        default:
            KERNEL_LOG_ERROR(
                "[%s] Data type of input is not support, input data type is [%s].", ctx.GetOpType().c_str(),
                DTypeStr(data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

template <typename T>
bool RealDivKernel::IsInputHasZero(T* input_data, const int64_t num_of_elems)
{
    for (int64_t i = 0; i < num_of_elems; ++i) {
        if (IsValueEqual<T>(input_data[i], T(0))) {
            return true;
        }
    }
    return false;
}

template <typename T>
uint32_t RealDivKernel::RealDivCompute(const CpuKernelContext& ctx, const bool verify_zero)
{
    Tensor* input0 = ctx.Input(kFirstInputIndex);
    Tensor* input1 = ctx.Input(kSecondInputIndex);

    if (verify_zero && IsInputHasZero<T>(static_cast<T*>(input1->GetData()), input1->NumElements())) {
        KERNEL_LOG_ERROR("Invalid argument, division by zero.");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto input0_shape = input0->GetTensorShape()->GetDimSizes();
    auto input1_shape = input1->GetTensorShape()->GetDimSizes();
    int64_t input0_elements = input0->NumElements();
    int64_t input1_elements = input1->NumElements();

    // Fast path: same shape, or one input is scalar — no broadcast needed.
    bool no_bcast = (input0_shape == input1_shape) || (input0_elements == 1) || (input1_elements == 1);
    if (no_bcast) {
        return NoBcastComputeImpl<T>(ctx);
    }

    // General broadcast path: stride-based iteration.
    BcastDivInfo info;
    if (!ComputeBcastDivInfo(input0_shape, input1_shape, info)) {
        KERNEL_LOG_ERROR("[%s] Broadcast shapes are incompatible.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return BcastComputeImpl<T>(ctx, info);
}

uint32_t RealDivKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, INPUT_NUM2, 1), "Div check input output params failed.");
    Tensor* input0 = ctx.Input(kFirstInputIndex);
    Tensor* input1 = ctx.Input(kSecondInputIndex);
    if ((input0->GetDataSize() == 0) || (input1->GetDataSize() == 0)) {
        KERNEL_LOG_INFO("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_OK;
    }

    DataType input0_type = input0->GetDataType();
    DataType input1_type = input1->GetDataType();
    KERNEL_CHECK_FALSE(
        (input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID, "input0 type[%s] is not equal to input1 type[%s]",
        DTypeStr(input0_type).c_str(), DTypeStr(input1_type).c_str());
    return RealDivSameTypeCompute(ctx, input0_type);
}

REGISTER_CPU_KERNEL(kRealDiv, RealDivKernel);
REGISTER_CPU_KERNEL(kDIV, RealDivKernel);
} // namespace aicpu
