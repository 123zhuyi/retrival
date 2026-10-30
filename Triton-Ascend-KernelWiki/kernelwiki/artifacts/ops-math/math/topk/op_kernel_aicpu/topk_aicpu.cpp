/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topk_aicpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"
#include "securec.h"
#include "status.h"

namespace {
const char* const TOPK = "TopK";
constexpr int32_t kSmallN = 64;
constexpr int32_t kSelectionMaxK = 8;
constexpr int64_t kHugeRowcountThreshold = 1LL << 20;
constexpr int32_t kMonotoneNonMonotone = 0;
constexpr int32_t kMonotoneAscending = 1;
constexpr int32_t kMonotoneDescending = 2;
constexpr int32_t kMonotoneAllEqual = 3;
constexpr int32_t kMonotoneProbeMaxN = 16;
constexpr int32_t kMonotoneComparableMinN = 2;
constexpr int64_t kParallelTargetShardBytes = 32LL * 1024LL;
constexpr int64_t kSerialInputBytesThreshold = 8LL * 1024LL;

inline int64_t ComputePerUnit(int64_t total_units, int64_t row_bytes, int64_t num_cores)
{
    if ((total_units <= 0) || (row_bytes <= 0) || (num_cores <= 0)) {
        return 1;
    }

    int64_t max_per_unit = kParallelTargetShardBytes / row_bytes;
    if (max_per_unit < 1) {
        max_per_unit = 1;
    }

    int64_t required_tasks = (total_units + max_per_unit - 1) / max_per_unit;
    int64_t aligned_tasks = num_cores * ((required_tasks + num_cores - 1) / num_cores);

    return (total_units + aligned_tasks - 1) / aligned_tasks;
}

template <typename T>
struct ValueIndex {
    T value;
    int32_t index;
};

template <typename T>
inline bool LessDescTieIdx(const ValueIndex<T>& a, const ValueIndex<T>& b)
{
    if (aicpu::IsValueEqual<T>(a.value, b.value)) {
        return a.index < b.index;
    }
    return a.value > b.value;
}

template <typename T>
inline bool LessAscTieIdx(const ValueIndex<T>& a, const ValueIndex<T>& b)
{
    if (aicpu::IsValueEqual<T>(a.value, b.value)) {
        return a.index < b.index;
    }
    return a.value < b.value;
}

// Path A: small-n selection / full-sort. Uses stack-resident scratch.
// largest_true is a compile-time bool so the inner comparator de-virtualizes.

// Monotone detector: returns kMonotoneNonMonotone, kMonotoneAscending,
// kMonotoneDescending or kMonotoneAllEqual. Used to fast-path the small-n
// selection-sort path when the row is already (anti-)sorted or all-equal.
//
// Two regimes:
//   - n <= kMonotoneProbeMaxN: 3-point cheap probe (head/mid/tail). On random
//   data the
//                first comparison pair flips both directions to false in
//                ~50% of cases, so we reject before touching the full row.
//                A full verifier only runs when the probe survives.
//   - n  > kMonotoneProbeMaxN: single-pass with dual-direction early exit.
//   Each step
//                kills at most one direction; the loop returns 0 as soon
//                as both have been killed, so on random data we typically
//                exit within 2-4 iterations.
template <typename T>
inline int32_t DetectMonotone(const T* __restrict__ in, int32_t jump, int32_t n)
{
    if (n < kMonotoneComparableMinN) {
        return kMonotoneAllEqual;
    }
    if (n <= kMonotoneProbeMaxN) {
        const T a = in[0];
        const T b = in[static_cast<int64_t>(n >> 1) * jump];
        const T c = in[static_cast<int64_t>(n - 1) * jump];
        if (aicpu::IsValueEqual<T>(a, b) && aicpu::IsValueEqual<T>(b, c)) {
            for (int32_t i = 1; i < n; ++i) {
                if (!aicpu::IsValueEqual<T>(in[static_cast<int64_t>(i) * jump], a)) {
                    return kMonotoneNonMonotone;
                }
            }
            return kMonotoneAllEqual;
        }
        const bool p_asc = (a <= b) && (b <= c);
        const bool p_desc = (a >= b) && (b >= c);
        if (!p_asc && !p_desc) {
            return kMonotoneNonMonotone;
        }
    }
    bool asc = true;
    bool desc = true;
    for (int32_t i = 1; i < n; ++i) {
        const T prev = in[static_cast<int64_t>(i - 1) * jump];
        const T cur = in[static_cast<int64_t>(i) * jump];
        if (cur < prev) {
            asc = false;
        } else if (cur > prev) {
            desc = false;
        }
        if (!asc && !desc) {
            return kMonotoneNonMonotone;
        }
    }
    if (asc && desc) {
        return kMonotoneAllEqual;
    }
    return asc ? kMonotoneAscending : kMonotoneDescending;
}

// Emit the top-k result for a row that DetectMonotone classified as
// monotone. Tie-break: equal values -> smaller original index ranks higher.
template <typename T, bool largest_true>
inline void EmitMonotoneTopK(const T* __restrict__ in, int32_t jump, T* __restrict__ out_v, int32_t* __restrict__ out_i,
                             int32_t n, int32_t k, int32_t mono)
{
    if (mono == kMonotoneAllEqual) {
        const T v = in[0];
        for (int32_t j = 0; j < k; ++j) {
            out_v[static_cast<int64_t>(j) * jump] = v;
            out_i[static_cast<int64_t>(j) * jump] = j;
        }
        return;
    }
    const bool head_is_largest = (mono == kMonotoneDescending);
    const bool take_head = (largest_true == head_is_largest);
    if (take_head) {
        for (int32_t j = 0; j < k; ++j) {
            out_v[static_cast<int64_t>(j) * jump] = in[static_cast<int64_t>(j) * jump];
            out_i[static_cast<int64_t>(j) * jump] = j;
        }
        return;
    }
    int32_t j = 0;
    int32_t r = n - 1;
    while (j < k && r >= 0) {
        int32_t g_left = r;
        const T gv = in[static_cast<int64_t>(r) * jump];
        while (g_left - 1 >= 0 && aicpu::IsValueEqual<T>(in[static_cast<int64_t>(g_left - 1) * jump], gv)) {
            --g_left;
        }
        const int32_t glen = r - g_left + 1;
        const int32_t take = (glen < (k - j)) ? glen : (k - j);
        for (int32_t t = 0; t < take; ++t) {
            out_v[static_cast<int64_t>(j) * jump] = gv;
            out_i[static_cast<int64_t>(j) * jump] = g_left + t;
            ++j;
        }
        r = g_left - 1;
    }
}

template <typename T, bool largest_true>
inline void SelectionTopK(const T* __restrict__ in_base, int32_t jump, T* __restrict__ out_v,
                          int32_t* __restrict__ out_i, int32_t n, int32_t k)
{
    const int32_t mono = DetectMonotone<T>(in_base, jump, n);
    if (mono != 0) {
        EmitMonotoneTopK<T, largest_true>(in_base, jump, out_v, out_i, n, k, mono);
        return;
    }
    T vs[kSmallN];
    int32_t is_[kSmallN];
    for (int32_t i = 0; i < n; ++i) {
        vs[i] = in_base[static_cast<int64_t>(i) * jump];
        is_[i] = i;
    }
    for (int32_t j = 0; j < k; ++j) {
        int32_t best = j;
        T bv = vs[j];
        int32_t bi = is_[j];
        for (int32_t i = j + 1; i < n; ++i) {
            T v = vs[i];
            int32_t ii = is_[i];
            bool eq = aicpu::IsValueEqual<T>(v, bv);
            bool take = largest_true ? (v > bv || (eq && ii < bi)) : (v < bv || (eq && ii < bi));
            if (take) {
                bv = v;
                bi = ii;
                best = i;
            }
        }
        if (best != j) {
            vs[best] = vs[j];
            is_[best] = is_[j];
            vs[j] = bv;
            is_[j] = bi;
        }
        out_v[static_cast<int64_t>(j) * jump] = bv;
        out_i[static_cast<int64_t>(j) * jump] = bi;
    }
}

template <typename T, bool largest_true>
inline void FullSortTopK(const T* __restrict__ in_base, int32_t jump, T* __restrict__ out_v,
                         int32_t* __restrict__ out_i, int32_t n)
{
    ValueIndex<T> aos[kSmallN];
    for (int32_t i = 0; i < n; ++i) {
        aos[i].value = in_base[static_cast<int64_t>(i) * jump];
        aos[i].index = i;
    }
    if (largest_true) {
        std::sort(aos, aos + n, LessDescTieIdx<T>);
    } else {
        std::sort(aos, aos + n, LessAscTieIdx<T>);
    }
    for (int32_t j = 0; j < n; ++j) {
        out_v[static_cast<int64_t>(j) * jump] = aos[j].value;
        out_i[static_cast<int64_t>(j) * jump] = aos[j].index;
    }
}

// Path B: k-sized heap with Floyd O(k) heapify + linear select.
// Tie-break: when two values are equal, the one with the smaller original
// index ranks higher. For the min-heap (largest=true) the heap root holds
// the element with the smallest value AND, among equal values, the largest
// index (so a candidate with smaller index can displace it).
template <typename T>
inline void SiftDownMin(T* val, int32_t* idx, int32_t pos, int32_t k, int32_t jump)
{
    while (true) {
        int32_t L = pos * 2 + 1;
        if (L >= k) {
            break;
        }
        int32_t R = L + 1;
        int64_t aL = static_cast<int64_t>(L) * jump;
        int32_t smallest = L;
        int64_t as = aL;
        if (R < k) {
            int64_t aR = static_cast<int64_t>(R) * jump;
            bool eqLR = aicpu::IsValueEqual<T>(val[aR], val[aL]);
            if (val[aR] < val[aL] || (eqLR && idx[aR] > idx[aL])) {
                smallest = R;
                as = aR;
            }
        }
        int64_t ap = static_cast<int64_t>(pos) * jump;
        bool eq = aicpu::IsValueEqual<T>(val[as], val[ap]);
        if (val[as] < val[ap] || (eq && idx[as] > idx[ap])) {
            T tv = val[ap];
            val[ap] = val[as];
            val[as] = tv;
            int32_t ti = idx[ap];
            idx[ap] = idx[as];
            idx[as] = ti;
            pos = smallest;
        } else {
            break;
        }
    }
}

template <typename T>
inline void SiftDownMax(T* val, int32_t* idx, int32_t pos, int32_t k, int32_t jump)
{
    while (true) {
        int32_t L = pos * 2 + 1;
        if (L >= k) {
            break;
        }
        int32_t R = L + 1;
        int64_t aL = static_cast<int64_t>(L) * jump;
        int32_t largest = L;
        int64_t as = aL;
        if (R < k) {
            int64_t aR = static_cast<int64_t>(R) * jump;
            bool eqLR = aicpu::IsValueEqual<T>(val[aR], val[aL]);
            if (val[aR] > val[aL] || (eqLR && idx[aR] > idx[aL])) {
                largest = R;
                as = aR;
            }
        }
        int64_t ap = static_cast<int64_t>(pos) * jump;
        bool eq = aicpu::IsValueEqual<T>(val[as], val[ap]);
        if (val[as] > val[ap] || (eq && idx[as] > idx[ap])) {
            T tv = val[ap];
            val[ap] = val[as];
            val[as] = tv;
            int32_t ti = idx[ap];
            idx[ap] = idx[as];
            idx[as] = ti;
            pos = largest;
        } else {
            break;
        }
    }
}

template <typename T, bool largest_true>
inline void HeapTopK(const T* __restrict__ in_base, int32_t jump, T* __restrict__ val, int32_t* __restrict__ idx,
                     int32_t n, int32_t k)
{
    for (int32_t i = 0; i < k; ++i) {
        int64_t a = static_cast<int64_t>(i) * jump;
        val[a] = in_base[a];
        idx[a] = i;
    }
    for (int32_t i = k / 2 - 1; i >= 0; --i) {
        if (largest_true) {
            SiftDownMin(val, idx, i, k, jump);
        } else {
            SiftDownMax(val, idx, i, k, jump);
        }
    }
    for (int32_t i = k; i < n; ++i) {
        int64_t a = static_cast<int64_t>(i) * jump;
        const T v = in_base[a];
        const T root = val[0];
        bool better = largest_true ? (v > root) : (v < root);
        if (better) {
            val[0] = v;
            idx[0] = i;
            if (largest_true) {
                SiftDownMin(val, idx, 0, k, jump);
            } else {
                SiftDownMax(val, idx, 0, k, jump);
            }
        }
    }
}

// In-place heap-sort of the k-sized heap. After Floyd heapify + select,
// for largest_true we have a min-heap whose root is the smallest of the
// top-k. Pop k-1 times: swap root with the last unsorted slot, sift-down
// on the shrunk heap. No final reversal needed in either case.
template <typename T, bool largest_true>
inline void InPlaceHeapSort(T* __restrict__ val, int32_t* __restrict__ idx, int32_t k, int32_t jump)
{
    for (int32_t end = k - 1; end > 0; --end) {
        int64_t a0 = 0;
        int64_t ae = static_cast<int64_t>(end) * jump;
        T tv = val[a0];
        val[a0] = val[ae];
        val[ae] = tv;
        int32_t ti = idx[a0];
        idx[a0] = idx[ae];
        idx[ae] = ti;
        if (largest_true) {
            SiftDownMin(val, idx, 0, end, jump);
        } else {
            SiftDownMax(val, idx, 0, end, jump);
        }
    }
}

} // namespace

namespace aicpu {

uint32_t TopkCpuKernel::Compute(CpuKernelContext& ctx)
{
    KernelStatus res = GetInputAndCheck(ctx);
    if (res != KERNEL_STATUS_OK) {
        return static_cast<uint32_t>(res);
    }
    KERNEL_LOG_INFO("[TopK] Compute begin, dtype=%d, head=%d, n=%d, k=%d, tail=%d, "
                    "largest=%d, sorted=%d.",
                    static_cast<int>(data_type_), head_, n_, k_, tail_, static_cast<int>(largest_),
                    static_cast<int>(sorted_));
    res = DispatchByDtype(ctx);
    if (res != KERNEL_STATUS_OK) {
        return static_cast<uint32_t>(res);
    }
    return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

KernelStatus TopkCpuKernel::DispatchByDtype(const CpuKernelContext& ctx)
{
    switch (data_type_) {
        case DT_FLOAT16:
            return DoCompute<Eigen::half>(ctx);
        case DT_FLOAT:
            return DoCompute<float>(ctx);
        case DT_DOUBLE:
            return DoCompute<double>(ctx);
        case DT_UINT8:
            return DoCompute<uint8_t>(ctx);
        case DT_INT8:
            return DoCompute<int8_t>(ctx);
        case DT_UINT16:
            return DoCompute<uint16_t>(ctx);
        case DT_INT16:
            return DoCompute<int16_t>(ctx);
        case DT_UINT32:
            return DoCompute<uint32_t>(ctx);
        case DT_INT32:
            return DoCompute<int32_t>(ctx);
        case DT_UINT64:
            return DoCompute<uint64_t>(ctx);
        case DT_INT64:
            return DoCompute<int64_t>(ctx);
        default:
            KERNEL_LOG_ERROR("[TopK] input tensor dtype=%d not supported.", static_cast<int>(data_type_));
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

template <typename T>
KernelStatus TopkCpuKernel::DoCompute(const CpuKernelContext& ctx)
{
    T* in = PtrToPtr<void, T>(input_tensor_->GetData());
    T* val = PtrToPtr<void, T>(output_values_->GetData());
    int32_t* indice = PtrToPtr<void, int32_t>(output_indices_->GetData());
    KERNEL_CHECK_NULLPTR(in, KERNEL_STATUS_PARAM_INVALID, "[TopK] input data pointer is null.");
    KERNEL_CHECK_NULLPTR(val, KERNEL_STATUS_PARAM_INVALID, "[TopK] output values pointer is null.");
    KERNEL_CHECK_NULLPTR(indice, KERNEL_STATUS_PARAM_INVALID, "[TopK] output indices pointer is null.");

    const int64_t total_units = static_cast<int64_t>(head_) * static_cast<int64_t>(tail_);
    if (total_units <= 0 || n_ <= 0 || k_ <= 0) {
        KERNEL_LOG_INFO("[TopK] empty work (head*tail=%ld, n=%d, k=%d), skip.", total_units, n_, k_);
        return KERNEL_STATUS_OK;
    }

    const int64_t row_bytes_in = static_cast<int64_t>(n_) * static_cast<int64_t>(sizeof(T));
    const int64_t num_cores = std::max(static_cast<int64_t>(1),
                                       static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx)));
    const int64_t per_unit = ComputePerUnit(total_units, row_bytes_in, num_cores);

    const int64_t total_input_bytes = total_units * static_cast<int64_t>(n_) * static_cast<int64_t>(sizeof(T));
    const bool go_serial = (total_input_bytes < kSerialInputBytesThreshold) || (total_units == 1);

    auto shard = [this, in, val, indice](size_t start, size_t end) {
        TopKForNVectorImpl<T>(in, val, indice, static_cast<int64_t>(start), static_cast<int64_t>(end));
    };

    if (go_serial) {
        KERNEL_LOG_INFO("[TopK] Serial path: total_units=%ld, total_input_bytes=%ld < %ld.", total_units,
                        total_input_bytes, kSerialInputBytesThreshold);
        shard(0u, static_cast<size_t>(total_units));
        return KERNEL_STATUS_OK;
    }

    KERNEL_LOG_INFO("[TopK] Parallel path: total_units=%ld, per_unit=%ld, row_bytes=%ld.", total_units, per_unit,
                    row_bytes_in);
    uint32_t ret = CpuKernelUtils::ParallelFor(ctx, total_units, per_unit, shard);
    if (ret != static_cast<uint32_t>(KERNEL_STATUS_OK)) {
        KERNEL_LOG_ERROR("[TopK] CpuKernelUtils::ParallelFor failed, rc=%u, total_units=%ld, "
                         "per_unit=%ld.",
                         ret, total_units, per_unit);
        return KERNEL_STATUS_INNER_ERROR;
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
void TopkCpuKernel::TopKForNVectorImpl(T* in, T* val, int32_t* indice, int64_t start, int64_t end) const
{
    const int64_t total_units = static_cast<int64_t>(head_) * static_cast<int64_t>(tail_);
    const bool k_almost_n = (k_ >= n_ - 1);
    const bool huge_rowcount_almost_full_sort = k_almost_n &&
                                                (total_units * static_cast<int64_t>(n_) >= kHugeRowcountThreshold);
    const bool use_small_n_path = (n_ <= kSmallN) && (k_ < n_) && (k_ <= kSelectionMaxK) &&
                                  !huge_rowcount_almost_full_sort;
    const bool use_full_sort_path = (n_ <= kSmallN) && (k_ == n_);

    for (int64_t u = start; u < end; ++u) {
        const int32_t head = static_cast<int32_t>(u / tail_);
        const int32_t tail = static_cast<int32_t>(u % tail_);
        const int64_t in_off = static_cast<int64_t>(head) * n_ * tail_ + tail;
        const int64_t out_off = static_cast<int64_t>(head) * k_ * tail_ + tail;
        const T* in_base = in + in_off;
        T* val_base = val + out_off;
        int32_t* idx_base = indice + out_off;

        if (use_small_n_path) {
            if (largest_) {
                SelectionTopK<T, true>(in_base, tail_, val_base, idx_base, n_, k_);
            } else {
                SelectionTopK<T, false>(in_base, tail_, val_base, idx_base, n_, k_);
            }
        } else if (use_full_sort_path) {
            if (largest_) {
                FullSortTopK<T, true>(in_base, tail_, val_base, idx_base, n_);
            } else {
                FullSortTopK<T, false>(in_base, tail_, val_base, idx_base, n_);
            }
        } else {
            if (largest_) {
                HeapTopK<T, true>(in_base, tail_, val_base, idx_base, n_, k_);
                if (sorted_) {
                    InPlaceHeapSort<T, true>(val_base, idx_base, k_, tail_);
                }
            } else {
                HeapTopK<T, false>(in_base, tail_, val_base, idx_base, n_, k_);
                if (sorted_) {
                    InPlaceHeapSort<T, false>(val_base, idx_base, k_, tail_);
                }
            }
        }
    }
}

template <typename T>
void TopkCpuKernel::TopKForNVector(size_t start, size_t end)
{
    T* in = PtrToPtr<void, T>(input_tensor_->GetData());
    T* val = PtrToPtr<void, T>(output_values_->GetData());
    int32_t* indice = PtrToPtr<void, int32_t>(output_indices_->GetData());
    if (in == nullptr || val == nullptr || indice == nullptr) {
        return;
    }
    TopKForNVectorImpl<T>(in, val, indice, static_cast<int64_t>(start), static_cast<int64_t>(end));
}

KernelStatus TopkCpuKernel::ParseShapeAndDim(const CpuKernelContext& ctx)
{
    input_tensor_ = ctx.Input(0);
    KERNEL_CHECK_NULLPTR(input_tensor_, KERNEL_STATUS_PARAM_INVALID, "[TopK] Get input[0] name[x] failed.");

    std::shared_ptr<TensorShape> input_shape = input_tensor_->GetTensorShape();
    KERNEL_CHECK_NULLPTR(input_shape, KERNEL_STATUS_PARAM_INVALID, "[TopK] Get shape of input[0] name[x] failed.");
    int32_t input_rank = input_shape->GetDims();
    if (input_rank < 1) {
        KERNEL_LOG_ERROR("[TopK] Input rank=%d must be >= 1.", input_rank);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    input_rank_ = input_rank;
    AttrValue* dim = ctx.GetAttr("dim");
    dim_ = static_cast<int32_t>(dim == nullptr ? -1 : (dim->GetInt()));
    dim_ = dim_ < 0 ? (input_rank + dim_) : dim_;
    KERNEL_CHECK_FALSE(((dim_ >= 0) && (dim_ < input_rank)), KERNEL_STATUS_PARAM_INVALID,
                       "[TopK] Invalid attr dim value=%d, must be in [%d, %d).", dim_, -input_rank, input_rank);

    head_ = 1;
    tail_ = 1;
    for (int32_t i = 0; i < input_rank; ++i) {
        if (i < dim_) {
            head_ *= static_cast<int32_t>(input_shape->GetDimSize(i));
        } else if (i == dim_) {
            n_ = static_cast<int32_t>(input_shape->GetDimSize(i));
        } else {
            tail_ *= static_cast<int32_t>(input_shape->GetDimSize(i));
            KERNEL_CHECK_FALSE((tail_ != 0), KERNEL_STATUS_PARAM_INVALID, "[TopK] input dim size cannot be 0.");
        }
    }
    data_type_ = static_cast<DataType>(input_tensor_->GetDataType());
    return KERNEL_STATUS_OK;
}

KernelStatus TopkCpuKernel::GetInputAndCheck(const CpuKernelContext& ctx)
{
    KernelStatus ret = ParseShapeAndDim(ctx);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    Tensor* k_tensor = ctx.Input(1);
    KERNEL_CHECK_NULLPTR(k_tensor, KERNEL_STATUS_PARAM_INVALID, "[TopK] Get input[1] name[k] failed.");
    KERNEL_CHECK_NULLPTR(k_tensor->GetData(), KERNEL_STATUS_PARAM_INVALID, "[TopK] Get input[1] name[k] data failed.");
    k_ = *static_cast<int32_t*>(k_tensor->GetData());
    if (k_ < 0) {
        KERNEL_LOG_ERROR("[TopK] k=%d must be >= 0.", k_);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (n_ < k_) {
        KERNEL_LOG_ERROR("[TopK] Input must have at least k=%d elements along dim, got n=%d.", k_, n_);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    AttrValue* sorted = ctx.GetAttr("sorted");
    sorted_ = (sorted == nullptr) ? true : (sorted->GetBool());

    AttrValue* largest = ctx.GetAttr("largest");
    largest_ = (largest == nullptr) ? true : (largest->GetBool());

    output_values_ = ctx.Output(0);
    KERNEL_CHECK_NULLPTR(output_values_, KERNEL_STATUS_PARAM_INVALID, "[TopK] Get output[0] name[values] failed.");

    output_indices_ = ctx.Output(1);
    KERNEL_CHECK_NULLPTR(output_indices_, KERNEL_STATUS_PARAM_INVALID, "[TopK] Get output[1] name[indices] failed.");

    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(TOPK, TopkCpuKernel);
} // namespace aicpu
