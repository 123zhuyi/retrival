/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DETAIL_GCD_KERNEL_SCALAR_H_
#define DETAIL_GCD_KERNEL_SCALAR_H_

template <typename T>
class GcdKernel {
public:
    __aicore__ inline GcdKernel() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdTilingData* tilingData)
    {
        tiling_ = *tilingData;
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), static_cast<uint64_t>(tiling_.totalNum));
    }

    __aicore__ inline void Process()
    {
        int64_t total = tiling_.totalNum;
        if (total <= 0) {
            return;
        }

        int64_t begin = 0;
        int64_t end = 0;
        CalcBlockRange(total, begin, end);
        if (begin >= end) {
            return;
        }

        if (IsContiguousElementwise(tiling_)) {
            for (int64_t linear = begin; linear < end; ++linear) {
                T lhs = x1Gm_.GetValue(static_cast<uint64_t>(linear));
                T rhs = x2Gm_.GetValue(static_cast<uint64_t>(linear));
                yGm_.SetValue(static_cast<uint64_t>(linear), GcdOp::GcdScalar<T>(lhs, rhs));
            }
            return;
        }

        OffsetCursor cursor;
        InitOffsetCursor(tiling_, begin, cursor);
        for (int64_t linear = begin; linear < end; ++linear) {
            T lhs = x1Gm_.GetValue(static_cast<uint64_t>(cursor.x1Offset));
            T rhs = x2Gm_.GetValue(static_cast<uint64_t>(cursor.x2Offset));
            yGm_.SetValue(static_cast<uint64_t>(linear), GcdOp::GcdScalar<T>(lhs, rhs));
            if (linear + 1 < end) {
                AdvanceOffsetCursor(tiling_, cursor);
            }
        }
    }

private:
    GcdTilingData tiling_;
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<T> yGm_;
};

#endif // DETAIL_GCD_KERNEL_SCALAR_H_
