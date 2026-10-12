/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cumsum_base.h
 * \brief cumsum base construct and function
 */

#ifndef CUMSUM_BASE_H
#define CUMSUM_BASE_H

#include "kernel_operator.h"

namespace Cumsum {
using namespace AscendC;

template <typename T>
class CumsumBase {
public:
    __aicore__ inline CumsumBase(TPipe& pipe) : pipe_(pipe)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        blockIdx_ = GetBlockIdx();
        xGm_.SetGlobalBuffer((__gm__ T*)x);
        yGm_.SetGlobalBuffer((__gm__ T*)y);
    }

protected:
    TPipe& pipe_;
    int32_t blockIdx_ = 0;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
}; // class CumsumBase

namespace __cumsumType {
template <typename DType>
struct GetPromoteType {};

template <>
struct GetPromoteType<half> {
    using T = float;
};

template <>
struct GetPromoteType<bfloat16_t> {
    using T = float;
};

template <>
struct GetPromoteType<float> {
    using T = float;
};

template <>
struct GetPromoteType<int8_t> {
    using T = int8_t;
};

template <>
struct GetPromoteType<uint8_t> {
    using T = uint8_t;
};

template <>
struct GetPromoteType<int32_t> {
    using T = int32_t;
};

template <>
struct GetPromoteType<int64_t> {
    using T = int64_t;
};

template <>
struct GetPromoteType<uint64_t> {
    using T = uint64_t;
};
} // namespace __cumsumType

namespace __CumsumUtil {

template <typename T>
__aicore__ inline T CeilDiv(T x, T y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

__aicore__ inline int32_t CalPow2(int32_t index)
{
    return 1 << index;
}

__aicore__ inline int32_t CalLog2(int32_t value)
{
    int32_t res = 0;
    int32_t len = value;
    while (value > 1) {
        value = value >> 1;
        res++;
    }
    if (len > CalPow2(res)) {
        res++;
    }
    return res;
}

__aicore__ inline int32_t min(int32_t a, int32_t b)
{
    return a > b ? b : a;
}

__aicore__ inline int32_t GetCacheID(const int32_t idx)
{
    return ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1;
}

} // namespace __CumsumUtil
} // namespace Cumsum
#endif // CANN_OPS_BUILD_IN_TBE_IMPL_ASCENDC_CUMSUM_CUMSUM_CORE_SS_UB_SS_TWOWAY_SS_H