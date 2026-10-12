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
 * \file radix_topk_util.h
 * \brief topk util
 */

#ifndef RADIX_TOP_K_UTIL_H
#define RADIX_TOP_K_UTIL_H

#include "radix_topk_constant.h"

template <typename T>
__aicore__ inline T TopkGetMax(T left, T right)
{
    return (left > right ? left : right);
}
template <typename T>
__aicore__ inline T TopkGetMin(T left, T right)
{
    return (left > right ? right : left);
}

template <typename T>
__aicore__ inline T GetTypeMaxValue() {
    if constexpr (is_same<uint32_t, T>::value) {
        return UINT_32_MAX_VALUE;
    } else if constexpr (is_same<uint16_t, T>::value) {
        return UINT_16_MAX_VALUE;
    } else if constexpr (is_same<uint8_t, T>::value) {
        return UINT_8_MAX_VALUE;
    } else if constexpr (is_same<int32_t, T>::value) {
        return INT_32_MAX_VALUE;
    } else if constexpr (is_same<int16_t, T>::value) {
        return INT_16_MAX_VALUE;
    } else if constexpr (is_same<int8_t, T>::value) {
        return INT_8_MAX_VALUE;
    } else if constexpr (is_same<float, T>::value) {
        return FLOAT_MAX_VALUE;
    } else if constexpr (is_same<half, T>::value) {
        return HALF_MAX_VALUE;
    } else if constexpr (is_same<bfloat16_t, T>::value) {
        return BFLOAT_16_MAX_VALUE;
    } else if constexpr (is_same<int64_t, T>::value) {
        return INT_64_MAX_VALUE;
    } else if constexpr (is_same<uint64_t, T>::value) {
        return UINT_64_MAX_VALUE;
    }
}

template <typename T>
__aicore__ inline T GetTypeMinValue() {
    if constexpr (is_same<uint32_t, T>::value) {
        return UINT_32_MIN_VALUE;
    } else if constexpr (is_same<uint16_t, T>::value) {
        return UINT_16_MIN_VALUE;
    } else if constexpr (is_same<uint8_t, T>::value) {
        return UINT_8_MIN_VALUE;
    } else if constexpr (is_same<int32_t, T>::value) {
        return INT_32_MIN_VALUE;
    } else if constexpr (is_same<int16_t, T>::value) {
        return INT_16_MIN_VALUE;
    } else if constexpr (is_same<int8_t, T>::value) {
        return INT_8_MIN_VALUE;
    } else if constexpr (is_same<float, T>::value) {
        return FLOAT_MIN_VALUE;
    } else if constexpr (is_same<half, T>::value) {
        return HALF_MIN_VALUE;
    } else if constexpr (is_same<bfloat16_t, T>::value) {
        return BFLOAT_16_MIN_VALUE;
    }  else if constexpr (is_same<int64_t, T>::value) {
        return INT_64_MIN_VALUE;
    } else if constexpr (is_same<uint64_t, T>::value) {
        return UINT_64_MIN_VALUE;
    }
}
#endif