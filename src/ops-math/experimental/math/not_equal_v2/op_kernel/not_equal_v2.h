/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "not_equal_v2_tiling_data.h"
#include "not_equal_v2_tiling_key.h"

using namespace AscendC;

template <typename T, typename... Ts>
struct is_one_of : std::false_type {};

template <typename T, typename U, typename... Ts>
struct is_one_of<T, U, Ts...> : std::conditional_t<std::is_same_v<T, U>, std::true_type, is_one_of<T, Ts...>> {};

template <typename T, typename... Ts>
constexpr bool is_one_of_v = is_one_of<T, Ts...>::value;

template <typename T>
__aicore__ inline constexpr T ceil_div(T x, T y)
{
    return (x - 1) / y + 1;
}

template <typename T>
__aicore__ inline constexpr T ceil_round(T x, T y)
{
    return ceil_div(x, y) * y;
}

#define NotEqual_0()                                             \
    if (i == 0) {                                                \
        PipeBarrier<PIPE_V>();                                   \
        Duplicate(x1, static_cast<U>(0), mask);                  \
        PipeBarrier<PIPE_V>();                                   \
        Duplicate(x2, static_cast<U>(1), mask);                  \
    }                                                            \
    PipeBarrier<PIPE_V>();                                       \
    Select<U, SELMODE::VSEL_CMPMASK_SPR>(y_U[i], x1, x2, 1, {}); \
    }                                                            \
    PipeBarrier<PIPE_V>();                                       \
    Cast(y_half, y_U, RoundMode::CAST_RINT, _);                  \
    PipeBarrier<PIPE_V>();                                       \
    Cast(y, y_half, RoundMode::CAST_TRUNC, _);                   \
    {
#define not_equal_0()                                                                                    \
    int block_index = GetBlockIdx();                                                                     \
    int block_dim = GetBlockNum();                                                                       \
    constexpr int MTE_BLOCK_SIZE = 512 / sizeof(uint8_t);                                                \
    int compute_blocks = ceil_div(tiling.size, MTE_BLOCK_SIZE);                                          \
    int compute_start = compute_blocks * block_index / block_dim * MTE_BLOCK_SIZE;                       \
    int compute_end = min(compute_blocks * (block_index + 1) / block_dim * MTE_BLOCK_SIZE, tiling.size); \
                                                                                                         \
    GlobalTensor<T> x1_global_tensor, x2_global_tensor;                                                  \
    GlobalTensor<int8_t> y_global_tensor;                                                                \
    x1_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));                                   \
    x2_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));                                   \
    y_global_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y));                                \
                                                                                                         \
    TPipe t_pipe;                                                                                        \
    TQue<TPosition::VECIN, 1> x1_t_que, x2_t_que;                                                        \
    TQue<TPosition::VECOUT, 1> y_t_que;                                                                  \
                                                                                                         \
    constexpr int MAX_TILE_SIZE = (30 << 10) / sizeof(U);                                                \
    t_pipe.InitBuffer(x1_t_que, 2, MAX_TILE_SIZE * sizeof(U));                                           \
    t_pipe.InitBuffer(x2_t_que, 2, MAX_TILE_SIZE * sizeof(U));                                           \
    t_pipe.InitBuffer(y_t_que, 2, MAX_TILE_SIZE * sizeof(U));

template <typename T, typename U>
__aicore__ inline std::enable_if_t<std::is_same_v<T, half>> NotEqual(
    LocalTensor<int8_t>& y, LocalTensor<U>& x1, LocalTensor<U>& x2, int _)
{
    Compare(y, x1, x2, CMPMODE::EQ, ceil_round(_, static_cast<int>(256 / sizeof(U))));
    Duplicate(x1, static_cast<U>(0), _);
    Duplicate(x2, static_cast<U>(1), _);
    Select(x1, y, x1, x2, SELMODE::VSEL_TENSOR_TENSOR_MODE, _);
    Cast(y, x1, RoundMode::CAST_TRUNC, _);
}

template <typename T, typename U>
__aicore__ inline std::enable_if_t<std::is_same_v<T, float>> NotEqual(
    LocalTensor<int8_t>& y, LocalTensor<U>& x1, LocalTensor<U>& x2, int _)
{
    LocalTensor<U> y_U = y.ReinterpretCast<U>();
    LocalTensor<half> y_half = y.ReinterpretCast<half>();
    PipeBarrier<PIPE_V>();
    for (int i = 0; i < _; i += 256 / sizeof(U)) {
        int mask = min(_ - i, static_cast<int>(256 / sizeof(U)));
        Compare<U, false>(x1[i], x2[i], CMPMODE::EQ, MASK_PLACEHOLDER, {});
        NotEqual_0()
    }
}

template <typename T, typename U>
__aicore__ inline std::enable_if_t<std::is_same_v<T, int>> NotEqual(
    LocalTensor<int8_t>& y, LocalTensor<U>& x1, LocalTensor<U>& x2, int _)
{
    LocalTensor<half> x1_half = x1.template ReinterpretCast<half>();
    LocalTensor<half> x2_half = x2.template ReinterpretCast<half>();
    Compare(y, x1, x2, CMPMODE::EQ, ceil_round(_, static_cast<int>(256 / sizeof(U))));
    Duplicate(x1_half, static_cast<half>(0), _);
    Duplicate(x2_half, static_cast<half>(1), _);
    Select(x1_half, y, x1_half, x2_half, SELMODE::VSEL_TENSOR_TENSOR_MODE, _);
    Cast(y, x1_half, RoundMode::CAST_TRUNC, _);
}

template <typename T, typename U>
__aicore__ inline std::enable_if_t<is_one_of_v<T, int8_t, uint8_t>> NotEqual(
    LocalTensor<int8_t>& y, LocalTensor<U>& x1, LocalTensor<U>& x2, int _)
{
    LocalTensor<U> y_U = y.ReinterpretCast<U>();
    Cast(y_U, x1.template ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
    Cast(x1, x2.template ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
    Sub(y_U, y_U, x1, _);
    Abs(y_U, y_U, _);
    Mins(y_U, y_U, static_cast<U>(0x1p-24), _);
    Muls(y_U, y_U, static_cast<U>(0x1p12), _);
    Muls(y_U, y_U, static_cast<U>(0x1p12), _);
    Cast(y, y_U, RoundMode::CAST_TRUNC, _);
}

template <typename T, typename U>
__aicore__ inline std::enable_if_t<std::is_same_v<T, bfloat16_t>> NotEqual(
    LocalTensor<int8_t>& y, LocalTensor<U>& x1, LocalTensor<U>& x2, int _)
{
    LocalTensor<U> y_U = y.ReinterpretCast<U>();
    LocalTensor<half> y_half = y.ReinterpretCast<half>();
    PipeBarrier<PIPE_V>();
    Cast(y_U, x1.template ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
    PipeBarrier<PIPE_V>();
    Cast(x1, x2.template ReinterpretCast<T>(), RoundMode::CAST_NONE, _);
    PipeBarrier<PIPE_V>();
    for (int i = 0; i < _; i += 256 / sizeof(U)) {
        int mask = min(_ - i, static_cast<int>(256 / sizeof(U)));
        Compare<U, false>(y_U[i], x1[i], CMPMODE::EQ, MASK_PLACEHOLDER, {});
        NotEqual_0()
    }
}

template <typename T, typename U>
__aicore__ inline void not_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const NotEqualTilingData& tiling)
{
    not_equal_0()

        for (int i = compute_start; i < compute_end; i += MAX_TILE_SIZE)
    {
        int _ = min(compute_end - i, MAX_TILE_SIZE);
        {
            LocalTensor<T> x1 = x1_t_que.AllocTensor<T>();
            LocalTensor<T> x2 = x2_t_que.AllocTensor<T>();
            int tile_size;
            if (_ < MAX_TILE_SIZE)
                tile_size = ceil_round(_, static_cast<int>(32 / sizeof(T)));
            else
                tile_size = _;
            DataCopy(x1, x1_global_tensor[i], tile_size);
            DataCopy(x2, x2_global_tensor[i], tile_size);
            x1_t_que.EnQue(x1);
            x2_t_que.EnQue(x2);
        }
        {
            LocalTensor<U> x1 = x1_t_que.DeQue<U>();
            LocalTensor<U> x2 = x2_t_que.DeQue<U>();
            LocalTensor<int8_t> y = y_t_que.AllocTensor<int8_t>();
            NotEqual<T, U>(y, x1, x2, _);
            x1_t_que.FreeTensor(x1);
            x2_t_que.FreeTensor(x2);
            y_t_que.EnQue(y);
        }
        {
            LocalTensor<int8_t> y = y_t_que.DeQue<int8_t>();
            int tile_size;
            if (_ < MAX_TILE_SIZE)
                tile_size = ceil_round(_, static_cast<int>(32 / sizeof(uint8_t)));
            else
                tile_size = _;
            DataCopy(y_global_tensor[i], y, tile_size);
            y_t_que.FreeTensor(y);
        }
    }
}

template <typename T>
__aicore__ inline std::enable_if_t<is_one_of_v<T, half, int, int8_t, uint8_t, bool>> not_equal(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const NotEqualTilingData& tiling)
{
    if constexpr (is_one_of_v<T, half, int>)
        not_equal<T, T>(x1, x2, y, tiling);
    else if constexpr (is_one_of_v<T, int8_t, uint8_t>)
        not_equal<T, half>(x1, x2, y, tiling);
    else if constexpr (std::is_same_v<T, bool>)
        not_equal<uint8_t, half>(x1, x2, y, tiling);
}

template <typename T>
__aicore__ std::enable_if_t<is_one_of_v<T, float, bfloat16_t>> not_equal(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const NotEqualTilingData& tiling)
{
    not_equal<T, float>(x1, x2, y, tiling);
}
