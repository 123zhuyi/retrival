/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file math_def_util.h
 * \brief op_def 使用的公共工具，提供多输入 data type 两两组合的编译期常量生成方法
 */
#ifndef MATH_COMMON_OP_HOST_MATH_DEF_UTIL_H
#define MATH_COMMON_OP_HOST_MATH_DEF_UTIL_H

#include <array>
#include <cstddef>

namespace Ops {
namespace Math {

namespace {
// 递归填充第 row 行组合数组（Prefix = 当前列表之前所有列表 size 之积）
template <typename T, size_t COUNT, size_t K, size_t Prefix, size_t N, size_t... RestNs>
static constexpr void CombineDataTypesImpl(std::array<std::array<T, COUNT>, K>& result, const std::array<T, N>& first,
                                           const std::array<T, RestNs>&... rest)
{
    constexpr size_t CURRENT_ROW = K - sizeof...(RestNs) - 1;
    for (size_t idx = 0; idx < COUNT; ++idx) {
        // 前面列表占据低位组合下标（变化更快），当前列表取值下标 = (idx / Prefix) % N
        result[CURRENT_ROW][idx] = first[(idx / Prefix) % N];
    }
    if constexpr (sizeof...(RestNs) > 0) {
        CombineDataTypesImpl<T, COUNT, K, Prefix * N>(result, rest...);
    }
}
} // namespace

/**
 * @brief 将 k（k >= 2）个 data type 列表两两组合，返回每路输入在各组合下标下的 data type 数组（编译期常量）
 *        返回的第 0/.../k-1 个数组分别对应第 0/.../k-1 个列表，每个数组长度 N0 * N1 * ... * N_{k-1}，
 *        组合顺序为第 0 个列表变化最快、最后一个列表变化最慢
 * @tparam T   data type 类型（如 ge::DataType）
 * @tparam Ns  各列表长度
 * @param lists 各输入 data type 取值列表（数量 >= 2）
 * @return 组合结果数组
 */
template <typename T, size_t... Ns>
static constexpr std::array<std::array<T, (Ns * ...)>, sizeof...(Ns)> CombineDataTypes(
    const std::array<T, Ns>&... lists)
{
    static_assert(sizeof...(Ns) >= 2, "CombineDataTypes requires at least two data type lists");
    constexpr size_t COUNT = (Ns * ...);
    constexpr size_t K = sizeof...(Ns);
    std::array<std::array<T, COUNT>, K> result{};
    CombineDataTypesImpl<T, COUNT, K, 1>(result, lists...);
    return result;
}

} // namespace Math
} // namespace Ops

#endif // MATH_COMMON_OP_HOST_MATH_DEF_UTIL_H
