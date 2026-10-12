/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATH_COMMON_OP_HOST_MATH_LOG_H
#define MATH_COMMON_OP_HOST_MATH_LOG_H

#include <cstddef>
#include <type_traits>
#include <vector>
#include <string>
#include <sstream>
#include "graph/types.h"
#include "graph/utils/type_utils.h"

#define CHECK_RET_SUCC(result)           \
    do {                                 \
        auto _ret = (result);            \
        if (_ret != ge::GRAPH_SUCCESS) { \
            return _ret;                 \
        }                                \
    } while (0)

namespace Ops::Math {

/**
 * 日志项的连接符
 */
enum class ItemConj {
    AND,
    OR,
};

namespace {
// 连接符
template <ItemConj Conj>
inline constexpr const char* CONJ_STR = "";
template <>
inline constexpr const char* CONJ_STR<ItemConj::AND> = " and ";
template <>
inline constexpr const char* CONJ_STR<ItemConj::OR> = " or ";
template <ItemConj Conj>
inline constexpr size_t CONJ_STR_LEN = std::char_traits<char>::length(CONJ_STR<Conj>);
// 分隔符
inline constexpr const char* SEP_STR = ", ";
inline constexpr size_t SEP_STR_LEN = std::char_traits<char>::length(SEP_STR);

template <typename>
struct AlwaysFalse : std::false_type {};

template <typename T>
static inline std::string ValueToString(const T& value)
{
    if constexpr (std::is_same_v<std::decay_t<T>, char>) {
        // char 类型 → 输出字符本身，而非数字
        return std::string(1, value);
    } else if constexpr (
        std::is_pointer_v<std::decay_t<T>> &&
        std::is_same_v<std::remove_const_t<std::remove_pointer_t<std::decay_t<T>>>, char>) {
        // C 风格字符串：const char*、char* 等
        return std::string(value); // 直接构造
    } else if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
        // std::string 直接拷贝即可
        return value;
    } else if constexpr (std::is_arithmetic_v<T>) {
        // 算术类型（此时 char 已被排除）→ 使用高速的 std::to_string
        return std::to_string(value);
    } else {
        // 其他未特化类型 → 编译期报错
        static_assert(AlwaysFalse<T>::value, "No conversion available. Need specialize.");
        return {};
    }
}

template <>
inline std::string ValueToString(const ge::DataType& value)
{
    return ge::TypeUtils::DataTypeToSerialString(value);
}
} // namespace

/**
 * @brief   将参数用连接符转为字符串
 * @tparam  Conj 连接类型
 * @param   args 值
 * @return  字符串
 */
template <ItemConj Conj, typename... Args>
static inline std::string Join(Args&&... args)
{
    // 所有参数转为字符串（数组存储，之后可以安全移动）
    std::string strings[] = {ValueToString(args)...};
    constexpr size_t N = sizeof...(Args);

    if constexpr (N == 0) {
        return "";
    }

    // 提前返回单参数，避免多余计算
    if constexpr (N == 1) {
        return std::move(strings[0]); // 直接移动，无拷贝
    }

    // 计算最终字符串长度
    size_t total = 0;
    if constexpr (N == 2) {
        total = strings[0].size() + CONJ_STR_LEN<Conj> + strings[1].size();
    } else {
        for (size_t i = 0; i < N; ++i)
            total += strings[i].size();
        total += (N - 2) * SEP_STR_LEN + CONJ_STR_LEN<Conj>; // 分隔符 + 连接符
    }

    std::string result;
    result.reserve(total);

    // 拼接（移动每个临时字符串）
    result += std::move(strings[0]);
    if constexpr (N > 2) {
        for (size_t i = 1; i < N - 1; ++i) {
            result.append(SEP_STR);
            result += std::move(strings[i]);
        }
    }
    result.append(CONJ_STR<Conj>);
    result += std::move(strings[N - 1]);
    return result;
}

template <typename... Args>
static inline std::string Join(Args&&... args)
{
    return Join<ItemConj::AND>(std::forward<Args>(args)...);
}

namespace {
template <ItemConj Conj, typename T, size_t N, size_t... Is>
static inline std::string JoinArrayImpl(const std::array<T, N>& arr, std::index_sequence<Is...>)
{
    // std::array 入参复用不定参数Join方法
    return Join<Conj>(arr[Is]...);
}
} // namespace

/**
 * @brief   数组用连接符转为字符串
 * @tparam  Conj 连接类型
 * @param   arr 值数组
 * @return  字符串
 */
template <ItemConj Conj, typename T, size_t N>
static inline std::string JoinArray(const std::array<T, N>& arr)
{
    return JoinArrayImpl<Conj>(arr, std::make_index_sequence<N>{});
}
} // namespace Ops::Math

#endif
