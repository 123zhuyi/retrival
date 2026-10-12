/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATH_COMMON_OP_HOST_INPUT_UTIL_H
#define MATH_COMMON_OP_HOST_INPUT_UTIL_H

#include <tuple>
#include <array>
#include "log/log.h"
#include "runtime/continuous_vector.h"
#include "graph/types.h"

namespace Ops::Math {
// NCHW格式维度索引
static constexpr size_t NCHW_N_DIM = 0U;
static constexpr size_t NCHW_C_DIM = 1U;
static constexpr size_t NCHW_H_DIM = 2U;
static constexpr size_t NCHW_W_DIM = 3U;
// NHWC格式维度索引
static constexpr size_t NHWC_N_DIM = 0U;
static constexpr size_t NHWC_H_DIM = 1U;
static constexpr size_t NHWC_W_DIM = 2U;
static constexpr size_t NHWC_C_DIM = 3U;
static constexpr size_t NHWC_DIM_NUM = 4U;
// unpack
static constexpr size_t UNPACK_TWO_ATTRS = 2U;
static constexpr size_t UNPACK_FOUR_ATTRS = 4U;

namespace {
// 内部方法

/**
 * @brief 检查 ListInt 类型属性的每个元素值
 * @tparam  T               context类型
 * @param   [in] context    infershape 或 tiling 上下文
 * @param   [in] attrName   属性名称
 * @param   [in] vec        属性vector
 * @param   [in] elementValidator 元素校验方法，对元素挨个进行校验
 * @param   [in] invalidValueReason 是无效元素的原因
 * @return  执行结果，false: 失败，true: 成功
 */
template <typename T>
static inline bool CheckListIntAttrValue(
    T* context, const char* attrName, const gert::TypedContinuousVector<int64_t>*& vec,
    const std::function<bool(int64_t)>& elementValidator, const char* invalidValueReason)
{
    for (size_t i = 0; i < vec->GetSize(); ++i) {
        int64_t value = vec->GetData()[i];
        if (!elementValidator(value)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), attrName, std::to_string(value).c_str(), invalidValueReason);
            return false;
        }
    }
    return true;
}

/**
 * @brief   检查固定长度的 ListInt 类型属性值
 * @tparam  unpackLen       解包长度
 * @tparam  T               context类型
 * @param   [in] context    infershape 或 tiling 上下文
 * @param   [in] attrName   属性名称
 * @param   [in] vec        属性vector
 * @param   [in] elementValidator 元素校验方法，对元素挨个进行校验
 * @param   [in] invalidValueReason 是无效元素的原因
 * @return  执行结果，false: 失败，true: 成功
 */
template <size_t unpackLen, typename T>
static inline bool CheckUnpackFixedDimListIntAttr(
    T* context, const char* attrName, const gert::TypedContinuousVector<int64_t>*& vec,
    const std::function<bool(int64_t)>& elementValidator, const char* invalidValueReason)
{
    static_assert(unpackLen > 0, "unpackLen should be positive");
    OP_CHECK_IF(vec == nullptr, OP_LOGE(context, "attr %s is nullptr!", attrName), return false);
    size_t packSize = vec->GetSize();
    OP_CHECK_IF(
        (packSize != unpackLen),
        OP_LOGE_FOR_INVALID_LISTSIZE(
            context->GetNodeName(), attrName, std::to_string(packSize).c_str(), std::to_string(unpackLen).c_str()),
        return false);
    return CheckListIntAttrValue(context, attrName, vec, elementValidator, invalidValueReason);
}

/**
 * @brief   检查自适应长度的 ListInt 类型属性值
 * @tparam  unpackLen       解包长度
 * @tparam  T               context类型
 * @param   [in] context    infershape 或 tiling 上下文
 * @param   [in] attrName   属性名称
 * @param   [in] vec        属性vector
 * @param   [in] elementValidator 元素校验方法，对元素挨个进行校验
 * @param   [in] invalidValueReason 是无效元素的原因
 * @return  执行结果，false: 失败，true: 成功
 */
template <size_t unpackLen, typename T>
static inline bool CheckUnpackAdaptDimListIntAttr(
    T* context, const char* attrName, const gert::TypedContinuousVector<int64_t>*& vec,
    const std::function<bool(int64_t)>& elementValidator, const char* invalidValueReason)
{
    static_assert(unpackLen > 0, "unpackLen should be positive");
    OP_CHECK_IF(vec == nullptr, OP_LOGE(context, "attr %s is nullptr!", attrName), return false);
    size_t packSize = vec->GetSize();
    if (unlikely(packSize == 0 || packSize > unpackLen || unpackLen % packSize != 0)) {
        if constexpr (unpackLen == UNPACK_TWO_ATTRS) {
            OP_LOGE_FOR_INVALID_LISTSIZE(
                context->GetNodeName(), attrName, std::to_string(packSize).c_str(), "in [1, 2]");
        } else if constexpr (unpackLen == UNPACK_FOUR_ATTRS) {
            OP_LOGE_FOR_INVALID_LISTSIZE(
                context->GetNodeName(), attrName, std::to_string(packSize).c_str(), "in [1, 2, 4]");
        } else {
            OP_LOGE_FOR_INVALID_LISTSIZE(
                context->GetNodeName(), attrName, std::to_string(packSize).c_str(),
                ("divisor of " + std::to_string(unpackLen)).c_str());
        }
        return false;
    }
    return CheckListIntAttrValue(context, attrName, vec, elementValidator, invalidValueReason);
}

/**
 * @brief   不安全解包函数。解包 ListInt 类型属性值，并将输入的元素自动扩展到输出参数长度，依次赋值到输出参数中。
 *          仅解包，不做参数检查，由调用者做检查。需满足约束： \n
 *          - 属性size 不为 0 \n
 *          - 属性size 必须为解包长度的约数
 * @param   [in] checkedVec     属性vector
 * @param   [in] Is...          索引序列，从0开始，长度等于args个数
 * @param   [out] args          解包结果
 * @return  解包结果，元素个数为 unpackLen
 */
template <size_t... Is, typename... Args>
static inline void UnsafeUnpackListIntAttr(
    const gert::TypedContinuousVector<int64_t>*& checkedVec, std::index_sequence<Is...>, Args&... args)
{
    static_assert(sizeof...(Args) == sizeof...(Is), "Number of arguments must match length of sequence");
    // 将输出参数分为size段，每段赋值为 vec 中的对应元素
    // 已校验 size 不为 0，无除0问题
    size_t partLen = sizeof...(Is) / checkedVec->GetSize();
    // 已校验 size 必须为 unpackLen 的约数，partLen 不为0，且 Is / partLen 最大为 size - 1，不会越界
    ((args = checkedVec->GetData()[Is / partLen]), ...);
}
} // namespace

/**
 * @brief   解包固定长度的 ListInt 类型属性
 * @tparam  unpackLen       解包长度
 * @tparam  T               context类型
 * @param   [in] context    infershape 或 tiling 上下文
 * @param   [in] attrName   属性名称
 * @param   [in] vec        属性vector
 * @param   [in] elementValidator 元素校验方法，对元素挨个进行校验
 * @param   [in] invalidValueReason 是无效元素的原因
 * @return
 *      - status:   执行结果，GRAPH_FAILED: 失败，GRAPH_SUCCESS: 成功
 *      - result:   解包结果，元素个数为 unpackLen
 */
template <size_t unpackLen, typename T>
static inline std::tuple<ge::graphStatus, std::array<int64_t, unpackLen>> UnpackFixedDimListIntAttr(
    T* context, const char* attrName, const gert::TypedContinuousVector<int64_t>*& vec,
    const std::function<bool(int64_t)>& elementValidator, const char* invalidValueReason)
{
    if (!CheckUnpackFixedDimListIntAttr<unpackLen>(context, attrName, vec, elementValidator, invalidValueReason)) {
        return {ge::GRAPH_FAILED, {}};
    }
    std::array<int64_t, unpackLen> value{};
    for (size_t i = 0; i < vec->GetSize(); ++i) {
        value[i] = vec->GetData()[i];
    }
    return {ge::GRAPH_SUCCESS, value};
}

/**
 * @brief   解包自适应长度的 ListInt 类型属性，将输入的元素自动扩展到解包长度
 * @tparam  unpackLen       解包长度
 * @tparam  T               context类型
 * @param   [in] context    infershape 或 tiling 上下文
 * @param   [in] attrName   属性名称
 * @param   [in] vec        属性vector
 * @param   [in] elementValidator 元素校验方法，对元素挨个进行校验
 * @param   [in] invalidValueReason 是无效元素的原因
 * @return
        - status:   执行结果，GRAPH_FAILED: 失败，GRAPH_SUCCESS: 成功
        - result:   解包结果，元素个数为 unpackLen
 */
template <size_t unpackLen, typename T>
static inline std::tuple<ge::graphStatus, std::array<int64_t, unpackLen>> UnpackAdaptDimListIntAttr(
    T* context, const char* attrName, const gert::TypedContinuousVector<int64_t>*& vec,
    const std::function<bool(int64_t)>& elementValidator, const char* invalidValueReason)
{
    if (!CheckUnpackAdaptDimListIntAttr<unpackLen>(context, attrName, vec, elementValidator, invalidValueReason)) {
        return {ge::GRAPH_FAILED, {}};
    }
    std::array<int64_t, unpackLen> value{};
    // 已校验 size 不为 0，无除0问题
    size_t partLen = unpackLen / vec->GetSize();
    for (size_t i = 0; i < unpackLen; ++i) {
        // 已校验 size 必须为 unpackLen 的约数，partLen 不为0，且 i / partLen 最大为 size - 1，不会越界
        value[i] = vec->GetData()[i / partLen];
    }
    return {ge::GRAPH_SUCCESS, value};
}

/**
 * @brief   解包固定长度的 ListInt 类型属性
 * @tparam  unpackLen       解包长度
 * @tparam  T               context类型
 * @param   [in] context    infershape 或 tiling 上下文
 * @param   [in] attrName   属性名称
 * @param   [in] vec        属性vector
 * @param   [in] elementValidator 元素校验方法，对元素挨个进行校验
 * @param   [in] invalidValueReason 是无效元素的原因
 * @param   [out] args      解包结果，参数个数为 unpackLen
 * @return  执行结果，GRAPH_FAILED: 失败，GRAPH_SUCCESS: 成功
 */
template <size_t unpackLen, typename T, typename... Args>
static inline ge::graphStatus UnpackFixedDimListIntAttr(
    T* context, const char* attrName, const gert::TypedContinuousVector<int64_t>*& vec,
    const std::function<bool(int64_t)>& elementValidator, const char* invalidValueReason, Args&... args)
{
    static_assert(unpackLen > 0, "unpackLen should be positive");
    static_assert(sizeof...(Args) == unpackLen, "Number of arguments must match template paremeter unpackLen");
    static_assert((std::is_same_v<Args, int64_t> && ...), "All arguments must be of type int64_t");
    if (!CheckUnpackFixedDimListIntAttr<unpackLen>(context, attrName, vec, elementValidator, invalidValueReason)) {
        return ge::GRAPH_FAILED;
    }
    UnsafeUnpackListIntAttr(vec, std::make_index_sequence<unpackLen>{}, args...);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief   解包自适应长度的 ListInt 类型属性，将输入的元素自动扩展到解包长度
 * @tparam  unpackLen       解包长度
 * @tparam  T               context类型
 * @param   [in] context    infershape 或 tiling 上下文
 * @param   [in] attrName   属性名称
 * @param   [in] vec        属性vector
 * @param   [in] elementValidator 元素校验方法，对元素挨个进行校验
 * @param   [in] invalidValueReason 是无效元素的原因
 * @param   [out] args      解包结果，参数个数为 unpackLen
 * @return  执行结果，GRAPH_FAILED: 失败，GRAPH_SUCCESS: 成功
 */
template <size_t unpackLen, typename T, typename... Args>
static inline ge::graphStatus UnpackAdaptDimListIntAttr(
    T* context, const char* attrName, const gert::TypedContinuousVector<int64_t>*& vec,
    const std::function<bool(int64_t)>& elementValidator, const char* invalidValueReason, Args&... args)
{
    static_assert(unpackLen > 0, "unpackLen should be positive");
    static_assert(sizeof...(Args) == unpackLen, "Number of arguments must match template paremeter unpackLen");
    static_assert((std::is_same_v<Args, int64_t> && ...), "All arguments must be of type int64_t");
    if (!CheckUnpackAdaptDimListIntAttr<unpackLen>(context, attrName, vec, elementValidator, invalidValueReason)) {
        return ge::GRAPH_FAILED;
    }
    UnsafeUnpackListIntAttr(vec, std::make_index_sequence<unpackLen>{}, args...);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief   按NCHW顺序获取图像数据shape的维度，仅支持 NCHW/NHWC 格式
 * @tparam  T               context类型
 * @param   [in] context    infershape 或 tiling 上下文
 * @param   [in] shape      输入形状
 * @param   [in] format     输入数据格式
 * @return  结果
 *      - status:   执行结果，GRAPH_FAILED: 失败，GRAPH_SUCCESS: 成功
 *      - result:   按NCHW顺序的shape的维度
 */
template <typename T>
static inline std::tuple<ge::graphStatus, std::array<int64_t, NHWC_DIM_NUM>> GetImgDataDimsByNCHWOrder(
    T* context, const char* paramName, const gert::Shape& shape, const ge::Format& format)
{
    std::array<int64_t, NHWC_DIM_NUM> dims{0};
    size_t dimNum = shape.GetDimNum();
    if (unlikely(dimNum != NHWC_DIM_NUM)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), paramName, std::to_string(dimNum).c_str(), "4");
        return {ge::GRAPH_FAILED, dims};
    }

    if (format == ge::Format::FORMAT_NCHW) {
        return {
            ge::GRAPH_SUCCESS,
            {
                shape.GetDim(NCHW_N_DIM),
                shape.GetDim(NCHW_C_DIM),
                shape.GetDim(NCHW_H_DIM),
                shape.GetDim(NCHW_W_DIM),
            }};
    }
    if (format == ge::Format::FORMAT_NHWC) {
        return {
            ge::GRAPH_SUCCESS,
            {
                shape.GetDim(NHWC_N_DIM),
                shape.GetDim(NHWC_C_DIM),
                shape.GetDim(NHWC_H_DIM),
                shape.GetDim(NHWC_W_DIM),
            }};
    }
    OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), paramName, ge::GetFormatName(format), "NCHW or NHWC");
    return {ge::GRAPH_FAILED, dims};
}

/**
 * @brief   按NCHW顺序获取图像数据shape的维度，仅支持 NCHW/NHWC 格式
 * @tparam  T               context类型
 * @param   [in] context    infershape 或 tiling 上下文
 * @param   [in] shape      输入形状
 * @param   [in] format     输入数据格式
 * @param   [out] n         输出N
 * @param   [out] c         输出C
 * @param   [out] h         输出H
 * @param   [out] w         输出W
 * @return  执行结果，GRAPH_FAILED: 失败，GRAPH_SUCCESS: 成功
 */
template <typename T>
static inline ge::graphStatus GetImgDataDimsByNCHWOrder(
    T* context, const char* paramName, const gert::Shape& shape, const ge::Format& format, int64_t& n, int64_t& c,
    int64_t& h, int64_t& w)
{
    size_t dimNum = shape.GetDimNum();
    if (unlikely(dimNum != NHWC_DIM_NUM)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), paramName, std::to_string(dimNum).c_str(), "4");
        return ge::GRAPH_FAILED;
    }

    if (format == ge::Format::FORMAT_NCHW) {
        n = shape.GetDim(NCHW_N_DIM);
        c = shape.GetDim(NCHW_C_DIM);
        h = shape.GetDim(NCHW_H_DIM);
        w = shape.GetDim(NCHW_W_DIM);
        return ge::GRAPH_SUCCESS;
    }
    if (format == ge::Format::FORMAT_NHWC) {
        n = shape.GetDim(NHWC_N_DIM);
        c = shape.GetDim(NHWC_C_DIM);
        h = shape.GetDim(NHWC_H_DIM);
        w = shape.GetDim(NHWC_W_DIM);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), paramName, ge::GetFormatName(format), "NCHW or NHWC");
    return ge::GRAPH_FAILED;
}
} // namespace Ops::Math

#endif
