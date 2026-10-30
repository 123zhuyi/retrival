/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATH_COMMON_DATA_TYPE_UTILS_H
#define MATH_COMMON_DATA_TYPE_UTILS_H

#include "opdev/data_type_utils.h"
#include "opdev/op_log.h"
#include "op_api/op_api_def.h"

namespace op {

constexpr int DTYPE_BITS_8 = 8;
constexpr int DTYPE_BITS_16 = 16;
constexpr int DTYPE_BITS_32 = 32;
constexpr int DTYPE_BITS_64 = 64;

// 浮点类型映射至对应复数类型，保证数值精度一致性
inline DataType InnerTypeToComplexType(const DataType input)
{
    switch (input) {
        case DataType::DT_BF16:
            return DataType::DT_COMPLEX64;
        case DataType::DT_FLOAT16:
            return DataType::DT_COMPLEX32;
        case DataType::DT_FLOAT:
            return DataType::DT_COMPLEX64;
        case DataType::DT_DOUBLE:
            return DataType::DT_COMPLEX128;
        case DataType::DT_COMPLEX32:
            return DataType::DT_COMPLEX32;
        case DataType::DT_COMPLEX64:
            return DataType::DT_COMPLEX64;
        case DataType::DT_COMPLEX128:
            return DataType::DT_COMPLEX128;
        default:
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unknown Complex ScalarType for [%s]", ToString(input).GetString());
            return DataType::DT_UNDEFINED;
    }
}

// 张量+标量混合运算类型推导，优先级complex>float>integral>bool，浮点遇复数时映射为复数
inline DataType CombineCategoriesWithComplex(const DataType higher, const DataType lower)
{
    if (IsComplexType(higher)) {
        return higher;
    } else if (IsComplexType(lower)) {
        if (IsFloatingType(higher)) {
            return InnerTypeToComplexType(higher);
        }
        return lower;
    } else if (IsFloatingType(higher)) {
        return higher;
    }
    if (higher == DataType::DT_BOOL || IsFloatingType(lower)) {
        return PromoteType(higher, lower);
    }
    if (higher != DataType::DT_UNDEFINED) {
        return higher;
    }
    return lower;
}

// 双输入类型推导，维度相同则标准提升，维度不同则混合推导
inline DataType BinaryOpTypePromote(
    const aclTensor* self, const aclTensor* other, bool promoteInt2Float = false)
{
    if (self == nullptr || other == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "self or other is nullptr");
        return DataType::DT_UNDEFINED;
    }
    auto selfDim = self->GetViewShape().GetDimNum();
    auto otherDim = other->GetViewShape().GetDimNum();
    DataType promoteType = DataType::DT_UNDEFINED;
    if ((selfDim > 0 && otherDim > 0) || (selfDim == 0 && otherDim == 0)) {
        promoteType = PromoteType(self->GetDataType(), other->GetDataType());
    } else {
        DataType higherDtype = DataType::DT_UNDEFINED;
        DataType lowerDtype = DataType::DT_UNDEFINED;
        if (selfDim > 0) {
            higherDtype = self->GetDataType();
            lowerDtype = other->GetDataType();
        } else {
            higherDtype = other->GetDataType();
            lowerDtype = self->GetDataType();
        }
        promoteType = CombineCategoriesWithComplex(higherDtype, lowerDtype);
    }
    if (promoteInt2Float && IsIntegralType(promoteType, true)) {
        promoteType = DataType::DT_FLOAT;
    }
    return promoteType;
}

// 判断有符号整型
inline bool IsSignedIntegerType(const DataType dtype)
{
    return dtype == DataType::DT_INT8 || dtype == DataType::DT_INT16 ||
           dtype == DataType::DT_INT32 || dtype == DataType::DT_INT64;
}

// 判断无符号整型
inline bool IsUnsignedIntegerType(const DataType dtype)
{
    return dtype == DataType::DT_UINT8 || dtype == DataType::DT_UINT16 ||
           dtype == DataType::DT_UINT32 || dtype == DataType::DT_UINT64;
}

// 获取数据类型位宽
inline int GetDataTypeBits(const DataType dtype)
{
    switch (dtype) {
        case DataType::DT_INT8:
        case DataType::DT_UINT8:
        case DataType::DT_BOOL:
            return DTYPE_BITS_8;
        case DataType::DT_INT16:
        case DataType::DT_UINT16:
        case DataType::DT_FLOAT16:
        case DataType::DT_BF16:
            return DTYPE_BITS_16;
        case DataType::DT_INT32:
        case DataType::DT_UINT32:
        case DataType::DT_FLOAT:
            return DTYPE_BITS_32;
        case DataType::DT_INT64:
        case DataType::DT_UINT64:
        case DataType::DT_DOUBLE:
            return DTYPE_BITS_64;
        default:
            return 0;
    }
}

// 判断输入类型直接计算安全性，需满足符号一致、位宽不溢出、类型兼容
inline bool IsMaxMinSafeInputDtype(const DataType inDtype, const DataType outDtype)
{
    if (inDtype == outDtype || inDtype == DataType::DT_BOOL) {
        return true;
    }
    if (IsIntegralType(inDtype) && IsIntegralType(outDtype)) {
        return IsSignedIntegerType(inDtype) == IsSignedIntegerType(outDtype) &&
               GetDataTypeBits(inDtype) <= GetDataTypeBits(outDtype);
    }
    if (IsFloatingType(outDtype)) {
        return IsIntegralType(inDtype) || IsFloatingType(inDtype);
    }
    return false;
}

} // namespace op

#endif // MATH_COMMON_DATA_TYPE_UTILS_H
