/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "op_plugin/utils/op_api_common.h"

namespace cann_ops_nn {
namespace quant {

constexpr int64_t BLOCK_SIZE_ = 32;
const int g_toAclOffset_ = 256;
const int SIZE_ = 8;

static std::unordered_map<const aclDataType, const at::ScalarType> ACL_TYPE_TO_SCALAR_TYPE_MAP_ = {
    {ACL_DT_UNDEFINED, at::ScalarType::Undefined},
    {ACL_FLOAT, at::ScalarType::Float},
    {ACL_FLOAT16, at::ScalarType::Half},
    {ACL_INT8, at::ScalarType::Char},
    {ACL_INT32, at::ScalarType::Int},
    {ACL_UINT8, at::ScalarType::Byte},
    {ACL_INT16, at::ScalarType::Short},
    {ACL_UINT16, at::ScalarType::UInt16},
    {ACL_UINT32, at::ScalarType::UInt32},
    {ACL_INT64, at::ScalarType::Long},
    {ACL_UINT64, at::ScalarType::UInt64},
    {ACL_DOUBLE, at::ScalarType::Double},
    {ACL_BOOL, at::ScalarType::Bool},
    {ACL_STRING, at::ScalarType::Undefined},
    {ACL_COMPLEX64, at::ScalarType::ComplexFloat},
    {ACL_COMPLEX128, at::ScalarType::ComplexDouble},
    {ACL_BF16, at::ScalarType::BFloat16},
    {ACL_INT4, at::ScalarType::Undefined},
    {ACL_UINT1, at::ScalarType::Undefined},
    {ACL_COMPLEX32, at::ScalarType::ComplexHalf},
    {ACL_HIFLOAT8, at::ScalarType::Byte},
    {ACL_FLOAT8_E5M2, at::ScalarType::Float8_e5m2},
    {ACL_FLOAT8_E4M3FN, at::ScalarType::Float8_e4m3fn},
    {ACL_FLOAT8_E8M0, at::ScalarType::Float8_e8m0fnu},
    {ACL_FLOAT6_E3M2, at::ScalarType::Byte},
    {ACL_FLOAT6_E2M3, at::ScalarType::Byte},
    {ACL_FLOAT4_E2M1, at::ScalarType::Byte},
    {ACL_FLOAT4_E1M2, at::ScalarType::Byte}};

at::ScalarType ConvertToScalarType_(const aclDataType data_type)
{
    auto iter = ACL_TYPE_TO_SCALAR_TYPE_MAP_.find(data_type);
    if (iter == ACL_TYPE_TO_SCALAR_TYPE_MAP_.end()) {
        TORCH_CHECK(false, std::string("aclDataType:") + std::to_string(data_type) + " has not been supported")
    }

    return iter->second;
}

at::ScalarType convert_to_scalar_type_(const aclDataType data_type) { return ConvertToScalarType_(data_type); }

#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR_(_)     \
    _(at::ScalarType::Byte, ACL_UINT8)                   \
    _(at::ScalarType::Char, ACL_INT8)                    \
    _(at::ScalarType::Short, ACL_INT16)                  \
    _(at::ScalarType::Int, ACL_INT32)                    \
    _(at::ScalarType::Long, ACL_INT64)                   \
    _(at::ScalarType::Half, ACL_FLOAT16)                 \
    _(at::ScalarType::Float, ACL_FLOAT)                  \
    _(at::ScalarType::Double, ACL_DOUBLE)                \
    _(at::ScalarType::ComplexHalf, ACL_COMPLEX32)        \
    _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)       \
    _(at::ScalarType::ComplexDouble, ACL_COMPLEX128)     \
    _(at::ScalarType::Bool, ACL_BOOL)                    \
    _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)          \
    _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)          \
    _(at::ScalarType::BFloat16, ACL_BF16)                \
    _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)        \
    _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)        \
    _(at::ScalarType::Bits1x8, ACL_DT_UNDEFINED)         \
    _(at::ScalarType::Bits2x4, ACL_DT_UNDEFINED)         \
    _(at::ScalarType::Bits4x2, ACL_DT_UNDEFINED)         \
    _(at::ScalarType::Bits8, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::Bits16, ACL_DT_UNDEFINED)          \
    _(at::ScalarType::Float8_e5m2, ACL_FLOAT8_E5M2)      \
    _(at::ScalarType::Float8_e4m3fn, ACL_FLOAT8_E4M3FN)  \
    _(at::ScalarType::Float8_e5m2fnuz, ACL_DT_UNDEFINED) \
    _(at::ScalarType::Float8_e4m3fnuz, ACL_DT_UNDEFINED) \
    _(at::ScalarType::UInt16, ACL_UINT16)                \
    _(at::ScalarType::UInt32, ACL_UINT32)                \
    _(at::ScalarType::UInt64, ACL_UINT64)                \
    _(at::ScalarType::UInt1, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::UInt2, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::UInt3, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::UInt4, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::UInt5, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::UInt6, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::UInt7, ACL_DT_UNDEFINED)           \
    _(at::ScalarType::Int1, ACL_DT_UNDEFINED)            \
    _(at::ScalarType::Int2, ACL_DT_UNDEFINED)            \
    _(at::ScalarType::Int3, ACL_DT_UNDEFINED)            \
    _(at::ScalarType::Int4, ACL_DT_UNDEFINED)            \
    _(at::ScalarType::Int5, ACL_DT_UNDEFINED)            \
    _(at::ScalarType::Int6, ACL_DT_UNDEFINED)            \
    _(at::ScalarType::Int7, ACL_DT_UNDEFINED)            \
    _(at::ScalarType::Float8_e8m0fnu, ACL_FLOAT8_E8M0)   \
    _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)       \
    _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

constexpr aclDataType kATenScalarTypeToAclDataTypeTable_[static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
    AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR_(DEFINE_ENUM)
#undef DEFINE_ENUM
};

inline aclDataType ConvertToAclDataType_(const at::ScalarType& data_type)
{
    int64_t dtype_index = static_cast<int64_t>(data_type);
    TORCH_CHECK(dtype_index >= 0 && dtype_index < static_cast<int64_t>(at::ScalarType::NumOptions) + 1,
                "data_type enum value (", dtype_index, ") is out of range: [0, ",
                static_cast<int64_t>(at::ScalarType::NumOptions), "]")
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable_[dtype_index];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED, std::string(c10::toString(data_type)) + " has not been supported")
    return acl_dtype;
}

inline aclDataType GetAclDataType_(int64_t t)
{
    if (t >= g_toAclOffset_) {
        return static_cast<aclDataType>(t - g_toAclOffset_);
    }
    return ConvertToAclDataType_(static_cast<at::ScalarType>(t));
}

inline c10::SmallVector<int64_t, SIZE_> array_to_small_vector_(c10::IntArrayRef shape)
{
    c10::SmallVector<int64_t, SIZE_> shape_small_vec;
    for (uint64_t i = 0; i < shape.size(); i++) {
        shape_small_vec.emplace_back(shape[i]);
    }
    return shape_small_vec;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mx_to_block_mx_quant(const at::Tensor& x, const at::Tensor& mxscale,
                                                                    int64_t dst_type, int64_t x_type)
{
    TORCH_CHECK(x.numel() > 0, "Input tensor x should not be empty.");
    TORCH_CHECK(mxscale.numel() > 0, "Input tensor mxscale should not be empty.");
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "Input x rank must be 2 or 3, got ", x.dim());
    TORCH_CHECK(mxscale.dim() == x.dim() + 1, "Input mxscale rank must be x rank + 1.");
    TORCH_CHECK(x_type == 296 || x_type == 297, "x_type must be one of {296, 297}, got ", x_type);

    at::Tensor y;
    at::Tensor scale1;
    at::Tensor scale2;
    auto y_shape = array_to_small_vector_(x.sizes());

    aclDataType src_acltype = GetAclDataType_(x_type);
    bool special_input_type = (src_acltype == ACL_FLOAT4_E2M1) || (src_acltype == ACL_FLOAT4_E1M2);
    if (special_input_type) {
        y_shape[x.dim() - 1] = x.size(x.dim() - 1) * FP4_IN_INT8;
    }

    aclDataType x_acltype = ACL_FLOAT4_E2M1;
    if (src_acltype == ACL_FLOAT4_E1M2) {
        x_acltype = ACL_FLOAT4_E1M2;
    }

    aclDataType y_acltype = GetAclDataType_(dst_type);
    at::ScalarType dtype = convert_to_scalar_type_(y_acltype);
    y = at::empty(y_shape, x.options().dtype(dtype));

    auto scale1_shape = array_to_small_vector_(mxscale.sizes());
    scale1 = at::empty(scale1_shape, x.options().dtype(at::ScalarType::Byte));

    int64_t rank = x.dim();
    int64_t last_dim = y.size(rank - 1);
    int64_t second_last_dim = x.size(rank - 2);
    int64_t scale2_dim0 = ((second_last_dim + BLOCK_SIZE_ - 1) / BLOCK_SIZE_ + 2 - 1) / 2;
    c10::SmallVector<int64_t, SIZE_> scale2_shape;
    if (rank == 3) {
        scale2_shape.push_back(x.size(0));
    }
    scale2_shape.push_back(scale2_dim0);
    scale2_shape.push_back(last_dim);
    scale2_shape.push_back(2);
    scale2 = at::empty(scale2_shape, x.options().dtype(at::ScalarType::Byte));

    TensorWrapper mxscale_wrapper = {mxscale, ACL_FLOAT8_E8M0};
    TensorWrapper x_wrapper = {x, x_acltype};
    TensorWrapper y_wrapper = {y, y_acltype};
    TensorWrapper scale1_wrapper = {scale1, ACL_FLOAT8_E8M0};
    TensorWrapper scale2_wrapper = {scale2, ACL_FLOAT8_E8M0};

    EXEC_NPU_CMD_EXT(aclnnMxToBlockMxQuant, x_wrapper, mxscale_wrapper, y_acltype, y_wrapper, scale1_wrapper,
                     scale2_wrapper);

    return std::make_tuple(y, scale1, scale2);
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mx_to_block_mx_quant", &cann_ops_nn::quant::mx_to_block_mx_quant, "MxToBlockMxQuant operator on NPU");
}
