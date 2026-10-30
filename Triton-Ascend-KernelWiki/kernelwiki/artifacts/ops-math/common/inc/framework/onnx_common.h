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
 * \file onnx_common.h
 * \brief
 */

#ifndef MATH_COMMON_ONNX_COMMON_H
#define MATH_COMMON_ONNX_COMMON_H

#include <string>
#include <vector>
#include <map>

#include "stub_ops.h"
#include "register/register.h"
#include "graph/operator.h"
#include "graph/graph.h"
#include "base/err_msg.h"
#include "log/log.h"
#include "onnx/proto/ge_onnx.pb.h"

namespace domi {
template <typename T>
inline std::string GetOpName(const T& op)
{
    ge::AscendString op_ascend_name;
    ge::graphStatus ret = op.GetName(op_ascend_name);
    if (ret != ge::GRAPH_SUCCESS) {
        std::string op_name = "None";
        return op_name;
    }
    return op_ascend_name.GetString();
}

template <typename T>
inline ge::Tensor Vec2Tensor(
    vector<T>& vals, const vector<int64_t>& dims, ge::DataType dtype, ge::Format format = ge::FORMAT_ND)
{
    ge::Shape shape(dims);
    ge::TensorDesc desc(shape, format, dtype);
    ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(vals.data()), vals.size() * sizeof(T));
    return tensor;
}

template <typename T>
inline ge::Tensor CreateScalar(T val, ge::DataType dtype, ge::Format format = ge::FORMAT_ND)
{
    vector<int64_t> dims_scalar = {};
    ge::Shape shape(dims_scalar);
    ge::TensorDesc desc(shape, format, dtype);
    ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(&val), sizeof(T));
    return tensor;
}

enum DataTypeOnnx {
    DTO_FLOAT = 1,       // float type
    DTO_UINT8 = 2,       // uint8 type
    DTO_INT8 = 3,        // int8 type
    DTO_UINT16 = 4,      // uint16 type
    DTO_INT16 = 5,       // int16 type
    DTO_INT32 = 6,       // int32 type
    DTO_INT64 = 7,       // int64 type
    DTO_STRING = 8,      // string type
    DTO_BOOL = 9,        // bool type
    DTO_FLOAT16 = 10,    // float16 type
    DTO_DOUBLE = 11,     // double type
    DTO_UINT32 = 12,     // uint32 type
    DTO_UINT64 = 13,     // uint64 type
    DTO_COMPLEX64 = 14,  // complex64 type
    DTO_COMPLEX128 = 15, // complex128 type
    DTO_BF16 = 16,       // bf16 type
    DTO_UNDEFINED
};

static std::map<int, ge::DataType> onnx2om_dtype_map = {
    {DTO_UINT8, ge::DT_UINT8},   {DTO_UINT16, ge::DT_UINT16},       {DTO_UINT32, ge::DT_UINT32},
    {DTO_UINT64, ge::DT_UINT64}, {DTO_INT8, ge::DT_INT8},           {DTO_INT16, ge::DT_INT16},
    {DTO_INT32, ge::DT_INT32},   {DTO_INT64, ge::DT_INT64},         {DTO_FLOAT16, ge::DT_FLOAT16},
    {DTO_FLOAT, ge::DT_FLOAT},   {DTO_DOUBLE, ge::DT_DOUBLE},       {DTO_STRING, ge::DT_STRING},
    {DTO_BOOL, ge::DT_BOOL},     {DTO_COMPLEX64, ge::DT_COMPLEX64}, {DTO_COMPLEX128, ge::DT_COMPLEX128},
    {DTO_BF16, ge::DT_BF16}};

inline ge::DataType GetOmDtypeFromOnnxDtype(int onnx_type)
{
    auto dto_type = static_cast<DataTypeOnnx>(onnx_type);
    if (onnx2om_dtype_map.find(dto_type) == onnx2om_dtype_map.end()) {
        return ge::DT_UNDEFINED;
    }
    return onnx2om_dtype_map.at(dto_type);
}

inline Status ChangeFormatFromOnnx(ge::Operator& op, const int idx, ge::Format format, bool is_input)
{
    if (is_input) {
        ge::TensorDesc org_tensor = op.GetInputDesc(idx);
        org_tensor.SetOriginFormat(format);
        org_tensor.SetFormat(format);
        auto ret = op.UpdateInputDesc(idx, org_tensor);
        if (ret != ge::GRAPH_SUCCESS) {
            OP_LOGE(GetOpName(op).c_str(), "change input format failed.");
            return FAILED;
        }
    } else {
        ge::TensorDesc org_tensor_y = op.GetOutputDesc(idx);
        org_tensor_y.SetOriginFormat(format);
        org_tensor_y.SetFormat(format);
        auto ret_y = op.UpdateOutputDesc(idx, org_tensor_y);
        if (ret_y != ge::GRAPH_SUCCESS) {
            OP_LOGE(GetOpName(op).c_str(), "change output format failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

} // namespace domi

#endif //  MATH_COMMON_ONNX_COMMON_H