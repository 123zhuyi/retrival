/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_INV_GRAD_EXAMPLES_ARCH35_INV_GRAD_GEIR_TEST_COMMON_H
#define OPS_MATH_INV_GRAD_EXAMPLES_ARCH35_INV_GRAD_GEIR_TEST_COMMON_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include "graph.h"
#include "tensor.h"
#include "types.h"
#include "array_ops.h"

#include "../../op_graph/inv_grad_proto.h"

namespace inv_grad_geir_test {

inline size_t NumElements(const std::vector<int64_t>& shape)
{
    size_t numel = 1;
    for (int64_t dim : shape) {
        numel *= static_cast<size_t>(dim);
    }
    return numel;
}

inline std::string ShapeToJson(const std::vector<int64_t>& shape)
{
    std::ostringstream stream;
    stream << "[";
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            stream << ",";
        }
        stream << shape[index];
    }
    stream << "]";
    return stream.str();
}

inline const char* DtypeName(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return "float32";
        case ge::DT_FLOAT16:
            return "float16";
        case ge::DT_BF16:
            return "bfloat16";
        case ge::DT_INT32:
            return "int32";
        default:
            return "unsupported";
    }
}

inline uint16_t FloatToHalfBits(float value)
{
    uint32_t source = 0;
    std::memcpy(&source, &value, sizeof(source));
    const uint32_t sign = (source >> 31U) & 1U;
    const uint32_t exponent = (source >> 23U) & 0xFFU;
    const uint32_t mantissa = source & 0x7FFFFFU;
    if (exponent == 0xFFU) {
        return static_cast<uint16_t>((sign << 15U) | 0x7C00U | (mantissa == 0U ? 0U : 0x0200U));
    }
    if (exponent == 0U) {
        return static_cast<uint16_t>(sign << 15U);
    }

    const int32_t halfExponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (halfExponent >= 31) {
        return static_cast<uint16_t>((sign << 15U) | 0x7C00U);
    }
    if (halfExponent <= 0) {
        return static_cast<uint16_t>(sign << 15U);
    }

    uint16_t result = static_cast<uint16_t>((sign << 15U) | (static_cast<uint32_t>(halfExponent) << 10U) |
                                            (mantissa >> 13U));
    const uint32_t remainder = mantissa & 0x1FFFU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (result & 1U) != 0U)) {
        ++result;
    }
    return result;
}

inline float HalfBitsToFloat(uint16_t value)
{
    const uint32_t sign = (static_cast<uint32_t>(value) >> 15U) & 1U;
    const uint32_t exponent = (static_cast<uint32_t>(value) >> 10U) & 0x1FU;
    const uint32_t mantissa = static_cast<uint32_t>(value) & 0x03FFU;
    uint32_t result = 0;
    if (exponent == 0U) {
        result = sign << 31U;
    } else if (exponent == 0x1FU) {
        result = (sign << 31U) | 0x7F800000U | (mantissa << 13U);
    } else {
        result = (sign << 31U) | ((exponent - 15U + 127U) << 23U) | (mantissa << 13U);
    }
    float output = 0.0F;
    std::memcpy(&output, &result, sizeof(output));
    return output;
}

inline uint16_t FloatToBf16Bits(float value)
{
    uint32_t source = 0;
    std::memcpy(&source, &value, sizeof(source));
    const uint32_t rounded = source + 0x7FFFU + ((source >> 16U) & 1U);
    return static_cast<uint16_t>(rounded >> 16U);
}

inline float Bf16BitsToFloat(uint16_t value)
{
    const uint32_t source = static_cast<uint32_t>(value) << 16U;
    float output = 0.0F;
    std::memcpy(&output, &source, sizeof(output));
    return output;
}

inline bool MakeFilledTensor(const std::vector<int64_t>& shape, ge::DataType dtype, double value, ge::Tensor& tensor)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(ge::FORMAT_ND);
    desc.SetRealDimCnt(shape.size());
    const size_t numel = NumElements(shape);

    if (dtype == ge::DT_FLOAT) {
        float* data = new (std::nothrow) float[numel];
        if (data == nullptr) {
            return false;
        }
        for (size_t index = 0; index < numel; ++index) {
            data[index] = static_cast<float>(value);
        }
        tensor = ge::Tensor(desc, reinterpret_cast<uint8_t*>(data), numel * sizeof(float));
        return true;
    }
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        uint16_t* data = new (std::nothrow) uint16_t[numel];
        if (data == nullptr) {
            return false;
        }
        const uint16_t bits = dtype == ge::DT_FLOAT16 ? FloatToHalfBits(static_cast<float>(value)) :
                                                        FloatToBf16Bits(static_cast<float>(value));
        for (size_t index = 0; index < numel; ++index) {
            data[index] = bits;
        }
        tensor = ge::Tensor(desc, reinterpret_cast<uint8_t*>(data), numel * sizeof(uint16_t));
        return true;
    }
    if (dtype == ge::DT_INT32) {
        int32_t* data = new (std::nothrow) int32_t[numel];
        if (data == nullptr) {
            return false;
        }
        for (size_t index = 0; index < numel; ++index) {
            data[index] = static_cast<int32_t>(value);
        }
        tensor = ge::Tensor(desc, reinterpret_cast<uint8_t*>(data), numel * sizeof(int32_t));
        return true;
    }
    return false;
}

inline bool ShapeMatches(const ge::Shape& actual, const std::vector<int64_t>& expected)
{
    if (actual.GetDimNum() != expected.size()) {
        return false;
    }
    for (size_t index = 0; index < expected.size(); ++index) {
        if (actual.GetDim(index) != expected[index]) {
            return false;
        }
    }
    return true;
}

inline bool ValidateOutput(const std::vector<ge::Tensor>& outputs, const std::vector<int64_t>& expectedShape,
                           ge::DataType expectedDtype, double expectedValue)
{
    if (outputs.size() != 1U) {
        return false;
    }
    const ge::TensorDesc& desc = outputs[0].GetTensorDesc();
    if (!ShapeMatches(desc.GetShape(), expectedShape) || desc.GetDataType() != expectedDtype) {
        return false;
    }

    const size_t numel = NumElements(expectedShape);
    const uint8_t* raw = outputs[0].GetData();
    if (numel != 0U && raw == nullptr) {
        return false;
    }
    for (size_t index = 0; index < numel; ++index) {
        if (expectedDtype == ge::DT_FLOAT) {
            const float actual = reinterpret_cast<const float*>(raw)[index];
            if (std::fabs(actual - static_cast<float>(expectedValue)) > 1.0e-5F) {
                return false;
            }
        } else if (expectedDtype == ge::DT_FLOAT16) {
            const float actual = HalfBitsToFloat(reinterpret_cast<const uint16_t*>(raw)[index]);
            if (std::fabs(actual - static_cast<float>(expectedValue)) > 5.0e-3F) {
                return false;
            }
        } else if (expectedDtype == ge::DT_BF16) {
            const float actual = Bf16BitsToFloat(reinterpret_cast<const uint16_t*>(raw)[index]);
            if (std::fabs(actual - static_cast<float>(expectedValue)) > 2.0e-2F) {
                return false;
            }
        } else if (expectedDtype == ge::DT_INT32) {
            if (reinterpret_cast<const int32_t*>(raw)[index] != static_cast<int32_t>(expectedValue)) {
                return false;
            }
        } else {
            return false;
        }
    }
    return true;
}

inline bool BuildGraph(ge::Graph& graph, const std::vector<int64_t>& declaredShape, ge::DataType dtype,
                       const std::string& tag)
{
    const std::string xName = "x_" + tag;
    const std::string gradName = "grad_" + tag;
    const std::string opName = "inv_grad_" + tag;

    auto x = ge::op::Data(xName.c_str()).set_attr_index(0);
    ge::TensorDesc xDesc(ge::Shape(declaredShape), ge::FORMAT_ND, dtype);
    xDesc.SetPlacement(ge::kPlacementHost);
    x.update_input_desc_x(xDesc);

    auto grad = ge::op::Data(gradName.c_str()).set_attr_index(1);
    ge::TensorDesc gradDesc(ge::Shape(declaredShape), ge::FORMAT_ND, dtype);
    gradDesc.SetPlacement(ge::kPlacementHost);
    grad.update_input_desc_x(gradDesc);

    auto invGrad = ge::op::InvGrad(opName.c_str());
    invGrad.set_input_x(x);
    invGrad.set_input_grad(grad);
    ge::TensorDesc outputDesc(ge::Shape(declaredShape), ge::FORMAT_ND, dtype);
    invGrad.update_output_desc_y(outputDesc);

    graph.AddOp(x);
    graph.AddOp(grad);
    graph.SetInputs({x, grad}).SetOutputs({invGrad});
    return true;
}

} // namespace inv_grad_geir_test

#endif // OPS_MATH_INV_GRAD_EXAMPLES_ARCH35_INV_GRAD_GEIR_TEST_COMMON_H
