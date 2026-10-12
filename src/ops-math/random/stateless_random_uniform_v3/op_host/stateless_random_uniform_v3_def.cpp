/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stateless_random_uniform_v3_def.cpp
 * \brief Outputs pseudorandom values from a uniform distribution in [from, to).
 *
 * \par Inputs:
 * \li shape: 1-D or empty tensor. The shape of the output tensor. Must be one of the following types: int32, int64.
 * \li key: shape[1]. Key for the counter-based RNG algorithm. Must be one of the following types: uint64.
 * \li counter: 1-D. Initial counter for the counter-based RNG algorithm. Must be one of the following types: uint64.
 * \li from: 0-D scalar. Lower bound of the random range (inclusive). Must be one of the following types: float.
 * \li to: 0-D scalar. Upper bound of the random range (exclusive). Must be one of the following types: float.
 *
 * \par Attributes:
 * \li dtype: Output data type. Must be one of the following types: float16, bfloat16, float32. Defaults to float32.
 * \li v3KernelMode: Range scaling mode. 0 = Uniform mode (from/to are float, formula: x*(to-from)+from),
 *               1 = Random mode (from/to are integer-valued floats, formula: x*to-(x*from-from)).
 *               Defaults to 0.
 *
 * \par Outputs:
 * y: Returns random values with specified shape. Values are in [from, to).
 * Must be one of the following types: float16, bfloat16, float32.
 *
 * \par Third-party framework compatibility
 * Compatible with TensorFlow StatelessRandomUniformV2 operator.
 *
 * \par Operator prototype
 * \code
 * REG_OP(StatelessRandomUniformV3)
 *     .INPUT(shape,   TensorType({DT_INT32, DT_INT64}))
 *     .INPUT(key,     TensorType({DT_UINT64}))
 *     .INPUT(counter, TensorType({DT_UINT64}))
 *     .INPUT(from,    TensorType({DT_FLOAT}))
 *     .INPUT(to,      TensorType({DT_FLOAT}))
 *     .OUTPUT(y,      TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
 *     .ATTR(dtype,    Type, DT_FLOAT)
 *     .ATTR(v3KernelMode, Int,  0)
 *     .OP_END_FACTORY_REG(StatelessRandomUniformV3)
 * \endcode
 */

#include "register/op_def_registry.h"

namespace ops {
class StatelessRandomUniformV3 : public OpDef {
public:
    explicit StatelessRandomUniformV3(const char* name) : OpDef(name)
    {
        this->Input("shape")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .ValueDepend(OPTIONAL);
        this->Input("counter")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .ValueDepend(OPTIONAL);
        this->Input("from")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("to")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("dtype").AttrType(OPTIONAL).Int(0);
        this->Attr("v3KernelMode").AttrType(OPTIONAL).Int(0);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(StatelessRandomUniformV3);
} // namespace ops