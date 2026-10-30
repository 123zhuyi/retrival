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
 * \file polar_def.cpp
 * \brief Polar 算子原型定义：out = input * (cos(angle) + i*sin(angle))
 *        input/angle 为 float32（可 numpy 广播），out 恒为 complex64。
 *        支持 Atlas A2 训练系列产品（ascend910b）/ Atlas A3 训练系列产品（ascend910_93）。
 */
#include "register/op_def_registry.h"

namespace ops {
class Polar : public OpDef {
public:
    explicit Polar(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("angle")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->AICore().AddConfig("ascend910b");   // Atlas A2 训练/推理系列产品
        this->AICore().AddConfig("ascend910_93"); // Atlas A3 训练/推理系列产品
    }
};
OP_ADD(Polar);
} // namespace ops
