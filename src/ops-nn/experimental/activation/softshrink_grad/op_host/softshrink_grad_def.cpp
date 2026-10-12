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
 * \file softshrink_grad_def.cpp
 * \brief SoftShrinkGrad 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> softShrinkGradDtypes = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};
static const std::vector<ge::Format> ndFormats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class SoftshrinkGrad : public OpDef {
public:
    explicit SoftshrinkGrad(const char* name) : OpDef(name)
    {
        this->Input("input_grad")
            .ParamType(REQUIRED)
            .DataType(softShrinkGradDtypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous();
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType(softShrinkGradDtypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous();
        this->Output("output_y")
            .ParamType(REQUIRED)
            .DataType(softShrinkGradDtypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous();
        this->Attr("lambd").AttrType(OPTIONAL).Float(0.5);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(SoftshrinkGrad);
} // namespace ops
