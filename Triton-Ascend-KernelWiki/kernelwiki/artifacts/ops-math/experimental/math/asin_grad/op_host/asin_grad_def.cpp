/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {
class AsinGrad : public OpDef {
public:
    explicit AsinGrad(const char* name) : OpDef(name)
    {
        auto setAsinGradTensorAttr = [](auto&& tensor) {
            tensor.ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
        };

        setAsinGradTensorAttr(this->Input("y"));
        setAsinGradTensorAttr(this->Input("dy"));
        setAsinGradTensorAttr(this->Output("z"));

        auto& aiCore = this->AICore();
        aiCore.AddConfig("ascend910b");
        aiCore.AddConfig("ascend910_93");
    }
};
OP_ADD(AsinGrad);
} // namespace ops
