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
class Im2col : public OpDef {
public:
    explicit Im2col(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> supportedDataTypes{ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BOOL};
        const std::vector<ge::Format> supportedDataFormats(supportedDataTypes.size(), ge::FORMAT_ND);
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(supportedDataTypes)
            .Format(supportedDataFormats)
            .UnknownShapeFormat(supportedDataFormats)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(supportedDataTypes)
            .Format(supportedDataFormats)
            .UnknownShapeFormat(supportedDataFormats);
        this->Attr("ksizes").AttrType(REQUIRED).ListInt();
        this->Attr("strides").AttrType(OPTIONAL).ListInt({1});
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1});
        this->Attr("padding_mode").AttrType(OPTIONAL).String("CALCULATED");
        this->Attr("pads").AttrType(OPTIONAL).ListInt({0});

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(Im2col);
} // namespace ops
