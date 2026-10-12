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
 * \file swin_attention_ffn.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops
{
    class SwinAttentionFFN : public OpDef
    {
        public:
            explicit SwinAttentionFFN(const char* name) : OpDef(name)
            {
                this->Input("x1")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
                this->Input("x2")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
                this->Input("bias")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
                this->Input("x3")
                    .ParamType(OPTIONAL)
                    .DataType({ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
                this->Output("y")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
                this->Attr("shifts")
                    .AttrType(OPTIONAL)
                    .ListInt({});
                this->AICore().AddConfig("ascend910b");
                this->AICore().AddConfig("kirinx90");
                this->AICore().AddConfig("kirin9030");
            }
    };

    OP_ADD(SwinAttentionFFN);
}
