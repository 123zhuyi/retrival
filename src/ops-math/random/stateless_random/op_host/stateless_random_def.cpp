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
 * \file stateless_random_def.cpp
 * \brief Operator definition for stateless_random
 */

#include "register/op_def_registry.h"
#include "../../random_common/op_host/random_dtype_fmt_gen.h"

namespace ops {
class StatelessRandom : public OpDef {
public:
    const std::vector<ge::DataType> int64Type = {ge::DT_INT64};
    const std::vector<ge::DataType> outputTypes = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32,
        ge::DT_INT64, ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL};
    const std::vector<ge::Format> baseFormat = {ge::FORMAT_ND};

    explicit StatelessRandom(const char* name) : OpDef(name)
    {
        randomdef::RandomDtypeFmtGen gen(
            {{"int64Type", int64Type}, {"outputTypes", outputTypes}, {"baseFormat", baseFormat}});
        const auto baseFormatSeq = gen.GetSequence<ge::Format>("baseFormat");

        this->Input("shape")
            .ParamType(REQUIRED)
            .DataType({gen.GetSequence("int64Type")})
            .Format({baseFormatSeq})
            .UnknownShapeFormat({baseFormatSeq});
        this->Input("seed")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType({gen.GetSequence("int64Type")})
            .Format({baseFormatSeq})
            .UnknownShapeFormat({baseFormatSeq});
        this->Input("offset")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType({gen.GetSequence("int64Type")})
            .Format({baseFormatSeq})
            .UnknownShapeFormat({baseFormatSeq});
        this->Input("from")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataType({gen.GetSequence("int64Type")})
            .Format({baseFormatSeq})
            .UnknownShapeFormat({baseFormatSeq});
        this->Input("to")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataType({gen.GetSequence("int64Type")})
            .Format({baseFormatSeq})
            .UnknownShapeFormat({baseFormatSeq});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({gen.GetSequence("outputTypes")})
            .Format({baseFormatSeq})
            .UnknownShapeFormat({baseFormatSeq});

        this->Attr("dtype").AttrType(OPTIONAL).Int(0);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(StatelessRandom);
} // namespace ops