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
 * \file stateless_normal_def.cpp
 * \brief StatelessNormal V4 OpDef definition
 */
#include "register/op_def_registry.h"
#include "../../random_common/op_host/random_dtype_fmt_gen.h"

namespace ops {
class StatelessNormal : public OpDef {
public:
    const std::vector<ge::DataType> shapeType = {ge::DT_INT64};
    const std::vector<ge::DataType> seedType = {ge::DT_INT64};
    const std::vector<ge::DataType> offsetType = {ge::DT_INT64};
    const std::vector<ge::DataType> meanType = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    const std::vector<ge::DataType> stdevType = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    const std::vector<ge::DataType> outType = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    const std::vector<ge::Format> baseFormat = {ge::FORMAT_ND};

    explicit StatelessNormal(const char* name) : OpDef(name)
    {
        randomdef::RandomDtypeFmtGen gen({
            {"seedType", seedType}, {"meanType", meanType}, {"shapeType", shapeType},
            {"offsetType", offsetType}, {"stdevType", stdevType}, {"outType", outType},
            {"baseFormat", baseFormat}});
        const auto baseFormatSeq = gen.GetSequence<ge::Format>("baseFormat");

        this->Input("shape")
            .ParamType(REQUIRED)
            .DataType(gen.GetSequence("shapeType"))
            .Format(baseFormatSeq)
            .UnknownShapeFormat(baseFormatSeq);
        this->Input("seed")
            .ParamType(REQUIRED)
            .DataType(gen.GetSequence("seedType"))
            .Format(baseFormatSeq)
            .UnknownShapeFormat(baseFormatSeq)
            .ValueDepend(OPTIONAL);
        this->Input("offset")
            .ParamType(REQUIRED)
            .DataType(gen.GetSequence("offsetType"))
            .Format(baseFormatSeq)
            .UnknownShapeFormat(baseFormatSeq)
            .ValueDepend(OPTIONAL);
        this->Input("mean")
            .ParamType(REQUIRED)
            .DataType(gen.GetSequence("meanType"))
            .Format(baseFormatSeq)
            .UnknownShapeFormat(baseFormatSeq)
            .ValueDepend(OPTIONAL);
        this->Input("stdev")
            .ParamType(REQUIRED)
            .DataType(gen.GetSequence("stdevType"))
            .Format(baseFormatSeq)
            .UnknownShapeFormat(baseFormatSeq)
            .ValueDepend(OPTIONAL);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(gen.GetSequence("outType"))
            .Format(baseFormatSeq)
            .UnknownShapeFormat(baseFormatSeq);

        this->Attr("dtype").AttrType(OPTIONAL).Int(0);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(StatelessNormal);
} // namespace ops
