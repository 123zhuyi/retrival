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
 * \file drop_out_do_mask_v3_d_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "../../random_common/op_host/random_dtype_fmt_gen.h"

namespace ops {
class DropOutDoMaskV3D : public OpDef {
public:
    const std::vector<ge::DataType> xDataType = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    const std::vector<ge::DataType> maskDataType = {ge::DT_UINT8, ge::DT_BOOL};
    const std::vector<ge::Format> baseFormat = {ge::FORMAT_ND};

    explicit DropOutDoMaskV3D(const char* name) : OpDef(name)
    {
        randomdef::RandomDtypeFmtGen gen(
            {{"xDataType", xDataType},
             {"maskDataType", maskDataType},
             {"baseFormat", baseFormat}});
        const auto baseFormatSeq = gen.GetSequence<ge::Format>("baseFormat");

        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(gen.GetSequence("xDataType"))
            .Format(baseFormatSeq)
            .UnknownShapeFormat(baseFormatSeq)
            .ValueDepend(OPTIONAL);
        this->Input("mask")
            .ParamType(REQUIRED)
            .DataType(gen.GetSequence("maskDataType"))
            .Format(baseFormatSeq)
            .UnknownShapeFormat(baseFormatSeq)
            .ValueDepend(OPTIONAL);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(gen.GetSequence("xDataType"))
            .Format(baseFormatSeq)
            .UnknownShapeFormat(baseFormatSeq);
        this->Attr("keep_prob").AttrType(REQUIRED).Float();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(DropOutDoMaskV3D);
} // namespace ops
