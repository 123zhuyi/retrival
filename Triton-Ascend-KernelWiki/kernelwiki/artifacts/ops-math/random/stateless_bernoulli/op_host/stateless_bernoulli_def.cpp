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
 * \file stateless_bernoulli_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "../../random_common/op_host/random_dtype_fmt_gen.h"

namespace ops {
class StatelessBernoulli : public OpDef {
public:
    const std::vector<ge::DataType> inputDataType = {ge::DT_INT32, ge::DT_INT64};
    const std::vector<ge::DataType> probDataType = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    const std::vector<ge::DataType> baseDataType = {ge::DT_INT64};
    const std::vector<ge::DataType> outputDataType = {ge::DT_INT8,  ge::DT_UINT8,   ge::DT_INT16, ge::DT_UINT16,
                                                      ge::DT_INT32, ge::DT_UINT32,  ge::DT_INT64, ge::DT_UINT64,
                                                      ge::DT_BOOL,  ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    const std::vector<ge::Format> baseFormat = {ge::FORMAT_ND};

    explicit StatelessBernoulli(const char* name) : OpDef(name)
    {
        randomdef::RandomDtypeFmtGen gen({
            {"inputDataType", inputDataType}, {"probDataType", probDataType}, {"baseDataType", baseDataType},
            {"outputDataType", outputDataType}, {"baseFormat", baseFormat}});
		const auto baseFormatSeq = gen.GetSequence<ge::Format>("baseFormat");
		
		this->Input("shape")
			.ParamType(REQUIRED)
			.DataType(gen.GetSequence("inputDataType"))
			.Format(baseFormatSeq)
			.UnknownShapeFormat(baseFormatSeq)
			.ValueDepend(OPTIONAL);
        this->Input("prob")
            .ParamType(REQUIRED)
            .DataType(gen.GetSequence("probDataType"))
            .Format(baseFormatSeq)
            .UnknownShapeFormat(baseFormatSeq);
        this->Input("seed")
            .ParamType(REQUIRED)
            .DataType(gen.GetSequence("baseDataType"))
            .Format(baseFormatSeq)
            .UnknownShapeFormat(baseFormatSeq)
            .ValueDepend(OPTIONAL);
        this->Input("offset")
            .ParamType(OPTIONAL)
            .DataType(gen.GetSequence("baseDataType"))
            .Format(baseFormatSeq)
            .UnknownShapeFormat(baseFormatSeq)
            .ValueDepend(OPTIONAL);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(gen.GetSequence("outputDataType"))
            .Format(baseFormatSeq)
            .UnknownShapeFormat(baseFormatSeq);
        this->Attr("dtype").AttrType(OPTIONAL).Int();

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

OP_ADD(StatelessBernoulli);
} // namespace ops
