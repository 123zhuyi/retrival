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
 * \file stateless_drop_out_gen_mask_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "../../random_common/op_host/random_dtype_fmt_gen.h"

namespace ops {
class StatelessDropOutGenMask : public OpDef {
public:
	const std::vector<ge::DataType> shapeDataType = {ge::DT_INT32, ge::DT_INT64};
	const std::vector<ge::DataType> probDataType = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
	const std::vector<ge::DataType> seedDataType = {ge::DT_INT32, ge::DT_INT64};
	const std::vector<ge::DataType> seed1DataType = {ge::DT_INT32, ge::DT_INT64};
	const std::vector<ge::DataType> offsetDataType = {ge::DT_INT64};
	const std::vector<ge::DataType> yDataType = {ge::DT_UINT8};
	const std::vector<ge::Format> baseFormat = {ge::FORMAT_ND};

	explicit StatelessDropOutGenMask(const char* name) : OpDef(name)
	{
		randomdef::RandomDtypeFmtGen gen({
			{"shapeDataType", shapeDataType}, {"probDataType", probDataType}, {"seedDataType", seedDataType},
			{"seed1DataType", seed1DataType}, {"offsetDataType", offsetDataType}, {"yDataType", yDataType},
			{"baseFormat", baseFormat}});
		const auto baseFormatSeq = gen.GetSequence<ge::Format>("baseFormat");

		this->Input("shape")
			.ParamType(REQUIRED)
			.DataType(gen.GetSequence("shapeDataType"))
			.Format(baseFormatSeq)
			.UnknownShapeFormat(baseFormatSeq)
			.ValueDepend(OPTIONAL);
		this->Input("prob")
			.ParamType(REQUIRED)
			.DataType(gen.GetSequence("probDataType"))
			.Format(baseFormatSeq)
			.UnknownShapeFormat(baseFormatSeq)
			.ValueDepend(OPTIONAL);
		this->Input("seed")
			.ParamType(REQUIRED)
			.DataType(gen.GetSequence("seedDataType"))
			.Format(baseFormatSeq)
			.UnknownShapeFormat(baseFormatSeq)
			.ValueDepend(OPTIONAL);
		this->Input("seed1")
			.ParamType(REQUIRED)
			.DataType(gen.GetSequence("seed1DataType"))
			.Format(baseFormatSeq)
			.UnknownShapeFormat(baseFormatSeq)
			.ValueDepend(OPTIONAL);
		this->Input("offset")
			.ParamType(OPTIONAL)
			.DataType(gen.GetSequence("offsetDataType"))
			.Format(baseFormatSeq)
			.UnknownShapeFormat(baseFormatSeq)
			.ValueDepend(OPTIONAL);
		this->Output("y")
			.ParamType(REQUIRED)
			.DataType(gen.GetSequence("yDataType"))
			.Format(baseFormatSeq)
			.UnknownShapeFormat(baseFormatSeq);

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

OP_ADD(StatelessDropOutGenMask);
}  // namespace ops
