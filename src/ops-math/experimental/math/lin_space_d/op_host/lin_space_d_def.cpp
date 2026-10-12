/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file lin_space_d.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class LinSpaceD : public OpDef {
public:
    explicit LinSpaceD(const char* name) : OpDef(name)
    {
        this->Input("start")
            .ParamType(REQUIRED)
            .DataTypeList(
                {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("end")
            .ParamType(REQUIRED)
            .DataTypeList(
                {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("num")
            .ValueDepend(REQUIRED)
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("z").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "lin_space_d");
        this->AICore()
            .AddConfig("ascend910b", aicoreConfig)
            .AddConfig("ascend310p", aicoreConfig)
            .AddConfig("ascend910", aicoreConfig)
            .AddConfig("ascend310b", aicoreConfig);
    }
};
OP_ADD(LinSpaceD);
} // namespace ops