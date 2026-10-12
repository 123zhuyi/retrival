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
 * \file stateless_randperm_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace  ops{
static const std::vector<ge::DataType> nDtype = {
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
};

static const std::vector<ge::DataType> outDtype = {
    ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
};

static const std::vector<ge::Format> format = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
};

class StatelessRandperm: public OpDef { 
public:
    explicit StatelessRandperm(const char* name) : OpDef(name) {
        StatelessRandpermFor950();
    }

private: 
    void StatelessRandpermFor950() {
        this->Input("n")
            .ParamType(REQUIRED)
            .DataType(nDtype)
            .Format(format)
            .UnknownShapeFormat(format)
            .ValueDepend(OPTIONAL);
        this->Input("seed")
            .ParamType(REQUIRED)
            .DataType(nDtype)
            .Format(format)
            .UnknownShapeFormat(format)
            .ValueDepend(OPTIONAL);
        this->Input("offset")
            .ParamType(REQUIRED)
            .DataType(nDtype)
            .Format(format)
            .UnknownShapeFormat(format)
            .ValueDepend(OPTIONAL);

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(outDtype)
            .Format(format)
            .UnknownShapeFormat(format);
        
        this->Attr("layout").AttrType(OPTIONAL).Int(0);
        this->Attr("dtype").AttrType(OPTIONAL).Int(ge::DT_INT64);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "stateless_randperm_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(StatelessRandperm);
};
