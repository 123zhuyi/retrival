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
 * \file expand_def.cpp
 * \brief op config of expand
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> xDType = {
    ge::DT_FLOAT,         ge::DT_FLOAT16,      ge::DT_INT32,     ge::DT_UINT8,
    ge::DT_INT8,          ge::DT_BOOL,         ge::DT_BF16,      ge::DT_INT64,
    ge::DT_FLOAT,         ge::DT_FLOAT16,      ge::DT_INT32,     ge::DT_UINT8,
    ge::DT_INT8,          ge::DT_BOOL,         ge::DT_BF16,      ge::DT_INT64};
static const std::vector<ge::Format> xFormat = {
    ge::FORMAT_ND,        ge::FORMAT_ND,       ge::FORMAT_ND,    ge::FORMAT_ND,
    ge::FORMAT_ND,        ge::FORMAT_ND,       ge::FORMAT_ND,    ge::FORMAT_ND,
    ge::FORMAT_ND,        ge::FORMAT_ND,       ge::FORMAT_ND,    ge::FORMAT_ND,
    ge::FORMAT_ND,        ge::FORMAT_ND,       ge::FORMAT_ND,    ge::FORMAT_ND};
static const std::vector<ge::DataType> constDType = {
    ge::DT_INT32,         ge::DT_INT32,        ge::DT_INT32,     ge::DT_INT32,
    ge::DT_INT32,         ge::DT_INT32,        ge::DT_INT32,     ge::DT_INT32,
    ge::DT_INT64,         ge::DT_INT64,        ge::DT_INT64,     ge::DT_INT64, 
    ge::DT_INT64,         ge::DT_INT64,        ge::DT_INT64,     ge::DT_INT64};

class Expand : public OpDef {
public:
    explicit Expand(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(xDType).Format(xFormat).UnknownShapeFormat(xFormat);
        this->Input("shape")
            .ParamType(REQUIRED)
            .DataType(constDType)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
            .ValueDepend(OPTIONAL);
        this->Output("y").ParamType(REQUIRED).DataType(xDType).Format(xFormat).UnknownShapeFormat(xFormat);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "expand_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(Expand);
} // namespace ops