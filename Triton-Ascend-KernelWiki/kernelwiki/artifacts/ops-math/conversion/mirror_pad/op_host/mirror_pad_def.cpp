/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mirror_pad.cpp
 * \brief mirror_pad op host
 */
#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::Format> format = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

static const std::vector<ge::DataType> valueDataType = {
    ge::DT_INT64,         ge::DT_INT32,        ge::DT_UINT32,   ge::DT_FLOAT,       ge::DT_FLOAT16,
    ge::DT_DOUBLE,        ge::DT_BF16,         ge::DT_INT16,    ge::DT_UINT16,      ge::DT_INT8,
    ge::DT_UINT8,         ge::DT_BOOL,         ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0,
    ge::DT_FLOAT8_E4M3FN, ge::DT_INT64,        ge::DT_INT32,    ge::DT_UINT32,      ge::DT_FLOAT,
    ge::DT_FLOAT16,       ge::DT_DOUBLE,       ge::DT_BF16,     ge::DT_INT16,       ge::DT_UINT16,
    ge::DT_INT8,          ge::DT_UINT8,        ge::DT_BOOL,     ge::DT_HIFLOAT8,    ge::DT_FLOAT8_E5M2,
    ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E4M3FN};

static const std::vector<ge::DataType> padDataType = {
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};

class MirrorPad : public OpDef {
public:
    explicit MirrorPad(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(valueDataType).Format(format).UnknownShapeFormat(format);
        this->Input("paddings")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(padDataType)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Output("y").ParamType(REQUIRED).DataType(valueDataType).Format(format).UnknownShapeFormat(format);
        this->Attr("mode").AttrType(REQUIRED).String("REFLECT");
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "mirror_pad_apt");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(MirrorPad);
} // namespace ops
