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
 * \file pad_v3_grad.cpp
 * \brief PadV3Grad
 */
#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::Format> format = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
static const std::vector<ge::DataType> valueDataType = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                                        ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};

static const std::vector<ge::DataType> padDataType = {ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                      ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};

class PadV3Grad : public OpDef {
public:
    explicit PadV3Grad(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(valueDataType).Format(format);
        this->Input("paddings").ParamType(REQUIRED).ValueDepend(OPTIONAL).DataType(padDataType).Format(format);
        this->Output("y").ParamType(REQUIRED).DataType(valueDataType).Format(format);
        this->Attr("mode").AttrType(OPTIONAL).String("reflect");
        this->Attr("paddings_contiguous").AttrType(OPTIONAL).Bool(true);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "pad_v3_grad_apt");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(PadV3Grad);
} // namespace ops
