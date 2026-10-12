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
 * \file log_add_exp_def.cpp
 * \brief log_add_exp op definition
 */
#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> dataType = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};

static const std::vector<ge::Format> format = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class LogAddExp : public OpDef {
public:
    explicit LogAddExp(const char* name) : OpDef(name)
    {
        this->Input("x1").ParamType(REQUIRED).DataType(dataType).Format(format).UnknownShapeFormat(format);
        this->Input("x2").ParamType(REQUIRED).DataType(dataType).Format(format).UnknownShapeFormat(format);
        this->Output("y").ParamType(REQUIRED).DataType(dataType).Format(format).UnknownShapeFormat(format);
        this->Attr("base").AttrType(OPTIONAL).Float(-1.0);
        this->Attr("scale").AttrType(OPTIONAL).Float(1.0);
        this->Attr("shift").AttrType(OPTIONAL).Float(0.0);
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "log_add_exp_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(LogAddExp);
} // namespace ops
