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
 * \file addr_v2_def.cpp
 * \brief op config of addr_v2 (arch22 / Ascend910B)
 *
 * Design ref: DESIGN.md §10.2 - 新增 ascend910b AICore 配置
 */

#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> xDType = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                                                 ge::DT_INT8,  ge::DT_UINT8,   ge::DT_BOOL};
static const std::vector<ge::Format> xFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class AddrV2 : public OpDef {
public:
    explicit AddrV2(const char* name) : OpDef(name)
    {
        this->Input("x1").ParamType(REQUIRED).DataType(xDType).Format(xFormat).UnknownShapeFormat(xFormat);
        this->Input("x2").ParamType(REQUIRED).DataType(xDType).Format(xFormat).UnknownShapeFormat(xFormat);
        this->Input("x3").ParamType(REQUIRED).DataType(xDType).Format(xFormat).UnknownShapeFormat(xFormat);
        this->Input("beta")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(xDType)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat);
        this->Input("alpha")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(xDType)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat);
        this->Output("y").ParamType(REQUIRED).DataType(xDType).Format(xFormat).UnknownShapeFormat(xFormat);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};

OP_ADD(AddrV2);
} // namespace ops
