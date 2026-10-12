/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file acos_grad_def.cpp
 * \brief AcosGrad 算子定义，声明输入输出和算子配置
 */

#include "register/op_def_registry.h"

namespace ops {
class AcosGrad : public OpDef {
public:
    explicit AcosGrad(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> AcosGradXDataType = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
        const std::vector<ge::Format> AcosGradXFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("y_grad")
            .ParamType(REQUIRED)
            .DataType(AcosGradXDataType)
            .Format(AcosGradXFormat)
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(AcosGradXDataType)
            .Format(AcosGradXFormat)
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("x_grad")
            .ParamType(REQUIRED)
            .DataType(AcosGradXDataType)
            .Format(AcosGradXFormat)
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(true)
            .NeedCheckSupportFlag(false)
            .DynamicRankSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "acos_grad");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(AcosGrad);
} // namespace ops
