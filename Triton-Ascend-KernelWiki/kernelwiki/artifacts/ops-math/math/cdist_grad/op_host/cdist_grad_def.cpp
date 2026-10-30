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
 * \file cdist_grad_def.cpp
 * \brief AICore info for CdistGrad op
 */

#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> dataType = {ge::DT_FLOAT16, ge::DT_FLOAT};
static const std::vector<ge::Format> format = {ge::FORMAT_ND, ge::FORMAT_ND};
static constexpr float DEFAULT_P_NORM = 2.0f;

class CdistGrad : public OpDef {
public:
    explicit CdistGrad(const char* name) : OpDef(name)
    {
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Input("cdist")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Attr("p").AttrType(OPTIONAL).Float(DEFAULT_P_NORM);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
                    .DynamicRankSupportFlag(true)
                    .DynamicShapeSupportFlag(true)
                    .ExtendCfgInfo("opFile.value", "cdist_grad_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(CdistGrad);
} // namespace ops
