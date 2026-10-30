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
 * \file confusion_transpose_d.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> confusionTransposeDDataType = {
    ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64,ge::DT_UINT8, 
    ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64,ge::DT_FLOAT16, ge::DT_FLOAT, 
    ge::DT_BF16};

static const std::vector<ge::Format> confusionTransposeDFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND};

class ConfusionTransposeD : public OpDef {
public:
    explicit ConfusionTransposeD(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(confusionTransposeDDataType)
            .Format(confusionTransposeDFormat)
            .UnknownShapeFormat(confusionTransposeDFormat);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(confusionTransposeDDataType)
            .Format(confusionTransposeDFormat)
            .UnknownShapeFormat(confusionTransposeDFormat);
        this->Attr("perm").AttrType(REQUIRED).ListInt();
        this->Attr("shape").AttrType(REQUIRED).ListInt();
        this->Attr("transpose_first").AttrType(REQUIRED).Bool();

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "confusion_transpose_d_apt");

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(ConfusionTransposeD);
}  // namespace ops