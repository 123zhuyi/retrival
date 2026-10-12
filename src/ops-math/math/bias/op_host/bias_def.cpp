/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> kBiasDataType = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                                                        ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
static const std::vector<ge::Format> kBiasFormat = {ge::FORMAT_ND,      ge::FORMAT_ND,      ge::FORMAT_ND,
                                                    ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0};
static const std::vector<ge::Format> kBiasUnknownShapeFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0};

class Bias : public OpDef {
public:
    explicit Bias(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(kBiasDataType)
            .Format(kBiasFormat)
            .UnknownShapeFormat(kBiasUnknownShapeFormat)
            .AutoContiguous();
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType(kBiasDataType)
            .Format(kBiasFormat)
            .UnknownShapeFormat(kBiasUnknownShapeFormat)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(kBiasDataType)
            .Format(kBiasFormat)
            .UnknownShapeFormat(kBiasUnknownShapeFormat)
            .AutoContiguous();
        this->Attr("axis").AttrType(OPTIONAL).Int(1);
        this->Attr("num_axes").AttrType(OPTIONAL).Int(1);
        this->Attr("bias_from_blob").AttrType(OPTIONAL).Bool(true);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "bias");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(Bias);
} // namespace ops
