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

static const std::vector<ge::DataType> kMaskDataTypes = {ge::DT_BF16,   ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_UINT8,
                                                         ge::DT_INT8,   ge::DT_INT16,   ge::DT_INT32, ge::DT_INT64,
                                                         ge::DT_DOUBLE, ge::DT_BOOL};
static const std::vector<ge::Format> kMaskFormats(kMaskDataTypes.size(), ge::FORMAT_ND);
static const std::vector<ge::DataType> kMaskInputTypes(kMaskDataTypes.size(), ge::DT_UINT8);
static const std::vector<ge::Format> kMaskInputFormats(kMaskDataTypes.size(), ge::FORMAT_ND);

class Bernoulli : public OpDef {
public:
    explicit Bernoulli(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(kMaskDataTypes)
            .Format(kMaskFormats)
            .UnknownShapeFormat(kMaskFormats);
        this->Input("mask")
            .ParamType(REQUIRED)
            .DataType(kMaskInputTypes)
            .Format(kMaskInputFormats)
            .UnknownShapeFormat(kMaskInputFormats);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(kMaskDataTypes)
            .Format(kMaskFormats)
            .UnknownShapeFormat(kMaskFormats);
        this->Attr("mode").AttrType(REQUIRED).Int(0);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "bernoulli")
            .Input("x")
            .ParamType(REQUIRED)
            .DataType(kMaskDataTypes)
            .Format(kMaskFormats)
            .UnknownShapeFormat(kMaskFormats);
        config.Input("mask")
            .ParamType(REQUIRED)
            .DataType(kMaskInputTypes)
            .Format(kMaskInputFormats)
            .UnknownShapeFormat(kMaskInputFormats);
        config.Output("y")
            .ParamType(REQUIRED)
            .DataType(kMaskDataTypes)
            .Format(kMaskFormats)
            .UnknownShapeFormat(kMaskFormats);
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
    }
};

OP_ADD(Bernoulli);

} // namespace ops
