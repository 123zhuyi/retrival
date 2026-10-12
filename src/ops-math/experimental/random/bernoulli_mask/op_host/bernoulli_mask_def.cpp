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
namespace {

const std::vector<ge::DataType> kMaskDtypes(10, ge::DT_UINT8);
const std::vector<ge::DataType> kOutputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_UINT8,
                                                 ge::DT_INT8,    ge::DT_INT16, ge::DT_INT32,  ge::DT_INT64,
                                                 ge::DT_BOOL,    ge::DT_BF16};
const std::vector<ge::Format> kNdFormats(10, ge::FORMAT_ND);

} // namespace

class BernoulliMask : public OpDef {
public:
    explicit BernoulliMask(const char* name) : OpDef(name)
    {
        this->Input("mask").ParamType(REQUIRED).DataType(kMaskDtypes).Format(kNdFormats).UnknownShapeFormat(kNdFormats);
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType(kOutputDtypes)
            .Format(kNdFormats)
            .UnknownShapeFormat(kNdFormats);
        this->Attr("output_shape").AttrType(REQUIRED).ListInt();
        this->Attr("mask_aliases_out").AttrType(OPTIONAL).Int(0);
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};

OP_ADD(BernoulliMask);
} // namespace ops
