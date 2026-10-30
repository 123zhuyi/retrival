/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> VALUE_DATA_TYPE_LIST{
    ge::DT_INT8,   ge::DT_UINT8,  ge::DT_INT16,  ge::DT_UINT16,    ge::DT_INT32,
    ge::DT_UINT32, ge::DT_INT64,  ge::DT_UINT64, ge::DT_BF16,      ge::DT_FLOAT16,
    ge::DT_FLOAT,  ge::DT_DOUBLE, ge::DT_BOOL,   ge::DT_COMPLEX32, ge::DT_COMPLEX64,
};
static const auto DATA_FORMAT_LIST = std::vector<ge::Format>(VALUE_DATA_TYPE_LIST.size(), ge::FORMAT_ND);

class Im2col : public OpDef {
public:
    explicit Im2col(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(VALUE_DATA_TYPE_LIST).Format(DATA_FORMAT_LIST);
        this->Output("y").ParamType(REQUIRED).DataType(VALUE_DATA_TYPE_LIST).Format(DATA_FORMAT_LIST);
        this->Attr("ksizes").AttrType(REQUIRED).ListInt();
        this->Attr("strides").AttrType(OPTIONAL).ListInt({1});
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1});
        this->Attr("padding_mode").AttrType(OPTIONAL).String("CALCULATED");
        this->Attr("pads").AttrType(OPTIONAL).ListInt({0});

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "im2col_apt");

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(Im2col);
} // namespace ops
