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
 * \file mem_set_v2_def.cpp
 * \brief mem_set_v2 op host
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> dataType = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,   ge::DT_BOOL,
                                                   ge::DT_INT8,  ge::DT_INT16,   ge::DT_INT32,  ge::DT_INT64,
                                                   ge::DT_UINT8, ge::DT_UINT16,  ge::DT_UINT32, ge::DT_UINT64};
static const std::vector<ge::Format> format = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class MemSetV2 : public OpDef {
public:
    explicit MemSetV2(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(DYNAMIC).DataType(dataType).Format(format).UnknownShapeFormat(format);

        this->Output("x").ParamType(DYNAMIC).DataType(dataType).Format(format).UnknownShapeFormat(format);

        this->Attr("values_int").AttrType(OPTIONAL).ListInt();
        this->Attr("values_float").AttrType(OPTIONAL).ListFloat();

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "mem_set_v2_apt");

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(MemSetV2);
} // namespace ops