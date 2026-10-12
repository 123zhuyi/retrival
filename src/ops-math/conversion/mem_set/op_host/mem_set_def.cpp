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
 * \file mem_set_def.cpp
 * \brief mem_set op host
 */
#include "register/op_def_registry.h"

namespace ops {

class MemSet : public OpDef {
public:
    explicit MemSet(const char* name) : OpDef(name)
    {
        this->Attr("sizes").AttrType(REQUIRED).ListInt();
        this->Attr("dtypes").AttrType(OPTIONAL).ListInt();
        this->Attr("values_int").AttrType(OPTIONAL).ListInt();
        this->Attr("values_float").AttrType(OPTIONAL).ListFloat();

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "mem_set_apt");

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(MemSet);
} // namespace ops