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
 * \file mod_def.cpp
 * \brief
 */
#include <vector>
#include "register/op_def_registry.h"

namespace ops {
// The kernel only exposes same-dtype prototypes. Cross-dtype inputs are promoted and cast by aclnn before Mod.
// Positional vectors remain one-to-one: lane k = (x1[k], x2[k], y[k]).
static const std::vector<ge::DataType> kModX1 = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16};
static const std::vector<ge::DataType> kModX2 = kModX1;
static const std::vector<ge::DataType> kModY = kModX1;
static const std::vector<ge::Format> kModFmt(kModX1.size(), ge::FORMAT_ND);

class Mod : public OpDef {
public:
    explicit Mod(const char* name) : OpDef(name)
    {
        this->Input("x1").ParamType(REQUIRED).DataType(kModX1).Format(kModFmt).UnknownShapeFormat(kModFmt);
        this->Input("x2").ParamType(REQUIRED).DataType(kModX2).Format(kModFmt).UnknownShapeFormat(kModFmt);

        this->Output("y").ParamType(REQUIRED).DataType(kModY).Format(kModFmt).UnknownShapeFormat(kModFmt);

        // 仅注册 Atlas A2 / A3 (均 arch22 / DAV_2201)
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(Mod); // 添加算子信息库
} // namespace ops
