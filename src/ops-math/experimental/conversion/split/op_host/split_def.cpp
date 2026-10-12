/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file split.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class Split : public OpDef {
public:
    explicit Split(const char* name) : OpDef(name)
    {
        // 输入参数说明
        this->Input("x")         // 输入x1定义
            .ParamType(REQUIRED) // 必选输入
            .DataType({ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous(); // 内存自动连续化
        this->Output("y")      // 输出y定义
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("indices_or_sections").AttrType(REQUIRED).ListInt({2});
        this->Attr("axis").AttrType(REQUIRED).Int(0);

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Split); // 添加算子信息库
} // namespace ops