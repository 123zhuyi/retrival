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
 * \file logical_and.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class LogicalAnd : public OpDef {
public:
    explicit LogicalAnd(const char* name) : OpDef(name)
    {
        // 输入参数说明
        this->Input("x1")            // 输入x1定义
            .ParamType(REQUIRED)     // 必选输入
            .DataType({ge::DT_BOOL}) // 支持数据类型
            .Format({ge::FORMAT_ND}) // 支持format格式
            .AutoContiguous();       // 内存自动连续化

        this->Input("x2")            // 输入x1定义
            .ParamType(REQUIRED)     // 必选输入
            .DataType({ge::DT_BOOL}) // 支持数据类型
            .Format({ge::FORMAT_ND}) // 支持format格式
            .AutoContiguous();       // 内存自动连续化

        /* ...此处补充其他输入输出参数说明 */

        // 输出参数说明
        this->Output("y") // 输出y定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(LogicalAnd); // 添加算子信息库
} // namespace ops