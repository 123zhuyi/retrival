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
 * \file log.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class Log : public OpDef {
public:
    explicit Log(const char* name) : OpDef(name)
    {
        // 输入参数说明
        this->Input("x")                                                        // 输入x1定义
            .ParamType(REQUIRED)                                                // 必选输入
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})              // 支持数据类型
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})              // 支持format格式
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}); // 未确定大小shape对应format格式;
        // 输出参数说明
        this->Output("y") // 输出y定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("base").AttrType(OPTIONAL).Float(-1.0);
        this->Attr("scale").AttrType(OPTIONAL).Float(1.0);
        this->Attr("shift").AttrType(OPTIONAL).Float(0.0);

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Log); // 添加算子信息库
} // namespace ops
