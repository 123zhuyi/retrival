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
 * \file axpy_v2.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class AxpyV2 : public OpDef {
public:
    explicit AxpyV2(const char* name) : OpDef(name)
    {
        // 输入参数说明
        this->Input("x1")                                                            // 输入x1定义
            .ParamType(REQUIRED)                                                     // 必选输入
            .DataTypeList({ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_BF16}) // 支持数据类型
            .FormatList({ge::FORMAT_ND})                                             // 支持format格式
            .AutoContiguous();                                                       // 内存自动连续化
        this->Input("x2")                                                            // 输入x1定义
            .ParamType(REQUIRED)                                                     // 必选输入
            .DataTypeList({ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_BF16}) // 支持数据类型
            .FormatList({ge::FORMAT_ND})                                             // 支持format格式
            .AutoContiguous();                                                       // 内存自动连续化
        this->Input("alpha")                                                         // 输入x1定义
            .ParamType(REQUIRED)                                                     // 必选输入
            .DataTypeList({ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_BF16}) // 支持数据类型
            .FormatList({ge::FORMAT_ND})                                             // 支持format格式
            .AutoContiguous()                                                        // 内存自动连续化
            .Scalar();
        /* ...此处补充其他输入输出参数说明 */

        // 输出参数说明
        this->Output("y") // 输出y定义
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_BF16}) // 支持数据类型
            .FormatList({ge::FORMAT_ND})                                             // 支持format格式
            .AutoContiguous();

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(AxpyV2); // 添加算子信息库
} // namespace ops