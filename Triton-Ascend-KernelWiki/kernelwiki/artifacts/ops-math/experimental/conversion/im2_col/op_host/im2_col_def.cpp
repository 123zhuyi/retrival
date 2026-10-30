/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Qiu Zhuang <@qiu-zhuang>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file im2_col_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
// 常量定义，用于替换魔法数字
namespace {
constexpr int32_t DEFAULT_KERNEL_H = 2;
constexpr int32_t DEFAULT_KERNEL_W = 2;
constexpr int32_t DEFAULT_STRIDE_H = 1;
constexpr int32_t DEFAULT_STRIDE_W = 1;
constexpr int32_t DEFAULT_PAD_H = 0;
constexpr int32_t DEFAULT_PAD_W = 0;
constexpr int32_t DEFAULT_DILATION_H = 1;
constexpr int32_t DEFAULT_DILATION_W = 1;
} // namespace
class Im2Col : public OpDef {
public:
    explicit Im2Col(const char* name) : OpDef(name)
    {
        this->Input("x")                                        // 输入x1定义
            .ParamType(REQUIRED)                                // 必选输入
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})           // 支持数据类型
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})             // 支持format格式
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}) // 未确定大小shape对应format格式
            .AutoContiguous();                                  // 内存自动连续化
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 定义卷积参数属性
        this->Attr("kernel_h").AttrType(OPTIONAL).Int(DEFAULT_KERNEL_H);
        this->Attr("kernel_w").AttrType(OPTIONAL).Int(DEFAULT_KERNEL_W);
        this->Attr("stride_h").AttrType(OPTIONAL).Int(DEFAULT_STRIDE_H);
        this->Attr("stride_w").AttrType(OPTIONAL).Int(DEFAULT_STRIDE_W);
        this->Attr("pad_h").AttrType(OPTIONAL).Int(DEFAULT_PAD_H);
        this->Attr("pad_w").AttrType(OPTIONAL).Int(DEFAULT_PAD_W);
        this->Attr("dilation_h").AttrType(OPTIONAL).Int(DEFAULT_DILATION_H);
        this->Attr("dilation_w").AttrType(OPTIONAL).Int(DEFAULT_DILATION_W);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "im2_col");        // 这里制定的值会对应到kernel入口文件名.cpp
        this->AICore().AddConfig("ascend910b", aicoreConfig); // 其他的soc版本补充部分配置项
    }
};
OP_ADD(Im2Col); // 添加算子信息库
} // namespace ops