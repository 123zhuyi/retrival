/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liu Jun <@kbryantttt>
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
 * \file strided_slice.cpp
 * \brief
 */
#include "register/op_def_registry.h"
namespace ops {
class StridedSlice : public OpDef {
public:
    explicit StridedSlice(const char* name) : OpDef(name)
    {
        // 输入参数说明
        this->Input("x") // 输入x1定义
            .DataType({ge::DT_FLOAT, ge::DT_INT32,  ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64, ge::DT_FLOAT16,
                       ge::DT_INT16, ge::DT_UINT16, ge::DT_BF16,   ge::DT_INT8,  ge::DT_UINT8,  ge::DT_BOOL,
                       ge::DT_FLOAT, ge::DT_INT32,  ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64, ge::DT_FLOAT16,
                       ge::DT_INT16, ge::DT_UINT16, ge::DT_BF16,   ge::DT_INT8,  ge::DT_UINT8,  ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous(); // 内存自动连续化
        /* ...此处补充其他输入输出参数说明 */
        this->Attr("start1").AttrType(OPTIONAL).Int();
        this->Attr("start2").AttrType(OPTIONAL).Int();
        this->Attr("end1").AttrType(OPTIONAL).Int();
        this->Attr("end2").AttrType(OPTIONAL).Int();
        this->Attr("stride1").AttrType(OPTIONAL).Int();
        this->Attr("stride2").AttrType(OPTIONAL).Int();
        // 输出参数说明
        this->Output("z") // 输出y定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT32,  ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64, ge::DT_FLOAT16,
                       ge::DT_INT16, ge::DT_UINT16, ge::DT_BF16,   ge::DT_INT8,  ge::DT_UINT8,  ge::DT_BOOL,
                       ge::DT_FLOAT, ge::DT_INT32,  ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64, ge::DT_FLOAT16,
                       ge::DT_INT16, ge::DT_UINT16, ge::DT_BF16,   ge::DT_INT8,  ge::DT_UINT8,  ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "strided_slice");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};
OP_ADD(StridedSlice); // 添加算子信息库
} // namespace ops