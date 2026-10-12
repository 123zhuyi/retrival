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
 * \file add_v2_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
namespace {
#define ADD_V2_FORMAT_LIST                                                      \
    {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, \
     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}
} // namespace

class AddV2 : public OpDef {
public:
    explicit AddV2(const char* name) : OpDef(name)
    {
        // 9 组同 dtype 组合。canonical AddV2 Verifier（canndev
        // elewise_calculation_ops.cc 的 CheckTwoInputDtypeSame）要求 x1/x2 同
        // dtype，异类型组合在 GE 图通路上不可达，故不注册。
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8,
                       ge::DT_INT64, ge::DT_COMPLEX64})
            .Format(ADD_V2_FORMAT_LIST)
            .UnknownShapeFormat(ADD_V2_FORMAT_LIST)
            .AutoContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8,
                       ge::DT_INT64, ge::DT_COMPLEX64})
            .Format(ADD_V2_FORMAT_LIST)
            .UnknownShapeFormat(ADD_V2_FORMAT_LIST)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8,
                       ge::DT_INT64, ge::DT_COMPLEX64})
            .Format(ADD_V2_FORMAT_LIST)
            .UnknownShapeFormat(ADD_V2_FORMAT_LIST)
            .AutoContiguous();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true);
        aicoreConfig.DynamicFormatFlag(false);
        aicoreConfig.DynamicRankSupportFlag(true);
        aicoreConfig.DynamicShapeSupportFlag(true);
        aicoreConfig.NeedCheckSupportFlag(false);
        aicoreConfig.PrecisionReduceFlag(true);
        aicoreConfig.ExtendCfgInfo("opFile.value", "add_v2");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(AddV2);
} // namespace ops
