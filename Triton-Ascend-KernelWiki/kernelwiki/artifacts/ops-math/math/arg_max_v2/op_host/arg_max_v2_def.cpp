/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file arg_max_v2_def.cpp
 * \brief arg_max_v2_def op host
 */
#include "register/op_def_registry.h"

namespace ops {

class ArgMaxV2 : public OpDef {
public:
    const std::vector<ge::DataType> argMaxXDataType = {
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64, ge::DT_BF16,
        ge::DT_BF16, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
        ge::DT_INT64, ge::DT_INT64, ge::DT_BF16,  ge::DT_BF16, ge::DT_INT32, ge::DT_INT32};
    const std::vector<ge::Format> argMaxFormat = {
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
    const std::vector<ge::DataType> dimensionDataType = {
        ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
        ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
        ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
        ge::DT_INT64, ge::DT_INT32, ge::DT_INT64};
    const std::vector<ge::DataType> argMaxYDataType = {
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
 explicit ArgMaxV2(const char* name) : OpDef(name) {
   this->Input("x")
       .ParamType(REQUIRED)
       .DataType(argMaxXDataType)
       .Format(argMaxFormat)
       .UnknownShapeFormat(argMaxFormat);
   this->Input("dimension")
       .ParamType(REQUIRED)
       .ValueDepend(OPTIONAL)
       .DataType(dimensionDataType)
       .Format(argMaxFormat)
       .UnknownShapeFormat(argMaxFormat);
   this->Output("y")
       .ParamType(REQUIRED)
       .DataType(argMaxYDataType)
       .Format(argMaxFormat)
       .UnknownShapeFormat(argMaxFormat);
    this->Attr("dtype").AttrType(OPTIONAL).Int(3);

   OpAICoreConfig aicore_config;
   aicore_config.DynamicCompileStaticFlag(true)
       .DynamicFormatFlag(false)
       .DynamicRankSupportFlag(true)
       .DynamicShapeSupportFlag(true)
       .NeedCheckSupportFlag(false)
       .ExtendCfgInfo("opFile.value", "arg_max_v2_apt");

   this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(ArgMaxV2);
}  // namespace ops