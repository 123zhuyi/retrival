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
 * \file masked_scale_def.cpp
 * \brief masked_scale op host
 */
#include "register/op_def_registry.h"

namespace ops {

class MaskedScale : public OpDef {
public:
  const std::vector<ge::DataType> xDataType = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16,
    ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
    ge::DT_FLOAT};
  const std::vector<ge::Format> xFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND};
  const std::vector<ge::DataType> maskDataType = {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_UINT8, ge::DT_UINT8,
    ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

  explicit MaskedScale(const char* name) : OpDef(name) {
  this->Input("x")
    .ParamType(REQUIRED)
    .DataType(xDataType)
    .Format(xFormat)
    .UnknownShapeFormat(xFormat)
    .AutoContiguous();
  this->Input("mask")
    .ParamType(REQUIRED)
    .DataType(maskDataType)
    .Format(xFormat)
    .UnknownShapeFormat(xFormat)
    .AutoContiguous();
  this->Output("y")
    .ParamType(REQUIRED)
    .Format(xFormat)
    .UnknownShapeFormat(xFormat)
    .DataType(xDataType)
    .AutoContiguous();
  this->Attr("value")
    .AttrType(REQUIRED)
    .Float();

  OpAICoreConfig aicore_config;
  aicore_config.DynamicCompileStaticFlag(true)
    .DynamicFormatFlag(false)
    .DynamicRankSupportFlag(true)
    .DynamicShapeSupportFlag(true)
    .NeedCheckSupportFlag(false)
    .ExtendCfgInfo("opFile.value", "masked_scale_apt");

  this->AICore().AddConfig("ascend950", aicore_config);
  }
};

OP_ADD(MaskedScale);
}  // namespace ops