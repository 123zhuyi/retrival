/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file atan2.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> supportedDtypes = {
    ge::DT_BF16,
    ge::DT_FLOAT16,
    ge::DT_FLOAT,
};
static const std::vector<ge::Format> supportedFormats = {
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
};
class Atan2 : public OpDef {
public:
  explicit Atan2(const char *name) : OpDef(name) {
    this->Input("x1")
        .ParamType(REQUIRED)
        .DataType(supportedDtypes)
        .Format(supportedFormats)
        .UnknownShapeFormat(supportedFormats);
    this->Input("x2")
        .ParamType(REQUIRED)
        .DataType(supportedDtypes)
        .Format(supportedFormats)
        .UnknownShapeFormat(supportedFormats);
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType(supportedDtypes)
        .Format(supportedFormats)
        .UnknownShapeFormat(supportedFormats);
    OpAICoreConfig aicoreConfig;
    aicoreConfig.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("opFile.value", "atan2_apt");
    this->AICore().AddConfig("ascend950", aicoreConfig);
  }
};

OP_ADD(Atan2);
} // namespace ops