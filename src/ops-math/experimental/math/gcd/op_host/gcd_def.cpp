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
 * \file gcd_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

#include <vector>

namespace ops {
namespace {
const std::vector<ge::DataType> GCD_X1_DTYPES = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_UINT8,  ge::DT_INT8,
                                                 ge::DT_INT16, ge::DT_INT32,   ge::DT_INT64, ge::DT_UINT8,  ge::DT_BF16,
                                                 ge::DT_INT8,  ge::DT_FLOAT,   ge::DT_INT16, ge::DT_FLOAT16};
const std::vector<ge::DataType> GCD_X2_DTYPES = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_UINT8, ge::DT_INT8, ge::DT_INT16,   ge::DT_INT32,
    ge::DT_INT64, ge::DT_BF16,    ge::DT_UINT8, ge::DT_FLOAT, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT16};
const std::vector<ge::DataType> GCD_Y_DTYPES = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_UINT8, ge::DT_INT8,
                                                ge::DT_INT16, ge::DT_INT32,   ge::DT_INT64, ge::DT_UINT8, ge::DT_UINT8,
                                                ge::DT_INT8,  ge::DT_INT8,    ge::DT_INT16, ge::DT_INT16};
const std::vector<ge::Format> GCD_FORMATS = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                             ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
} // namespace

class Gcd : public OpDef {
public:
    explicit Gcd(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType(GCD_X1_DTYPES)
            .Format(GCD_FORMATS)
            .UnknownShapeFormat(GCD_FORMATS);
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType(GCD_X2_DTYPES)
            .Format(GCD_FORMATS)
            .UnknownShapeFormat(GCD_FORMATS);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(GCD_Y_DTYPES)
            .Format(GCD_FORMATS)
            .UnknownShapeFormat(GCD_FORMATS);
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "gcd");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};

OP_ADD(Gcd);
} // namespace ops
