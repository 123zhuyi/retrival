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
 * \file not_equal_def.cpp
 * \brief aicore info for not equal op
 */
#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> inputDataType = {ge::DT_BF16,  ge::DT_FLOAT16, ge::DT_FLOAT,
                                                        ge::DT_INT8,  ge::DT_UINT8,   ge::DT_INT32,
                                                        ge::DT_INT64, ge::DT_UINT64,  ge::DT_BOOL};

static const std::vector<ge::DataType> outputDataType = {ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL,
                                                         ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL,
                                                         ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL};

static const std::vector<ge::Format> format = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class NotEqual : public OpDef {
public:
    explicit NotEqual(const char* name) : OpDef(name)
    {
        this->Input("x1").ParamType(REQUIRED).DataType(inputDataType).Format(format).UnknownShapeFormat(format);

        this->Input("x2").ParamType(REQUIRED).DataType(inputDataType).Format(format).UnknownShapeFormat(format);

        this->Output("y").ParamType(REQUIRED).DataType(outputDataType).Format(format).UnknownShapeFormat(format);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "not_equal_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(NotEqual);
} // namespace ops