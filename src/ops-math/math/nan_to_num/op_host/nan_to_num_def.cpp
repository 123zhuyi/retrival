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
 * \file nan_to_num_def.cpp
 * \brief NanToNum def
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> nanToNumDataType = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};
static const std::vector<ge::Format> nanToNumFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
class NanToNum : public OpDef {
public:
    explicit NanToNum(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(nanToNumDataType)
            .Format(nanToNumFormat)
            .UnknownShapeFormat(nanToNumFormat);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(nanToNumDataType)
            .Format(nanToNumFormat)
            .UnknownShapeFormat(nanToNumFormat);
        this->Attr("nan").AttrType(REQUIRED).Float();
        this->Attr("posinf").AttrType(REQUIRED).Float();
        this->Attr("neginf").AttrType(REQUIRED).Float();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "nan_to_num_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(NanToNum);
} // namespace ops
