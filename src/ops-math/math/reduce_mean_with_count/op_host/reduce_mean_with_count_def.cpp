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
 * \file reduce_mean_with_count_def.cpp
 * \brief AICore info for ReduceMeanWithCount op
 */

#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> dataType = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};

static const std::vector<ge::Format> format = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class ReduceMeanWithCount : public OpDef {
public:
    explicit ReduceMeanWithCount(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(dataType).UnknownShapeFormat(format);

        this->Input("count").ParamType(REQUIRED).DataType(dataType).UnknownShapeFormat(format);

        this->Input("count_sum").ParamType(REQUIRED).DataType(dataType).UnknownShapeFormat(format);

        this->Output("y").ParamType(REQUIRED).DataType(dataType).UnknownShapeFormat(format);

        this->Attr("axes").AttrType(REQUIRED).ListInt({});
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "reduce_mean_with_count_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(ReduceMeanWithCount);
}  // namespace ops
