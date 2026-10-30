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
 * \file reduce_log_sum_exp.cpp
 * \brief aicore info for reduceLogSumExp op
 */
#include "register/op_def_registry.h"
 
namespace ops {
static const std::vector<ge::DataType> dataType = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                                                   ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

static const std::vector<ge::Format> format = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

static const std::vector<ge::DataType> axesDataType = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

class ReduceLogSumExp : public OpDef {
   public:
    explicit ReduceLogSumExp(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .UnknownShapeFormat(format);

        this->Input("axes")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(axesDataType)
            .UnknownShapeFormat(format);

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .UnknownShapeFormat(format);

        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);
 
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "reduce_log_sum_exp_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
 
OP_ADD(ReduceLogSumExp);
}  // namespace ops