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
 * \file radix_top_k_def.cpp
 * \brief radix_top_k def
 */
#include <cstdint>
#include "register/op_def_registry.h"
namespace ops {
static const std::vector<ge::DataType> valuesDataType = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                                                         ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16};

static const std::vector<ge::Format> topKFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

static const std::vector<ge::DataType> indicesDataType = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64,
                                                          ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64};

static const std::vector<ge::DataType> kDataType = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};  

class RadixTopK: public OpDef {
public:
    explicit RadixTopK(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(valuesDataType)
            .Format({topKFormat});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType(kDataType)
            .Format({topKFormat})
            .ValueDepend(OPTIONAL);
        this->Output("values")
            .ParamType(REQUIRED)
            .DataType(valuesDataType)
            .Format({topKFormat});
        this->Output("indices" )
            .ParamType(REQUIRED)
            .DataType(indicesDataType)
            .Format(topKFormat);
        this->Attr("sorted").AttrType(OPTIONAL).Bool(true);
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);
        this->Attr("largest").AttrType(OPTIONAL).Bool(true);
        this->Attr("indices_dtype").AttrType(OPTIONAL).Int(ge::DT_INT32);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "radix_top_k")
            .ExtendCfgInfo("opInterface.value",  "radix_top_k");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};
OP_ADD(RadixTopK);
} // namespace ops
