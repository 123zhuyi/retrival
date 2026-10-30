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
 * \file top_k_v2_def.cpp
 * \brief top_k_v2 def
 */
#include <cstdint>
#include "register/op_def_registry.h"
namespace ops {
static const std::vector<ge::DataType> valuesDataType = {ge::DT_INT64,   ge::DT_INT32,  ge::DT_INT16, ge::DT_INT8,
                                                         ge::DT_UINT32,  ge::DT_UINT16, ge::DT_UINT8, ge::DT_BF16,
                                                         ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_UINT64,
                                                         ge::DT_INT64,   ge::DT_INT32,  ge::DT_INT16, ge::DT_INT8,
                                                         ge::DT_UINT32,  ge::DT_UINT16, ge::DT_UINT8, ge::DT_BF16,
                                                         ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_UINT64,
                                                         ge::DT_INT64,   ge::DT_INT32,  ge::DT_INT16, ge::DT_INT8,
                                                         ge::DT_UINT32,  ge::DT_UINT16, ge::DT_UINT8, ge::DT_BF16,
                                                         ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_UINT64,
                                                         ge::DT_INT64,   ge::DT_INT32,  ge::DT_INT16, ge::DT_INT8,
                                                         ge::DT_UINT32,  ge::DT_UINT16, ge::DT_UINT8, ge::DT_BF16,
                                                         ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_UINT64};

static const std::vector<ge::Format> topKFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

static const std::vector<ge::DataType> indicesDataType = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                          ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                          ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                          ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                          ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                          ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                          ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                          ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                          ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                          ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                          ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                          ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

static const std::vector<ge::DataType> kDataType = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};  

class TopKV2: public OpDef {
public:
    explicit TopKV2(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(valuesDataType)
            .Format({topKFormat});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType(kDataType)
            .Format({topKFormat});
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
            .ExtendCfgInfo("opFile.value", "top_k_v2_apt")
            .ExtendCfgInfo("opInterface.value",  "top_k_v2");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(TopKV2);
} // namespace ops
