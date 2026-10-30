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
 * \file sort_with_index_def.cpp
 * \brief SortWithIndex op_host
 */
#include <cstdint>
#include "register/op_def_registry.h"
namespace ops
{
static const std::vector<ge::DataType> DataTypeValue = {ge::DT_INT32,  ge::DT_INT16, ge::DT_INT8,  ge::DT_UINT32,
                                                        ge::DT_UINT16, ge::DT_UINT8, ge::DT_BF16,  ge::DT_FLOAT16,
                                                        ge::DT_FLOAT,  ge::DT_INT64, ge::DT_UINT64,
                                                        ge::DT_INT32,  ge::DT_INT16, ge::DT_INT8,  ge::DT_UINT32,
                                                        ge::DT_UINT16, ge::DT_UINT8, ge::DT_BF16,  ge::DT_FLOAT16,
                                                        ge::DT_FLOAT,  ge::DT_INT64, ge::DT_UINT64};
                                                        
static const std::vector<ge::Format> format = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

static const std::vector<ge::DataType> DataTypeIndex = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

class SortWithIndex : public OpDef
{
public:
    explicit SortWithIndex(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(DataTypeValue)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType(DataTypeIndex)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(DataTypeValue)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Output("sorted_index")
            .ParamType(REQUIRED)
            .DataType(DataTypeIndex)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Attr("axis").AttrType(OPTIONAL).Int(-1);
        this->Attr("descending").AttrType(OPTIONAL).Bool(false);
        this->Attr("stable").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "sort_with_index_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(SortWithIndex);
}  // namespace ops