/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file strided_slice_assign_v2_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> inputDataType = {ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_BF16, ge::DT_INT32,
                                                        ge::DT_INT64,   ge::DT_DOUBLE, ge::DT_INT8};

static const std::vector<ge::DataType> idxDataType = {ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                      ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

static const std::vector<ge::Format> format = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

static const std::vector<ge::DataType> inputDataTypeKirin = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32,
                                                        ge::DT_INT64,   ge::DT_DOUBLE, ge::DT_INT8};

static const std::vector<ge::DataType> idxDataTypeKirin = {ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                                                      ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

static const std::vector<ge::Format> formatKirin = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class StridedSliceAssignV2 : public OpDef {
public:
    explicit StridedSliceAssignV2(const char* name) : OpDef(name)
    {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType(inputDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Input("input_value")
            .ParamType(REQUIRED)
            .DataType(inputDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Input("begin")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType(idxDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Input("end")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType(idxDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Input("strides")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType(idxDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Input("axes")
            .ParamType(OPTIONAL)
            .ValueDepend(REQUIRED)
            .DataType(idxDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Output("var")
            .ParamType(REQUIRED)
            .DataType(inputDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend950");

        OpAICoreConfig config_kirin = GetKirinCoreConfig();
        this->AICore().AddConfig("kirinx90", config_kirin);
        this->AICore().AddConfig("kirin9030", config_kirin);
    }

private:
    OpAICoreConfig GetKirinCoreConfig() const
    {
        OpAICoreConfig config_kirin;
        config_kirin.DynamicCompileStaticFlag(true).DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(true);
        config_kirin.Input("var")
            .ParamType(REQUIRED)
            .DataType(inputDataTypeKirin)
            .Format(formatKirin).UnknownShapeFormat(formatKirin)
            .AutoContiguous();
        config_kirin.Input("input_value")
            .ParamType(REQUIRED)
            .DataType(inputDataTypeKirin)
            .Format(formatKirin).UnknownShapeFormat(formatKirin)
            .AutoContiguous();
        config_kirin.Input("begin")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType(idxDataTypeKirin)
            .Format(formatKirin).UnknownShapeFormat(formatKirin)
            .AutoContiguous();
        config_kirin.Input("end")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType(idxDataTypeKirin)
            .Format(formatKirin).UnknownShapeFormat(formatKirin)
            .AutoContiguous();
        config_kirin.Input("strides")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType(idxDataTypeKirin)
            .Format(formatKirin).UnknownShapeFormat(formatKirin)
            .AutoContiguous();
        config_kirin.Input("axes")
            .ParamType(OPTIONAL)
            .ValueDepend(REQUIRED)
            .DataType(idxDataTypeKirin)
            .Format(formatKirin).UnknownShapeFormat(formatKirin)
            .AutoContiguous();
        config_kirin.Output("var")
            .ParamType(REQUIRED)
            .DataType(inputDataTypeKirin)
            .Format(formatKirin).UnknownShapeFormat(formatKirin)
            .AutoContiguous();
        return config_kirin;
    }
};

OP_ADD(StridedSliceAssignV2);
} // namespace ops