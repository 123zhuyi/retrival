/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {
class SliceWithAxes : public OpDef {
public:
    const std::vector<ge::DataType> baseDataType = {
        ge::DT_INT8,  ge::DT_UINT8,  ge::DT_INT16, ge::DT_UINT16,  ge::DT_INT32, ge::DT_UINT32,
        ge::DT_INT64, ge::DT_UINT64, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_BOOL,
        ge::DT_INT8,  ge::DT_UINT8,  ge::DT_INT16, ge::DT_UINT16,  ge::DT_INT32, ge::DT_UINT32,
        ge::DT_INT64, ge::DT_UINT64, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_BOOL};

    const std::vector<ge::Format> baseFormat = {
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

    const std::vector<ge::DataType> indexDataType = {
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

    explicit SliceWithAxes(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(baseDataType).Format(baseFormat).UnknownShapeFormat(baseFormat);
        this->Input("offsets")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(indexDataType)
            .Format(baseFormat)
            .UnknownShapeFormat(baseFormat);
        this->Input("size")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(indexDataType)
            .Format(baseFormat)
            .UnknownShapeFormat(baseFormat);
        this->Output("y").ParamType(REQUIRED).DataType(baseDataType).Format(baseFormat).UnknownShapeFormat(baseFormat);

        this->Attr("axes").AttrType(REQUIRED).ListInt();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "slice_with_axes_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(SliceWithAxes);
} // namespace ops
