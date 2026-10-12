/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lin_space.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class LinSpace : public OpDef {
public:
    explicit LinSpace(const char* name) : OpDef(name)
    {
        this->Input("start")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(
                {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_UINT8,
                 ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("stop")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(
                {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_UINT8,
                 ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("num")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(
                {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64,
                 ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_UINT8,
                 ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend310p");
        this->AICore().AddConfig("kirinx90");
        this->AICore().AddConfig("kirin9030");

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false);
        config.Input("start")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(
                {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                 ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND});
        config.Input("stop")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(
                {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                 ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND});
        config.Input("num")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(
                {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                 ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND});
        config.Output("output")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                 ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);

        OpAICoreConfig regBaseConfig;
        regBaseConfig.Input("start")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(
                {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                 ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND});
        regBaseConfig.Input("stop")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(
                {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                 ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND});
        regBaseConfig.Input("num")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(
                {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                 ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND});
        regBaseConfig.Output("output")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                 ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND});
        regBaseConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "lin_space_apt");
        this->AICore().AddConfig("ascend950", regBaseConfig);
    }
};

OP_ADD(LinSpace);
} // namespace ops