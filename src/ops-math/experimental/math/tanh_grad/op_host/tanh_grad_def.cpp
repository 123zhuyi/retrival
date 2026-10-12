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
 * \file tanh_grad_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class TanhGrad : public OpDef {
public:
    explicit TanhGrad(const char* name) : OpDef(name)
    {
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                 ge::DT_FLOAT16, ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
            .AutoContiguous();

        this->Input("dy")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                 ge::DT_FLOAT16, ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
            .AutoContiguous();

        this->Output("dx")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                 ge::DT_FLOAT16, ge::DT_BF16})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
            .AutoContiguous();

        // AICore配置
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "tanh_grad"); // 对应kernel入口文件名

        // 支持的芯片版本
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};
OP_ADD(TanhGrad);
} // namespace ops