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
 * \file acos_grad_v2_def.cpp
 * \brief AcosGradV2 算子定义（A2 / Ascend910B 适配）
 *
 * 对齐 math/acos_grad README:
 *   INPUT(y)  : 前向 Acos 的输入张量，值域期望 [-1, 1]
 *   INPUT(dy) : 上游梯度，shape/dtype 与 y 一致
 *   OUTPUT(z) : 对原始输入的梯度，shape/dtype 与 y 一致
 *
 * 公式: z = -dy / sqrt(1 - y^2)
 */
#include "register/op_def_registry.h"

namespace ops {
class AcosGradV2 : public OpDef {
public:
    explicit AcosGradV2(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> AcosGradV2DataType = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
        const std::vector<ge::Format> AcosGradV2Format = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("y")
            .ParamType(REQUIRED)
            .DataType(AcosGradV2DataType)
            .Format(AcosGradV2Format)
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("dy")
            .ParamType(REQUIRED)
            .DataType(AcosGradV2DataType)
            .Format(AcosGradV2Format)
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType(AcosGradV2DataType)
            .Format(AcosGradV2Format)
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(true)
            .NeedCheckSupportFlag(false)
            .DynamicRankSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "acos_grad_v2");
        // A2 适配：仅注册 ascend910b (Atlas A2 训练/推理系列产品, DAV_2201)
        this->AICore().AddConfig("ascend910b", aiCoreConfig);
    }
};
OP_ADD(AcosGradV2);
} // namespace ops
