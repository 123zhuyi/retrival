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
 * \file asinh_def.cpp
 * \brief Asinh 算子定义（OpDef + OpAICoreConfig）
 *
 * 与 DESIGN.md v1.1 §3.3 对齐（参数名与 op_graph/asinh_proto.h REG_OP(Asinh) 一致）：
 *   - Input("x") / Output("y")
 *   - 支持 dtype: FLOAT / FLOAT16 / BFLOAT16
 *   - format: ND
 *   - PrecisionReduceFlag(false)：19 步内部已升 FP32 精度，无需框架降精度
 *   - 仅 ascend950（DAV_3510 / arch35）
 *
 * 迭代一范围：仅 FP32 完整实现；FP16/BF16 在迭代二落地。
 * 但 DataType 列表声明 3 个 dtype 保证 TilingKey 接口稳定。
 */
#include "register/op_def_registry.h"

namespace ops {
class Asinh : public OpDef {
public:
    explicit Asinh(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)    // 必须为 true 否则不会进入二进制编译流程
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)        // 19 步内部已 Cast 升精，无需框架降精度
            .ExtendCfgInfo("opFile.value", "asinh_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig950);
    }
};
OP_ADD(Asinh);
} // namespace ops
