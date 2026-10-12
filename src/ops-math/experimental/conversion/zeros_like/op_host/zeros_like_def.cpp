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
 * \file zeros_like_def.cpp
 * \brief experimental 自包含 ZerosLike 算子信息库定义（仅 ascend910b / DAV_2201）。
 *        私有 8 种 dtype（FP16/BF16/FP32/INT32/INT64/INT8/UINT8/BOOL），
 *        与 op_api L0 AICORE910B_DTYPE_SUPPORT_LIST 一致。
 *        显式设置 opFile.value="zeros_like"（与默认 snake-case 等价，显式声明以提升鲁棒性、
 *        不依赖未文档化的默认 kernel 文件扫描行为）→ 入口文件 op_kernel/zeros_like.cpp。
 */
#include "register/op_def_registry.h"

namespace ops {
// ascend910b（DAV_2201）私有 8 种 dtype。
static const std::vector<ge::DataType> dataType910b = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32,
                                                       ge::DT_INT64,   ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL};

static const std::vector<ge::Format> dataFormat910b(8, ge::FORMAT_ND);

class ZerosLike : public OpDef {
public:
    explicit ZerosLike(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(dataType910b)
            .Format(dataFormat910b)
            .UnknownShapeFormat(dataFormat910b);

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(dataType910b)
            .Format(dataFormat910b)
            .UnknownShapeFormat(dataFormat910b);

        OpAICoreConfig config910b;
        config910b.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("op.pattern", "formatAgnostic")
            .ExtendCfgInfo("opFile.value", "zeros_like"); // 对应 kernel 入口 op_kernel/zeros_like.cpp
        config910b.Input("x")
            .ParamType(REQUIRED)
            .DataType(dataType910b)
            .Format(dataFormat910b)
            .UnknownShapeFormat(dataFormat910b);
        config910b.Output("y")
            .ParamType(REQUIRED)
            .DataType(dataType910b)
            .Format(dataFormat910b)
            .UnknownShapeFormat(dataFormat910b);
        this->AICore().AddConfig("ascend910b", config910b);
    }
};

OP_ADD(ZerosLike);
} // namespace ops
