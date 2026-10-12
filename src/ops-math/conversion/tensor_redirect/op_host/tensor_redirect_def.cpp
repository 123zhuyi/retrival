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
 * \file tensor_redirect_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
// 11 种 dtype，与 REG_OP 严格一致
static const std::vector<ge::DataType> dataType = {ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_INT8,  ge::DT_INT32,
                                                   ge::DT_UINT8,   ge::DT_INT64,  ge::DT_INT16, ge::DT_UINT16,
                                                   ge::DT_UINT64,  ge::DT_UINT32, ge::DT_BF16};

static const std::vector<ge::Format> dataFormat(11, ge::FORMAT_ND);

class TensorRedirect : public OpDef {
public:
    explicit TensorRedirect(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(dataType).Format(dataFormat).UnknownShapeFormat(dataFormat);

        this->Output("output_x")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(dataFormat)
            .UnknownShapeFormat(dataFormat);

        OpAICoreConfig aicoreConfig;
        aicoreConfig
            .DynamicCompileStaticFlag(true) // GE 静态 shape
            .DynamicFormatFlag(false)       // 仅 ND
            .DynamicRankSupportFlag(true)   // dynamic rank
            .DynamicShapeSupportFlag(true)  // 动态 shape
            .NeedCheckSupportFlag(false)
            // 恒等拷贝语义要求逐 bit 相等（含负零/NaN payload/非规格化数），
            // 不允许 allow_mix_precision 下把 FP32 节点降为 FP16
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("op.pattern", "formatAgnostic")
            .ExtendCfgInfo("opFile.value", "tensor_redirect_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(TensorRedirect);
} // namespace ops
