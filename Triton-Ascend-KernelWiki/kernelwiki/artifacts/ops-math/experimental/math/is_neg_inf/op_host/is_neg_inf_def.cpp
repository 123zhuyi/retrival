/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {
class IsNegInf : public OpDef {
public:
    explicit IsNegInf(const char *name) : OpDef(name)
    {
        static const std::vector<ge::DataType> inputDtypes = {
            ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BOOL, ge::DT_INT32,
            ge::DT_INT64,   ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8};
        static const std::vector<ge::DataType> outputDtypes(inputDtypes.size(), ge::DT_BOOL);
        static const std::vector<ge::Format> formats(inputDtypes.size(), ge::FORMAT_ND);

        this->Input("x").ParamType(REQUIRED).DataType(inputDtypes).Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Output("y").ParamType(REQUIRED).DataType(outputDtypes).Format(formats).UnknownShapeFormat(formats).AutoContiguous();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "is_neg_inf");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};
OP_ADD(IsNegInf);
}  // namespace ops
