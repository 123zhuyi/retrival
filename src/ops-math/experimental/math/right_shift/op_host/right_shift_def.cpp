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
 * \file right_shift_def.cpp
 * \brief RightShift op definition
 */

#include <cstdint>
#include <vector>
#include "register/op_def_registry.h"

namespace ops {
namespace {
const std::vector<ge::DataType> RIGHT_SHIFT_DATA_TYPES = {ge::DT_INT8,  ge::DT_UINT8,  ge::DT_INT16, ge::DT_UINT16,
                                                          ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64};
const std::vector<ge::Format> RIGHT_SHIFT_FORMATS(RIGHT_SHIFT_DATA_TYPES.size(), ge::FORMAT_ND);
} // namespace

class RightShift : public OpDef {
public:
    explicit RightShift(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(RIGHT_SHIFT_DATA_TYPES)
            .Format(RIGHT_SHIFT_FORMATS)
            .UnknownShapeFormat(RIGHT_SHIFT_FORMATS)
            .AutoContiguous();
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType(RIGHT_SHIFT_DATA_TYPES)
            .Format(RIGHT_SHIFT_FORMATS)
            .UnknownShapeFormat(RIGHT_SHIFT_FORMATS)
            .AutoContiguous();
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType(RIGHT_SHIFT_DATA_TYPES)
            .Format(RIGHT_SHIFT_FORMATS)
            .UnknownShapeFormat(RIGHT_SHIFT_FORMATS)
            .AutoContiguous();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "right_shift");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};

OP_ADD(RightShift);
} // namespace ops
