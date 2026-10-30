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
 * \file eye.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class Eye : public OpDef {
public:
    explicit Eye(const char *name) : OpDef(name)
    {
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_UINT8, ge::DT_INT8, ge::DT_INT16,
            ge::DT_INT32, ge::DT_INT64, ge::DT_BOOL })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Attr("num_rows").AttrType(REQUIRED).Int();
        this->Attr("num_columns").AttrType(OPTIONAL).Int(0);
        this->Attr("batch_shape").AttrType(OPTIONAL).ListInt({});
        this->Attr("dtype").AttrType(OPTIONAL).Int(0);
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "eye_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(Eye);
}