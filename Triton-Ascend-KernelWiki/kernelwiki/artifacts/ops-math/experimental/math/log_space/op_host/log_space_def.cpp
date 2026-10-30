/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file log_space_def.cpp
 * \brief LogSpace 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class LogSpace : public OpDef {
public:
    explicit LogSpace(const char* name) : OpDef(name)
    {
        // LogSpace 是纯生成类算子：无输入 tensor，仅输出 result
        this->Output("result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 属性：start/end 已在 Host 侧由 aclScalar 转为 float 传入；base 同样转为 float
        this->Attr("start").AttrType(REQUIRED).Float();
        this->Attr("end").AttrType(REQUIRED).Float();
        this->Attr("steps").AttrType(REQUIRED).Int();
        this->Attr("base").AttrType(REQUIRED).Float();

        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "log_space");
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};
OP_ADD(LogSpace);
} // namespace ops
