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
 * \file add_mat_mat_elements_def.cpp
 * \brief AddMatMatElements 算子定义
 *
 * 算子规格：
 *   输入：a, b, c（tensor），alpha, beta（scalar，aclScalar）
 *   输出：cOut（tensor）
 *   支持 dtype：fp16 / fp32 / bfloat16
 *   目标芯片：Ascend950（arch35，DAV_3510）
 */

#include "register/op_def_registry.h"

namespace ops {

class AddMatMatElements : public OpDef {
public:
    explicit AddMatMatElements(const char* name) : OpDef(name)
    {
        // 输入 tensor a：与 b、c 同 shape 和 dtype
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 输入 tensor b：与 a 同 shape 和 dtype
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 输入 tensor c：与 a 同 shape 和 dtype
        this->Input("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 标量 alpha：缩放系数，作为 float Attr 传入（不参与 kernel binary key 计算）
        this->Attr("alpha").AttrType(REQUIRED).Float();

        // 标量 beta：缩放系数，作为 float Attr 传入（不参与 kernel binary key 计算）
        this->Attr("beta").AttrType(REQUIRED).Float();

        // 输出 tensor cOut：计算结果，shape 与 a/b/c 相同
        this->Output("c_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Ascend950（arch35，DAV_3510）配置
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "add_mat_mat_elements_apt");

        this->AICore().AddConfig("ascend950", aicoreConfig950);
    }
};

OP_ADD(AddMatMatElements);

} // namespace ops
