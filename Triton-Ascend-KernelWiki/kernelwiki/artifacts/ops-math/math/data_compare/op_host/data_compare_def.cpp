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
 * \file data_compare_def.cpp
 * \brief DataCompare 算子定义
 *
 * 逐元素比较两个张量，统计差异超出容差范围的元素总个数（All Reduce）。
 * 输入：x1, x2（同 shape 同 dtype）
 * 输出：num（float32 标量）
 * 属性：atol（默认 1e-5），rtol（默认 1e-3）
 * 无 dim/keep_dims 属性：归约轴固定为 All Reduce（spec.yaml reduction.axis_source: fixed）
 */
#include "register/op_def_registry.h"

namespace ops {

class DataCompare : public OpDef {
public:
    explicit DataCompare(const char* name) : OpDef(name)
    {
        // 输入 x1：6 种 dtype
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 输入 x2：与 x1 同 dtype 同 shape
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 输出 num：固定 float32 标量
        this->Output("num")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 无 dim/keep_dims 属性：归约轴固定为 All Reduce（spec.yaml reduction.axis_source: fixed）

        // Attr atol：绝对容差，默认 1e-5
        this->Attr("atol").AttrType(OPTIONAL).Float(1e-5f);

        // Attr rtol：相对容差，默认 1e-3
        this->Attr("rtol").AttrType(OPTIONAL).Float(1e-3f);

        // AI Core 编译配置（仅支持 ascend950）
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "data_compare_apt");

        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(DataCompare);
} // namespace ops
