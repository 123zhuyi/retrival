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
 * \file reciprocal_grad_def.cpp
 * \brief ReciprocalGrad 算子定义，声明输入输出和算子配置
 *
 * 算子功能：计算 reciprocal 函数的梯度
 * 数学公式：z = -y * y * dy
 *
 * 输入：
 *   - y:  前向输出（倒数值，第一个输入）
 *   - dy: 上游梯度（损失函数对 reciprocal 输出的梯度，第二个输入）
 *
 * 输出：
 *   - z: 最终梯度（损失函数对 reciprocal 输入的梯度）
 *
 * 约束：
 *   - y、dy、z 三者的 shape 必须完全相同
 *   - y、dy、z 三者的 dtype 必须一致
 *
 * 数据类型支持：FLOAT16, FLOAT32, BFLOAT16
 * 芯片支持：Ascend950 (DAV_3510)
 */
#include "register/op_def_registry.h"

namespace ops {
class ReciprocalGrad : public OpDef {
public:
    explicit ReciprocalGrad(const char* name) : OpDef(name)
    {
        // 输入 y 定义（第一个输入）
        this->Input("y")
            .ParamType(REQUIRED)                                               // 必选输入
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})             // 支持数据类型
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})             // 支持format格式
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}) // 未确定大小shape对应format格式
            .AutoContiguous();                                                 // 内存自动连续化

        // 输入 dy 定义（第二个输入）
        this->Input("dy")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 输出 z 定义
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Ascend950 (DAV_3510) 配置
        // 使用 atvoss 框架，Kernel 入口文件为 reciprocal_grad.cpp
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true) // 支持0-8维
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "reciprocal_grad"); // Kernel 入口文件名
        this->AICore().AddConfig("ascend950", aicoreConfig950);
    }
};
OP_ADD(ReciprocalGrad); // 添加算子信息库
} // namespace ops
