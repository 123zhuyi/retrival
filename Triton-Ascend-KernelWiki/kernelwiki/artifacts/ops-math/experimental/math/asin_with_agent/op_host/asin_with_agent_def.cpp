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
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file asin_with_agent_def.cpp
 * \brief AsinWithAgent 算子定义，声明输入输出和算子配置
 *
 * 迭代二：激活全部 9 种 dtype（TilingKey 0-8）
 *
 * 输出 dtype 规则：
 *   - FLOAT/FLOAT16/DOUBLE -> 与输入相同 dtype
 *   - INT8/INT16/INT32/INT64/UINT8/BOOL -> 输出 FLOAT32
 *
 * 注意：DOUBLE 输出 DOUBLE（op_api 层做 Host 端 fp64->fp32->fp64 转换），
 *       Kernel 侧实际处理的是 fp32 数据。
 */
#include "register/op_def_registry.h"

namespace ops {
class AsinWithAgent : public OpDef {
public:
    explicit AsinWithAgent(const char* name) : OpDef(name)
    {
        // 输入 x：支持全部 9 种 dtype
        // DataType、Format、UnknownShapeFormat 列表长度必须一致
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,   // fp32  - TilingKey=0（Group A）
                ge::DT_FLOAT16, // fp16  - TilingKey=1（Group A）
                ge::DT_DOUBLE,  // fp64  - TilingKey=2（Group B，op_api 层 Host 端转换）
                ge::DT_INT8,    // int8  - TilingKey=3（Group C）
                ge::DT_INT16,   // int16 - TilingKey=4（Group C）
                ge::DT_INT32,   // int32 - TilingKey=5（Group C）
                ge::DT_INT64,   // int64 - TilingKey=6（Group C）
                ge::DT_UINT8,   // uint8 - TilingKey=7（Group C）
                ge::DT_BOOL,    // bool  - TilingKey=8（Group C）
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
            })
            .AutoContiguous();

        // 输出 y：浮点类型与输入相同，整数/BOOL 类型输出 FLOAT32
        // DataType 列表中的 dtype 与 Input.DataType 一一对应（按位置匹配）
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,   // 对应 fp32 输入
                ge::DT_FLOAT16, // 对应 fp16 输入
                ge::DT_DOUBLE,  // 对应 fp64 输入（DOUBLE 输出 DOUBLE）
                ge::DT_FLOAT,   // 对应 int8 输入（输出 fp32）
                ge::DT_FLOAT,   // 对应 int16 输入（输出 fp32）
                ge::DT_FLOAT,   // 对应 int32 输入（输出 fp32）
                ge::DT_FLOAT,   // 对应 int64 输入（输出 fp32）
                ge::DT_FLOAT,   // 对应 uint8 输入（输出 fp32）
                ge::DT_FLOAT,   // 对应 bool 输入（输出 fp32）
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
            })
            .AutoContiguous();

        // Ascend910B 配置（arch32）
        OpAICoreConfig aicoreConfig910B;
        aicoreConfig910B.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "asin_with_agent");
        this->AICore().AddConfig("ascend910b", aicoreConfig910B);
    }
};
OP_ADD(AsinWithAgent);
} // namespace ops
