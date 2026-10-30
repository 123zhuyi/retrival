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
 * \file trace_def.cpp
 * \brief Operator definition for trace operator
 *
 * REG_OP(Trace) from canndev (SE section 5.2):
 *   INPUT(x): DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16,
 *             DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
 *             DT_UINT32, DT_INT64, DT_UINT64, DT_BF16  (15 types in REG_OP)
 *   OUTPUT(y): DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16,
 *              DT_INT64, DT_UINT64, DT_BF16  (8 possible output types)
 *
 * Ascend950 SIMT implementation supports 13 dtype combinations (see SE §5.7):
 *   complex64->complex64, float32->float32, float16->float16, bfloat16->bfloat16,
 *   int8->int64, int16->int64, int32->int64, int64->int64,
 *   uint8->int64, uint16->int64, uint32->int64, uint64->uint64, bool->int64
 *
 * Hardware limitation note (review issue #1):
 *   SE §5.2 REG_OP 声明包含 DT_DOUBLE 和 DT_COMPLEX128，但 Ascend950 aicore
 *   不支持 double precision 运算（编译器报错 "double precision operation is
 *   not allowed in aicore function"），且 complex128 在 AscendC SIMT DTYPE_MAP
 *   中映射为 "unknown"（无对应 C++ 类型）。因此本算子在 Ascend950 上仅注册
 *   13 种可编译的 dtype 组合，DT_DOUBLE 和 DT_COMPLEX128 暂不支持。
 *   若需支持这两种类型，需在其他支持 double 的芯片（如 ascend910b）上实现，
 *   或使用软件模拟（性能代价较大）。
 *
 * uint64 output type note (review issue #2):
 *   SE §5.2 REG_OP OUTPUT 声明包含 DT_UINT64，SE §5.5 推导规则说 uint64→int64。
 *   按 SE §5.2 REG_OP 定义为准，Output 保留 DT_UINT64（与 REG_OP 一致）。
 */
#include "register/op_def_registry.h"

namespace ops {
class Trace : public OpDef {
public:
    explicit Trace(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64,
                        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                        ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64,
                        ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64,
                        ge::DT_BOOL})
            .Format({ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64,
                        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_UINT64,
                        ge::DT_INT64})
            .Format({ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND})
            .AutoContiguous();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "trace_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(Trace);
}  // namespace ops
