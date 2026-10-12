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
 * \file arange.cpp
 * \brief
 */
#include "register/op_def_registry.h"

#define ARANGE_SCALAR_DTYPE_LIST \
    {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64}

#define ARANGE_ND_FORMAT_LIST                                    \
    {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, \
     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}

namespace ops {
class Arange : public OpDef {
public:
    explicit Arange(const char* name) : OpDef(name)
    {
        // start, end and step are scalar operands. Each dtype entry forms one
        // positional registration lane and must use the same ND format.
        this->Input("start")
            .ParamType(REQUIRED)
            .DataType(ARANGE_SCALAR_DTYPE_LIST)
            .Format(ARANGE_ND_FORMAT_LIST)
            .UnknownShapeFormat(ARANGE_ND_FORMAT_LIST)
            .Scalar();

        // Keep the scalar input lanes aligned so that host-side dtype matching
        // selects the same type for the complete arithmetic progression.
        this->Input("end")
            .ParamType(REQUIRED)
            .DataType(ARANGE_SCALAR_DTYPE_LIST)
            .Format(ARANGE_ND_FORMAT_LIST)
            .UnknownShapeFormat(ARANGE_ND_FORMAT_LIST)
            .Scalar();

        this->Input("step")
            .ParamType(REQUIRED)
            .DataType(ARANGE_SCALAR_DTYPE_LIST)
            .Format(ARANGE_ND_FORMAT_LIST)
            .UnknownShapeFormat(ARANGE_ND_FORMAT_LIST)
            .Scalar();

        // The output registration mirrors the scalar dtype lanes and describes
        // the generated one-dimensional arithmetic-progression tensor.
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType(ARANGE_SCALAR_DTYPE_LIST)
            .Format(ARANGE_ND_FORMAT_LIST)
            .UnknownShapeFormat(ARANGE_ND_FORMAT_LIST);

        // This implementation is registered only for the A2/A3 targets that
        // provide the kernel variants declared by the Arange operator.
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};
OP_ADD(Arange); // 添加算子信息库
} // namespace ops

#undef ARANGE_ND_FORMAT_LIST
#undef ARANGE_SCALAR_DTYPE_LIST
