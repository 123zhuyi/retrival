/**
 * This file is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 Yang Zhenze, Chongqing University of Posts and Telecommunications (CQUPT).
 * All Rights Reserved.
 *
 * Author (account):
 * - Yang Zhenze <@gcw_5x5Ew5Ms>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bias_add_def.cpp
 * \brief aicore info for bias_add op
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> dataType = {
    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,  ge::DT_BF16,  ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,   ge::DT_FLOAT,
    ge::DT_FLOAT,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};

static const std::vector<ge::Format> baseFormat = {
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NDHWC, ge::FORMAT_NCDHW, ge::FORMAT_ND,
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NDHWC, ge::FORMAT_NCDHW, ge::FORMAT_ND,
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NDHWC, ge::FORMAT_NCDHW, ge::FORMAT_ND,
    ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NDHWC, ge::FORMAT_NCDHW, ge::FORMAT_ND};

static const std::vector<ge::Format> biasFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class BiasAdd : public OpDef {
public:
    explicit BiasAdd(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(dataType).Format(baseFormat).UnknownShapeFormat(baseFormat);
        this->Input("bias").ParamType(REQUIRED).DataType(dataType).Format(biasFormat).UnknownShapeFormat(biasFormat);
        this->Output("y").ParamType(REQUIRED).DataType(dataType).Format(baseFormat).UnknownShapeFormat(baseFormat);
        this->Attr("data_format").AttrType(OPTIONAL).String("NHWC");

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(BiasAdd);
} // namespace ops
