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
 * \file tile_def.cpp
 * \brief
 */

#include <vector>

#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> kXDataTypes = {
    ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_BF16,      ge::DT_FLOAT,    ge::DT_FLOAT16, ge::DT_BF16,

    ge::DT_INT32,  ge::DT_INT32,   ge::DT_INT16,     ge::DT_INT16,    ge::DT_INT8,    ge::DT_INT8,

    ge::DT_UINT8,  ge::DT_UINT8,   ge::DT_UINT16,    ge::DT_UINT16,   ge::DT_UINT32,  ge::DT_UINT32,
    ge::DT_UINT64, ge::DT_UINT64,

    ge::DT_BOOL,   ge::DT_BOOL,    ge::DT_COMPLEX64, ge::DT_COMPLEX64};

static const std::vector<ge::DataType> kMultiplesDataTypes = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,

    ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,

    ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,

    ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64};

static const std::vector<ge::Format> kNdFormats(24, ge::FORMAT_ND);

class Tile : public OpDef {
public:
    explicit Tile(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(kXDataTypes).Format(kNdFormats).UnknownShapeFormat(kNdFormats);

        this->Input("multiples")
            .ParamType(REQUIRED)
            .DataType(kMultiplesDataTypes)
            .Format(kNdFormats)
            .UnknownShapeFormat(kNdFormats)
            .ValueDepend(REQUIRED);

        this->Output("y").ParamType(REQUIRED).DataType(kXDataTypes).Format(kNdFormats).UnknownShapeFormat(kNdFormats);

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Tile);

} // namespace ops
