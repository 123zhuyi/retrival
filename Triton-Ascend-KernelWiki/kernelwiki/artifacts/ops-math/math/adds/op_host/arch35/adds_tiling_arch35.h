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
 * \file adds_tiling.cpp
 * \brief Adds 算子 Host Tiling 实现（atvoss 框架 - Elewise 模式）
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ADDS_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ADDS_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "math/adds/op_kernel/arch35/adds_tiling_data.h"

namespace optiling {
using namespace Ops::Base;

struct AddsCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class AddsTiling {
public:
    explicit AddsTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus RunTiling();
    AddsTilingData* tiling = nullptr;

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus SetTilingData();

private:
    ge::graphStatus CheckShape();

    gert::TilingContext *tilingContext;
    ge::DataType outputDtype;
};

}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_ADDS_TILING_H