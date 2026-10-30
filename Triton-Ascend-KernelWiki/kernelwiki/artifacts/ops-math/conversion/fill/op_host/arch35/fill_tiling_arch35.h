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
 * \file fill_tiling.h
 * \brief
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_FILL_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_FILL_TILING_H

#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/elewise/elewise_base_struct.h"
#include "../../op_kernel/fill_struct.h"

namespace optiling {
using namespace Ops::Base;

struct FillCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class FillTiling {
public:
    explicit FillTiling(gert::TilingContext *context) : context_(context){};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus SetTilingData();

private:
    ge::graphStatus CheckInputDims();
    ge::graphStatus CheckInputValue();
    FillStruct::FillTilingDataStruct *tilingData;
    gert::TilingContext *context_;
    ge::DataType outputDtype_;
};

}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_FILL_TILING_H