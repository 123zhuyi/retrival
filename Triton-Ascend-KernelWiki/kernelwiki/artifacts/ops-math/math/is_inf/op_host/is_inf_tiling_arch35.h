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
 * \file is_inf_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_IS_INF_REGBASE_OPTILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_IS_INF_REGBASE_OPTILING_H

#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"

namespace optiling {
using namespace Ops::Base;

class IsInfRegbaseTiling
{
public:
    explicit IsInfRegbaseTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus SetTilingData();

private:
    gert::TilingContext* tilingContext;
    ge::DataType inputDtype;
    ge::DataType outputDtype;
    EleBaseTilingDataV2* tiling_ = nullptr;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_IS_INF_REGBASE_OPTILING_H