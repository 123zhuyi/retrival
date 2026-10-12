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
 * \file expm1_tiling_arch35.h
 * \brief
 */
#ifndef OPS_MATH_EXPM1_TILING_ARCH35_H
#define OPS_MATH_EXPM1_TILING_ARCH35_H

#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "math/expm1/op_kernel/arch35/expm1_struct.h"

namespace optiling {
using namespace Ops::Base;

class Expm1Tiling {
public:
    explicit Expm1Tiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus SetTilingData();

private:
    EleBaseTilingDataV2* tiling_ = nullptr;
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype;
    ge::DataType inputDtype;
    uint64_t dType = 0;
};
} // namespace optiling
#endif // OPS_MATH_EXPM1_TILING_ARCH35_H