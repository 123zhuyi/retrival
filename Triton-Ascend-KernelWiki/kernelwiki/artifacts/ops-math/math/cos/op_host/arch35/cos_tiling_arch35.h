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
 * \file cos_tiling_arch35.h
 * \brief
 */
#ifndef OPS_MATH_COS_OP_HOST_COS_ARCH35_H_
#define OPS_MATH_COS_OP_HOST_COS_ARCH35_H_

#include "atvoss/elewise/elewise_tiling.h"
#include "math/cos/op_kernel/arch35/cos_tilingdata.h"

namespace optiling {

class CosTiling
{
public:
    explicit CosTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();
    CosTilingData *tiling = nullptr;

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus SetTilingData();

private:
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    ge::DataType inputDtype = ge::DT_UNDEFINED;
    uint64_t dType = 0;
};
}  // namespace optiling
#endif  // OPS_MATH_COS_OP_HOST_COS_TILING_ARCH35_H_