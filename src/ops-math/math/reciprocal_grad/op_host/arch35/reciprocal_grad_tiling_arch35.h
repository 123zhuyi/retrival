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
 * \file reciprocal_grad_tiling_arch35.h
 * \brief ReciprocalGrad 算子 Host Tiling 类与 CompileInfo 定义（atvoss 框架 - Elewise 模式）
 */

#ifndef RECIPROCAL_GRAD_TILING_H
#define RECIPROCAL_GRAD_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "math/reciprocal_grad/op_kernel/arch35/reciprocal_grad_tiling_data.h"

namespace optiling {
using namespace Ops::Base;

struct ReciprocalGradCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};
class ReciprocalGradTiling {
public:
    explicit ReciprocalGradTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();
    ReciprocalGradTilingData* tiling = nullptr;

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus SetTilingData();

private:
    ge::graphStatus CheckShape();

    gert::TilingContext* tilingContext;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
};

} // namespace optiling

#endif // RECIPROCAL_GRAD_TILING_H
