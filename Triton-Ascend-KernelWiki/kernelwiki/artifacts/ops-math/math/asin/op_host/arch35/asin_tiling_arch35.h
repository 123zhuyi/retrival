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
 * \file asin_tiling_arch35.h
 * \brief Asin tiling header
 */
#ifndef ASIN_TILING_ARCH35_H
#define ASIN_TILING_ARCH35_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "math/asin/op_kernel/arch35/asin_tilingdata.h"

namespace optiling {
class AsinTiling {
public:
    explicit AsinTiling(gert::TilingContext* context) : tilingContext(context) {};
    ~AsinTiling() = default;
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus SetTilingData();
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CheckShape();

private:
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype;
    AsinTilingData* tiling;
};
}  // namespace optiling

#endif  // ASIN_TILING_ARCH35_H
