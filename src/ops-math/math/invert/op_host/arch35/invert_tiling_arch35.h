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
 * \file invert_tiling_arch35.h
 * \brief invert ascendc tiling def
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ELEMENTWISE_TEMPLATE_INVERT_OPTILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ELEMENTWISE_TEMPLATE_INVERT_OPTILING_H

#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"

using namespace Ops::Base;
namespace optiling {

class InvertTiling : public ElewiseBaseTiling {
public:
    explicit InvertTiling(gert::TilingContext* context) : ElewiseBaseTiling(context), context_(context) {};
    ge::graphStatus RunTiling();

private:
    ge::graphStatus SetTilingData();
    ge::graphStatus CheckAndGetOutputDtype(ge::DataType& outputDtype);

private:
    gert::TilingContext* context_;
    EleBaseTilingData* td_;
};

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ELEMENTWISE_TEMPLATE_INVERT_OPTILING_H