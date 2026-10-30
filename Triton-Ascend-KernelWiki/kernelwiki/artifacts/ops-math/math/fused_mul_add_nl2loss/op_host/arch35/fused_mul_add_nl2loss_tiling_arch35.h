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
 * \file fused_mul_add_nl2loss_tiling_arch35.h
 * \brief FusedMulAddNL2loss arch35 tiling
 */

#ifndef FUSED_MUL_ADD_NL2LOSS_TILING_ARCH35_H
#define FUSED_MUL_ADD_NL2LOSS_TILING_ARCH35_H

#include <cstdint>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/fused_mul_add_nl2loss_tiling_data.h"

namespace optiling {

struct FusedMulAddNL2lossCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

class FusedMulAddNL2lossTiling {
public:
    explicit FusedMulAddNL2lossTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetPlatformInfo();

    gert::TilingContext* context_ = nullptr;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
};

} // namespace optiling

#endif // FUSED_MUL_ADD_NL2LOSS_TILING_ARCH35_H
