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
 * \file random_standard_normal_v2_tiling_arch35.h
 * \brief
 */
#ifndef RANDOM_STANDARD_NORMAL_V2_TILING_ARCH35_H
#define RANDOM_STANDARD_NORMAL_V2_TILING_ARCH35_H

#include <string>
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "util/math_util.h"
#include "../../../random_common/op_host/arch35/random_tiling_arch35.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"

namespace optiling {
class RandomStandardNormalV2Tiling : public RandomTilingArch35 {
public:
    explicit RandomStandardNormalV2Tiling(gert::TilingContext* context);
    ~RandomStandardNormalV2Tiling() override = default;

protected:
private:
    // 构建当前算子的配置（仅需配置规则+字段映射）
    static OpTilingConfig BuildOpConfig();
};
struct RandomStandardNormalV2CompileInfo {
    int64_t totalCoreNum = 0;
    int64_t ubSize = 0;
};
} // namespace optiling
#endif // RANDOM_STANDARD_NORMAL_V2_TILING_ARCH35_H
