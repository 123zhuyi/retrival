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
* \file stateless_random_normal_v3_tiling.h
* \brief
*/

#ifndef STATELESS_RANDOM_NORMAL_V3_TILING_ARCH35_H
#define STATELESS_RANDOM_NORMAL_V3_TILING_ARCH35_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "util/math_util.h"
#include  "../../../random_common/op_host/arch35/random_tiling_arch35.h"

namespace optiling {
class StatelessRandomNormalV3Tiling : public RandomTilingArch35
{
public:
    explicit StatelessRandomNormalV3Tiling(gert::TilingContext* ctx);
    ~StatelessRandomNormalV3Tiling() override = default;

protected:
    ge::graphStatus UniqueProcess() override;

private:
    static OpTilingConfig BuildOpConfig();
};

} // namespace optiling
#endif // STATELESS_RANDOM_NORMAL_V3_TILING_ARCH35_H