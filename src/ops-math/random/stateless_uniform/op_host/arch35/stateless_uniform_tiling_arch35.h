/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stateless_uniform_tiling_arch35.h
 * \brief StatelessUniform Tiling类声明
 */

#ifndef STATELESS_UNIFORM_TILING_ARCH35_H
#define STATELESS_UNIFORM_TILING_ARCH35_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "util/math_util.h"
#include  "../../../random_common/op_host/arch35/random_tiling_arch35.h"

namespace optiling {
class StatelessUniformTiling : public RandomTilingArch35
{
public:
    explicit StatelessUniformTiling(gert::TilingContext* ctx);
    ~StatelessUniformTiling() override = default;

protected:
    ge::graphStatus UniqueProcess() override;
private:
    static OpTilingConfig BuildOpConfig();
};

} // namespace optiling
#endif // STATELESS_UNIFORM_TILING_ARCH35_H
