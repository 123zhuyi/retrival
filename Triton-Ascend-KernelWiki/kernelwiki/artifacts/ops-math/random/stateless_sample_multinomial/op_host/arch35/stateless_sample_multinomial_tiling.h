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
 * \file stateless_sample_multinomial_tiling.h
 * \brief Tiling for StatelessSampleMultinomial kernel
 */
#ifndef STATELESS_SAMPLE_MULTINOMIAL_TILING_H
#define STATELESS_SAMPLE_MULTINOMIAL_TILING_H

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "../../../random_common/op_host/arch35/random_tiling_arch35.h"

namespace optiling {

class StatelessSampleMultinomialTiling : public RandomTilingArch35 {
public:
    explicit StatelessSampleMultinomialTiling(gert::TilingContext* context)
        : RandomTilingArch35(context, BuildOpConfig()) {}

protected:
    ge::graphStatus UniqueProcess() override;

private:
    static OpTilingConfig BuildOpConfig();
};

} // namespace optiling
#endif // STATELESS_SAMPLE_MULTINOMIAL_TILING_H
