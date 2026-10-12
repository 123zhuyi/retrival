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
 * \file stateless_bernoulli_tiling_arch35.h
 * \brief
 */
#ifndef STATELESS_BERNOULLI_TILING_ARCH35_H
#define STATELESS_BERNOULLI_TILING_ARCH35_H

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "../../../random_common/op_host/arch35/random_tiling_arch35.h"

namespace optiling {

class StatelessBernoulliTiling : public RandomTilingArch35 {
public:
    explicit StatelessBernoulliTiling(gert::TilingContext* context)
        : RandomTilingArch35(context, BuildOpConfig()) {}

protected:
    ge::graphStatus DoSimtBlockTiling() override;

private:
    static OpTilingConfig BuildOpConfig();
};
} // namespace optiling
#endif // STATELESS_BERNOULLI_TILING_ARCH35_H
