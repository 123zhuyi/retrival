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
 * \file truncated_normal_v2_tiling_arch35.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_Truncated_Normal_V2_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_Truncated_Normal_V2_H_

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "../../../random_common/op_host/arch35/random_tiling_arch35.h"

namespace optiling {

class TruncatedNormalV2Tiling : public RandomTilingArch35 {
public:
    explicit TruncatedNormalV2Tiling(gert::TilingContext* context)
        : RandomTilingArch35(context, BuildOpConfig()) {}

protected:
    ge::graphStatus DoSimtBlockTiling() override;

private:
    static OpTilingConfig BuildOpConfig();
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_Truncated_Normal_V2_H_
