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
 * \file bias_add_grad_tiling_arch35.h
 * \brief tiling for bias add grad ops
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_BIAS_ADD_GRAD_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_BIAS_ADD_GRAD_H

#include "atvoss/reduce/reduce_tiling.h"

namespace optiling {
ge::graphStatus Tiling4BiasAddGradForAscendC(gert::TilingContext* context);
} // namespace optiling

#endif