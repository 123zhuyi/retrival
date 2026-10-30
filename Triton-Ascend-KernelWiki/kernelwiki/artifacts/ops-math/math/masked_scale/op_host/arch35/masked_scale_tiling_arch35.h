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
 * \file masked_scale_tiling_arch35.h
 * \brief
 */
#ifndef OPS_OP_TILING_RUNTIME_MASKED_SCALE_TILING_H
#define OPS_OP_TILING_RUNTIME_MASKED_SCALE_TILING_H

#include <cstdint>

namespace optiling {

struct MaskedScaleCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

}  // namespace optiling
#endif  // OPS_OP_TILING_RUNTIME_MASKED_SCALE_TILING_H
