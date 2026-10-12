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
 * \file zeros_like_tiling.h
 * \brief experimental 自包含 ZerosLike tiling：CompileInfo（TilingPrepare 阶段缓存平台信息）。
 */
#ifndef ZEROS_LIKE_TILING_H
#define ZEROS_LIKE_TILING_H

#include <cstdint>
#include "graph/types.h"

namespace optiling {

// TilingPrepare 阶段缓存的平台信息
struct ZerosLikeCompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
};

} // namespace optiling

#endif // ZEROS_LIKE_TILING_H
