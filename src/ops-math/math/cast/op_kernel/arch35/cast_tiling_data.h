/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cast_tiling_data.h
 * \brief Cast tiling data struct (plain struct, shared by host tiling and device kernel)
 */

#ifndef CAST_TILING_DATA_H
#define CAST_TILING_DATA_H

#include <cstdint>

struct CastTilingData {
    int64_t shapeSize = 0; // Total number of elements
    int32_t coreNum = 0;   // Number of cores to use
    int32_t ubFormer = 0;  // Number of elements per UB iteration
};

#endif // CAST_TILING_DATA_H
