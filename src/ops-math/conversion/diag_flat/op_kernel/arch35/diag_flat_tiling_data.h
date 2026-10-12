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
 * \file diag_flat_tiling_data.h
 * \brief DiagFlat TilingData struct definition (arch35, DAV_3510)
 *
 * Design basis: DESIGN.md v2.1 Sec 3.2
 * Standard C++ POD struct.
 * (ref: diag_v2/op_kernel/arch35/diag_v2_tiling_data.h)
 */

#ifndef __DIAG_FLAT_ARCH35_TILING_DATA_H__
#define __DIAG_FLAT_ARCH35_TILING_DATA_H__

#include <cstdint>

struct DiagFlatArch35TilingData {
    int64_t numInput;         // N = numel(x)
    int64_t diagonal;         // k (signed)
    int64_t outWidth;         // W = N + |k|
    int64_t outTotal;         // W * W
    int64_t outPerCore;       // ceil(outTotal / realCoreNum), elements per core
    int64_t tileLength;       // max elements per tile
    int64_t realCoreNum;      // actual number of cores used
};

#endif // __DIAG_FLAT_ARCH35_TILING_DATA_H__
