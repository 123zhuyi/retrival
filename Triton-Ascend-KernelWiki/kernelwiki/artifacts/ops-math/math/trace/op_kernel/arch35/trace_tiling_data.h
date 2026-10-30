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
 * \file trace_tiling_data.h
 * \brief Tiling data struct for trace operator
 */

#ifndef TRACE_TILING_DATA_H_
#define TRACE_TILING_DATA_H_

struct TraceTilingData {
    int64_t diagSize;       // Number of diagonal elements = min(M, N)
    int64_t diagStride;     // Stride between diagonal elements = stride0 + stride1
};

#endif  // TRACE_TILING_DATA_H_
