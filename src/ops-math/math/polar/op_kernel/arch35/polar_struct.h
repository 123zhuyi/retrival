/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file polar_struct.h
 * \brief polar struct
 */
#ifndef POLAR_STRUCT_H_
#define POLAR_STRUCT_H_

#include <cstdint>

constexpr int64_t POLAR_MAX_DIM = 8;

#pragma pack(push, 8)
struct PolarTilingData {
    int64_t totalElements;
    int64_t elementsPerCore;
    int64_t coreNum;
    int64_t formerCore;
    int64_t dimNum;
    int64_t mergedStride[POLAR_MAX_DIM];
    int64_t absStride[POLAR_MAX_DIM];
    int64_t angleStride[POLAR_MAX_DIM];
    int64_t yStride[POLAR_MAX_DIM];
};
#pragma pack(pop)

#endif // POLAR_STRUCT_H_
