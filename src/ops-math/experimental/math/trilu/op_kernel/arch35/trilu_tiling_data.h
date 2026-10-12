/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TRILU_TILING_DATA_H__
#define __TRILU_TILING_DATA_H__

struct TriluTilingData {
    int32_t needCoreNum = 0;
    int64_t totalElements = 0;
    int64_t perCoreElements = 0;
    int64_t lastCoreElements = 0;
    int64_t diagonal = 0;
    int32_t upper = 0;
    int64_t h = 0;
    int64_t w = 0;
};

#endif // __TRILU_TILING_DATA_H__
