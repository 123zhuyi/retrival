/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CUMULATIVE_LOGSUMEXP_TILING_DATA_H
#define CUMULATIVE_LOGSUMEXP_TILING_DATA_H

#include <cstdint>

struct CumulativeLogsumexpTilingData {
    int64_t totalNum = 0;
    int64_t outerNum = 0;
    int64_t axisNum = 0;
    int64_t innerNum = 0;
    int64_t exclusive = 0;
    int64_t reverse = 0;
};

#endif // CUMULATIVE_LOGSUMEXP_TILING_DATA_H
