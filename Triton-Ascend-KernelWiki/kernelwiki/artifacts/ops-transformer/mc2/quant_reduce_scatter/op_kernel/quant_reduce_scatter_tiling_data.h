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
 * \file quant_reduce_scatter_tiling_data.h
 * \brief 定义TilingData
 */

#ifndef QUANT_REDUCE_SCATTER_TILING_DATA_H
#define QUANT_REDUCE_SCATTER_TILING_DATA_H

#include <kernel_tiling/kernel_tiling.h>

struct QuantReduceScatterTilingInfo {
    uint64_t bs;
    uint64_t hiddenSize;
    uint64_t scaleHiddenSize;
    uint64_t aivNum;
    uint64_t totalWinSize;   // Win区总大小，即HCCL_BUFFER_SIZE
};

struct QuantReduceScatterTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    QuantReduceScatterTilingInfo quantReduceScatterTilingInfo;
};

#endif // QUANT_REDUCE_SCATTER_TILING_DATA_H