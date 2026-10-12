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
 * \file tabulate_fusion_tiling_data.h
 * \brief Tiling data struct for tabulate_fusion operator
 */

#ifndef TABULATE_FUSION_TILING_DATA_H_
#define TABULATE_FUSION_TILING_DATA_H_

struct TabulateFusionTilingData {
    int32_t needCoreNum = 0;   // 实际启动核数
    int32_t nloc = 0;          // 原子数（em.shape[0]）
    int32_t nnei = 0;          // 邻居数（em.shape[1]）
    int32_t lastLayerSize = 0; // 输出最后一维大小（ATTR last_layer_size）
    int32_t lastSizeAlign = 0; // 64 对齐大小 = ceil(lastLayerSize/64)*64
    int32_t tableRowSize = 0;  // table 每行元素数 = lastSizeAlign * 6
    int32_t tableRows = 0;     // table 行数 = table.shape[0]（用于 tableIdx 越界保护）
};

#endif // TABULATE_FUSION_TILING_DATA_H_
