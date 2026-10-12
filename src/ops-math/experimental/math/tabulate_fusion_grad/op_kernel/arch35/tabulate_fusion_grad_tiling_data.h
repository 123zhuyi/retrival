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
 * \file tabulate_fusion_grad_tiling_data.h
 * \brief Tiling data struct for tabulate_fusion_grad operator
 *
 * TilingData 不包含 threadNum（VF 线程数是 kernel 侧编译期常量）。
 * TilingData 不包含 table_info 的值（无值依赖，kernel 内从 GM 读取）。
 */

#ifndef TABULATE_FUSION_GRAD_TILING_DATA_H_
#define TABULATE_FUSION_GRAD_TILING_DATA_H_

struct TabulateFusionGradTilingData {
    int32_t needCoreNum = 0;    // 实际启动核数
    int32_t perCoreNloc = 0;    // 每核处理的 loc 数
    int32_t nloc = 0;           // 总 loc 数（em.shape[0]）
    int32_t nnei = 0;           // 邻居数（em.shape[1]）
    int32_t lastLayerSize = 0;  // 末层大小（descriptor.shape[2]）
    int32_t sizeAlign64 = 0;    // size 维 64 对齐长度 = ceil(lastLayerSize/64)*64
    int32_t tableDim0 = 0;      // table 行数（table.shape[0]）
    int32_t locStartOffset = 0; // loc 起始偏移（split_count=2 时 Vector Core 用）
    int32_t splitCount = 1;     // 1=单核，2=双核并行
    int32_t splitIndex = 0;     // 0=AI Core，1=Vector Core
};

#endif // TABULATE_FUSION_GRAD_TILING_DATA_H_
