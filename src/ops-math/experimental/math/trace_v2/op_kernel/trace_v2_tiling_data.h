/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua<@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file trace_v2_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _ROTARY_POSITION_EMBEDDING_GRAD_TILING_DATA_H_
#define _ROTARY_POSITION_EMBEDDING_GRAD_TILING_DATA_H_

struct TraceV2TilingData {
    uint64_t alignNum;        // 对齐数
    uint64_t typeSize;        // 数据类型大小
    uint64_t matrixOrder;     // 矩阵存储顺序 0-行优先 1-列优先
    uint64_t rowLength;       // 矩阵行数
    uint64_t columnLength;    // 矩阵列数
    uint64_t diagLen;         // 对角线长度
    uint64_t fullBlockLength; // 每个核处理的对角线长度（向上取整）
    uint64_t tailBlockLength; // 每个核处理的对角线长度（向下取整）
    uint64_t fullBlockNum;    // 处理满块的核数量
    uint64_t tailBlockNum;    // 处理尾块的核数量
};
#endif
