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
 * \file arange_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __ARANGE_TILLING_DATA_H__
#define __ARANGE_TILLING_DATA_H__

struct ArangeTilingData {
    uint32_t dtypeSize; // 目标 dtype 字节数（1/2/4/8）
    int64_t totalNum;   // 总元素数 N（= out shape size），放宽 int64 防 N 接近 2³² 时溢出
    uint32_t unitNum;   // 单次 UB API 计算元素数（全核共用一个值）
    // —— 多核切分字段（former/tail 模型）——
    uint32_t coreNum;     // 实际使用核数（SetBlockDim 值）
    uint32_t formerNum;   // 处理 formerLength 的核个数（前 formerNum 个核多分 1 个 32B 块）
    int64_t formerLength; // 前段核每核元素数（32B 对齐），放宽 int64（全局偏移链）
    int64_t tailLength;   // 后段核每核元素数（32B 对齐），放宽 int64（全局偏移链）
    // —— 每核内 UB 子循环（former/tail 各一套，因每核元素数不同）——
    uint32_t formerUnitLoops; // former 核的 UB 循环次数
    uint32_t formerTailNum;   // former 核最后一个 UB 块的元素数
    uint32_t tailUnitLoops;   // tail 核的 UB 循环次数
    uint32_t tailTailNum;     // tail 核最后一个 UB 块的元素数
    // 扩展其他tilling参数
};
#endif
