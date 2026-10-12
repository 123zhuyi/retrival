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
 * \file radix_top_k_struct.h
 * \brief Radix TopK TilingData 结构体定义
 */

#ifndef RADIX_TOPK_STRUCT_H
#define RADIX_TOPK_STRUCT_H

#pragma pack(push, 8)
/**
 * @brief Radix TopK Tiling 数据结构
 *        包含数据切分、核分配、tile 参数等编译期确定的常量信息
 */
struct RadixTopKTilingData {
    uint64_t coreNum;            /**< 总 core 数 */
    uint64_t formerCoreNum;      /**< 主核数量（处理 formerTileNum 个 tile） */
    uint64_t tailCoreNum;         /**< 尾核数量（处理 tailTileNum 个 tile） */
    uint64_t totalTileNum;        /**< 总 tile 数 */
    uint64_t formerTileNum;       /**< 主核 tile 数 */
    uint64_t tailTileNum;         /**< 尾核 tile 数 */
    uint64_t formerTileLen;    /**< 每个 tile 的数据长度（主核/尾核常规tile） */
    uint64_t tailTileLen;      /**< 最后一个 tail tile 的数据长度 */
    uint64_t batch;               /**< batch 数（除最后一维外各维度乘积） */
    uint64_t sortLen;       /**< 排序轴长度（输入最后一维长度） */
    uint64_t kValue;          /**< topK 值 */

    bool largest;              /**< 是否求最大 k 值（true: largest, false: smallest） */
    bool sorted;               /**< 输出结果是否排序（当前仅支持 false） */
    bool needWorkspace;        /**< 是否使用 workspace（false: 复用 indices 内存做暂存） */
};

#pragma pack(pop)

#endif