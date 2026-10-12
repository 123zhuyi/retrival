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
 * \file cdist_brc_tilingdata.h
 * \brief POD tiling struct for the M==1 Broadcast fast path (方案2: 静态 rank UB-BRC + OneDim).
 *        y[b,p,r] = |x1[b,p] - x2[b,r]| (M==1: p-norm degenerates to |d|), p!=1 normalizes on
 *        the single element in scalar. Layout after batch-merge: out (B,P,R), x1 (B,P), x2 (B,R).
 *
 *  Branch selection (brcBranch):
 *    0 = OneDim  : P==1 && R==1  -> output collapses to 1D length B; elementwise |x1-x2| then p-norm.
 *    1 = UB-BRC  : P>1 || R>1    -> per-batch (rowsP, R) tile, static-rank(=2) Broadcast on x1/x2.
 */
#ifndef CDIST_BRC_TILINGDATA_H
#define CDIST_BRC_TILINGDATA_H

#include <cstdint>

struct CdistBrcTilingData {
    int64_t B = 0;  // merged batch
    int64_t P = 0;  // x1 rows
    int64_t R = 0;  // x2 rows
    float p = 1.0f; // norm order (M==1 => |d| for p in {1,2,inf}; 0 => min(ceil,1); else exp(ln|d|))

    int32_t brcBranch = 1; // 0=OneDim, 1=UB-BRC, 2=UB-BRC multi-batch (small P*R tile)

    // ---- OneDim (brcBranch==0) : 1D length = B ----
    int64_t dimLen = 0;   // = B (P==R==1)
    int32_t tileNum = 0;  // elements per UB block (aligned)
    int32_t blockNum = 0; // active cores

    // ---- UB-BRC (brcBranch==1) ----
    // Tiling unit = (rowsP within one batch) x R plane. Static in-tile broadcast rank = 2.
    int64_t elemNum = 0;      // per-buffer element capacity (aligned)
    int64_t ubFormerP = 0;    // rows of P per main tile (rowsP)
    int64_t ubOuterP = 0;     // number of P-tiles per batch = ceil(P/ubFormerP)
    int64_t ubTailP = 0;      // rows in the last P-tile
    int64_t rSeg = 0;         // R segment per tile ( == R when R fits UB; else <=elemNum split )
    int64_t rOuter = 0;       // number of R segments = ceil(R/rSeg)
    int64_t rTail = 0;        // last R segment length
    int64_t totalTiles = 0;   // = B * ubOuterP * rOuter (brcBranch==1) / = ceil(B/batchPerTile) (brcBranch==2)
    int64_t blockFormerT = 0; // tiles per main core
    int64_t coreNumT = 0;     // active cores for UB-BRC
    int64_t blockTailT = 0;   // tiles on last core

    // ---- UB-BRC multi-batch (brcBranch==2): P*R small, pack batchPerTile full batches per UB tile ----
    // 缺陷② 修复：小 tile(P*R 小)场景下按 batch 打包，把 30896 个 256 元素小 tile 合并成少量大 tile，
    // 大幅减少 per-tile DataCopyPad/Broadcast/EnQue 固定开销（181: 525us→接近方案1）。
    int64_t batchPerTile = 0; // batches packed into one UB tile (G)
};
#endif // CDIST_BRC_TILINGDATA_H
