/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liu Jun <@kbryantttt>
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
 * \file strided_slice.h
 * \brief
 */
#ifndef __STRIDED_SLICE_H__
#define __STRIDED_SLICE_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "strided_slice_tiling_data.h"
#include "strided_slice_tiling_key.h"

namespace NsStridedSlice {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t MAX_TILE_NUM = 64;
constexpr uint32_t MAX_TILE_OUT_ROWS = 64;

template <typename T>
class StridedSlice {
public:
    __aicore__ inline StridedSlice(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, const StridedSliceTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t tile_idx);
    __aicore__ inline void CopyOut(int32_t tile_idx);
    __aicore__ inline uint32_t CalcCoreOutRows(uint32_t core_start, uint32_t core_end);
    __aicore__ inline uint32_t CalcPreCoreOutTotal(uint32_t current_core_id, uint32_t total_core_num);
    __aicore__ inline void InitTileArrays();

private:
    AscendC::TPipe pipe;
    AscendC::TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> inQueueX;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> zGm;
    uint32_t tile_start_rows[MAX_TILE_NUM];
    uint32_t tile_end_rows[MAX_TILE_NUM];
    uint32_t tile_out_rows[MAX_TILE_NUM];
    uint32_t tile_out_start_indices[MAX_TILE_NUM];
    uint32_t tile_in_row_offsets[MAX_TILE_NUM][MAX_TILE_OUT_ROWS];
    uint32_t cols;
    uint32_t rows;
    uint32_t tileRows;
    uint32_t baseRowsPerCore;
    uint32_t bigTotalRows;
    uint32_t tailRows;
    uint32_t tileTailRows;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t start1;
    uint32_t end1;
    uint32_t stride1;
    uint32_t start2;
    uint32_t end2;
    uint32_t stride2;
    uint32_t outNum_cols;
    uint32_t core_start_row;
    uint32_t core_last_row;
};

template <typename T>
__aicore__ inline void StridedSlice<T>::Init(GM_ADDR x, GM_ADDR z, const StridedSliceTilingData* tilingData)
{
    uint32_t coreId = AscendC::GetBlockIdx();
    uint32_t totalCoreNum = AscendC::GetBlockNum();
    this->cols = tilingData->cols;
    this->rows = tilingData->rows;
    this->tileRows = tilingData->tileRows;
    this->baseRowsPerCore = tilingData->baseRowsPerCore;
    this->bigTotalRows = tilingData->bigTotalRows;
    this->tailRows = tilingData->tailRows;
    this->start1 = tilingData->start1;
    this->end1 = tilingData->end1;
    this->stride1 = tilingData->stride1;
    this->start2 = tilingData->start2;
    this->end2 = tilingData->end2;
    this->stride2 = tilingData->stride2;
    this->outNum_cols = ((this->end2 - this->start2 + this->stride2 - 1) / this->stride2);
    uint32_t globalBufferIndex = bigTotalRows * this->cols * coreId;
    this->tileDataNum = tileRows * this->cols;
    uint32_t coreRows;
    if (coreId < tailRows) {
        coreRows = bigTotalRows;
        this->tileNum = tilingData->finalBigTileNum;
        this->core_start_row = coreId * bigTotalRows;
        this->tileTailRows = tilingData->bigTailRows;
    } else {
        coreRows = baseRowsPerCore;
        this->tileNum = tilingData->finalSmallTileNum;
        this->core_start_row = tailRows * bigTotalRows + (coreId - tailRows) * baseRowsPerCore;
        globalBufferIndex -= (bigTotalRows - baseRowsPerCore) * this->cols * (coreId - tailRows);
        this->tileTailRows = tilingData->smallTailRows;
    }
    this->core_last_row = this->core_start_row + coreRows - 1;
    uint32_t coreOutRows = CalcCoreOutRows(core_start_row, core_last_row);
    uint32_t core_out_count = coreOutRows * this->outNum_cols;
    uint32_t core_out_offset = CalcPreCoreOutTotal(coreId, totalCoreNum);
    InitTileArrays();

    xGm.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, coreRows * this->cols);
    zGm.SetGlobalBuffer((__gm__ T*)z + core_out_offset, core_out_count);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * (32 / sizeof(T)) * sizeof(T));
}

template <typename T>
__aicore__ inline void StridedSlice<T>::CopyIn(int32_t tile_idx)
{
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    AscendC::DataCopyExtParams copyParams{
        static_cast<uint16_t>(this->outNum_cols), static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>((this->stride2 - 1) * sizeof(T)), 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>((32 / sizeof(T)) - 1), 0};
    for (int i = 0; i < this->tileRows; i++) {
        AscendC::DataCopyPad(
            xLocal[i * this->outNum_cols * (32 / sizeof(T))],
            xGm[tile_idx * this->tileDataNum + i * this->cols + this->start2], copyParams, padParams);
    }
    inQueueX.EnQue(xLocal);
    AscendC::SyncAll();
}

template <typename T>
__aicore__ inline void StridedSlice<T>::CopyOut(int32_t tile_idx)
{
    uint32_t tile_out_row_cnt = tile_out_rows[tile_idx];
    AscendC::SyncAll();
    uint32_t tile_out_start = tile_out_start_indices[tile_idx];
    AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    if (tile_out_row_cnt == 0) {
        inQueueX.FreeTensor(xLocal);
        return;
    }
    AscendC::DataCopyExtParams copyParams{
        static_cast<uint16_t>(this->outNum_cols), static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    for (uint32_t out_row_idx = 0; out_row_idx < tile_out_row_cnt; out_row_idx++) {
        uint32_t local_in_offset = tile_in_row_offsets[tile_idx][out_row_idx] * this->outNum_cols * (32 / sizeof(T));
        uint32_t core_global_out_row = tile_out_start + out_row_idx;
        uint32_t global_out_offset = core_global_out_row * this->outNum_cols;
        AscendC::DataCopyPad(zGm[global_out_offset], xLocal[local_in_offset], copyParams);
    }
    inQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline uint32_t StridedSlice<T>::CalcCoreOutRows(uint32_t core_start, uint32_t core_end)
{
    const uint32_t slice_start = this->start1;
    const uint32_t slice_end = this->end1 - 1;
    if (core_start > slice_end || core_end < slice_start) {
        return 0;
    }
    uint32_t clip_start = AscendC::Std::max(core_start, slice_start);
    uint32_t clip_end = AscendC::Std::min(core_end, slice_end);
    uint32_t k_first = (clip_start - slice_start + this->stride1 - 1) / this->stride1;
    uint32_t k_last = (clip_end - slice_start) / this->stride1;
    return k_last - k_first + 1;
}

template <typename T>
__aicore__ inline uint32_t StridedSlice<T>::CalcPreCoreOutTotal(uint32_t current_core_id, uint32_t total_core_num)
{
    uint32_t pre_total = 0;
    for (uint32_t pre_core_id = 0; pre_core_id < current_core_id; pre_core_id++) {
        uint32_t pre_core_rows = (pre_core_id < tailRows) ? bigTotalRows : baseRowsPerCore;
        uint32_t pre_core_start = (pre_core_id < tailRows) ?
                                      (pre_core_id * bigTotalRows) :
                                      (tailRows * bigTotalRows + (pre_core_id - tailRows) * baseRowsPerCore);
        uint32_t pre_core_end = pre_core_start + pre_core_rows - 1;
        uint32_t pre_core_out_rows = CalcCoreOutRows(pre_core_start, pre_core_end);
        pre_total += pre_core_out_rows * this->outNum_cols;
    }
    return pre_total;
}

template <typename T>
__aicore__ inline void StridedSlice<T>::InitTileArrays(/*参数列表*/)
{
    uint32_t current_core_row = this->core_start_row;
    uint32_t core_global_out_idx = 0;
    for (uint32_t tile_idx = 0; tile_idx < this->tileNum; tile_idx++) {
        uint32_t tile_rows = (tile_idx == this->tileNum - 1) ? this->tileTailRows : this->tileRows;
        tile_start_rows[tile_idx] = current_core_row;
        tile_end_rows[tile_idx] = current_core_row + tile_rows - 1;
        tile_end_rows[tile_idx] = AscendC::Std::min(tile_end_rows[tile_idx], this->core_last_row);
        const uint32_t slice_start = this->start1;
        const uint32_t slice_end = this->end1 - 1;
        const uint32_t tile_clip_start = AscendC::Std::max(tile_start_rows[tile_idx], slice_start);
        const uint32_t tile_clip_end = AscendC::Std::min(tile_end_rows[tile_idx], slice_end);
        if (tile_clip_start > tile_clip_end || this->stride1 == 0) {
            tile_out_rows[tile_idx] = 0;
            tile_out_start_indices[tile_idx] = core_global_out_idx;
            current_core_row = tile_end_rows[tile_idx] + 1;
            continue;
        }
        const uint32_t offset_from_slice_start = tile_clip_start - slice_start;
        const uint32_t k_first = (offset_from_slice_start + this->stride1 - 1) / this->stride1;
        const uint32_t k_last = (tile_clip_end - slice_start) / this->stride1;
        tile_out_rows[tile_idx] = k_last - k_first + 1;
        tile_out_start_indices[tile_idx] = core_global_out_idx;
        for (uint32_t out_row_idx = 0; out_row_idx < tile_out_rows[tile_idx]; out_row_idx++) {
            uint32_t global_in_row = slice_start + (k_first + out_row_idx) * this->stride1;
            tile_in_row_offsets[tile_idx][out_row_idx] = global_in_row - tile_start_rows[tile_idx];
        }
        core_global_out_idx += tile_out_rows[tile_idx];
        current_core_row = tile_end_rows[tile_idx] + 1;
    }
}
template <typename T>
__aicore__ inline void StridedSlice<T>::Process()
{
    int32_t loopCount = this->tileNum;
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);
        CopyOut(i);
    }
}

} // namespace NsStridedSlice
#endif // STRIDED_SLICE_H