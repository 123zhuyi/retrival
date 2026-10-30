/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Tu Yuanhang <@TuYHAAAAAA>
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
 * \file concat_v2.h
 * \brief
 * */
#ifndef CONCATV2_H
#define CONCATV2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "concat_v2_tiling_data.h"
#include "concat_v2_tiling_key.h"

namespace NsConcatV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class ConcatV2 {
public:
    __aicore__ inline ConcatV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const ConcatV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress, int32_t row);
    __aicore__ inline void CopyOut(int32_t progress, int32_t row);
    __aicore__ inline void Compute(int32_t progress, int32_t row);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    AscendC::GlobalTensor<T> zGm;
    AscendC::DataCopyPadExtParams<T> padParams;
    AscendC::DataCopyPadExtParams<T> padParams0;
    AscendC::DataCopyExtParams copyParams;
    AscendC::DataCopyExtParams copyParams0;
    AscendC::DataCopyExtParams copyParams1;
    AscendC::DataCopyExtParams copyParams2;
    AscendC::DataCopyExtParams copyParams3;

    uint32_t partnum;
    uint32_t partnumx;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tailNum;
    uint32_t processNum;
    uint32_t tileLength;
    ConcatV2TilingData tiling;
};

template <typename T>
__aicore__ inline void ConcatV2<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const ConcatV2TilingData* tilingData)
{
    this->tiling = *tilingData;
    partnum = tiling.partnum;
    partnumx = tiling.partnumX;
    int32_t* startx = tiling.startX;
    int32_t* rowsx = tiling.rowsX;
    int32_t* starty = tiling.startY;
    int32_t* rowsy = tiling.rowsY;

    if (AscendC::GetBlockIdx() < tiling.sbig_core_num) {
        this->blockLength = tiling.sbig_tile_length;
        this->tileNum = tiling.sbig_tile_times;
        this->tailNum = tiling.sbig_tail_num;
    } else {
        this->blockLength = tiling.ssmall_tile_length;
        this->tileNum = tiling.ssmall_tile_times;
        this->tailNum = tiling.ssmall_tail_num;
    }

    if (AscendC::GetBlockIdx() < tiling.sbig_core_num) {
        // 判断逻辑 --> 就是用block去求要办的x和y的行数
        xGm.SetGlobalBuffer(
            (__gm__ T*)x + startx[AscendC::GetBlockIdx()] * tiling.x2, rowsx[AscendC::GetBlockIdx()] * tiling.x2);
        yGm.SetGlobalBuffer(
            (__gm__ T*)y + starty[AscendC::GetBlockIdx()] * tiling.y2, rowsy[AscendC::GetBlockIdx()] * tiling.y2);
        zGm.SetGlobalBuffer(
            (__gm__ T*)z + this->blockLength * AscendC::GetBlockIdx() * tiling.z2, this->blockLength * tiling.z2);
    } else {
        xGm.SetGlobalBuffer((__gm__ T*)x + startx[AscendC::GetBlockIdx()] * tiling.x2, this->blockLength * tiling.x2);
        yGm.SetGlobalBuffer((__gm__ T*)y + starty[AscendC::GetBlockIdx()] * tiling.y2, this->blockLength * tiling.y2);
        zGm.SetGlobalBuffer(
            (__gm__ T*)z + (this->blockLength * AscendC::GetBlockIdx() + tiling.sbig_core_num) * tiling.z2,
            this->blockLength * tiling.z2);
    }

    pipe.InitBuffer(inQueueX, BUFFER_NUM, tiling.core_tile_s1 * tiling.x2 * sizeof(T));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, tiling.core_tile_s1 * tiling.y2 * sizeof(T));
}

template <typename T>
__aicore__ inline void ConcatV2<T>::CopyIn(int32_t progress, int32_t row)
{
    if (tiling.d == tiling.dimNum - 1) {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        AscendC::PRINTF("");

        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(tiling.x2 * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[(progress * tiling.core_tile_s1 + row) * tiling.x2], copyParams, padParams);

        AscendC::DataCopyExtParams copyParams0{1, static_cast<uint32_t>(tiling.y2 * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams0{true, 0, 0, 0};
        AscendC::DataCopyPad(yLocal, yGm[(progress * tiling.core_tile_s1 + row) * tiling.y2], copyParams0, padParams0);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    } else {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        AscendC::PRINTF("");
        int inx = 0;
        int iny = 0;
        int rowsss = 0;
        // 判断是否搬完了x，就是要得到行数
        if (AscendC::GetBlockIdx() < tiling.sbig_core_num) {
            rowsss = this->blockLength * AscendC::GetBlockIdx();
        } else {
            rowsss = this->blockLength * AscendC::GetBlockIdx() + tiling.sbig_core_num;
        }
        int inpartnum = (rowsss + progress * tiling.core_tile_s1 + row) % (partnum);
        // 求部分内偏移
        for (int i = rowsss; i < rowsss + progress * tiling.core_tile_s1 + row; i++) {
            int inpartnum = i % (partnum);
            if (inpartnum < partnumx) {
                inx++;
            } else {
                iny++;
            }
        }
        if (inpartnum < partnumx) { //
            AscendC::DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(tiling.x2 * sizeof(T)), 0, 0,
                0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
            AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            AscendC::DataCopyPad(xLocal, xGm[inx * tiling.x2], copyParams, padParams);
        } else {
            AscendC::DataCopyExtParams copyParams0{
                1, static_cast<uint32_t>(tiling.y2 * sizeof(T)), 0, 0,
                0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
            AscendC::DataCopyPadExtParams<T> padParams0{true, 0, 0, 0};
            AscendC::DataCopyPad(yLocal, yGm[iny * tiling.y2], copyParams0, padParams0);
        }
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
}

template <typename T>
__aicore__ inline void ConcatV2<T>::CopyOut(int32_t progress, int32_t row)
{
    if (tiling.d == tiling.dimNum - 1) {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.DeQue<T>();

        AscendC::DataCopyExtParams copyParams1{
            static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T) * tiling.x2), 0, 0,
            0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        AscendC::DataCopyPad(zGm[progress * tiling.z2 * tiling.core_tile_s1 + tiling.z2 * row], xLocal, copyParams1);

        AscendC::DataCopyExtParams copyParams2{
            static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T) * tiling.y2), 0, 0,
            0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        AscendC::DataCopyPad(
            zGm[progress * tiling.z2 * tiling.core_tile_s1 + tiling.z2 * row + tiling.x2], yLocal, copyParams2);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    } else {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.DeQue<T>();

        int rowsss = 0;
        if (AscendC::GetBlockIdx() < tiling.sbig_core_num) {
            rowsss = this->blockLength * AscendC::GetBlockIdx();
        } else {
            rowsss = this->blockLength * AscendC::GetBlockIdx() + tiling.sbig_core_num;
        }
        int inpartnum = (rowsss + progress * tiling.core_tile_s1 + row) % (partnum);
        if (inpartnum < partnumx) { // 如果行数小于x1，则随便搬x
            AscendC::DataCopyExtParams copyParams1{
                static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T) * tiling.x2), 0, 0,
                0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
            AscendC::DataCopyPad(
                zGm[progress * tiling.z2 * tiling.core_tile_s1 + tiling.z2 * row], xLocal, copyParams1);
        } else {
            AscendC::DataCopyExtParams copyParams2{
                static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T) * tiling.y2), 0, 0,
                0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
            AscendC::DataCopyPad(
                zGm[progress * tiling.z2 * tiling.core_tile_s1 + tiling.z2 * row], yLocal, copyParams2);
        }
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
}

template <typename T>
__aicore__ inline void ConcatV2<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processNum = tiling.core_tile_s1;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        for (int32_t j = 0; j < this->processNum; j++) {
            CopyIn(i, j);
            CopyOut(i, j);
        }
    }
    this->processNum = this->tailNum;
    for (int32_t j = 0; j < this->processNum; j++) {
        CopyIn(loopCount - 1, j);
        CopyOut(loopCount - 1, j);
    }
}

} // namespace NsConcatV2
#endif // ConcatV2_H
