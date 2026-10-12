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
 * \file Pad_v2.h
 * \brief
 */
#ifndef PADV2_H
#define PADV2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "pad_v2_tiling_data.h"
#include "pad_v2_tiling_key.h"

namespace NsPadV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class PadV2 {
public:
    __aicore__ inline PadV2(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, const PadV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress, int32_t row);
    __aicore__ inline void CopyOut(int32_t progress, int32_t row);
    __aicore__ inline void Compute(int32_t progress, int32_t row);
    __aicore__ inline int32_t Ispad(int32_t row);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> zGm;
    AscendC::DataCopyPadExtParams<T> padParams;
    AscendC::DataCopyExtParams copyParams;
    AscendC::DataCopyExtParams copyParams1;

    uint32_t rowsss = 0;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tailNum;
    uint32_t processNum;
    uint32_t tileLength;
    uint32_t lastdimnum;
    uint32_t dimNum;
    int32_t* dimarrz; // 获取通用数据类型数组变量arrSample
    PadV2TilingData tiling;
};

template <typename T>
__aicore__ inline void PadV2<T>::Init(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, const PadV2TilingData* tilingData)
{
    this->tiling = *tilingData;
    dimNum = tiling.dimNum;
    int32_t* dimarrz = tiling.dimarrz;
    lastdimnum = dimarrz[dimNum - 1];
    if (AscendC::GetBlockIdx() < tiling.big_core_num) {
        this->blockLength = tiling.big_tile_length;
        this->tileNum = tiling.big_tile_times;
        this->tailNum = tiling.big_tail_num;
    } else {
        this->blockLength = tiling.small_tile_length;
        this->tileNum = tiling.small_tile_times;
        this->tailNum = tiling.small_tail_num;
    }
    if (AscendC::GetBlockIdx() < tiling.big_core_num) {
        xGm.SetGlobalBuffer((__gm__ T*)x, tiling.sumspace);
        zGm.SetGlobalBuffer(
            (__gm__ T*)z + this->blockLength * AscendC::GetBlockIdx() * lastdimnum, this->blockLength * lastdimnum);
    } else {
        xGm.SetGlobalBuffer((__gm__ T*)x, tiling.sumspace);
        zGm.SetGlobalBuffer(
            (__gm__ T*)z + (this->blockLength * AscendC::GetBlockIdx() + tiling.big_core_num) * lastdimnum,
            this->blockLength * lastdimnum);
    }
    pipe.InitBuffer(inQueueX, BUFFER_NUM, tiling.xlastdim * sizeof(T));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, lastdimnum * sizeof(T));
}
template <typename T>
__aicore__ inline int32_t PadV2<T>::Ispad(int32_t row)
{
    int32_t* bias = tiling.bias;
    int32_t* orign_bias = tiling.orign_bias;
    int32_t* pad = tiling.pad;
    int32_t* dimarr = tiling.dimarr;
    int32_t* idx = tiling.idx;
    int32_t orig_now_row = 0;
    int tmp = row;
    // 1) 拆 pad 后坐标
    for (int d = 0; d < tiling.dimNum - 1; ++d) {
        idx[d] = tmp / bias[d];
        tmp = tmp % bias[d];
    }
    // 2) 判断是否有效（只用 pad_left）
    bool valid = true;
    for (int d = 0; d < tiling.dimNum - 1; ++d) {
        int pad_left = pad[2 * d];
        if (idx[d] < pad_left || idx[d] >= pad_left + dimarr[d]) {
            valid = false;
            break;
        }
    }
    if (!valid) {
        return -1;
    } else {
        // 3) 映射回原始行号
        int orig_row = 0;
        for (int d = 0; d < tiling.dimNum - 1; ++d) {
            int pad_left = pad[2 * d];
            int orig_idx = idx[d] - pad_left;
            orig_row += orig_idx * orign_bias[d];
        }
        return orig_row;
    }
}
template <typename T>
__aicore__ inline void PadV2<T>::CopyIn(int32_t progress, int32_t row)
{
    if (AscendC::GetBlockIdx() < tiling.big_core_num) {
        rowsss = this->blockLength * AscendC::GetBlockIdx();
    } else {
        rowsss = this->blockLength * AscendC::GetBlockIdx() + tiling.big_core_num;
    }
    int32_t now_row = rowsss + progress * tiling.core_tile_x1 + row;
    now_row = Ispad(now_row);
    if (now_row != -1) {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(tiling.xlastdim * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[now_row * tiling.xlastdim], copyParams, padParams);
        AscendC::DumpTensor(xLocal, 16, 16);
        inQueueX.EnQue(xLocal);
    }
}

template <typename T>
__aicore__ inline void PadV2<T>::CopyOut(int32_t progress, int32_t row)
{
    AscendC::LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
    AscendC::DataCopyExtParams copyParams1{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T) * lastdimnum), 0, 0,
        0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
    AscendC::DataCopyPad(zGm[progress * lastdimnum * tiling.core_tile_x1 + lastdimnum * row], zLocal, copyParams1);
    outQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void PadV2<T>::Compute(int32_t progress, int32_t row)
{
    if (AscendC::GetBlockIdx() < tiling.big_core_num) {
        rowsss = this->blockLength * AscendC::GetBlockIdx();
    } else {
        rowsss = this->blockLength * AscendC::GetBlockIdx() + tiling.big_core_num;
    }
    int32_t now_row = rowsss + progress * tiling.core_tile_x1 + row;
    now_row = Ispad(now_row);
    if (now_row == -1) {
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        AscendC::Duplicate<T>(zLocal, tiling.value, lastdimnum);
        outQueueZ.EnQue<T>(zLocal);
    } else {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        AscendC::Duplicate<T>(zLocal, tiling.value, lastdimnum);
        for (int i = 0; i < tiling.xlastdim; i++) {
            zLocal.SetValue(i + tiling.lpad, xLocal.GetValue(i));
        }
        outQueueZ.EnQue<T>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
}

template <typename T>
__aicore__ inline void PadV2<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processNum = tiling.core_tile_x1;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        for (int32_t j = 0; j < this->processNum; j++) {
            CopyIn(i, j);
            Compute(i, j);
            CopyOut(i, j);
        }
    }
    this->processNum = this->tailNum;
    for (int32_t j = 0; j < this->processNum; j++) {
        CopyIn(loopCount - 1, j);
        Compute(loopCount - 1, j);
        CopyOut(loopCount - 1, j);
    }
}
} // namespace NsPadV2
#endif // PadV2_H