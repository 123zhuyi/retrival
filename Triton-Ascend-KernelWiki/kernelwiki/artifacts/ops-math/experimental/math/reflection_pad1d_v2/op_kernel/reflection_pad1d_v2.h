/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
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
 * \file reflection_pad1d_v2.h
 * \brief
 */
#ifndef REFLECTION_PAD1D_V2_H
#define REFLECTION_PAD1D_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "reflection_pad1d_v2_tiling_data.h"
#include "reflection_pad1d_v2_tiling_key.h"

namespace NsReflectionPad1dV2 {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;

template <typename T>
class ReflectionPad1dV2 {
public:
    __aicore__ inline ReflectionPad1dV2() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReflectionPad1dV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const ReflectionPad1dV2TilingData* tilingData);
    __aicore__ inline void CopyIn(uint32_t inGmOffset);
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut(uint32_t outGmOffset);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    uint32_t left = 0;      // 左填充量
    uint32_t right = 0;     // 右填充量
    uint32_t wSize = 0;     // 目标W维度大小
    uint32_t outWSize = 0;  // 输出W维度大小
    uint32_t alignWSize;    // 目标维度32字节对齐后大小
    uint32_t alignOutWSize; // 输出目标维度对齐后大小

    uint32_t startNC = 0;  // 起始N×C索引
    uint32_t endNC = 0;    // 结束N×C索引
    uint32_t blockIdx = 0; // 当前核索引
};

template <typename T>
__aicore__ inline void ReflectionPad1dV2<T>::Init(GM_ADDR x, GM_ADDR y, const ReflectionPad1dV2TilingData* tilingData)
{
    ParseTilingData(tilingData);

    // 绑定GM内存
    const uint32_t currentTaskCount = this->endNC - this->startNC;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + this->startNC * this->wSize, currentTaskCount * this->wSize);
    yGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(y) + this->startNC * this->outWSize, currentTaskCount * this->outWSize);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->alignWSize * sizeof(T));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->outWSize * sizeof(T));
}

template <typename T>
__aicore__ inline void ReflectionPad1dV2<T>::Process()
{
    for (uint32_t nc = startNC; nc < endNC; ++nc) {
        uint32_t localNcIdx = nc - startNC;
        uint32_t inGmOffset = localNcIdx * wSize;
        uint32_t outGmOffset = localNcIdx * outWSize;

        CopyIn(inGmOffset);
        Compute();
        CopyOut(outGmOffset);
    }
}

template <typename T>
__aicore__ inline void ReflectionPad1dV2<T>::ParseTilingData(const ReflectionPad1dV2TilingData* tilingData)
{
    this->left = tilingData->padLeft;
    this->right = tilingData->padRight;
    this->wSize = tilingData->wSize;
    this->outWSize = this->wSize + this->left + this->right;
    this->alignWSize = tilingData->alignWSize;
    this->blockIdx = GetBlockIdx();

    if (this->blockIdx < tilingData->tailNC) {
        this->startNC = this->blockIdx * (tilingData->ncPerCore + 1);
        this->endNC = this->startNC + (tilingData->ncPerCore + 1);
    } else {
        this->startNC = tilingData->tailNC * (tilingData->ncPerCore + 1) +
                        (this->blockIdx - tilingData->tailNC) * tilingData->ncPerCore;
        this->endNC = this->startNC + tilingData->ncPerCore;
    }
}

template <typename T>
__aicore__ inline void ReflectionPad1dV2<T>::CopyIn(uint32_t inGmOffset)
{
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint16_t>(this->wSize * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(this->alignWSize - this->wSize), 0};
    DataCopyPad(xLocal, xGm[inGmOffset], copyParams, padParams);
    inQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void ReflectionPad1dV2<T>::Compute()
{
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

    // 左填充
    for (int32_t i = 0; i < left; ++i) {
        int32_t refIdx = left - i;
        refIdx = refIdx < 0 ? 0 : refIdx;
        refIdx = Std::max(0, Std::min(refIdx, static_cast<int32_t>(wSize - 1)));
        T val = xLocal.GetValue(static_cast<uint32_t>(refIdx));
        yLocal.SetValue(static_cast<uint32_t>(i), val);
    }

    // 中间原始数据
    for (int32_t i = 0; i < wSize; ++i) {
        T val = xLocal.GetValue(static_cast<uint32_t>(i));
        yLocal.SetValue(static_cast<uint32_t>(left + i), val);
    }

    // 右填充
    for (int32_t i = 0; i < right; ++i) {
        int32_t refIdx = wSize - 2 - i;
        refIdx = refIdx < 0 ? 0 : refIdx;
        refIdx = Std::max(0, Std::min(refIdx, static_cast<int32_t>(wSize - 1)));
        T val = xLocal.GetValue(static_cast<uint32_t>(refIdx));
        yLocal.SetValue(static_cast<uint32_t>(left + wSize + i), val);
    }

    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void ReflectionPad1dV2<T>::CopyOut(uint32_t outGmOffset)
{
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint16_t>(this->outWSize * sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm[outGmOffset], yLocal, copyParams);
    outQueueY.FreeTensor(yLocal);
}

} // namespace NsReflectionPad1dV2
#endif // REFLECTION_PAD1D_V2_H