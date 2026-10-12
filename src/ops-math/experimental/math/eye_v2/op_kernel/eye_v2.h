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
 * \file eye_v2.h
 * \brief
 */
#ifndef EYE_V2_H
#define EYE_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "eye_v2_tiling_data.h"
#include "eye_v2_tiling_key.h"

namespace NsEyeV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class EyeV2 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const EyeV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyOut();
    __aicore__ inline void Compute();

private:
    TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<T> yGm;

    EyeV2TilingData tiling;
    uint64_t rowLength;       // 矩阵行数
    uint64_t columnLength;    // 矩阵列数
    uint64_t diagLen;         // 对角线长度
    uint64_t fullBlockLength; // 每个核处理的对角线长度（向上取整）
    uint64_t tailBlockLength; // 每个核处理的对角线长度（向下取整）
    uint64_t fullBlockNum;    // 处理大核的核数
    uint64_t tailBlockNum;    // 处理小核的核数
    uint64_t typeSize;        // 每个元素的字节数
    uint64_t matrixOrder;     // 当前核处理的矩阵阶数
    uint64_t blockIdx;        // 当前核索引
};

template <typename T>
__aicore__ inline void EyeV2<T>::Init(GM_ADDR x, GM_ADDR y, const EyeV2TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    this->tiling = *tilingData;
    this->blockIdx = AscendC::GetBlockIdx();
    matrixOrder =
        (this->blockIdx >= tilingData->fullBlockNum) ? tilingData->tailBlockLength : tilingData->fullBlockLength;
    if (matrixOrder == 0) {
        return;
    }
    yGm.SetGlobalBuffer((__gm__ T*)y, tilingData->rowLength * tilingData->columnLength);
    uint32_t bufSize = matrixOrder * sizeof(T);
    uint32_t alignSize = 32;
    uint32_t allocSize = (bufSize + alignSize - 1) / alignSize * alignSize;
    pipe.InitBuffer(outQueueY, BUFFER_NUM, allocSize);
}

template <typename T>
__aicore__ inline void EyeV2<T>::CopyOut()
{
    AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    if (matrixOrder == 0) {
        outQueueY.FreeTensor(yLocal);
        return;
    }
    uint64_t idx = this->blockIdx;
    uint64_t diagStart = (idx < tiling.fullBlockNum) ?
                             static_cast<uint64_t>(idx) * tiling.fullBlockLength :
                             static_cast<uint64_t>(tiling.fullBlockNum) * tiling.fullBlockLength +
                                 static_cast<uint64_t>(idx - tiling.fullBlockNum) * tiling.tailBlockLength;
    uint64_t globalOffset = diagStart * tiling.columnLength + (static_cast<int64_t>(diagStart));
    AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    for (uint64_t i = 0; i < matrixOrder; ++i) {
        uint64_t dst = globalOffset + static_cast<uint64_t>(i * (tiling.columnLength + 1));
        AscendC::DataCopyPad(yGm[dst], yLocal, copyParams);
    }
    outQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void EyeV2<T>::Compute()
{
    AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    AscendC::Duplicate(yLocal, static_cast<T>(1), matrixOrder);
    outQueueY.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void EyeV2<T>::Process()
{
    if (matrixOrder > 0) {
        Compute();
        CopyOut();
    }
}

} // namespace NsEyeV2
#endif // EyeV2_H