/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Zhi <@hitLeechi>
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
 * \file range.cpp
 * \brief
 */

#ifndef RANGE_H
#define RANGE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "range_tiling_data.h"
#include "range_tiling_key.h"
using namespace std;
constexpr uint32_t OUTPUT_DATA_TYPE_SIZE = 4;                          // 输出float的字节数
constexpr uint32_t BLOCK_SIZE = 32;                                    // 32B内存对齐基准
constexpr uint32_t TILE_ELEM_NUM = BLOCK_SIZE / OUTPUT_DATA_TYPE_SIZE; // float的32B对齐元素数
constexpr uint32_t BUFFER_NUM = 2;                                     // 双缓冲
namespace NsRange {
using namespace AscendC;
template <typename TStart, typename TEnd, typename TStep>
class Range {
public:
    __aicore__ inline Range() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR s, GM_ADDR z, uint32_t _totalLength, uint32_t _blockLength, uint32_t _tileNum,
        uint32_t _lastTileLength, uint32_t _formerLength, uint32_t _formerTileNum, uint32_t _formerLastTileLength,
        uint32_t _formerNum, uint32_t _tailLength, uint32_t _tailTileNum, uint32_t _tailLastTileLength,
        uint32_t _isEvenCore)
    {
        // 不论输入是什么类型，kernel侧始终使用float进行计算
        this->start = *reinterpret_cast<const __gm__ float*>(x);
        this->end = *reinterpret_cast<const __gm__ float*>(y);
        this->step = *reinterpret_cast<const __gm__ float*>(s);
        this->origSize = _totalLength;

        uint32_t coreId = AscendC::GetBlockIdx();
        uint32_t currentBlockLength = 0;

        if (_isEvenCore) {
            this->blockLength = _blockLength;
            this->tileNum = _tileNum;
            this->lastTileLength = _lastTileLength;
            this->blockOffset = blockLength * coreId;
            currentBlockLength = blockLength;
        } else {
            if (coreId < _formerNum) {
                this->blockLength = _formerLength;
                this->tileNum = _formerTileNum;
                this->lastTileLength = _formerLastTileLength;
                this->blockOffset = _formerLength * coreId;
                currentBlockLength = _formerLength;
            } else {
                this->blockLength = _tailLength;
                this->tileNum = _tailTileNum;
                this->lastTileLength = _tailLastTileLength;
                this->blockOffset = _formerLength * _formerNum + _tailLength * (coreId - _formerNum);
                currentBlockLength = _tailLength;
            }
        }
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(z) + this->blockOffset, currentBlockLength);
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_ELEM_NUM * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (origSize == 1) {
            AscendC::LocalTensor<float> tileOutput = outQueueZ.AllocTensor<float>();
            for (int32_t i = 0; i < TILE_ELEM_NUM; i++)
                tileOutput.SetValue(i, (i == 0) ? start : 0.0f);
            outQueueZ.EnQue(tileOutput);
            CopyOut(0);
            return;
        }
        for (int32_t tileIdx = 0; tileIdx < tileNum; tileIdx++) {
            Compute(tileIdx);
            CopyOut(tileIdx);
        }
    }

private:
    __aicore__ inline void CopyIn() {}

    __aicore__ inline void Compute(int32_t tileIdx)
    {
        AscendC::LocalTensor<float> tileOutput = outQueueZ.AllocTensor<float>();
        int32_t tileElemNum = TILE_ELEM_NUM; // 8
        int32_t tileGlobalStart = blockOffset + tileIdx * tileElemNum;

        for (int32_t i = 0; i < tileElemNum; i++) {
            int32_t globalIdx = tileGlobalStart + i;
            if (globalIdx < origSize)
                tileOutput.SetValue(i, start + step * static_cast<float>(globalIdx));
            else
                tileOutput.SetValue(i, 0.0f);
        }
        outQueueZ.EnQue(tileOutput);
    }

    __aicore__ inline void CopyOut(int32_t tileIdx)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        int32_t copyElemNum;
        if (tileIdx == tileNum - 1)
            copyElemNum = lastTileLength;
        else
            copyElemNum = TILE_ELEM_NUM;

        int32_t copyOffset = tileIdx * TILE_ELEM_NUM;
        if (copyOffset + copyElemNum > blockLength)
            copyElemNum = blockLength - copyOffset;

        AscendC::DataCopy(zGm[copyOffset], zLocal, copyElemNum);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> zGm;

    uint32_t blockLength;    // 当前核心处理的float总元素数
    uint32_t tileNum;        // 当前核心的Tile总数（每个Tile 8个float元素）
    uint32_t lastTileLength; // 最后一个Tile的float元素数
    uint32_t blockOffset;    // 当前核心的全局float元素偏移
    uint32_t origSize;       // 原始有效元素数
    float start;
    float end;
    float step;
};

} // namespace NsRange
#endif // RANGE_H
