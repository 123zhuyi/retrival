/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "rope_matrix_cube.h"

namespace npu_ops_transformer_ext {
namespace RopeMatrix {

constexpr int32_t DOUBLE_BUFFER = 2;

using AscendC::GetBlockIdx;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::WaitFlag;
using AscendC::DataCopy;
using AscendC::Add;
using AscendC::Mul;
using AscendC::Cast;
using AscendC::HardEvent;
using AscendC::PipeBarrier; // <PIPE_V>();

__aicore__ inline void CopyTiling(TCubeTiling *tiling, RopeMatrixTiling *ropeTiling, 
                                    TCubeTiling *mmTilingDataGm, RopeMatrixTiling *ropeTilingGm)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    uint32_t *ropePtr = reinterpret_cast<uint32_t *>(ropeTiling);
    auto mmtiling32 = reinterpret_cast<uint32_t *>(mmTilingDataGm);
    auto vectiling32 = reinterpret_cast<uint32_t *>(ropeTilingGm);

    uint32_t i = 0;
    for (; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(mmtiling32 + i);
    }

    for (uint32_t j = 0; j < sizeof(RopeMatrixTiling) / sizeof(uint32_t); ropePtr++, j++) {
        *ropePtr = *(vectiling32 + j);
    }
    return;
}

template <typename T>
class RopeVec{
public :
    __aicore__ inline RopeVec(){}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR xNew, GM_ADDR sin, GM_ADDR cos, GM_ADDR out,
                                RopeMatrixTiling *ropeTiling, TPipe *tpipe)
    {
        this->aivIdx = GetBlockIdx();
        this->aivNum = GetBlockNum();
        this->subIdx = GetSubBlockIdx();
        this->subNum = GetSubBlockNum();
        // doc say aivNum has already mul subNum, by we test are not, thus need this
        this->aivNum *= this->subNum;

        // init bsnd
        this->CopyTiling(ropeTiling);
        this->bnSize = this->b * this->n;
        // init pipe
        this->pipe = tpipe;
        // set ub size
        uint32_t sinSize = 10 * 1024;
        uint32_t sinSize32 = 20 * 1024;

        // vector tiling
        // global
        this->sdSizeTotal = this->seqLen * this->d;
        // this core only
        this->sdBlockSize = sinSize / sizeof(T);
        this->sBlockSize = this->sdBlockSize / this->d;

        this->sSize = this->CeilDiv(this->seqLen, this->aivNum);
        this->sdSize = this->sSize * this->d;

        uint32_t remainSize = this->seqLen - this->sSize * this->aivIdx;
        this->startSize = this->sSize * this->aivIdx;
        // update sSize for last core, need after this->startSize set
        this->sSize = (this->sSize < remainSize) ? (this->sSize) : (remainSize);
        this->sBlockNum = this->CeilDiv(sSize, this->sBlockSize);
        this->sLastBlockSize = this->sSize % this->sBlockSize;
        this->sLastBlockSize = (this->sLastBlockSize == 0) ? this->sBlockSize : this->sLastBlockSize;
        this->sdLastBlockSize = this->sLastBlockSize * this->d;

        uint32_t startNumel = this->startSize * this->d;
        xGm.SetGlobalBuffer((__gm__ T*) x + startNumel);
        xNewGm.SetGlobalBuffer((__gm__ T*) xNew + startNumel);
        yGm.SetGlobalBuffer((__gm__ T*) out + startNumel);
        cosGm.SetGlobalBuffer((__gm__ T*) cos + startNumel);
        sinGm.SetGlobalBuffer((__gm__ T*) sin + startNumel);

        pipe->InitBuffer(inQueueX, DOUBLE_BUFFER, sinSize);
        pipe->InitBuffer(inQueueXnew, DOUBLE_BUFFER, sinSize);
        pipe->InitBuffer(outQueueY, DOUBLE_BUFFER, sinSize);
        pipe->InitBuffer(inQueueCos, DOUBLE_BUFFER, sinSize);
        pipe->InitBuffer(inQueueSin, DOUBLE_BUFFER, sinSize);

        pipe->InitBuffer(xBuf, sinSize32);
        pipe->InitBuffer(cosBuf, sinSize32);
        pipe->InitBuffer(sinBuf, sinSize32);
        pipe->InitBuffer(xNewBuf, sinSize32);
    }

    __aicore__ inline int CeilDiv(int a, int b)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    }

    __aicore__ inline void CopyTiling(RopeMatrixTiling *ropeTiling)
    {
        this->seqLen = ropeTiling->s;
        this->n = ropeTiling->n;
        this->b = ropeTiling->b;
        this->d = ropeTiling->d;
        return;
    }

    __aicore__ inline void Debug(){
        // add debug code using AscendC::printf(), like the code below
        // AscendC::printf("... debug %d, %d, %d, %d\n", this->aivIdx, this->aivNum, this->subIdx, this->subNum);
    }

    __aicore__ inline void Process() {
        uint32_t ubLoop = this->sBlockNum - 1;
        for (uint32_t progress = 0; progress < ubLoop; progress++) {
            SingleStepProcess(progress, this->sdBlockSize, this->sdBlockSize);
        }
        // last block
        SingleStepProcess(ubLoop, this->sdLastBlockSize, this->sdLastBlockSize);
    }

    __aicore__ inline void SingleStepProcess(uint32_t progress, uint64_t copyLength, uint64_t calcLength) {
        uint64_t xOffset;
        uint64_t rOffset;
        uint64_t bnLoopXStartOffset;
        uint64_t progressOffset;
        uint64_t batchOffset;
        rOffset = progress * this->sdBlockSize;
        CopyInR(rOffset, copyLength);
        LocalTensor<T> cosLocal = inQueueCos.DeQue<T>();
        LocalTensor<T> sinLocal = inQueueSin.DeQue<T>();
        LocalTensor<float> cosFp32 = cosBuf.Get<float>();
        LocalTensor<float> sinFp32 = sinBuf.Get<float>();

        Cast(cosFp32, cosLocal, RoundMode::CAST_NONE, calcLength);
        Cast(sinFp32, sinLocal, RoundMode::CAST_NONE, calcLength);
        inQueueCos.FreeTensor<T>(cosLocal);
        inQueueSin.FreeTensor<T>(sinLocal);

        bnLoopXStartOffset = progress * this->sdBlockSize;
        for (uint32_t bnLoop = 0; bnLoop < this->bnSize; bnLoop++) {
            xOffset = bnLoopXStartOffset + bnLoop * this->sdSizeTotal;
            CopyInX(xOffset, copyLength);
            Compute(cosFp32, sinFp32, calcLength);
            CopyOut(xOffset, copyLength);
        }
    }

    __aicore__ inline void CopyInR(uint64_t rStartOffset, uint32_t copyLength) {
        LocalTensor<T> sinLocal = inQueueSin.AllocTensor<T>();
        LocalTensor<T> cosLocal = inQueueCos.AllocTensor<T>();
        DataCopy(sinLocal, sinGm[rStartOffset], copyLength);
        DataCopy(cosLocal, cosGm[rStartOffset], copyLength);
        inQueueSin.EnQue(sinLocal);
        inQueueCos.EnQue(cosLocal);
    }

    __aicore__ inline void CopyInX(uint64_t xStartOffset, uint32_t copyLength) {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> xNewLocal = inQueueXnew.AllocTensor<T>();
        DataCopy(xLocal, xGm[xStartOffset], copyLength);
        DataCopy(xNewLocal, xNewGm[xStartOffset], copyLength);
        inQueueX.EnQue(xLocal);
        inQueueXnew.EnQue(xNewLocal);
    }

    __aicore__ inline void CopyOut(uint64_t yStartOffset, uint32_t copyLength) {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopy(yGm[yStartOffset], yLocal, copyLength);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Compute(LocalTensor<float>& cos, LocalTensor<float>& sin, uint32_t calcLength) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> xNewLocal = inQueueXnew.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<float> xFp32 = xBuf.Get<float>();
        LocalTensor<float> xNewFp32 = xNewBuf.Get<float>();

        Cast(xFp32, xLocal, RoundMode::CAST_NONE, calcLength);
        Cast(xNewFp32, xNewLocal, RoundMode::CAST_NONE, calcLength);
        inQueueX.FreeTensor(xLocal);
        inQueueXnew.FreeTensor(xNewLocal);

        this->ComputeInner(xFp32, xNewFp32, cos, sin, calcLength);
        Cast(yLocal, xNewFp32, RoundMode::CAST_RINT, calcLength);
        outQueueY.EnQue(yLocal);
    }

    __aicore__ inline void ComputeInner(LocalTensor<float>& x, LocalTensor<float>& xNew,
                                        LocalTensor<float>& cos , LocalTensor<float>& sin,
                                        uint32_t calcLength) {
        Mul(x, x, cos, calcLength);
        Mul(xNew, xNew, sin, calcLength);
        Add(xNew, xNew, x, calcLength);
    }

protected:
    TPipe *pipe;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueX;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueXnew;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueCos;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueSin;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQueueY;
    TBuf<AscendC::TPosition::VECCALC> xBuf;
    TBuf<AscendC::TPosition::VECCALC> xNewBuf;
    TBuf<AscendC::TPosition::VECCALC> cosBuf;
    TBuf<AscendC::TPosition::VECCALC> sinBuf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> xNewGm;
    GlobalTensor<T> sinGm;
    GlobalTensor<T> cosGm;
    GlobalTensor<T> yGm;

    uint32_t seqLen;
    uint32_t b;
    uint32_t n;
    uint32_t d;
    uint32_t bnSize;

    uint32_t aivIdx;
    uint32_t aivNum;
    uint32_t subIdx;
    uint32_t subNum;

    // global
    uint64_t sdSizeTotal;
    // for this core
    uint32_t sBlockSize;
    uint32_t sdBlockSize;
    uint32_t sBlockNum;
    uint32_t startSize;
    uint32_t sSize;
    uint32_t sdSize;
    uint32_t sLastBlockSize;
    uint32_t sdLastBlockSize;
};
} // namespace RopeMatrix
} //namespace npu_ops_transformer_ext
