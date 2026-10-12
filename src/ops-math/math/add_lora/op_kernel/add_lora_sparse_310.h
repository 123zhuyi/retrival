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
 * \file add_lora_sparse_310.h
 * \brief
 */
#ifndef ADD_LORA_SPARSE_310_H
#define ADD_LORA_SPARSE_310_H

#include <vector>
#include "kernel_operator.h"
#include "add_lora_base_310.h"

using namespace AscendC;
using namespace AddLora310;

__aicore__ inline void AddLoraSparse310::InitVectorBuffers()
{
    pipe_->InitBuffer(ubBuf_, TOTAL_USED_UB_SIZE);
    mm1ResUb = ubBuf_.GetWithOffset<half>(CUBE_BASE_M * R, 0);
    mm2ResUb[0] = ubBuf_.GetWithOffset<half>(DOUBLE_NUM * CUBE_BASE_M * CUBE_BASE_M, 0);
    mm2ResUb[1] = ubBuf_.GetWithOffset<half>(
        DOUBLE_NUM * CUBE_BASE_M * CUBE_BASE_M, DOUBLE_NUM * CUBE_BASE_M * CUBE_BASE_M * sizeof(half));
    allZeroTensor = ubBuf_.GetWithOffset<int32_t>(
        coreNum * DEFAULT_SYNCALL_NEED_SIZE, DOUBLE_NUM * DOUBLE_NUM * CUBE_BASE_M * CUBE_BASE_M * sizeof(half));
}

__aicore__ inline void AddLoraSparse310::ComputeBatchIndiceCount()
{
    for (uint32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        int32_t loraIdOfBatch = int32_t(indicesGm_.GetValue(batchIdx));
        if (loraIdOfBatch < 0) {
            loraIdOfBatch = wBatch;
        }
        batchOffsetInLora[batchIdx] = batchNumOfLora[loraIdOfBatch];
        batchNumOfLora[loraIdOfBatch] += 1;
    }

    batchAccumulationByCurLora[0] = 0;
    for (uint32_t loraIdx = 1; loraIdx <= wBatch; loraIdx++) {
        batchAccumulationByCurLora[loraIdx] = batchAccumulationByCurLora[loraIdx - 1] + batchNumOfLora[loraIdx - 1];
    }

    uint32_t indice = 0;
    for (uint32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        int32_t loraIdOfBatch = int32_t(indicesGm_.GetValue(batchIdx));
        if (loraIdOfBatch < 0) {
            loraIdOfBatch = wBatch;
        }
        oriBatchIdx[batchAccumulationByCurLora[loraIdOfBatch] + batchOffsetInLora[batchIdx]] = batchIdx;
    }
}

__aicore__ inline void AddLoraSparse310::Process()
{
    InitVectorBuffers();

    AscendC::SetAtomicNone();
    Duplicate(allZeroTensor, 0, coreNum * DEFAULT_SYNCALL_NEED_SIZE);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(softSyncGm_, allZeroTensor, coreNum * DEFAULT_SYNCALL_NEED_SIZE);

    ComputeBatchIndiceCount();

    if (coreId < batchSize) {
        UpdateByWeightA();
    }

    // 全核同步，等待MM1完成再计算MM2
    LocalTensor<int32_t> workLocal2 = workQueue_.AllocTensor<int32_t>();
    SyncAll(softSyncGm_, workLocal2);
    workQueue_.FreeTensor(workLocal2);

    SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID3);

    // ComputeMm2
    UpdateByWeightB();

    WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
}

__aicore__ inline void AddLoraSparse310::UpdateByWeightA()
{
    int32_t loraId = int32_t(indicesGm_.GetValue(coreId));
    if (loraId < 0 || loraId >= wBatch) {
        return;
    }

    uint32_t maxMN = RoundUp16(R);
    // matmul时一次计算K的值，128倍数向下对齐
    uint32_t maxKL0Round = maxMN == 0 ? 0 : CUBE_BASE_M * CUBE_BASE_M / maxMN / CUBE_BASE_M * CUBE_BASE_M;
    // L1的预留空间是L0的 512k/64k=8 倍
    uint32_t maxKL1 = maxKL0Round * 8;

    AscendC::LocalTensor<float> l0cLocal = outQueueCO1.AllocTensor<float>();

    for (int kIdx = 0; kIdx < DivCeil(H1, maxKL0Round); kIdx++) {
        uint32_t kOffset = kIdx * maxKL0Round;
        uint32_t kCurLoop = (H1 - kOffset) < maxKL0Round ? (H1 - kOffset) : maxKL0Round;
        CopyInL1A(xGm_[coreId * H1 + kOffset], 1, kCurLoop, H1);

        uint32_t aWeightOffset = loraId * layerNum * H1 * R + layer_idx * H1 * R + kOffset;
        CopyInL1BNd(waGm_[aWeightOffset], kCurLoop, R, R, H1);

        SplitA(1, kCurLoop);
        SplitB(kCurLoop, R);

        AscendC::LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        AscendC::LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        AscendC::PipeBarrier<PIPE_M>();
        doMatmul(l0cLocal, a2Local, b2Local, kIdx, 1, kCurLoop, R);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }
    outQueueCO1.EnQue<float>(l0cLocal);

    CopyMm1ResOut();
}

__aicore__ inline void AddLoraSparse310::UpdateByWeightB()
{
    uint32_t pingPongFlag = 0;

    for (uint32_t loraIdx = 0; loraIdx < wBatch; loraIdx++) {
        uint32_t batchNumCurLora = batchNumOfLora[loraIdx];
        uint32_t batchOffsetCurLora = batchAccumulationByCurLora[loraIdx];
        uint32_t maxN = CUBE_BASE_M * CUBE_BASE_M / R;
        uint32_t H2BlockNum = DivCeil(H2PerCore, maxN);

        for (uint32_t H2BlockIdx = 0; H2BlockIdx < H2BlockNum; H2BlockIdx++) {
            uint32_t H2OffsetCurLoop = H2OffsetCurCore + H2BlockIdx * maxN;
            uint32_t bWeightOffset = loraIdx * layerNum * H2 * R + layer_idx * H2 * R + H2OffsetCurLoop * R;
            uint32_t nToProcess = (H2BlockIdx == (H2BlockNum - 1)) ? (H2PerCore - H2BlockIdx * maxN) : maxN;

            // GM -> L1B
            CopyInL1BNd(wbGm_[bWeightOffset], R, nToProcess, H2, R);
            // L1B -> L0B
            SplitB(R, nToProcess);
            // GM -> L1A
            CopyInL1A(mm1ResGm_[batchOffsetCurLora * R], batchNumCurLora, R, R);
            // L1A -> L0A
            SplitA(batchNumCurLora, R);

            // Mmad
            AscendC::LocalTensor<float> l0cLocal = outQueueCO1.AllocTensor<float>();
            AscendC::LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
            AscendC::LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
            doMatmul(l0cLocal, a2Local, b2Local, 0, batchNumCurLora, R, nToProcess);
            inQueueA2.FreeTensor(a2Local);
            inQueueB2.FreeTensor(b2Local);
            outQueueCO1.EnQue<float>(l0cLocal);

            // L0C -> UB
            WaitFlag<HardEvent::MTE3_V>(pingPongFlag + DOUBLE_NUM);
            CopyL0cToUb(mm2ResUb[pingPongFlag], nToProcess);

            // scale
            AscendC::PipeBarrier<PIPE_V>();
            uint32_t dataCountRound = CeilCubeBlock(batchNumCurLora) * CeilCubeBlock(nToProcess) * CUBE_BLOCK_SIZE;
            Muls(mm2ResUb[pingPongFlag], mm2ResUb[pingPongFlag], static_cast<half>(scale), dataCountRound);

            // copyOut
            SetFlag<HardEvent::V_MTE3>(pingPongFlag);
            WaitFlag<HardEvent::V_MTE3>(pingPongFlag);
            CopyFinalResOutAdd(batchNumCurLora, nToProcess, batchOffsetCurLora, H2OffsetCurLoop, pingPongFlag);
            SetFlag<HardEvent::MTE3_V>(pingPongFlag + DOUBLE_NUM);

            pingPongFlag = !pingPongFlag;
        }
    }
}

__aicore__ inline void AddLoraSparse310::CopyMm1ResOut()
{
    // L0C -> UB
    CopyL0cToUb(mm1ResUb, R);

    int32_t loraIdx = indicesGm_.GetValue(coreId);
    if (loraIdx < 0) {
        loraIdx = wBatch;
    }
    uint32_t newBatchIdx = batchAccumulationByCurLora[loraIdx] + batchOffsetInLora[coreId];

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    // UB -> GM
    if (R == CUBE_BLOCK) {
        DataCopy(mm1ResGm_[newBatchIdx * R], mm1ResUb, R);
    } else {
        DataCopyParams nz2NdParams;
        nz2NdParams.blockCount = CeilCubeBlock(R);
        nz2NdParams.blockLen = 1;
        nz2NdParams.srcStride = RoundUp16(1) - 1;
        nz2NdParams.dstStride = 0;

        DataCopy(mm1ResGm_[newBatchIdx * R], mm1ResUb, nz2NdParams);
    }
}

__aicore__ inline void AddLoraSparse310::CopyL0cToUb(LocalTensor<half> ubTensor, uint32_t nToProcess)
{
    AscendC::LocalTensor<float> l0cLocal = outQueueCO1.DeQue<float>();

    DataCopyParams intriParams{1, static_cast<uint16_t>(CeilCubeBlock(nToProcess)), 0, 0};
    DataCopyEnhancedParams enhancedParams;
    enhancedParams.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;

    DataCopy(ubTensor, l0cLocal, intriParams, enhancedParams);

    outQueueCO1.FreeTensor(l0cLocal);
}

__aicore__ inline void AddLoraSparse310::CopyFinalResOutAdd(
    uint32_t batchToProcess, uint32_t H2ToProcess, uint32_t batchOffset, uint32_t H2Offset, uint32_t pingPongFlag)
{
    AscendC::SetAtomicAdd<half>();

    DataCopyParams nz2NdParams;
    nz2NdParams.blockCount = CeilCubeBlock(H2ToProcess);
    nz2NdParams.blockLen = 1;
    nz2NdParams.srcStride = RoundUp16(batchToProcess) - 1;
    nz2NdParams.dstStride = 0;
    for (uint32_t mIdx = 0; mIdx < batchToProcess; mIdx++) {
        uint32_t batchIdxOri = oriBatchIdx[batchOffset + mIdx];
        uint32_t dstGmOffset = batchIdxOri * y_column + H2Offset + y_offset;
        uint32_t srcOffset = mIdx * CUBE_BLOCK;

        DataCopy(yGm_[dstGmOffset], mm2ResUb[pingPongFlag][srcOffset], nz2NdParams);
    }
    AscendC::SetAtomicNone();
}

#endif // ADD_LORA_SPARSE_310_H
