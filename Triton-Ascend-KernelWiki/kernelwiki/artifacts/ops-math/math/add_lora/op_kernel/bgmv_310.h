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
 * \file bgmv_310.h
 * \brief
 */
#ifndef BGMV_310_H
#define BGMV_310_H

#include <vector>
#include "kernel_operator.h"
#include "add_lora_base_310.h"

using namespace AscendC;
using namespace AddLora310;

__aicore__ inline void BgmvKernel310::InitVectorBuffers()
{
    pipe_->InitBuffer(ubBuf_, TOTAL_USED_UB_SIZE);
    mm2ResUb[0] = ubBuf_.GetWithOffset<half>(DOUBLE_NUM * CUBE_BASE_M * CUBE_BASE_M, 0);
    mm2ResUb[1] = ubBuf_.GetWithOffset<half>(
        DOUBLE_NUM * CUBE_BASE_M * CUBE_BASE_M, DOUBLE_NUM * CUBE_BASE_M * CUBE_BASE_M * sizeof(half));
    allZeroTensor = ubBuf_.GetWithOffset<int32_t>(
        coreNum * DEFAULT_SYNCALL_NEED_SIZE, DOUBLE_NUM * DOUBLE_NUM * CUBE_BASE_M * CUBE_BASE_M * sizeof(half));
}

__aicore__ inline void BgmvKernel310::ComputeIndicesInfo()
{
    for (uint32_t bIdx = 0; bIdx < batchSize; bIdx++) {
        int32_t loraIdCurBatch = int32_t(indicesGm_.GetValue(bIdx));
        if (loraIdCurBatch < 0) {
            loraIdCurBatch = wBatch;
        }
        batchOffsetInLoraList_.SetValue(bIdx, batchNumOfLora[loraIdCurBatch]);
        batchNumOfLora[loraIdCurBatch] += 1;
    }

    batchAccumulationByCurLora[0] = 0;
    bool startFlag = true;
    bool endFlag = true;
    for (uint32_t loraIdx = 1; loraIdx <= wBatch; loraIdx++) {
        batchAccumulationByCurLora[loraIdx] = batchAccumulationByCurLora[loraIdx - 1] + batchNumOfLora[loraIdx - 1];
        if (startFlag && (batchAccumulationByCurLora[loraIdx] > batchOffsetCurCore)) {
            startLoraIdCurCore = loraIdx - 1;
            startFlag = false;
        }
        if (endFlag && (batchAccumulationByCurLora[loraIdx] >= (batchOffsetCurCore + batchNumCurCore))) {
            endLoraIdCurCore = loraIdx - 1;
            endFlag = false;
        }
    }

    if (startFlag) {
        startLoraIdCurCore = wBatch;
    }
    if (endFlag) {
        endLoraIdCurCore = wBatch;
    }

    uint32_t indice = 0;
    for (uint32_t bIdx = 0; bIdx < batchSize; bIdx++) {
        int32_t loraIdCurBatch = int32_t(indicesGm_.GetValue(bIdx));
        if (loraIdCurBatch < 0) {
            loraIdCurBatch = wBatch;
        }
        uint32_t batchOffsetInLora = batchOffsetInLoraList_.GetValue(bIdx);
        oriBatchIdxList_.SetValue(batchAccumulationByCurLora[loraIdCurBatch] + batchOffsetInLora, bIdx);
    }
}

__aicore__ inline void BgmvKernel310::Process()
{
    AscendC::SetAtomicNone();
    InitVectorBuffers();

    Duplicate(allZeroTensor, 0, coreNum * DEFAULT_SYNCALL_NEED_SIZE);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    DataCopy(softSyncGm_, allZeroTensor, coreNum * DEFAULT_SYNCALL_NEED_SIZE);

    ComputeIndicesInfo();

    // 对x根据Lora分组进行重排
    xDataRearrange();

    // 等待所有核都完成计算
    LocalTensor<int32_t> syncLocal = workQueue_.AllocTensor<int32_t>();
    SyncAll(softSyncGm_, syncLocal);
    workQueue_.FreeTensor(syncLocal);

    SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID3);

    // ComputeMm2
    UpdateByWeightB();

    WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
}

__aicore__ inline void BgmvKernel310::UpdateByWeightB()
{
    uint32_t l0cPingPongFlag = 0;

    for (uint32_t loraIdx = 0; loraIdx < wBatch; loraIdx++) {
        uint32_t batchNumCurLora = batchNumOfLora[loraIdx];
        uint32_t batchOffsetCurLora = batchAccumulationByCurLora[loraIdx];
        uint32_t batchBlockNum = DivCeil(batchNumCurLora, CUBE_BASE_M);
        uint32_t H2BlockNum = DivCeil(H2PerCore, CUBE_BASE_M);
        uint32_t H1BlockNum = DivCeil(H1, CUBE_BASE_M);

        for (uint32_t H2BlockIdx = 0; H2BlockIdx < H2BlockNum; H2BlockIdx++) {
            uint32_t nToProcess =
                (H2BlockIdx == (H2BlockNum - 1)) ? (H2PerCore - H2BlockIdx * CUBE_BASE_M) : CUBE_BASE_M;
            uint32_t H2OffsetCurLoop = H2OffsetCurCore + H2BlockIdx * CUBE_BASE_M;

            for (uint32_t batchBlockIdx = 0; batchBlockIdx < batchBlockNum; batchBlockIdx++) {
                uint32_t mToProcess = (batchBlockIdx == (batchBlockNum - 1)) ?
                                          (batchNumCurLora - batchBlockIdx * CUBE_BASE_M) :
                                          CUBE_BASE_M;
                uint32_t batchOffsetCurLoop = batchOffsetCurLora + batchBlockIdx * CUBE_BASE_M;

                AscendC::LocalTensor<float> l0cLocal = outQueueCO1.AllocTensor<float>();
                for (uint32_t H1BlockIdx = 0; H1BlockIdx < H1BlockNum; H1BlockIdx++) {
                    uint32_t kToProcess =
                        (H1BlockIdx == (H1BlockNum - 1)) ? (H1 - H1BlockIdx * CUBE_BASE_M) : CUBE_BASE_M;
                    uint32_t H1Offset = H1BlockIdx * CUBE_BASE_M;

                    uint32_t xOffset = batchOffsetCurLoop * H1 + H1Offset;
                    CopyInL1A(xRearrangeGm_[xOffset], mToProcess, kToProcess, H1);

                    uint32_t weightOffset =
                        loraIdx * layerNum * H2 * H1 + layer_idx * H2 * H1 + H1Offset + H2OffsetCurLoop * H1;
                    CopyInL1BNd(wbGm_[weightOffset], kToProcess, nToProcess, H2, H1);

                    SplitA(mToProcess, kToProcess);
                    SplitB(kToProcess, nToProcess);

                    AscendC::LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
                    AscendC::LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
                    AscendC::PipeBarrier<PIPE_M>();
                    doMatmul(l0cLocal, a2Local, b2Local, H1BlockIdx, mToProcess, kToProcess, nToProcess);
                    inQueueA2.FreeTensor(a2Local);
                    inQueueB2.FreeTensor(b2Local);
                }
                outQueueCO1.EnQue<float>(l0cLocal);

                WaitFlag<HardEvent::MTE3_V>(l0cPingPongFlag + DOUBLE_NUM);

                CopyL0cToUb(mm2ResUb[l0cPingPongFlag], mToProcess, nToProcess);
                AscendC::PipeBarrier<PIPE_V>();
                uint32_t dataCountRound = CeilCubeBlock(mToProcess) * CeilCubeBlock(nToProcess) * CUBE_BLOCK_SIZE;
                Muls(mm2ResUb[l0cPingPongFlag], mm2ResUb[l0cPingPongFlag], static_cast<half>(scale), dataCountRound);

                SetFlag<HardEvent::V_MTE3>(l0cPingPongFlag);
                WaitFlag<HardEvent::V_MTE3>(l0cPingPongFlag);

                CopyFinalResOutAdd(mToProcess, nToProcess, batchOffsetCurLoop, H2OffsetCurLoop, l0cPingPongFlag);

                SetFlag<HardEvent::MTE3_V>(l0cPingPongFlag + DOUBLE_NUM);
                l0cPingPongFlag = !l0cPingPongFlag;
            }
        }
    }
}

__aicore__ inline void BgmvKernel310::CopyL0cToUb(LocalTensor<half> ubTensor, uint32_t mToProcess, uint32_t nToProcess)
{
    AscendC::LocalTensor<float> l0cLocal = outQueueCO1.DeQue<float>();

    DataCopyParams dataCopyParams{
        1, static_cast<uint16_t>(CeilCubeBlock(mToProcess) * CeilCubeBlock(nToProcess)), 0, 0};
    DataCopyEnhancedParams enhancedParams{
        AscendC::BlockMode::BLOCK_MODE_MATRIX, AscendC::DeqScale::DEQ_NONE, 0, 0, false, pad_t::PAD_NONE, 0};
    DataCopy(ubTensor, l0cLocal, dataCopyParams, enhancedParams);

    outQueueCO1.FreeTensor(l0cLocal);
}

__aicore__ inline void BgmvKernel310::CopyFinalResOutAdd(
    uint32_t batchToProcess, uint32_t H2ToProcess, uint32_t batchOffset, uint32_t H2Offset, uint32_t pingPongFlag)
{
    AscendC::SetAtomicAdd<half>();

    DataCopyParams copyNz2NdParams;
    copyNz2NdParams.blockCount = CeilCubeBlock(H2ToProcess);
    copyNz2NdParams.blockLen = 1;
    copyNz2NdParams.srcStride = RoundUp16(batchToProcess) - 1;
    copyNz2NdParams.dstStride = 0;
    for (uint32_t mIdx = 0; mIdx < batchToProcess; mIdx++) {
        uint32_t oriBatchIdx = uint32_t(oriBatchIdxList_.GetValue(batchOffset + mIdx));
        uint32_t dstGmOffset = oriBatchIdx * y_column + H2Offset + y_offset;
        uint32_t srcUbOffset = mIdx * CUBE_BLOCK;

        DataCopy(yGm_[dstGmOffset], mm2ResUb[pingPongFlag][srcUbOffset], copyNz2NdParams);
    }
    AscendC::SetAtomicNone();
}
#endif // BGMV_310_H