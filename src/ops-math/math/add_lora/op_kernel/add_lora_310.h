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
 * \file add_lora_310.h
 * \brief
 */
#ifndef ADD_LORA_310_H
#define ADD_LORA_310_H

#include <vector>
#include "kernel_operator.h"
#include "add_lora_base_310.h"

using namespace AscendC;
using namespace AddLora310;

__aicore__ inline void AddLoraKernel310::InitVectorBuffers()
{
    pipe_->InitBuffer(ubBuf_, TOTAL_USED_UB_SIZE);

    mm1ResUb[0] = ubBuf_.GetWithOffset<half>(CUBE_BASE_M * R, 0);
    mm1ResUb[1] = ubBuf_.GetWithOffset<half>(CUBE_BASE_M * R, CUBE_BASE_M * R * sizeof(half));
    mm2ResUb[0] = ubBuf_.GetWithOffset<half>(CUBE_BASE_M * CUBE_BASE_M * DOUBLE_NUM, 0);
    mm2ResUb[1] = ubBuf_.GetWithOffset<half>(
        CUBE_BASE_M * CUBE_BASE_M * DOUBLE_NUM, CUBE_BASE_M * CUBE_BASE_M * DOUBLE_NUM * sizeof(half));
    allZeroTensor = ubBuf_.GetWithOffset<int32_t>(
        coreNum * DEFAULT_SYNCALL_NEED_SIZE, DOUBLE_NUM * DOUBLE_NUM * CUBE_BASE_M * CUBE_BASE_M * sizeof(half));
}

__aicore__ inline void AddLoraKernel310::Process()
{
    InitVectorBuffers();

    // 软同步初始化
    AscendC::SetAtomicNone();
    Duplicate(allZeroTensor, 0, coreNum * DEFAULT_SYNCALL_NEED_SIZE);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(softSyncGm_, allZeroTensor, coreNum * DEFAULT_SYNCALL_NEED_SIZE);

    // 根据Indices获取计算信息
    ComputeBatchIndiceCount();

    // 对x根据Lora分组进行重排
    xDataRearrange();

    // 全局软同步
    LocalTensor<int32_t> workLocal = workQueue_.AllocTensor<int32_t>();
    SyncAll(softSyncGm_, workLocal);
    workQueue_.FreeTensor(workLocal);

    SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID3);

    // 根据weightA计算第一个矩阵乘
    UpdateByWeightA();

    // 全核同步，等待MM1完成再计算MM2
    LocalTensor<int32_t> workLocal2 = workQueue_.AllocTensor<int32_t>();
    SyncAll(softSyncGm_, workLocal2);
    workQueue_.FreeTensor(workLocal2);

    // 根据weightB计算第二个矩阵乘
    UpdateByWeightB();

    WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
}

__aicore__ inline void AddLoraKernel310::UpdateByWeightA()
{
    uint32_t l0cPingPongFlag = 0;

    for (int loraId = startLoraIdCurCore; loraId <= endLoraIdCurCore; ++loraId) {
        if (loraId == wBatch) {
            continue;
        }

        uint32_t batchNumCurLora = batchNumOfLora[loraId];
        uint32_t batchOffsetCurLora = batchAccumulationByCurLora[loraId];
        uint32_t processStart = batchOffsetCurCore < batchOffsetCurLora ? batchOffsetCurLora : batchOffsetCurCore;
        uint32_t processEnd = (batchOffsetCurCore + batchNumCurCore) < (batchOffsetCurLora + batchNumCurLora) ?
                                  (batchOffsetCurCore + batchNumCurCore) :
                                  (batchOffsetCurLora + batchNumCurLora);
        uint32_t processBatchNum = processEnd - processStart;

        // 如果处理的batch数大于128，要进行循环
        for (int batchBlockId = 0; batchBlockId < DivCeil(processBatchNum, CUBE_BASE_M); batchBlockId++) {
            // 当前这轮处理的batch数
            uint32_t batchOffsetInnerLora = batchBlockId * CUBE_BASE_M;
            uint32_t batchOffset = processStart + batchOffsetInnerLora;
            uint32_t processNumCurLoop = (processBatchNum - batchOffsetInnerLora) < CUBE_BASE_M ?
                                             (processBatchNum - batchOffsetInnerLora) :
                                             CUBE_BASE_M;
            // M和N当中的较大值，决定K的大小
            uint32_t maxMN = processNumCurLoop > R ? RoundUp16(processNumCurLoop) : RoundUp16(R);
            // matmul时一次计算K的值，128倍数向下对齐
            uint32_t maxKL0Round = CUBE_BASE_M * CUBE_BASE_M / maxMN / CUBE_BASE_M * CUBE_BASE_M;
            // L1的预留空间是L0的 512k/64k=8 倍
            uint32_t maxKL1 = maxKL0Round * 8;

            AscendC::LocalTensor<float> l0cLocal = outQueueCO1.AllocTensor<float>();

            for (int kIdx = 0; kIdx < DivCeil(H1, maxKL0Round); kIdx++) {
                uint32_t kOffset = kIdx * maxKL0Round;
                uint32_t kCurLoop = (H1 - kOffset) < maxKL0Round ? (H1 - kOffset) : maxKL0Round;
                CopyInL1A(xRearrangeGm_[batchOffset * H1 + kOffset], processNumCurLoop, kCurLoop, H1);

                // ND
                uint32_t aWeightOffset = loraId * layerNum * H1 * R + layer_idx * H1 * R + kOffset;
                CopyInL1BNd(waGm_[aWeightOffset], kCurLoop, R, R, H1);

                SplitA(processNumCurLoop, kCurLoop);
                SplitB(kCurLoop, R);

                AscendC::LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
                AscendC::LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
                AscendC::PipeBarrier<PIPE_M>();
                doMatmul(l0cLocal, a2Local, b2Local, kIdx, processNumCurLoop, kCurLoop, R);
                inQueueA2.FreeTensor(a2Local);
                inQueueB2.FreeTensor(b2Local);
            }
            outQueueCO1.EnQue<float>(l0cLocal);

            WaitFlag<HardEvent::MTE3_V>(l0cPingPongFlag + DOUBLE_NUM);
            CopyMm1ResOut(processNumCurLoop, batchOffset, l0cPingPongFlag);
            SetFlag<HardEvent::MTE3_V>(l0cPingPongFlag + DOUBLE_NUM);
            l0cPingPongFlag = !l0cPingPongFlag;
        }
    }
}

__aicore__ inline void AddLoraKernel310::UpdateByWeightB()
{
    uint32_t pingPongFlag = 0;
    for (uint32_t loraIdx = 0; loraIdx < wBatch; loraIdx++) {
        uint32_t batchNumCurLora = batchNumOfLora[loraIdx];
        uint32_t batchOffsetCurLora = batchAccumulationByCurLora[loraIdx];
        uint32_t maxN = CUBE_BASE_M * CUBE_BASE_M / R < 256 ? CUBE_BASE_M * CUBE_BASE_M / R : 256;
        uint32_t batchBlockNum = DivCeil(batchNumCurLora, CUBE_BASE_M);
        uint32_t H2BlockNum = DivCeil(H2PerCore, maxN);

        // wb >> mm1Res, 所以先循环右矩阵
        for (uint32_t H2BlockIdx = 0; H2BlockIdx < H2BlockNum; H2BlockIdx++) {
            uint32_t H2OffsetCurLoop = H2OffsetCurCore + H2BlockIdx * maxN;
            // ND
            uint32_t bWeightOffset = loraIdx * layerNum * H2 * R + layer_idx * H2 * R + H2OffsetCurLoop * R;
            uint32_t nToProcess = (H2BlockIdx == (H2BlockNum - 1)) ? (H2PerCore - H2BlockIdx * maxN) : maxN;

            // GM -> L1B
            CopyInL1BNd(wbGm_[bWeightOffset], R, nToProcess, H2, R);
            // L1B -> L0B
            SplitB(R, nToProcess);

            AscendC::LocalTensor<half> b2Local = inQueueB2.DeQue<half>();

            for (uint32_t batchBlockIdx = 0; batchBlockIdx < batchBlockNum; batchBlockIdx++) {
                uint32_t mToProcess = (batchBlockIdx == (batchBlockNum - 1)) ?
                                          (batchNumCurLora - batchBlockIdx * CUBE_BASE_M) :
                                          CUBE_BASE_M;
                // GM -> L1A
                CopyInL1A(mm1ResGm_[(batchOffsetCurLora + batchBlockIdx * CUBE_BASE_M) * R], mToProcess, R, R);
                // L1A -> L0A
                SplitA(mToProcess, R);

                // Mmad
                AscendC::LocalTensor<float> l0cLocal = outQueueCO1.AllocTensor<float>();
                AscendC::LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
                doMatmul(l0cLocal, a2Local, b2Local, 0, mToProcess, R, nToProcess);
                inQueueA2.FreeTensor(a2Local);
                outQueueCO1.EnQue<float>(l0cLocal);

                // L0C -> UB
                WaitFlag<HardEvent::MTE3_V>(pingPongFlag + DOUBLE_NUM);
                CopyL0cToUb(mm2ResUb[pingPongFlag], mToProcess, nToProcess);

                // scale
                AscendC::PipeBarrier<PIPE_V>();
                uint32_t dataCountRound = CeilCubeBlock(mToProcess) * CeilCubeBlock(nToProcess) * CUBE_BLOCK_SIZE;
                Muls(mm2ResUb[pingPongFlag], mm2ResUb[pingPongFlag], static_cast<half>(scale), dataCountRound);

                // copyOut
                SetFlag<HardEvent::V_MTE3>(pingPongFlag);
                WaitFlag<HardEvent::V_MTE3>(pingPongFlag);
                CopyFinalResOutAdd(
                    mToProcess, nToProcess, batchOffsetCurLora + batchBlockIdx * CUBE_BASE_M, H2OffsetCurLoop,
                    pingPongFlag);
                SetFlag<HardEvent::MTE3_V>(pingPongFlag + DOUBLE_NUM);

                pingPongFlag = !pingPongFlag;
            }

            inQueueB2.FreeTensor(b2Local);
        }
    }
}

__aicore__ inline void AddLoraKernel310::ComputeBatchIndiceCount()
{
    for (uint32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        int32_t loraIdOfBatch = int32_t(indicesGm_.GetValue(batchIdx));
        if (loraIdOfBatch < 0) {
            loraIdOfBatch = wBatch;
        }
        batchOffsetInLoraList_.SetValue(batchIdx, batchNumOfLora[loraIdOfBatch]);
        batchNumOfLora[loraIdOfBatch] += 1;
    }

    batchAccumulationByCurLora[0] = 0;
    bool calcStartFlag = true;
    bool calcEndFlag = true;
    for (uint32_t loraIdx = 1; loraIdx <= wBatch; loraIdx++) {
        batchAccumulationByCurLora[loraIdx] = batchAccumulationByCurLora[loraIdx - 1] + batchNumOfLora[loraIdx - 1];
        if (calcStartFlag && (batchAccumulationByCurLora[loraIdx] > batchOffsetCurCore)) {
            startLoraIdCurCore = loraIdx - 1;
            calcStartFlag = false;
        }
        if (calcEndFlag && (batchAccumulationByCurLora[loraIdx] >= (batchOffsetCurCore + batchNumCurCore))) {
            endLoraIdCurCore = loraIdx - 1;
            calcEndFlag = false;
        }
    }

    if (calcStartFlag) {
        startLoraIdCurCore = wBatch;
    }
    if (calcEndFlag) {
        endLoraIdCurCore = wBatch;
    }

    uint32_t indice = 0;
    for (uint32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        int32_t loraIdOfBatch = int32_t(indicesGm_.GetValue(batchIdx));
        if (loraIdOfBatch < 0) {
            loraIdOfBatch = wBatch;
        }
        uint32_t batchOffsetInLora = batchOffsetInLoraList_.GetValue(batchIdx);
        oriBatchIdxList_.SetValue(batchAccumulationByCurLora[loraIdOfBatch] + batchOffsetInLora, batchIdx);
    }
}

__aicore__ inline void AddLoraKernel310::CopyMm1ResOut(uint32_t mToProcess, uint32_t mOffset, uint32_t l0cPingPongFlag)
{
    // L0C -> UB
    CopyL0cToUb(mm1ResUb[l0cPingPongFlag], mToProcess, R);

    SetFlag<HardEvent::V_MTE3>(l0cPingPongFlag);
    WaitFlag<HardEvent::V_MTE3>(l0cPingPongFlag);

    if (R == CUBE_BLOCK) {
        DataCopy(mm1ResGm_[mOffset * R], mm1ResUb[l0cPingPongFlag], mToProcess * R);
    } else {
        DataCopyParams nz2NdParams;
        nz2NdParams.blockCount = CeilCubeBlock(R);
        nz2NdParams.blockLen = 1;
        nz2NdParams.srcStride = RoundUp16(mToProcess) - 1;
        nz2NdParams.dstStride = 0;

        for (uint32_t mIdx = 0; mIdx < mToProcess; mIdx++) {
            DataCopy(mm1ResGm_[(mOffset + mIdx) * R], mm1ResUb[l0cPingPongFlag][mIdx * CUBE_BLOCK], nz2NdParams);
        }
    }
}

__aicore__ inline void AddLoraKernel310::CopyL0cToUb(
    LocalTensor<half> ubTensor, uint32_t mToProcess, uint32_t nToProcess)
{
    AscendC::LocalTensor<float> l0cLocal = outQueueCO1.DeQue<float>();

    DataCopyParams intriParams{1, static_cast<uint16_t>(CeilCubeBlock(mToProcess) * CeilCubeBlock(nToProcess)), 0, 0};
    DataCopyEnhancedParams enhancedParams{
        AscendC::BlockMode::BLOCK_MODE_MATRIX, AscendC::DeqScale::DEQ_NONE, 0, 0, false, pad_t::PAD_NONE, 0};
    DataCopy(ubTensor, l0cLocal, intriParams, enhancedParams);

    outQueueCO1.FreeTensor(l0cLocal);
}

__aicore__ inline void AddLoraKernel310::CopyFinalResOutAdd(
    uint32_t batchToProcess, uint32_t H2ToProcess, uint32_t batchOffset, uint32_t H2Offset, uint32_t pingPongFlag)
{
    AscendC::SetAtomicAdd<half>();

    DataCopyParams nz2NdCopyParams;
    nz2NdCopyParams.blockCount = CeilCubeBlock(H2ToProcess);
    nz2NdCopyParams.blockLen = 1;
    nz2NdCopyParams.srcStride = RoundUp16(batchToProcess) - 1;
    nz2NdCopyParams.dstStride = 0;
    for (uint32_t rowIdx = 0; rowIdx < batchToProcess; rowIdx++) {
        uint32_t batchIdxOri = uint32_t(oriBatchIdxList_.GetValue(batchOffset + rowIdx));
        uint32_t dstGmOffset = batchIdxOri * y_column + H2Offset + y_offset;
        uint32_t srcOffset = rowIdx * CUBE_BLOCK;

        DataCopy(yGm_[dstGmOffset], mm2ResUb[pingPongFlag][srcOffset], nz2NdCopyParams);
    }
    AscendC::SetAtomicNone();
}

#endif // ADD_LORA_310_H