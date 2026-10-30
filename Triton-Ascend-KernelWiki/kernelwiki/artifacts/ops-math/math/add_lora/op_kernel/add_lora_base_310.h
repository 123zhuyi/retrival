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
 * \file add_lora_base_310.h
 * \brief
 */

#ifndef ADD_LORA_BASE_310_H
#define ADD_LORA_BASE_310_H

#include <vector>
#include "kernel_operator.h"

namespace AddLora310 {
using namespace AscendC;

class AddLoraKernelBase310
{
public:
    __aicore__ inline AddLoraKernelBase310(){};
    __aicore__ inline void Init(
        GM_ADDR y, GM_ADDR x, GM_ADDR weightA, GM_ADDR weightB, GM_ADDR indices, GM_ADDR y_out, GM_ADDR workSpace,
        const AddLoraTilingData& tilingData, TPipe* pipe);
    __aicore__ inline void Process();

protected:
    // CONST
    static constexpr uint32_t CUBE_BLOCK = 16;
    static constexpr uint32_t CUBE_BLOCK_SIZE = 16 * 16;
    static constexpr uint32_t MAX_WEIGHT_SIZE = 32;
    static constexpr uint32_t DOUBLE_NUM = 2;
    static constexpr uint32_t FRACTAL_SIZE = 512;
    static constexpr uint32_t DEFAULT_SYNCALL_NEED_SIZE = 8;
    static constexpr uint32_t CUBE_BASE_M = 128;
    static constexpr uint32_t L1_ADDR = 0;
    static constexpr uint32_t TOTAL_USED_UB_SIZE = 192 * 1024;
    static constexpr uint32_t TOTAL_USED_L0A_SIZE = 64 * 1024;
    static constexpr uint32_t TOTAL_USED_L0B_SIZE = 64 * 1024;
    static constexpr uint32_t TOTAL_USED_L0C_SIZE = 256 * 1024;
    static constexpr uint32_t TOTAL_USED_L1_SIZE = 1024 * 1024;

    // parms
    uint32_t usedCoreNum;
    uint32_t layerNum;
    uint32_t batchSize;
    uint32_t H1;
    uint32_t H2;
    uint32_t R;
    uint32_t y_column;
    uint32_t wBatch;
    uint32_t layer_idx;
    uint32_t y_offset;
    uint32_t y_slice_size;
    uint32_t taskNumPerCore;
    uint32_t H2PerCore;
    uint32_t H2OffsetCurCore;
    uint32_t batchNumCurCore;
    uint32_t batchOffsetCurCore;
    uint32_t startLoraIdCurCore;
    uint32_t endLoraIdCurCore;
    int32_t coreId;
    int32_t coreNum;
    float scale;
    uint32_t batchAccumulationByCurLora[MAX_WEIGHT_SIZE] = {0};
    uint32_t batchNumOfLora[MAX_WEIGHT_SIZE] = {0};

    // TQue
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_NUM> xMoveBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> workQueue_;
    AscendC::TQue<AscendC::QuePosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::QuePosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::QuePosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::QuePosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::QuePosition::CO1, 1> outQueueCO1;

    // GM
    AscendC::GlobalTensor<half> yGm_;
    AscendC::GlobalTensor<half> xGm_;
    AscendC::GlobalTensor<half> waGm_;
    AscendC::GlobalTensor<half> wbGm_;
    AscendC::GlobalTensor<half> outputGm_;
    AscendC::GlobalTensor<int32_t> indicesGm_;

    // workspace
    AscendC::GlobalTensor<half> xRearrangeGm_;
    AscendC::GlobalTensor<half> mm1ResGm_;
    AscendC::GlobalTensor<int32_t> batchOffsetInLoraList_;
    AscendC::GlobalTensor<int32_t> oriBatchIdxList_;
    AscendC::GlobalTensor<int32_t> softSyncGm_;

    // UB
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<half> mm1ResUb[DOUBLE_NUM];
    LocalTensor<half> mm2ResUb[DOUBLE_NUM];
    LocalTensor<half> finalResTransUb[DOUBLE_NUM];
    LocalTensor<int32_t> allZeroTensor;

    __aicore__ inline void InitTilingData(const AddLoraTilingData& tilingData);
    __aicore__ inline void InitGlobalTensor(
        GM_ADDR y, GM_ADDR x, GM_ADDR wa_t_all, GM_ADDR wb_t_all, GM_ADDR indices, GM_ADDR output);
    __aicore__ inline void InitWorkSpace(GM_ADDR workSpace);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void xDataRearrange();
    __aicore__ inline void CopyInL1A(
        GlobalTensor<half> srcGm, uint32_t mToProcess, uint32_t kToProcess, uint32_t srcDvalue);
    __aicore__ inline void CopyInL1BNd(
        GlobalTensor<half> srcWeightGm, uint32_t kToProcess, uint32_t nToProcess, uint32_t srcN, uint32_t srcDvalue);
    __aicore__ inline void SplitA(uint32_t mToProcess, uint32_t kToProcess);
    __aicore__ inline void SplitB(uint32_t kToProcess, uint32_t nToProcess);
    __aicore__ inline void doMatmul(
        LocalTensor<float> l0cLocal, LocalTensor<half> l0aLocal, LocalTensor<half> l0bLocal, uint32_t kIdx_,
        uint32_t mToProcess, uint32_t kToProcess, uint32_t nToProcess);
    __aicore__ inline uint32_t RoundUp16(uint32_t len);
    __aicore__ inline uint32_t CeilCubeBlock(uint32_t len);
};

class AddLoraKernel310 : public AddLoraKernelBase310
{
public:
    __aicore__ inline AddLoraKernel310()
    {}
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitVectorBuffers();
    __aicore__ inline void ComputeBatchIndiceCount();
    __aicore__ inline void UpdateByWeightA();
    __aicore__ inline void UpdateByWeightB();
    __aicore__ inline void CopyL0cToUb(LocalTensor<half> ubTensor, uint32_t mToProcess, uint32_t nToProcess);
    __aicore__ inline void CopyMm1ResOut(uint32_t mToProcess, uint32_t mOffset, uint32_t l0cPingPongFlag);
    __aicore__ inline void CopyFinalResOutAdd(
        uint32_t batchToProcess, uint32_t H2ToProcess, uint32_t batchOffset, uint32_t H2Offset, uint32_t pingPongFlag);
};

class AddLoraSparse310 : public AddLoraKernelBase310
{
public:
    __aicore__ inline AddLoraSparse310()
    {}
    __aicore__ inline void Process();

protected:
    static constexpr uint32_t MAX_SMALL_BATCH_SIZE = 8;

    LocalTensor<half> mm1ResUb;

    uint32_t batchOffsetInLora[MAX_SMALL_BATCH_SIZE] = {0};
    uint32_t oriBatchIdx[MAX_SMALL_BATCH_SIZE] = {0};

    __aicore__ inline void InitVectorBuffers();
    __aicore__ inline void ComputeBatchIndiceCount();
    __aicore__ inline void UpdateByWeightA();
    __aicore__ inline void UpdateByWeightB();
    __aicore__ inline void CopyL0cToUb(LocalTensor<half> ubTensor, uint32_t nToProcess);
    __aicore__ inline void CopyMm1ResOut();
    __aicore__ inline void CopyFinalResOutAdd(
        uint32_t batchToProcess, uint32_t H2ToProcess, uint32_t batchOffset, uint32_t H2Offset, uint32_t pingPongFlag);
};

class BgmvKernel310 : public AddLoraKernelBase310
{
public:
    __aicore__ inline BgmvKernel310(){};
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitVectorBuffers();
    __aicore__ inline void ComputeIndicesInfo();
    __aicore__ inline void UpdateByWeightB();
    __aicore__ inline void CopyL0cToUb(LocalTensor<half> ubTensor, uint32_t mToProcess, uint32_t nToProcess);
    __aicore__ inline void CopyFinalResOutAdd(
        uint32_t batchToProcess, uint32_t H2ToProcess, uint32_t batchOffset, uint32_t H2Offset, uint32_t pingPongFlag);
};

__aicore__ inline void AddLoraKernelBase310::Init(
    GM_ADDR y, GM_ADDR x, GM_ADDR weightA, GM_ADDR weightB, GM_ADDR indices, GM_ADDR y_out, GM_ADDR workSpace,
    const AddLoraTilingData& tilingData, TPipe* pipe)
{
    pipe_ = pipe;
    coreId = GetBlockIdx();
    coreNum = GetBlockNum();

    InitTilingData(tilingData);
    InitGlobalTensor(y, x, weightA, weightB, indices, y_out);
    InitWorkSpace(workSpace);
    InitBuffers();
}

__aicore__ inline void AddLoraKernelBase310::InitTilingData(const AddLoraTilingData& tilingData)
{
    batchSize = tilingData.batch;
    layerNum = tilingData.layer;
    H1 = tilingData.H1;
    H2 = tilingData.H2;
    R = tilingData.R;
    y_column = tilingData.y_column;
    wBatch = tilingData.wBatch;
    layer_idx = tilingData.layer_idx;
    scale = tilingData.scale;
    y_offset = tilingData.y_offset;
    y_slice_size = tilingData.y_slice_size;
    taskNumPerCore = tilingData.taskNumPerCore;
    H2PerCore = tilingData.H2PerCore;

    H2OffsetCurCore = H2PerCore * coreId;

    if (coreId < (batchSize % coreNum)) {
        batchNumCurCore = taskNumPerCore + 1;
        batchOffsetCurCore = batchNumCurCore * coreId;
    } else {
        batchNumCurCore = taskNumPerCore;
        batchOffsetCurCore = batchSize - ((coreNum - coreId) * batchNumCurCore);
    }

    if (coreId == coreNum - 1) {
        H2PerCore = H2 - coreId * H2PerCore;
    }
}

__aicore__ inline void AddLoraKernelBase310::InitGlobalTensor(
    GM_ADDR y, GM_ADDR x, GM_ADDR wa_t_all, GM_ADDR wb_t_all, GM_ADDR indices, GM_ADDR output)
{
    yGm_.SetGlobalBuffer((__gm__ half*)y, batchSize * y_column);
    xGm_.SetGlobalBuffer((__gm__ half*)x, batchSize * H1);
    waGm_.SetGlobalBuffer((__gm__ half*)wa_t_all, wBatch * layerNum * R * H1);
    wbGm_.SetGlobalBuffer((__gm__ half*)wb_t_all, wBatch * layerNum * R * H2);
    indicesGm_.SetGlobalBuffer((__gm__ int32_t*)indices, batchSize);
    outputGm_.SetGlobalBuffer((__gm__ half*)output, batchSize * y_column);
}

__aicore__ inline void AddLoraKernelBase310::InitWorkSpace(GM_ADDR workSpace)
{
    uint32_t listOfBatchInLoraOffset = CUBE_BLOCK_SIZE;
    uint32_t listOfOriBatchIdxOffset = listOfBatchInLoraOffset + RoundUp16(batchSize) * sizeof(int32_t);
    uint32_t xDataRearrangeOffset = listOfOriBatchIdxOffset + RoundUp16(batchSize) * sizeof(int32_t);
    uint32_t mm1ResWorkspaceOffset = xDataRearrangeOffset + RoundUp16(batchSize) * H1 * sizeof(half);
    uint32_t globalSoftSyncOffset = mm1ResWorkspaceOffset + RoundUp16(batchSize) * R * sizeof(half) + CUBE_BLOCK_SIZE;

    batchOffsetInLoraList_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workSpace + listOfBatchInLoraOffset));
    oriBatchIdxList_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workSpace + listOfOriBatchIdxOffset));
    xRearrangeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workSpace + xDataRearrangeOffset));
    mm1ResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workSpace + mm1ResWorkspaceOffset));
    softSyncGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workSpace + globalSoftSyncOffset));
}

__aicore__ inline void AddLoraKernelBase310::InitBuffers()
{
    // L1
    pipe_->InitBuffer(inQueueA1, DOUBLE_NUM, TOTAL_USED_L1_SIZE / DOUBLE_NUM / DOUBLE_NUM);
    pipe_->InitBuffer(inQueueB1, DOUBLE_NUM, TOTAL_USED_L1_SIZE / DOUBLE_NUM / DOUBLE_NUM);

    // L0
    pipe_->InitBuffer(inQueueA2, DOUBLE_NUM, TOTAL_USED_L0A_SIZE / DOUBLE_NUM);
    pipe_->InitBuffer(inQueueB2, DOUBLE_NUM, TOTAL_USED_L0B_SIZE / DOUBLE_NUM);
    pipe_->InitBuffer(outQueueCO1, DOUBLE_NUM, TOTAL_USED_L0C_SIZE / DOUBLE_NUM);

    // UB
    pipe_->InitBuffer(xMoveBuf_, DOUBLE_NUM, H1 * sizeof(half));

    // 全局软同步
    pipe_->InitBuffer(workQueue_, 1, coreNum * DEFAULT_SYNCALL_NEED_SIZE * sizeof(int32_t));
}

__aicore__ inline void AddLoraKernelBase310::xDataRearrange()
{
    uint32_t dataMovePingPong = 0;
    event_t eventdataMove = dataMovePingPong ? EVENT_ID0 : EVENT_ID1;
    for (uint64_t taskLoop = 0; taskLoop < batchNumCurCore; taskLoop++) {
        uint64_t currBatchIdx = batchOffsetCurCore + taskLoop;
        int32_t indiceIdx = indicesGm_.GetValue(currBatchIdx);
        if (indiceIdx < 0 || indiceIdx == wBatch) {
            continue;
        }
        uint32_t loraGroupOffset = batchAccumulationByCurLora[indiceIdx];
        uint32_t xIndiceAddr = loraGroupOffset * H1 + batchOffsetInLoraList_.GetValue(currBatchIdx) * H1;

        SetFlag<HardEvent::S_MTE2>(eventdataMove);
        WaitFlag<HardEvent::S_MTE2>(eventdataMove);

        uint64_t currQueryAddr = currBatchIdx * H1;
        LocalTensor<half> xIndiceTensor = xMoveBuf_.AllocTensor<half>();
        DataCopy(xIndiceTensor, xGm_[currQueryAddr], H1);
        xMoveBuf_.EnQue(xIndiceTensor);
        xIndiceTensor = xMoveBuf_.DeQue<half>();

        SetFlag<HardEvent::MTE2_MTE3>(eventdataMove);
        WaitFlag<HardEvent::MTE2_MTE3>(eventdataMove);
        DataCopy(xRearrangeGm_[xIndiceAddr], xIndiceTensor, H1);

        SetFlag<HardEvent::MTE3_S>(eventdataMove);
        WaitFlag<HardEvent::MTE3_S>(eventdataMove);
        xMoveBuf_.FreeTensor(xIndiceTensor);

        dataMovePingPong = 1 - dataMovePingPong;
    }
}

__aicore__ inline void AddLoraKernelBase310::CopyInL1A(
    GlobalTensor<half> srcGm, uint32_t mToProcess, uint32_t kToProcess, uint32_t srcDvalue)
{
    AscendC::LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();

    if (mToProcess == 1) {
        DataCopy(a1Local, srcGm, kToProcess);
    } else {
        Nd2NzParams copyInA1Params;
        copyInA1Params.ndNum = 1;
        copyInA1Params.nValue = mToProcess;
        copyInA1Params.dValue = kToProcess;
        copyInA1Params.srcNdMatrixStride = 0;
        copyInA1Params.srcDValue = srcDvalue; // 源操作数同一nd矩阵的相邻行起始地址间的偏移
        copyInA1Params.dstNzC0Stride = RoundUp16(mToProcess);
        copyInA1Params.dstNzNStride = 1;
        copyInA1Params.dstNzMatrixStride = 0;

        DataCopy(a1Local, srcGm, copyInA1Params); // nd -> nz
    }

    inQueueA1.EnQue(a1Local);
}

__aicore__ inline void AddLoraKernelBase310::CopyInL1BNd(
    GlobalTensor<half> srcWeightGm, uint32_t kToProcess, uint32_t nToProcess, uint32_t srcN, uint32_t srcDvalue)
{
    AscendC::LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();

    if (kToProcess == CUBE_BLOCK && srcDvalue == CUBE_BLOCK) {
        DataCopy(b1Local, srcWeightGm, nToProcess * kToProcess);
    } else {
        Nd2NzParams nd2nzB1Params;
        nd2nzB1Params.ndNum = 1;
        nd2nzB1Params.nValue = nToProcess;
        nd2nzB1Params.dValue = kToProcess;
        nd2nzB1Params.srcNdMatrixStride = 0;
        nd2nzB1Params.srcDValue = srcDvalue;
        nd2nzB1Params.dstNzC0Stride = RoundUp16(nToProcess);
        nd2nzB1Params.dstNzNStride = 1;
        nd2nzB1Params.dstNzMatrixStride = 0;

        DataCopy(b1Local, srcWeightGm, nd2nzB1Params);
    }

    inQueueB1.EnQue(b1Local);
}

__aicore__ inline void AddLoraKernelBase310::SplitA(uint32_t mToProcess, uint32_t kToProcess)
{
    AscendC::LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
    AscendC::LocalTensor<half> a2Local = inQueueA2.AllocTensor<half>();

    if (mToProcess == 1) {
        LoadData(a2Local, a1Local, LoadData2dParams(0, DivCeil(kToProcess, CUBE_BLOCK_SIZE), 1, 0, 0, false, 0));
    } else {
        uint32_t dstOffset = CeilCubeBlock(kToProcess) * CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;
        // Nz -> Zz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = CeilCubeBlock(kToProcess);
        loadDataParams.srcStride = CeilCubeBlock(mToProcess);
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        for (int i = 0; i < CeilCubeBlock(mToProcess); i++) {
            LoadData(a2Local[i * dstOffset], a1Local[i * srcOffset], loadDataParams);
        }
    }

    inQueueA2.EnQue<half>(a2Local);
    inQueueA1.FreeTensor(a1Local);
}

__aicore__ inline void AddLoraKernelBase310::SplitB(uint32_t kToProcess, uint32_t nToProcess)
{
    AscendC::LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
    AscendC::LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();

    // Nz -> Zn
    AscendC::LoadData2DParams loadDataParams;
    loadDataParams.repeatTimes = CeilCubeBlock(nToProcess) * CeilCubeBlock(kToProcess);
    loadDataParams.srcStride = 1;
    loadDataParams.dstGap = 0;
    loadDataParams.ifTranspose = false;
    LoadData(b2Local, b1Local, loadDataParams);

    inQueueB2.EnQue<half>(b2Local);
    inQueueB1.FreeTensor(b1Local);
}

__aicore__ inline void AddLoraKernelBase310::doMatmul(
    LocalTensor<float> l0cLocal, LocalTensor<half> l0aLocal, LocalTensor<half> l0bLocal, uint32_t kIdx_,
    uint32_t mToProcess, uint32_t kToProcess, uint32_t nToProcess)
{
    AscendC::MmadParams mmadParams;
    mmadParams.m = mToProcess;
    mmadParams.n = nToProcess;
    mmadParams.k = kToProcess;
    if (kIdx_ == 0) {
        mmadParams.cmatrixInitVal = true;
    } else {
        mmadParams.cmatrixInitVal = false;
    }

    AscendC::Mmad(l0cLocal, l0aLocal, l0bLocal, mmadParams);
}

__aicore__ inline uint32_t AddLoraKernelBase310::RoundUp16(uint32_t len)
{
    return (len + CUBE_BLOCK - 1) / CUBE_BLOCK * CUBE_BLOCK;
}

__aicore__ inline uint32_t AddLoraKernelBase310::CeilCubeBlock(uint32_t len)
{
    return (len + CUBE_BLOCK - 1) / CUBE_BLOCK;
}
} // namespace AddLora310

#endif // ADD_LORA_BASE_310_H
