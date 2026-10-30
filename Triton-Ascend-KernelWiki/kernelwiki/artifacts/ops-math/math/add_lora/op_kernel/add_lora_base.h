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
 * \file add_lora_base.h
 * \brief
 */
#ifndef ADD_LORA_BASE_H
#define ADD_LORA_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
class AddLoraKernelBase
{
public:
    __aicore__ inline AddLoraKernelBase(){};
    __aicore__ inline void Init(
        GM_ADDR y, GM_ADDR x, GM_ADDR weightB, GM_ADDR indices, GM_ADDR weightA, GM_ADDR y_out, GM_ADDR workSpace,
        const AddLoraTilingData& tiling, TPipe* pipe);
    __aicore__ inline void Process();

protected:
    // params
    uint32_t usedCoreNum;
    uint32_t layer;
    uint32_t batch;
    uint32_t H1;
    uint32_t H2;
    uint32_t R;
    uint32_t wBatch;
    uint32_t layer_idx;
    float scale;
    uint32_t y_offset;
    uint32_t y_slice_size;
    int32_t coreId;
    int32_t coreNum;
    uint32_t taskNumPerCore;
    uint32_t subBlockIdx_;
    uint32_t addLoraFlag;
    uint32_t y_column;

    static constexpr uint32_t MAX_ND_NZ_STRIDE_SIZE = 65535;
    static constexpr uint32_t B32_BLOCK_ALIGN_NUM = 16;
    static constexpr uint32_t CUBE_BLOCK = 16;
    static constexpr uint32_t CUBE_BLOCK_SIZE = 16 * 16;
    static constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
    static constexpr uint32_t HALF_SIZE = 2;
    static constexpr uint32_t FRACTAL_SIZE = 1024;
    static constexpr int64_t BLOCK_SIZE = 32;
    static constexpr uint32_t BLOCK_HALF_SIZE = 32 / sizeof(int16_t);
    static constexpr uint64_t SYNC_MODE0 = 0;
    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_AIV_ONLY_ALL_FLAG = 0;
    static constexpr uint64_t SYNC_AIC_ONLY_ALL_FLAG = 1;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 2;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 4;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG2 = 6;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG2 = 8;
    static constexpr uint32_t MAX_WEIGHT_SIZE = 40;
    static constexpr uint32_t DEFAULT_UB_SPLIT_NUM = 12 * 1024;
    static constexpr uint32_t MIN_SPLIT = 128;
    static constexpr uint32_t SINGLE_L0_SIZE = 16 * 1024;
    static constexpr uint32_t OFFSET_TWO = 2;
    static constexpr uint32_t OFFSET_THREE = 3;
    static constexpr uint32_t OFFSET_FOUR = 4;
    static constexpr uint32_t OFFSET_SIX = 6;
    static constexpr uint32_t AIV_AIC_RATIO = 2;

    static constexpr uint32_t TOTAL_USED_UB_SIZE = 192 * 1024;
    static constexpr uint32_t TOTAL_USED_L0A_SIZE = 64 * 1024;
    static constexpr uint32_t TOTAL_USED_L0B_SIZE = 64 * 1024;
    static constexpr uint32_t TOTAL_USED_L0C_SIZE = 128 * 1024;
    static constexpr uint32_t TOTAL_USED_L1_SIZE = 512 * 1024;
    // L1
    static constexpr uint32_t L1_ADDR = 0;
    event_t eventId;

    uint32_t sumIndiceCount[MAX_WEIGHT_SIZE];
    uint32_t batchIndiceCount[MAX_WEIGHT_SIZE] = {0};
    // GM
    GlobalTensor<half> yGm_;
    GlobalTensor<half> xGm_;
    GlobalTensor<half> waGm_;
    GlobalTensor<half> wbGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<half> youtputGm_;
    GlobalTensor<half> xIndiceGm_;
    GlobalTensor<half> tmpGm_;
    GlobalTensor<half> mm2Gm_;
    GlobalTensor<uint32_t> mOffsetBatch;
    GlobalTensor<uint32_t> IndicebymOffset;
    GlobalTensor<uint32_t> countIndiceBatch;

    TPipe* pipe_;
    TQue<TPosition::VECOUT, DOUBLE_BUFFER_NUM> xMoveBuf_;
    //  // UB
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<half> mmResUb[DOUBLE_BUFFER_NUM];
    LocalTensor<half> yUb[DOUBLE_BUFFER_NUM];
    LocalTensor<half> yOffsetUb[DOUBLE_BUFFER_NUM];
    // L1
    TBuf<TPosition::A1> A1Buf_;
    LocalTensor<half> queryL1_[DOUBLE_BUFFER_NUM];
    TBuf<TPosition::B1> B1Buf_;
    LocalTensor<half> keyL1_[DOUBLE_BUFFER_NUM];
    // L0A
    TBuf<TPosition::A2> l0ABuf_;
    LocalTensor<half> queryL0A_[DOUBLE_BUFFER_NUM];
    // L0B
    TBuf<TPosition::B2> l0BBuf_;
    LocalTensor<half> keyL0B_[DOUBLE_BUFFER_NUM];
    // L0C
    TBuf<TPosition::CO1> l0CBuf_;
    LocalTensor<float> matmull0C_;
    __aicore__ inline void DataCopyL1Nz(uint32_t rowNum, uint32_t colNum, uint32_t colSize,
        const GlobalTensor<half> &srcGm, const LocalTensor<half> &dstTensor);
    __aicore__ inline void CopyIn(
        uint32_t weightIdx, uint32_t kIdx_, uint32_t mInCore, uint32_t k, uint32_t mInCoreOffset, uint32_t kOffset,
        uint32_t pingPongFlag);
    __aicore__ inline void SplitA(uint32_t mInCore, uint32_t k, uint32_t mInCoreOffset, uint32_t pingPongFlag);
    __aicore__ inline void SplitB(
        uint32_t weightIdx, uint32_t kIdx_, uint32_t mInCore, uint32_t k, uint32_t mInCoreOffset,
        uint32_t pingPongFlag);
    __aicore__ inline void Compute(
        uint32_t weightIdx, uint32_t kIdx_, uint32_t mInCore, uint32_t k, uint32_t n, uint32_t pingPongFlag);
    __aicore__ inline void ComputeMM2(
        uint32_t mInCoreOffset, uint32_t mInCore, uint32_t k, uint32_t n, uint32_t nIncoreOffset, uint32_t kIdx_,
        uint32_t nidx, uint32_t pingPongFlag);
    __aicore__ inline void CopyWbB(
        uint32_t weightIdx, uint32_t nIdx, uint32_t nInCore, uint32_t k, uint32_t kidx, uint32_t nIncoreOffset,
        uint32_t nSplit, uint32_t kSplit, uint32_t pingPongFlag);
    __aicore__ inline void GlobalTensorInit(
        GM_ADDR y, GM_ADDR x, GM_ADDR weightB, GM_ADDR indices, GM_ADDR weightA, GM_ADDR yOut, GM_ADDR workSpace);
    __aicore__ inline void CopyL0C2GM(
        uint32_t mInCoreOffset, uint32_t mInCore, uint32_t n, uint32_t nidx, uint32_t nSplit, uint32_t nIncoreOffset);
    __aicore__ inline void CopyInA2(
        uint32_t mInCoreOffset, uint32_t mInCore, uint32_t nInCore, uint32_t nId, uint32_t nSplit,
        uint32_t pingPongFlag);
    __aicore__ inline void CopyOutRes(
        uint32_t mInVector, uint32_t mVectorOffset, uint32_t nInCore, uint32_t nIncoreOffset, uint32_t pingPongFlag);
    __aicore__ inline void InitMatmulBuffer();
    __aicore__ inline void InitDataMoveBuffer();
    __aicore__ inline void InitVectorBuffer();
    __aicore__ inline void QueryDataMove();
    __aicore__ inline void ComputeBatchIndice();

    template <pipe_t pipe>
    __aicore__ inline void SyncAicOnly(uint16_t eventId);

    template <pipe_t pipe>
    __aicore__ inline void SyncAivOnly(uint16_t eventId);

    template <pipe_t pipe>
    __aicore__ inline void VectorNotifyCube(uint16_t aivOnlyEventId, uint16_t aiv2AicEventId);

    __aicore__ inline void CubeWaitVector(uint16_t aiv2AicEventId);

    template <pipe_t pipe>
    __aicore__ inline void CubeNotifyVector(uint16_t aic2AivEventId);

    __aicore__ inline void VectorWaitCube(uint16_t aic2AivEventId);
    __aicore__ inline uint32_t CeilCubeBlock(uint32_t len);
};

__aicore__ inline void AddLoraKernelBase::Init(
    GM_ADDR y, GM_ADDR x, GM_ADDR weightB, GM_ADDR indices, GM_ADDR weightA, GM_ADDR y_out, GM_ADDR workSpace,
    const AddLoraTilingData& tiling, TPipe* pipe)
{
    pipe_ = pipe;
    coreId = GetBlockIdx();
    coreNum = GetBlockNum();
    subBlockIdx_ = GetSubBlockIdx();
    batch = tiling.batch;
    layer = tiling.layer;
    H1 = tiling.H1;
    H2 = tiling.H2;
    R = tiling.R;
    wBatch = tiling.wBatch;
    layer_idx = tiling.layer_idx;
    scale = tiling.scale;
    y_offset = tiling.y_offset;
    y_slice_size = tiling.y_slice_size;
    taskNumPerCore = tiling.taskNumPerCore;
    addLoraFlag = tiling.addLoraFlag;
    y_column = tiling.y_column;
    GlobalTensorInit(y, x, weightB, indices, weightA, y_out, workSpace);
}

__aicore__ inline void AddLoraKernelBase::GlobalTensorInit(
    GM_ADDR y, GM_ADDR x, GM_ADDR weightB, GM_ADDR indices, GM_ADDR weightA, GM_ADDR y_out, GM_ADDR workSpace)
{
    yGm_.SetGlobalBuffer((__gm__ half*)y, batch * y_column);
    xGm_.SetGlobalBuffer((__gm__ half*)x, batch * H1);
    waGm_.SetGlobalBuffer((__gm__ half*)weightA, wBatch * layer * R * H1);
    wbGm_.SetGlobalBuffer((__gm__ half*)weightB, wBatch * layer * R * H2);
    indicesGm_.SetGlobalBuffer((__gm__ int32_t*)indices, batch);
    youtputGm_.SetGlobalBuffer((__gm__ half*)y_out, batch * y_column);
    uint32_t dataMoveWorkspaceOffset = 0;
    uint32_t xIndiceWorkspaceOffset = dataMoveWorkspaceOffset + batch * H1 * sizeof(half);
    uint32_t mm1WorkspaceOffset = xIndiceWorkspaceOffset + CeilCubeBlock(batch) * CUBE_BLOCK * R * sizeof(half);
    uint32_t mm2WorkspaceOffset = mm1WorkspaceOffset + CeilCubeBlock(batch) * CUBE_BLOCK * H2 * sizeof(half);
    uint32_t mOffsetBatchOffset = mm2WorkspaceOffset + batch * sizeof(uint32_t);
    uint32_t countIndiceBatchOffset = mOffsetBatchOffset + batch * sizeof(uint32_t);
    xIndiceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workSpace + dataMoveWorkspaceOffset));
    tmpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workSpace + xIndiceWorkspaceOffset));
    mm2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workSpace + mm1WorkspaceOffset));
    mOffsetBatch.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(workSpace + mm2WorkspaceOffset));
    countIndiceBatch.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(workSpace + mOffsetBatchOffset));
    IndicebymOffset.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(workSpace + countIndiceBatchOffset));
}

__aicore__ inline void AddLoraKernelBase::InitMatmulBuffer()
{
    // L1
    pipe_->InitBuffer(A1Buf_, TOTAL_USED_L1_SIZE / HALF_SIZE);
    queryL1_[0] = A1Buf_.GetWithOffset<half>(SINGLE_L0_SIZE, L1_ADDR);
    queryL1_[1] = A1Buf_.GetWithOffset<half>(SINGLE_L0_SIZE, L1_ADDR + SINGLE_L0_SIZE * sizeof(half));
    pipe_->InitBuffer(B1Buf_, TOTAL_USED_L1_SIZE / HALF_SIZE);
    keyL1_[0] = B1Buf_.GetWithOffset<half>(SINGLE_L0_SIZE, L1_ADDR);
    keyL1_[1] = B1Buf_.GetWithOffset<half>(SINGLE_L0_SIZE, L1_ADDR + SINGLE_L0_SIZE * sizeof(half));
    // L0A
    pipe_->InitBuffer(l0ABuf_, TOTAL_USED_L0A_SIZE);
    queryL0A_[0] = l0ABuf_.GetWithOffset<half>(SINGLE_L0_SIZE, 0);
    queryL0A_[1] = l0ABuf_.GetWithOffset<half>(SINGLE_L0_SIZE, SINGLE_L0_SIZE * sizeof(half));
    // L0B
    pipe_->InitBuffer(l0BBuf_, TOTAL_USED_L0B_SIZE);
    keyL0B_[0] = l0BBuf_.GetWithOffset<half>(SINGLE_L0_SIZE, 0);
    keyL0B_[1] = l0BBuf_.GetWithOffset<half>(SINGLE_L0_SIZE, SINGLE_L0_SIZE * sizeof(half));
    // L0C
    pipe_->InitBuffer(l0CBuf_, TOTAL_USED_L0C_SIZE);
    matmull0C_ = l0CBuf_.GetWithOffset<float>(SINGLE_L0_SIZE, 0);
}

__aicore__ inline void AddLoraKernelBase::InitDataMoveBuffer()
{
    pipe_->InitBuffer(xMoveBuf_, DOUBLE_BUFFER_NUM, H1 * sizeof(half));
}

__aicore__ inline void AddLoraKernelBase::InitVectorBuffer()
{ // 12 * 1024 DEFAULT_UB_SPLIT_NUM 表示缓冲区的大小
    pipe_->InitBuffer(ubBuf_, TOTAL_USED_UB_SIZE);
    mmResUb[0] = ubBuf_.GetWithOffset<half>(DEFAULT_UB_SPLIT_NUM, 0);
    mmResUb[1] = ubBuf_.GetWithOffset<half>(
        DEFAULT_UB_SPLIT_NUM, DEFAULT_UB_SPLIT_NUM * sizeof(half)); // 12 * 1024 * sizeof(half) 表示偏移量
    yUb[0] = ubBuf_.GetWithOffset<half>(
        DEFAULT_UB_SPLIT_NUM, OFFSET_TWO * DEFAULT_UB_SPLIT_NUM * sizeof(half)); // 24 * 1024 * sizeof(half) 表示偏移量
    yUb[1] = ubBuf_.GetWithOffset<half>(
        DEFAULT_UB_SPLIT_NUM,
        OFFSET_THREE * DEFAULT_UB_SPLIT_NUM * sizeof(half)); // 36 * 1024 * sizeof(half) 表示偏移量
    yOffsetUb[0] = ubBuf_.GetWithOffset<half>(
        OFFSET_TWO * DEFAULT_UB_SPLIT_NUM,
        OFFSET_FOUR * DEFAULT_UB_SPLIT_NUM * sizeof(half)); // 48 * 1024 * sizeof(half) 表示偏移量
    yOffsetUb[1] = ubBuf_.GetWithOffset<half>(
        OFFSET_TWO * DEFAULT_UB_SPLIT_NUM,
        OFFSET_SIX * DEFAULT_UB_SPLIT_NUM * sizeof(half)); // 72 * 1024 * sizeof(half) 表示偏移量
}

template <pipe_t pipe = PIPE_S>
__aicore__ inline void AddLoraKernelBase::SyncAicOnly(uint16_t eventId)
{
    CrossCoreSetFlag<SYNC_MODE0, pipe>(eventId);
    CrossCoreWaitFlag(eventId);
}

template <pipe_t pipe = PIPE_S>
__aicore__ inline void AddLoraKernelBase::SyncAivOnly(uint16_t eventId)
{
    CrossCoreSetFlag<SYNC_MODE0, pipe>(eventId);
    CrossCoreWaitFlag(eventId);
}

template <pipe_t pipe = PIPE_S>
__aicore__ inline void AddLoraKernelBase::VectorNotifyCube(uint16_t aivOnlyEventId, uint16_t aiv2AicEventId)
{
    SyncAivOnly<pipe>(aivOnlyEventId);
    CrossCoreSetFlag<SYNC_MODE2, pipe>(aiv2AicEventId);
}

__aicore__ inline void AddLoraKernelBase::CubeWaitVector(uint16_t aiv2AicEventId)
{
    CrossCoreWaitFlag(aiv2AicEventId);
}

template <pipe_t pipe = PIPE_S>
__aicore__ inline void AddLoraKernelBase::CubeNotifyVector(uint16_t aic2AivEventId)
{
    CrossCoreSetFlag<SYNC_MODE2, pipe>(aic2AivEventId);
}

__aicore__ inline void AddLoraKernelBase::VectorWaitCube(uint16_t aic2AivEventId)
{
    CrossCoreWaitFlag(aic2AivEventId);
}

__aicore__ inline uint32_t AddLoraKernelBase::CeilCubeBlock(uint32_t len)
{
    return (len + CUBE_BLOCK - 1) / CUBE_BLOCK;
}

__aicore__ inline void AddLoraKernelBase::DataCopyL1Nz(uint32_t rowNum, uint32_t colNum, uint32_t colSize,
    const GlobalTensor<half> &srcGm, const LocalTensor<half> &dstTensor)
{
    uint32_t blockNum = CeilDiv(colNum, CUBE_BLOCK);
    for (uint32_t blockIdx = 0; blockIdx < blockNum; blockIdx++) {
        uint32_t copyLen = CUBE_BLOCK;
        uint64_t gmOffset = blockIdx * CUBE_BLOCK;
        AscendC::DataCopyParams dataCopyParams{static_cast<uint16_t>(rowNum),
            static_cast<uint16_t>(copyLen / B32_BLOCK_ALIGN_NUM),
            static_cast<uint16_t>((colSize - CUBE_BLOCK) / B32_BLOCK_ALIGN_NUM), 0};
        uint32_t rowNumAlign = CeilDiv(rowNum, CUBE_BLOCK) * CUBE_BLOCK;
        AscendC::DataCopy(dstTensor[CUBE_BLOCK * rowNumAlign * blockIdx], srcGm[gmOffset], dataCopyParams);
    }
}

__aicore__ inline void AddLoraKernelBase::CopyIn(
    uint32_t weightIdx, uint32_t kIdx_, uint32_t mInCore, uint32_t k, uint32_t mInCoreOffset, uint32_t kOffset,
    uint32_t pingPongFlag)
{
    eventId = pingPongFlag ? EVENT_ID0 : EVENT_ID1;
    SetFlag<HardEvent::S_MTE2>(eventId);
    WaitFlag<HardEvent::S_MTE2>(eventId);
    SetFlag<HardEvent::FIX_MTE2>(eventId);
    WaitFlag<HardEvent::FIX_MTE2>(eventId);
    uint32_t xGmOffset = (mInCoreOffset) * H1 + kIdx_ * kOffset;
    if (mInCore == 1) {
        DataCopy(queryL1_[pingPongFlag], xIndiceGm_[xGmOffset], k);
    } else {
        if (H1 > MAX_ND_NZ_STRIDE_SIZE) {
            DataCopyL1Nz(mInCore, k, H1, xIndiceGm_[xGmOffset], queryL1_[pingPongFlag]);
        } else {
            Nd2NzParams copyinA1Params;
            copyinA1Params.ndNum = 1;
            copyinA1Params.nValue = mInCore;
            copyinA1Params.dValue = k;
            copyinA1Params.srcNdMatrixStride = 0;
            copyinA1Params.srcDValue = H1; // 源操作数同一个ND的相邻行起始地址偏移
            copyinA1Params.dstNzC0Stride = CeilCubeBlock(mInCore) * CUBE_BLOCK;
            copyinA1Params.dstNzNStride = 1;
            copyinA1Params.dstNzMatrixStride = 0;
            DataCopy(queryL1_[pingPongFlag], xIndiceGm_[xGmOffset], copyinA1Params); // nd -> nz
        }
    }
    AscendC::PipeBarrier<PIPE_MTE2>();
    uint32_t weightOffset = weightIdx * layer * H1 * R + layer_idx * H1 * R + kIdx_ * kOffset;
    if (H1 > MAX_ND_NZ_STRIDE_SIZE) {
        DataCopyL1Nz(R, k, H1, waGm_[weightOffset], keyL1_[pingPongFlag]);
    } else {
        Nd2NzParams nd2nzB1Params;
        nd2nzB1Params.ndNum = 1;
        nd2nzB1Params.nValue = R;
        nd2nzB1Params.dValue = k;
        nd2nzB1Params.srcNdMatrixStride = 0;
        nd2nzB1Params.srcDValue = H1;
        nd2nzB1Params.dstNzC0Stride = CeilCubeBlock(R) * CUBE_BLOCK;
        nd2nzB1Params.dstNzNStride = 1;
        nd2nzB1Params.dstNzMatrixStride = 0;
        DataCopy(keyL1_[pingPongFlag], waGm_[weightOffset], nd2nzB1Params);
    }
    SetFlag<HardEvent::MTE2_MTE1>(eventId);
    WaitFlag<HardEvent::MTE2_MTE1>(eventId);
}

__aicore__ inline void AddLoraKernelBase::SplitA(
    uint32_t mInCore, uint32_t k, uint32_t mInCoreOffset, uint32_t pingPongFlag)
{
    eventId = pingPongFlag ? EVENT_ID0 : EVENT_ID1;
    if (mInCore == 1) {
        LoadData(
            queryL0A_[pingPongFlag], queryL1_[pingPongFlag],
            LoadData2dParams(0, DivCeil(k, 256), 1, 0, 0, false, 0)); // 256 is the block size for loading data
    } else {
        uint32_t dstOffset = CeilCubeBlock(k) * CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;
        // Nz -> Zz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = CeilCubeBlock(k);
        loadDataParams.srcStride = CeilCubeBlock(mInCore);
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        for (int i = 0; i < CeilCubeBlock(mInCore); i++) {
            LoadData(queryL0A_[pingPongFlag][i * dstOffset], queryL1_[pingPongFlag][i * srcOffset], loadDataParams);
        }
    }
    AscendC::PipeBarrier<PIPE_MTE1>();
    SetFlag<HardEvent::MTE1_MTE2>(eventId);
    WaitFlag<HardEvent::MTE1_MTE2>(eventId);
}

__aicore__ inline void AddLoraKernelBase::SplitB(
    uint32_t weightIdx, uint32_t kIdx_, uint32_t mInCore, uint32_t k, uint32_t mInCoreOffset, uint32_t pingPongFlag)
{
    // Nz -> Zn
    AscendC::LoadData2DParams loadDataParams;
    loadDataParams.repeatTimes = CeilCubeBlock(R) * CeilCubeBlock(k);
    ;
    loadDataParams.srcStride = 1;
    loadDataParams.dstGap = 0;
    loadDataParams.ifTranspose = false;
    LoadData(keyL0B_[pingPongFlag], keyL1_[pingPongFlag], loadDataParams);
    AscendC::PipeBarrier<PIPE_MTE1>();
}

__aicore__ inline void AddLoraKernelBase::Compute(
    uint32_t weightIdx, uint32_t kIdx_, uint32_t mInCore, uint32_t k, uint32_t n, uint32_t pingPongFlag)
{
    AscendC::MmadParams mmadParams;
    mmadParams.m = mInCore;
    mmadParams.n = n;
    mmadParams.k = k;
    if (kIdx_ == 0) {
        mmadParams.cmatrixInitVal = true;
    } else {
        mmadParams.cmatrixInitVal = false;
        mmadParams.cmatrixSource = false;
    }
    eventId = pingPongFlag ? EVENT_ID0 : EVENT_ID1;
    SetFlag<HardEvent::MTE1_M>(eventId);
    WaitFlag<HardEvent::MTE1_M>(eventId);
    Mmad(matmull0C_, queryL0A_[pingPongFlag], keyL0B_[pingPongFlag], mmadParams);
    AscendC::PipeBarrier<PIPE_M>();
}

__aicore__ inline void AddLoraKernelBase::CopyWbB(
    uint32_t weightIdx, uint32_t nIdx, uint32_t nInCore, uint32_t k, uint32_t kidx, uint32_t nIncoreOffset,
    uint32_t nSplit, uint32_t kSplit, uint32_t pingPongFlag)
{
    eventId = pingPongFlag ? EVENT_ID0 : EVENT_ID1;
    SetFlag<HardEvent::S_MTE2>(eventId);
    WaitFlag<HardEvent::S_MTE2>(eventId);
    Nd2NzParams nd2nzB1Params;
    nd2nzB1Params.ndNum = 1;
    nd2nzB1Params.nValue = nInCore;
    nd2nzB1Params.dValue = k;
    nd2nzB1Params.srcNdMatrixStride = 0;
    nd2nzB1Params.srcDValue = R; // 源操作数同一个ND矩阵的相邻行起始地址偏移
    nd2nzB1Params.dstNzC0Stride = CeilCubeBlock(nInCore) * CUBE_BLOCK;
    nd2nzB1Params.dstNzNStride = 1;
    nd2nzB1Params.dstNzMatrixStride = 0;
    uint32_t weightOffset = weightIdx * layer * H2 * R + layer_idx * H2 * R;
    if (R == CUBE_BLOCK && k == CUBE_BLOCK) { // NZ 直接搬
        // R 和 k 的值 16时, ND格式 和NZ格式的数据在GM上 顺序一致
        DataCopy(keyL1_[pingPongFlag], wbGm_[weightOffset + (nIncoreOffset + nIdx * nSplit) * R], nInCore * k);
    } else {
        DataCopy(
            keyL1_[pingPongFlag], wbGm_[weightOffset + (nIncoreOffset + nIdx * nSplit) * R + kidx * kSplit],
            nd2nzB1Params);
    }
    AscendC::PipeBarrier<PIPE_MTE2>();
    SetFlag<HardEvent::MTE2_MTE1>(eventId);
    WaitFlag<HardEvent::MTE2_MTE1>(eventId);
    AscendC::LoadData2DParams loadDataParams;
    loadDataParams.repeatTimes = CeilCubeBlock(nInCore) * CeilCubeBlock(k);
    loadDataParams.srcStride = 1;
    loadDataParams.dstGap = 0;
    loadDataParams.ifTranspose = false;
    LoadData(keyL0B_[pingPongFlag], keyL1_[pingPongFlag], loadDataParams);
    SetFlag<HardEvent::MTE1_M>(eventId);
    WaitFlag<HardEvent::MTE1_M>(eventId);
    AscendC::PipeBarrier<PIPE_MTE1>();
}

__aicore__ inline void AddLoraKernelBase::ComputeMM2(
    uint32_t mInCoreOffset, uint32_t mInCore, uint32_t k, uint32_t n, uint32_t nIncoreOffset, uint32_t kIdx_,
    uint32_t nidx, uint32_t pingPongFlag)
{
    AscendC::MmadParams mmad2Params;
    mmad2Params.m = mInCore;
    mmad2Params.n = n;
    mmad2Params.k = k;
    if (kIdx_ == 0) {
        mmad2Params.cmatrixInitVal = true;
    } else {
        mmad2Params.cmatrixInitVal = false;
        mmad2Params.cmatrixSource = false;
    }
    eventId = pingPongFlag ? EVENT_ID0 : EVENT_ID1;
    SetFlag<HardEvent::MTE1_M>(eventId);
    WaitFlag<HardEvent::MTE1_M>(eventId);
    Mmad(matmull0C_, queryL0A_[pingPongFlag], keyL0B_[pingPongFlag], mmad2Params);
    AscendC::PipeBarrier<PIPE_M>();
    SetFlag<HardEvent::M_MTE2>(eventId);
    WaitFlag<HardEvent::M_MTE2>(eventId);
    SetFlag<HardEvent::M_FIX>(eventId);
    WaitFlag<HardEvent::M_FIX>(eventId);
}

__aicore__ inline void AddLoraKernelBase::CopyL0C2GM(
    uint32_t mInCoreOffset, uint32_t mInCore, uint32_t n, uint32_t nidx, uint32_t nSplit, uint32_t nIncoreOffset)
{
    uint32_t resOffset = (mInCoreOffset)*H2 + nIncoreOffset + nidx * nSplit;
    FixpipeParams<float> fixpipeParams;
    fixpipeParams.cburstNum = CeilCubeBlock(n); // coutBlocks = (n + 16 - 1) / 16;
    fixpipeParams.burstLen =
        static_cast<uint16_t>(mInCore * 16 * sizeof(float) / 32); // static_cast<uint16_t>(m * 16 * sizeof(float) / 32)
    fixpipeParams.srcStride = 0;
    fixpipeParams.dstStride = H2; // 同一个ND矩阵的相邻行起始地址偏移
    fixpipeParams.quantParams = {QuantMode_t::F322F16};
    fixpipeParams.nz2ndParams.nz2ndEn = true;
    fixpipeParams.nz2ndParams.ndNum = 1;
    fixpipeParams.nz2ndParams.srcNdStride = 0; // ndNum 为1
    fixpipeParams.nz2ndParams.dstNdStride = 0; // ndNum 为1
    fixpipeParams.nz2ndParams.originalNSize = n;
    Fixpipe(mm2Gm_[resOffset], matmull0C_, fixpipeParams);
}

__aicore__ inline void AddLoraKernelBase::CopyInA2(
    uint32_t mInCoreOffset, uint32_t mInCore, uint32_t nInCore, uint32_t nId, uint32_t nSplit, uint32_t pingPongFlag)
{
    eventId = pingPongFlag ? EVENT_ID0 : EVENT_ID1;
    SetFlag<HardEvent::S_MTE2>(eventId);
    WaitFlag<HardEvent::S_MTE2>(eventId);
    SetFlag<HardEvent::M_MTE2>(eventId);
    WaitFlag<HardEvent::M_MTE2>(eventId);

    if (addLoraFlag) {
        if (R == CUBE_BLOCK && nInCore == CUBE_BLOCK) {
            // R 和 nInCore 的值 16时, ND格式 和NZ格式的数据在GM上 顺序一致
            DataCopy(
                queryL1_[pingPongFlag], tmpGm_[mInCoreOffset * nInCore],
                CeilCubeBlock(mInCore) * CeilCubeBlock(nInCore) * CUBE_BLOCK * CUBE_BLOCK);
        } else {
            Nd2NzParams copyinA1Params;
            copyinA1Params.ndNum = 1;
            copyinA1Params.nValue = mInCore;
            copyinA1Params.dValue = nInCore;
            copyinA1Params.srcNdMatrixStride = 0;
            copyinA1Params.srcDValue = R; // 源操作数同一个ND矩阵的相邻行起始地址偏移
            copyinA1Params.dstNzC0Stride = CeilCubeBlock(mInCore) * CUBE_BLOCK;
            copyinA1Params.dstNzNStride = 1;
            copyinA1Params.dstNzMatrixStride = 0;
            DataCopy(queryL1_[pingPongFlag], tmpGm_[mInCoreOffset * R + nId * nSplit], copyinA1Params); // nd -> nz
        }
    } else {
        if (mInCore == 1) {
            DataCopy(queryL1_[pingPongFlag], xIndiceGm_[mInCoreOffset * H1 + nId * nSplit], nInCore);
        } else {
            Nd2NzParams copyinA1Params;
            copyinA1Params.ndNum = 1;
            copyinA1Params.nValue = mInCore;
            copyinA1Params.dValue = nInCore;
            copyinA1Params.srcNdMatrixStride = 0;
            copyinA1Params.srcDValue = H1; // 源操作数同一个ND矩阵的相邻行起始地址偏移
            copyinA1Params.dstNzC0Stride = CeilCubeBlock(mInCore) * CUBE_BLOCK;
            copyinA1Params.dstNzNStride = 1;
            copyinA1Params.dstNzMatrixStride = 0;
            DataCopy(queryL1_[pingPongFlag], xIndiceGm_[mInCoreOffset * H1 + nId * nSplit], copyinA1Params); // nd -> nz
        }
    }
    AscendC::PipeBarrier<PIPE_MTE2>();
    SetFlag<HardEvent::MTE2_MTE1>(eventId);
    WaitFlag<HardEvent::MTE2_MTE1>(eventId);
}

__aicore__ inline void AddLoraKernelBase::QueryDataMove()
{
    uint32_t taskNum_ = (coreId < (batch % (coreNum * 2))) ? taskNumPerCore + 1 : taskNumPerCore;
    uint32_t startTaskOffset_ =
        (coreId < (batch % (coreNum * 2))) ? taskNum_ * coreId : batch - ((coreNum * 2 - coreId) * taskNum_);
    uint32_t dataMovePingPang = 0;
    for (uint64_t taskLoop = 0; taskLoop < taskNum_; taskLoop++) {
        event_t eventdataMove = dataMovePingPang ? EVENT_ID0 : EVENT_ID1;
        uint64_t currBatchIdx = startTaskOffset_ + taskLoop;
        int32_t indiceIdx = indicesGm_.GetValue(currBatchIdx);
        if (indiceIdx < 0) {
            indiceIdx = wBatch;
        }
        uint32_t indiceOffset = (indiceIdx == 0) ? 0 : sumIndiceCount[indiceIdx - 1];
        uint32_t xIndiceAddr = indiceOffset * H1 + countIndiceBatch.GetValue(currBatchIdx) * H1;

        SetFlag<HardEvent::S_MTE2>(eventdataMove);
        WaitFlag<HardEvent::S_MTE2>(eventdataMove);
        uint64_t currQueryAddr = currBatchIdx * H1;
        LocalTensor<half> xIndiceTensor = xMoveBuf_.AllocTensor<half>();
        DataCopyExtParams queryCopyInParams{1, H1 * static_cast<uint32_t>(HALF_SIZE), 0, 0, 0};
        DataCopyPadExtParams<half> queryCopyInPadParams{false, 0, 0, 0};
        DataCopyPad(xIndiceTensor, xGm_[currQueryAddr], queryCopyInParams, queryCopyInPadParams);
        xMoveBuf_.EnQue(xIndiceTensor);
        xIndiceTensor = xMoveBuf_.DeQue<half>();
        DataCopyExtParams copyQOutParams{1, H1 * static_cast<uint32_t>(HALF_SIZE), 0, 0, 0};
        SetFlag<HardEvent::MTE2_MTE3>(eventdataMove);
        WaitFlag<HardEvent::MTE2_MTE3>(eventdataMove);
        DataCopyPad(xIndiceGm_[xIndiceAddr], xIndiceTensor, copyQOutParams);
        SetFlag<HardEvent::MTE3_S>(eventdataMove);
        WaitFlag<HardEvent::MTE3_S>(eventdataMove);
        xMoveBuf_.FreeTensor(xIndiceTensor);
        dataMovePingPang = 1 - dataMovePingPang;
    }
}

__aicore__ inline void AddLoraKernelBase::ComputeBatchIndice()
{
    for (uint32_t batchIdx = 0; batchIdx < batch; batchIdx++) {
        int32_t indiceBatch = int32_t(indicesGm_.GetValue(batchIdx));
        if (indiceBatch < 0) {
            indiceBatch = wBatch;
        }
        countIndiceBatch.SetValue(batchIdx, batchIndiceCount[indiceBatch]);
        batchIndiceCount[indiceBatch] += 1;
    }
    uint32_t indice = 0;
    sumIndiceCount[0] = batchIndiceCount[0];
    for (uint32_t indiceId = 1; indiceId <= wBatch; indiceId++) {
        sumIndiceCount[indiceId] = sumIndiceCount[indiceId - 1] + batchIndiceCount[indiceId];
    }
    for (uint32_t batchIdx = 0; batchIdx < batch; batchIdx++) {
        int32_t indiceBatch = int32_t(indicesGm_.GetValue(batchIdx));
        if (indiceBatch < 0) {
            indiceBatch = wBatch;
        }
        uint32_t indiceOffset = (indiceBatch == 0) ? 0 : sumIndiceCount[indiceBatch - 1];
        mOffsetBatch.SetValue(indiceOffset + countIndiceBatch.GetValue(batchIdx), batchIdx);
        uint32_t indiceValue = (sumIndiceCount[indice] > batchIdx) ? indice : ++indice;
        IndicebymOffset.SetValue(batchIdx, indiceValue);
    }
}

__aicore__ inline void AddLoraKernelBase::CopyOutRes(
    uint32_t mInVector, uint32_t mVectorOffset, uint32_t nInCore, uint32_t nIncoreOffset, uint32_t pingPongFlag)
{
    for (uint32_t midx = 0; midx < mInVector; ++midx) {
        uint32_t mOffset = mVectorOffset + midx;
        for (uint32_t nidx = 0; nidx < DivCeil(nInCore, DEFAULT_UB_SPLIT_NUM); ++nidx) {
            uint32_t nInner = (nidx < (nInCore / DEFAULT_UB_SPLIT_NUM)) ? DEFAULT_UB_SPLIT_NUM :
                                                                          nInCore - nidx * DEFAULT_UB_SPLIT_NUM;
            eventId = pingPongFlag ? EVENT_ID0 : EVENT_ID1;
            uint32_t dstOffset =
                mOffsetBatch.GetValue(mOffset) * y_column + y_offset + nidx * DEFAULT_UB_SPLIT_NUM + nIncoreOffset;
            uint32_t resOffset = mOffset * H2 + nidx * DEFAULT_UB_SPLIT_NUM + nIncoreOffset;
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            SetFlag<HardEvent::S_MTE2>(eventId);
            WaitFlag<HardEvent::S_MTE2>(eventId);
            DataCopy(mmResUb[pingPongFlag], mm2Gm_[resOffset], CeilCubeBlock(nInner) * CUBE_BLOCK);
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
            Muls(
                mmResUb[pingPongFlag], mmResUb[pingPongFlag], static_cast<half>(scale),
                CeilCubeBlock(nInner) * CUBE_BLOCK);
            SetFlag<HardEvent::V_MTE3>(eventId);
            WaitFlag<HardEvent::V_MTE3>(eventId);
            SetAtomicAdd<half>();
            if (nInner % CUBE_BLOCK == 0) {
                DataCopy(yGm_[dstOffset], mmResUb[pingPongFlag], nInner);
            } else {
                DataCopyExtParams DataCopyParams{1, static_cast<uint32_t>(nInner * HALF_SIZE), 0, 0, 0};
                DataCopyPad(yGm_[dstOffset], mmResUb[pingPongFlag], DataCopyParams);
            }
            SetAtomicNone();
            PipeBarrier<PIPE_MTE3>();
            pingPongFlag = !pingPongFlag;
        }
    }
}
#endif // ADD_LORA_BASE_H