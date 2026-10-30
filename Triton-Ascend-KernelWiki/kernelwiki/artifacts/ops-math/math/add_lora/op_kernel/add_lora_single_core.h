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
 * \file add_lora_single_core.h
 * \brief
 */
#ifndef ADD_LORA_SINGLE_CORE_H
#define ADD_LORA_SINGLE_CORE_H
#include "add_lora_base.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;

class AddLoraSingleCoreBatchKernel : public AddLoraKernelBase
{
public:
    __aicore__ inline AddLoraSingleCoreBatchKernel(){};
    __aicore__ inline void Process();

protected:
    int32_t eventIDM_FIX = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::M_FIX));
    int32_t eventIDFIX_MTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_MTE2));
    __aicore__ inline void CopyL0CGM(uint32_t mInCore, uint32_t mInCoreOffset, uint32_t nInCore, uint32_t xIndiceAddr);

    __aicore__ inline void CopyInA(
        uint32_t weightIdx, uint32_t kIdx_, uint32_t mInCore, uint32_t k, uint32_t mInCoreOffset, uint32_t kOffset,
        uint32_t pingPongFlag);

    __aicore__ inline void ComputeWeightBMM(
        uint32_t nSplit, uint32_t nIncore, uint32_t nIncoreOffset, uint32_t pingPongFlag);

    __aicore__ inline void ComputeWeightAMM(
        uint32_t mInCore, uint32_t nSplit, uint32_t mInCoreOffset, uint32_t pingPongFlag);
};

__aicore__ inline void AddLoraSingleCoreBatchKernel::CopyL0CGM(
    uint32_t mInCore, uint32_t mInCoreOffset, uint32_t nInCore, uint32_t xIndiceAddr)
{
    SetFlag<HardEvent::M_FIX>(eventIDM_FIX);
    WaitFlag<HardEvent::M_FIX>(eventIDM_FIX);
    FixpipeParams<float> fixpipeParams;
    fixpipeParams.cburstNum = CeilCubeBlock(nInCore); // coutBlocks = (n + 16 - 1) / 16;
    fixpipeParams.burstLen = static_cast<uint16_t>(
        mInCore * CUBE_BLOCK * sizeof(float) / 32); // static_cast<uint16_t>(m * 16 * sizeof(float) / 32)
    fixpipeParams.srcStride = 0;
    fixpipeParams.dstStride = R; // 同一nd矩阵的相邻行起始地址间的偏移
    fixpipeParams.quantParams = {QuantMode_t::F322F16};
    fixpipeParams.nz2ndParams.nz2ndEn = true;
    fixpipeParams.nz2ndParams.ndNum = 1;
    fixpipeParams.nz2ndParams.srcNdStride = static_cast<uint16_t>(mInCore * nInCore * sizeof(float) / FRACTAL_SIZE);
    fixpipeParams.nz2ndParams.dstNdStride = static_cast<uint16_t>(mInCore * nInCore);
    fixpipeParams.nz2ndParams.originalNSize = nInCore;
    Fixpipe(tmpGm_[xIndiceAddr * nInCore], matmull0C_, fixpipeParams);
    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIX_MTE2);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(eventIDFIX_MTE2);
}

__aicore__ inline void AddLoraSingleCoreBatchKernel::CopyInA(
    uint32_t weightIdx, uint32_t kIdx_, uint32_t mInCore, uint32_t k, uint32_t mInCoreOffset, uint32_t kOffset,
    uint32_t pingPongFlag)
{
    eventId = pingPongFlag ? EVENT_ID0 : EVENT_ID1;
    SetFlag<HardEvent::S_MTE2>(eventId);
    WaitFlag<HardEvent::S_MTE2>(eventId);
    uint32_t xGmOffset = (mInCoreOffset) * H1 + kIdx_ * kOffset;
    if (mInCore == 1) {
        DataCopy(queryL1_[pingPongFlag], xGm_[xGmOffset], k);
    } else {
        if (H1 > MAX_ND_NZ_STRIDE_SIZE) {
            DataCopyL1Nz(mInCore, k, H1, xGm_[xGmOffset], queryL1_[pingPongFlag]);
        } else {
            Nd2NzParams copyinXGmParams;
            copyinXGmParams.ndNum = 1;
            copyinXGmParams.nValue = mInCore;
            copyinXGmParams.dValue = k;
            copyinXGmParams.srcNdMatrixStride = 0;
            copyinXGmParams.srcDValue = H1; // 源操作数同一nd矩阵的相邻行起始地址间的偏移
            copyinXGmParams.dstNzC0Stride = CeilCubeBlock(mInCore) * CUBE_BLOCK;
            copyinXGmParams.dstNzNStride = 1;
            copyinXGmParams.dstNzMatrixStride = 0;
            DataCopy(queryL1_[pingPongFlag], xGm_[xGmOffset], copyinXGmParams); // nd -> nz
        }
    }
    AscendC::PipeBarrier<PIPE_MTE2>();
    uint32_t weightOffset = weightIdx * layer * H1 * R + layer_idx * H1 * R + kIdx_ * kOffset;
    if (H1 > MAX_ND_NZ_STRIDE_SIZE) {
        DataCopyL1Nz(R, k, H1, waGm_[weightOffset], keyL1_[pingPongFlag]);
    } else {
        Nd2NzParams copyinB1Params;
        copyinB1Params.ndNum = 1;
        copyinB1Params.nValue = R;
        copyinB1Params.dValue = k;
        copyinB1Params.srcNdMatrixStride = 0;
        copyinB1Params.srcDValue = H1;
        copyinB1Params.dstNzC0Stride = CeilCubeBlock(R) * CUBE_BLOCK;
        copyinB1Params.dstNzNStride = 1;
        copyinB1Params.dstNzMatrixStride = 0;
        DataCopy(keyL1_[pingPongFlag], waGm_[weightOffset], copyinB1Params);
    }

    SetFlag<HardEvent::MTE2_MTE1>(eventId);
    WaitFlag<HardEvent::MTE2_MTE1>(eventId);
}

__aicore__ inline void AddLoraSingleCoreBatchKernel::Process()
{
    ComputeBatchIndice();
    if ASCEND_IS_AIV {
        if (!addLoraFlag) {
            InitDataMoveBuffer();
            QueryDataMove();
            pipe_->Reset();
            InitVectorBuffer();
            VectorNotifyCube<PIPE_MTE3>(SYNC_AIV_ONLY_ALL_FLAG, SYNC_AIV_AIC_FLAG);
        } else {
            InitVectorBuffer();
        }
    }
    if ASCEND_IS_AIV {
        coreId = GetBlockIdx() / 2;
    }
    int32_t mInCore = batch / coreNum;
    uint32_t mInCoreOffset = 0;
    mInCore = (coreId < batch % coreNum) ? mInCore + 1 : mInCore;
    mInCoreOffset = (coreId < batch % coreNum) ? mInCore * coreId : batch - ((coreNum - coreId) * mInCore);
    uint32_t pingPongFlag = 0;
    uint32_t nSplit = (SINGLE_L0_SIZE / R / CUBE_BLOCK_SIZE) * CUBE_BLOCK_SIZE;
    nSplit = nSplit > MIN_SPLIT ? nSplit : MIN_SPLIT;
    if ASCEND_IS_AIC {
        InitMatmulBuffer();
        if (addLoraFlag) {
            ComputeWeightAMM(mInCore, nSplit, mInCoreOffset, pingPongFlag);
        } else {
            CubeWaitVector(SYNC_AIV_AIC_FLAG);
        }
    }
    uint32_t nCoreId = coreId;
    int32_t nBlockNum = DivCeil(DivCeil(H2, BLOCK_HALF_SIZE), coreNum);
    nBlockNum = (coreId < DivCeil(H2, BLOCK_HALF_SIZE) % coreNum) ? nBlockNum : nBlockNum - 1;
    uint32_t nIncore = nBlockNum * BLOCK_HALF_SIZE;
    uint32_t nIncoreOffset =
        (coreId < DivCeil(H2, BLOCK_HALF_SIZE) % coreNum) ? nIncore * nCoreId : H2 - ((coreNum - coreId) * nIncore);
    if (nIncoreOffset + nIncore > H2) {
        nIncore = (H2 - nIncoreOffset) > 0 ? H2 - nIncoreOffset : 0;
    }
    if (nIncore <= 0) {
        return;
    } else {
        ComputeWeightBMM(nSplit, nIncore, nIncoreOffset, pingPongFlag);
    }
}

__aicore__ inline void AddLoraSingleCoreBatchKernel::ComputeWeightAMM(
    uint32_t mInCore, uint32_t nSplit, uint32_t mInCoreOffset, uint32_t pingPongFlag)
{
    if (mInCore == 0) {
        SyncAicOnly<PIPE_FIX>(SYNC_AIC_ONLY_ALL_FLAG);
    } else {
        uint32_t n = R;
        int32_t indiceIdx = int32_t(indicesGm_.GetValue(coreId));
        if (indiceIdx < 0 || indiceIdx == wBatch) {
            SyncAicOnly<PIPE_FIX>(SYNC_AIC_ONLY_ALL_FLAG);
        } else {
            uint32_t indiceOffset = (indiceIdx == 0) ? 0 : sumIndiceCount[indiceIdx - 1];
            uint32_t xIndiceAddr = indiceOffset + countIndiceBatch.GetValue(coreId);
            for (int kidx = 0; kidx < DivCeil(H1, nSplit); ++kidx) {
                if (nSplit == 0) {
                    continue;
                }
                uint32_t k = (kidx < (H1 / nSplit)) ? nSplit : H1 - kidx * nSplit;
                CopyInA(indiceIdx, kidx, mInCore, k, mInCoreOffset, nSplit, pingPongFlag);
                SplitA(mInCore, k, mInCoreOffset, pingPongFlag);
                SplitB(indiceIdx, kidx, mInCore, k, mInCoreOffset, pingPongFlag);
                Compute(indiceIdx, kidx, mInCore, k, n, pingPongFlag);
                pingPongFlag = !pingPongFlag;
            }
            CopyL0CGM(mInCore, mInCoreOffset, n, xIndiceAddr);
            SyncAicOnly<PIPE_FIX>(SYNC_AIC_ONLY_ALL_FLAG);
        }
    }
}

__aicore__ inline void AddLoraSingleCoreBatchKernel::ComputeWeightBMM(
    uint32_t nSplit, uint32_t nIncore, uint32_t nIncoreOffset, uint32_t pingPongFlag)
{
    for (int weightIndex = 0; weightIndex <= wBatch; ++weightIndex) {
        uint32_t m = batchIndiceCount[weightIndex];
        uint32_t indiceOffset_ = (weightIndex == 0) ? 0 : sumIndiceCount[weightIndex - 1];
        uint32_t mInCoreOffset = indiceOffset_;
        if (m == 0 || weightIndex == wBatch) {
            continue;
        }
        if ASCEND_IS_AIC {
            for (int nIn = 0; nIn < DivCeil(nIncore, nSplit); ++nIn) {
                uint32_t nInner = (nIn < (nIncore / nSplit)) ? nSplit : nIncore - nIn * nSplit;
                uint32_t RSplit = 16;
                if (!addLoraFlag) {
                    RSplit = 128;
                }
                for (int Ridx = 0; Ridx < DivCeil(R, RSplit); ++Ridx) {
                    uint32_t RInner = (Ridx < (R / RSplit)) ? RSplit : R - Ridx * RSplit;
                    CopyInA2(mInCoreOffset, m, RInner, Ridx, RSplit, pingPongFlag);
                    SplitA(m, RInner, mInCoreOffset, pingPongFlag);
                    CopyWbB(weightIndex, nIn, nInner, RInner, Ridx, nIncoreOffset, nSplit, RSplit, pingPongFlag);
                    ComputeMM2(mInCoreOffset, m, RInner, nInner, nIncoreOffset, Ridx, nIn, pingPongFlag);
                    pingPongFlag = !pingPongFlag;
                }
                SetFlag<HardEvent::M_FIX>(eventIDM_FIX);
                WaitFlag<HardEvent::M_FIX>(eventIDM_FIX);
                CopyL0C2GM(mInCoreOffset, m, nInner, nIn, nSplit, nIncoreOffset);
                AscendC::PipeBarrier<PIPE_FIX>();
                SetFlag<HardEvent::FIX_MTE2>(eventIDFIX_MTE2);
                WaitFlag<HardEvent::FIX_MTE2>(eventIDFIX_MTE2);
            }
            CubeNotifyVector<PIPE_FIX>(SYNC_AIC_AIV_FLAG2);
        }
        if ASCEND_IS_AIV {
            uint32_t mInVector = m;
            uint32_t mVectorOffset = mInCoreOffset;
            if (subBlockIdx_ == 0) {
                mInVector = m / 2 + m % 2;
            } else {
                mVectorOffset = mInCoreOffset + (m / 2 + m % 2);
                mInVector = m / 2;
            }
            if (mInVector == 0) {
                VectorWaitCube(SYNC_AIC_AIV_FLAG2);
            } else {
                VectorWaitCube(SYNC_AIC_AIV_FLAG2);
                CopyOutRes(mInVector, mVectorOffset, nIncore, nIncoreOffset, pingPongFlag);
            }
        }
    }
}
#endif // ADD_LORA_SINGLE_CORE_H