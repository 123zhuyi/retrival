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
 * \file flash_attention_score_grad_kernel_fp8.h
 * \brief
 */ 

#ifndef FLASH_ATTENTION_SCORE_GRAD_KERNEL_FP8_H
#define FLASH_ATTENTION_SCORE_GRAD_KERNEL_FP8_H
 
#include "flash_attention_score_grad_common.h"
#include "flash_attention_score_grad_kernel_base.h"
#include "flash_attention_score_grad_tiling_data_regbase.h"
#include "deter.h"
 
namespace FagBaseApi {
 
template <typename CubeBlockType, typename VecBlockType>
 
class FlashAttentionScoreGradKernelQuant
    : public FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>,
                                               CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    constexpr static uint32_t CUBE_BASEM = static_cast<uint32_t>(s1TemplateType);
    constexpr static uint32_t CUBE_BASEN = static_cast<uint32_t>(s2TemplateType);

    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicy4buff<BufferType::L1, SyncType::NO_SYNC> pL1Buf;

    BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> dsL1Buf[4][4];

    GlobalTensor<OUTDTYPE> dqGm, dkGm, dvGm;
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    GlobalTensor<float> deqScaleQGm, deqScaleKGm, deqScaleVGm, deqScaleDyGm, pScaleGm, dsScaleGm;

    FagTilingType tilingData;
    PreloadArgs<IS_ROPE> preloadArgs;
    TPipe *pipe;
    using BaseClass = FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>,
                                                        CubeBlockType, VecBlockType>;
    __aicore__ inline void Init(
            GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR pseShift, GM_ADDR dropMask, GM_ADDR attenMask,
            GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR prefixN, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen,
            GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR dsScale, GM_ADDR pScale, GM_ADDR queryRope, GM_ADDR keyRope,
            GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dpse, GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR workspace,
            FagTilingType ordTilingData, TPipe *pipeIn);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void SetUniqueRunInfo(FagRunInfo &runInfo);
    __aicore__ inline void SetUniqueConstInfo(FagConstInfo &constInfo);
    __aicore__ inline void SetConstInfo();
    __aicore__ inline void InitCVCommonBuffer();
    __aicore__ inline void InitCVCommonGlobalBuffer(GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR deqScaleQ, GM_ADDR deqScaleK,
                                                    GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR dsScale, GM_ADDR pScale, GM_ADDR workspace);
    __aicore__ inline void SetRunInfo(FagRunInfo &runInfo, FagRunInfo &lastRunInfo, int64_t taskId, CoordinateInfo &coordinateInfo, CoordinateInfo &nextCoordinateInfo);
    template<bool isSetSize>
    __aicore__ inline void SetQuantRunInfo(FagRunInfo &runInfo, int64_t taskId, int64_t s1Idx, int64_t s2Idx);
    __aicore__ inline int64_t GetDeqScaleKOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t GetDeqScaleQOffset(FagRunInfo &runInfo);
    __aicore__ inline int64_t CalDeterMaxLoopNum();
    __aicore__ inline void CalDeterIndex(uint32_t roundId, uint32_t maxLoopNum, int64_t &nextValidRoundId, int64_t &nextValidIndex, int64_t taskId,
        CoordinateInfo &coordinateInfo, FagRunInfo &runInfo);
    __aicore__ inline int64_t CalDenseDeterIndex(uint32_t roundId, uint32_t maxLoopNum, CoordinateInfo &coordinateInfo);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessFirstS2(FagRunInfo &runInfo, FagRunInfo &lastRunInfo, int8_t kvInnerId);
    __aicore__ inline void ProcessSP(FagRunInfo &runInfo, int8_t sdpId);
    __aicore__ inline void ProcessPDs(FagRunInfo &runInfo, int8_t sdpId);
    template<bool isDK>
    __aicore__ inline void ProcessDKV(FagRunInfo &runInfo, int8_t sdpId);
    __aicore__ inline void ProcessDQ(FagRunInfo &runInfo, int8_t sdpId);
    __aicore__ inline void ComputeDQ(FagRunInfo &runInfo, int8_t sdpId);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
protected:
    GlobalTensor<float> deterGm;
    uint64_t dkvWorkSpaceOffet{0};
    uint64_t dAlign16 = 0;
    uint64_t dvAlign16 = 0;
    int64_t specialDkGmOffset = 0;
    int64_t specialDvGmOffset = 0;
    int16_t deterPpFlag = 1;
    int64_t deterGmOffset = 0;
    int64_t specialHalfS2RealSize = 0;
    int64_t specialFirstHalfS2RealSize = 0;
    int64_t lastS2Idx = -1;
    int64_t lastBatchIdx = -1;
    int64_t lastN2Idx = -1;

    int64_t innerN2GD;

    TBuf<> vecQue;
    LocalTensor<CALC_TYPE> spTensors; // 64*128*4*2
    LocalTensor<CALC_TYPE> spTensor[2];
    LocalTensor<CALC_TYPE> dpdsTensors; // 64*128*4*2
    LocalTensor<CALC_TYPE> dpdsTensor[2];

    LocalTensor<CALC_TYPE> dkvTensors; // 64*128*4*2
    LocalTensor<CALC_TYPE> commonTensors; // 64*128*4*2
    LocalTensor<CALC_TYPE> dqTensor; // 64*128*4
    LocalTensor<CALC_TYPE> dkTensor; // 64*128*4
    LocalTensor<CALC_TYPE> dvTensor; // 64*128*4

    CoordinateInfo coordinateInfos[2];
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::Init(
    GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR query, GM_ADDR pseShift, GM_ADDR dropMask, GM_ADDR attenMask,
    GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR prefixN, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKvlen,
    GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR dsScale, GM_ADDR pScale, GM_ADDR queryRope, GM_ADDR keyRope,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dpse, GM_ADDR dqRope, GM_ADDR dkRope, GM_ADDR workspace,
    FagTilingType ordTilingData, TPipe *pipeIn)
{
    if ASCEND_IS_AIV {
        this->vBlockIdx = GetBlockIdx();
        this->cBlockIdx = this->vBlockIdx / CV_CORE_RATIO;
        this->vSubBlockIdx = GetSubBlockIdx();
    } else {
        this->cBlockIdx = GetBlockIdx();
    }
    tilingData = ordTilingData;
    pipe = pipeIn;

    InitCVCommonGlobalBuffer(dq, dk, dv, deqScaleQ, deqScaleK, deqScaleV, deqScaleDy, dsScale, pScale, workspace);
    SetConstInfo();
    
    InitCVCommonBuffer();
    this->vecBlock.SetVecBlockParams(pipeIn, tilingData, this->vBlockIdx, this->cBlockIdx, this->vSubBlockIdx, this->constInfo);
    this->vecBlock.InitUbBuffer(commonTensors);
    this->vecBlock.InitGlobalBuffer(value, dy, y, softmaxMax, softmaxSum, deqScaleQ, deqScaleK,
                              deqScaleV, deqScaleDy, dq, dk, dv, workspace, this->tilingData->postTilingData.sfmgWorkSpaceOffset);
 
    this->cubeBlock.SetCubeBlockParams(pipeIn, tilingData, &l1BufferManager);
    this->cubeBlock.InitCubeBuffer(this->constInfo);
    this->cubeBlock.InitGlobalBuffer(query, key, value, dy);
}
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::SetConstInfo()
{
    // split info
    this->constInfo.s1Outer = tilingData->s1s2BNGS1S2SplitCoreParams.s1Outer;
    this->constInfo.s1CvTail = tilingData->s1s2BNGS1S2SplitCoreParams.s1CvTail;
    this->constInfo.s1Tail = tilingData->s1s2BNGS1S2SplitCoreParams.s1Tail;
    this->constInfo.s2Tail = tilingData->s1s2BNGS1S2SplitCoreParams.s2Tail;
    this->constInfo.s2Outer = tilingData->s1s2BNGS1S2SplitCoreParams.s2Outer;
 
    this->constInfo.commonConstInfo.s1BaseSize = CUBE_BASEM;
    this->constInfo.commonConstInfo.s2BaseSize = CUBE_BASEN;
    this->constInfo.bSize = tilingData->s1s2BNGS1S2BaseParams.b;
    this->constInfo.n2Size = tilingData->s1s2BNGS1S2BaseParams.n2;
    this->constInfo.commonConstInfo.gSize = tilingData->s1s2BNGS1S2BaseParams.g;
    this->constInfo.commonConstInfo.s1Size = tilingData->s1s2BNGS1S2BaseParams.s1;
    this->constInfo.commonConstInfo.s2Size = tilingData->s1s2BNGS1S2BaseParams.s2;
    this->constInfo.commonConstInfo.dSize = tilingData->s1s2BNGS1S2BaseParams.d;
    this->constInfo.commonConstInfo.dSizeV = tilingData->s1s2BNGS1S2BaseParams.d1;
    this->constInfo.commonConstInfo.layoutType = tilingData->s1s2BNGS1S2BaseParams.layout;
 
    this->constInfo.commonConstInfo.s1D = this->constInfo.commonConstInfo.s1Size * this->constInfo.commonConstInfo.dSize;
    this->constInfo.commonConstInfo.gS1D = this->constInfo.commonConstInfo.gSize * this->constInfo.commonConstInfo.s1D;
    this->constInfo.commonConstInfo.n2GS1D = this->constInfo.n2Size * this->constInfo.commonConstInfo.gS1D;
    this->constInfo.commonConstInfo.s2D = this->constInfo.commonConstInfo.s2Size * this->constInfo.commonConstInfo.dSize;
    this->constInfo.commonConstInfo.n2S2D = this->constInfo.n2Size * this->constInfo.commonConstInfo.s2D;
    this->constInfo.commonConstInfo.s1S2 = this->constInfo.commonConstInfo.s1Size * this->constInfo.commonConstInfo.s2Size;
    this->constInfo.commonConstInfo.gS1 = this->constInfo.commonConstInfo.gSize * this->constInfo.commonConstInfo.s1Size;
    this->constInfo.commonConstInfo.gD = this->constInfo.commonConstInfo.gSize * this->constInfo.commonConstInfo.dSize;
    this->constInfo.commonConstInfo.n2D = this->constInfo.n2Size * this->constInfo.commonConstInfo.dSize;
    this->constInfo.commonConstInfo.bN2D = this->constInfo.bSize * this->constInfo.commonConstInfo.n2D;
    this->constInfo.commonConstInfo.n2G = this->constInfo.n2Size * this->constInfo.commonConstInfo.gSize;
    this->constInfo.commonConstInfo.n2GD = this->constInfo.commonConstInfo.n2G * this->constInfo.commonConstInfo.dSize;
    this->constInfo.commonConstInfo.bN2GD = this->constInfo.bSize * this->constInfo.commonConstInfo.n2GD;
    this->constInfo.commonConstInfo.gS2 = this->constInfo.commonConstInfo.gSize * this->constInfo.commonConstInfo.s2Size;
    // for D_V
    this->constInfo.commonConstInfo.s1Dv = this->constInfo.commonConstInfo.s1D;
    this->constInfo.commonConstInfo.gS1Dv = this->constInfo.commonConstInfo.gS1D;
    this->constInfo.commonConstInfo.n2GS1Dv = this->constInfo.commonConstInfo.n2GS1D;
    this->constInfo.commonConstInfo.s2Dv = this->constInfo.commonConstInfo.s2D;
    this->constInfo.commonConstInfo.n2S2Dv = this->constInfo.commonConstInfo.n2S2D;
    this->constInfo.commonConstInfo.gDv = this->constInfo.commonConstInfo.gD;
    this->constInfo.commonConstInfo.n2Dv = this->constInfo.commonConstInfo.n2D;
    this->constInfo.commonConstInfo.bN2Dv = this->constInfo.commonConstInfo.bN2D;
    this->constInfo.commonConstInfo.n2GDv = this->constInfo.commonConstInfo.n2GD;
    this->constInfo.commonConstInfo.bN2GDv = this->constInfo.commonConstInfo.bN2GD;
    
    this->constInfo.scaleValue = tilingData->s1s2BNGS1S2BaseParams.scaleValue;
    this->constInfo.n2GS1oS2o = this->constInfo.commonConstInfo.n2G * this->constInfo.s1Outer * this->constInfo.s2Outer;
    this->constInfo.gS1oS2o = this->constInfo.commonConstInfo.gSize * this->constInfo.s1Outer * this->constInfo.s2Outer;
    this->constInfo.s1oS2o = this->constInfo.s1Outer * this->constInfo.s2Outer;
    if ASCEND_IS_AIC {
        this->constInfo.commonConstInfo.mm1Ka = this->constInfo.commonConstInfo.n2GDv;
        this->constInfo.commonConstInfo.mm1Kb = this->constInfo.commonConstInfo.n2Dv;
        this->constInfo.mm2Ka = this->constInfo.commonConstInfo.n2GD;
        this->constInfo.mm2Kb = this->constInfo.commonConstInfo.n2D;
        this->constInfo.mm3Ka = this->constInfo.mm2Ka;
        this->constInfo.mm4Kb = this->constInfo.mm2Kb;
    }
    this->constInfo.commonConstInfo.subBlockIdx = this->vSubBlockIdx;
 
    uint32_t tmp = 0xFF7FFFFF;

    uint32_t maxContinuousBlockNum = this->constInfo.commonConstInfo.s1Size < MIN_SWIZZLE_S1 ?
                                         MAX_CONTINUOUS_BLOCK_NUM :
                                         (this->constInfo.commonConstInfo.s1Size / MIN_SWIZZLE_S1) * BASE_SWIZZLE_BLOCK_NUM;
    this->constInfo.continuousBlockNum = tilingData->s1s2BNGS1S2SplitCoreParams.maxValidBBLen > maxContinuousBlockNum ?
                                       maxContinuousBlockNum :
                                       tilingData->s1s2BNGS1S2SplitCoreParams.maxValidBBLen;
    this->innerN2GD = this->constInfo.commonConstInfo.n2GD * 128;

    if ASCEND_IS_AIV {
        this->constInfo.pScale = pScaleGm.GetValue(0);
        this->constInfo.dsScale = dsScaleGm.GetValue(0);
        this->constInfo.pScaleD = (float)1.0 / this->constInfo.pScale;
        this->constInfo.dsScaleD = (float)1.0 / this->constInfo.dsScale;    
        this->constInfo.copyOutDStride = (this->constInfo.commonConstInfo.n2GD - 64) * sizeof(CALC_TYPE);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::InitCVCommonBuffer() 
{
    l1BufferManager.Init(pipe, L1_MAX_SIZE);
    pL1Buf.Init(l1BufferManager, 128 * 128 * sizeof(INPUT_TYPE));
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j ++) {
            dsL1Buf[i][j].Init(l1BufferManager, 128 * 128 * sizeof(INPUT_TYPE));
        }
    }
    uint32_t totalUbSpace = 224 * 1024;
    pipe->InitBuffer(vecQue, totalUbSpace);
    uint32_t ubOffset = 0;
    spTensors = vecQue.GetWithOffset<CALC_TYPE>(64 * 128 * 2, ubOffset);
    ubOffset += 64 * 128 * sizeof(CALC_TYPE) * 2;
    dpdsTensors = vecQue.GetWithOffset<CALC_TYPE>(64 * 128 * 2, ubOffset);
    ubOffset += 64 * 128 * sizeof(CALC_TYPE) * 2;
    dkvTensors = vecQue.GetWithOffset<CALC_TYPE>(64 * 128 * 2, ubOffset);
    ubOffset += 64 * 128 * sizeof(CALC_TYPE) * 2;
    commonTensors = vecQue.GetWithOffset<CALC_TYPE>(64 * 128 * 2, ubOffset);

    spTensor[0] = spTensors[0];
    spTensor[1] = spTensors[64];
    dpdsTensor[0] = dpdsTensors[0];
    dpdsTensor[1] = dpdsTensors[64];
    dvTensor = dkvTensors[0];
    dkTensor = dkvTensors[64];
    dqTensor = commonTensors[0];
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::InitCVCommonGlobalBuffer(GM_ADDR dq,
                                                                                         GM_ADDR dk,
                                                                                         GM_ADDR dv,
                                                                                         GM_ADDR deqScaleQ,
                                                                                         GM_ADDR deqScaleK,
                                                                                         GM_ADDR deqScaleV,
                                                                                         GM_ADDR deqScaleDy,
                                                                                         GM_ADDR dsScale,
                                                                                         GM_ADDR pScale,
                                                                                         GM_ADDR workspace)
{
    dqGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
    dkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
    dvGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
    deqScaleQGm.SetGlobalBuffer((__gm__ float *)deqScaleQ);
    deqScaleKGm.SetGlobalBuffer((__gm__ float *)deqScaleK);
    deqScaleVGm.SetGlobalBuffer((__gm__ float *)deqScaleV);
    deqScaleDyGm.SetGlobalBuffer((__gm__ float *)deqScaleDy);
    dsScaleGm.SetGlobalBuffer((__gm__ float *)dsScale);
    pScaleGm.SetGlobalBuffer((__gm__ float *)pScale);

    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                          tilingData->postTilingData.dqWorkSpaceOffset / sizeof(float));
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                    tilingData->postTilingData.dkWorkSpaceOffset / sizeof(float));
    dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                    tilingData->postTilingData.dvWorkSpaceOffset / sizeof(float));
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::SetUniqueRunInfo(FagRunInfo &runInfo)
{
}
 
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::SetUniqueConstInfo(FagConstInfo &constInfo)
{
}
 
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::CalDenseDeterIndex(
    uint32_t roundId, uint32_t maxLoopNum, CoordinateInfo &coordinateInfo)
{
    int64_t j = this->cBlockIdx + 1;
    int64_t r = roundId + 1;
 
    int64_t k = static_cast<int64_t>(this->tilingData->s1s2BNGS1S2BaseParams.coreNum / NUM_TWO);
    int64_t b = this->constInfo.bSize * this->constInfo.n2Size;
    if (this->constInfo.s2Outer == 1) {
        CalDenseIndexForSingleN(k, this->constInfo.s1Outer, b, j, r, maxLoopNum, coordinateInfo);
    } else {
        CalDenseIndex(k, this->constInfo.s1Outer, this->constInfo.s2Outer, b, j, r, coordinateInfo);
    }
 
    int64_t w = coordinateInfo.batchId;
    int64_t n1 = this->constInfo.commonConstInfo.gSize * this->constInfo.n2Size;
    coordinateInfo.batchId = Ceil<int64_t>(w, n1) - 1;
    int64_t n1Idx = w - coordinateInfo.batchId * n1 - 1;
    coordinateInfo.n2Idx = n1Idx / this->constInfo.commonConstInfo.gSize;
    coordinateInfo.gIdx = n1Idx % this->constInfo.commonConstInfo.gSize;
    coordinateInfo.s1Idx -= 1;
    coordinateInfo.s2Idx -= 1;
    if (!(w > 0 && coordinateInfo.batchId < this->constInfo.bSize && coordinateInfo.n2Idx < this->constInfo.n2Size &&
          coordinateInfo.gIdx < this->constInfo.commonConstInfo.gSize && coordinateInfo.s1Idx >= 0 &&
          coordinateInfo.s1Idx < coordinateInfo.s1Outer && coordinateInfo.s2Idx >= 0 &&
          coordinateInfo.s2Idx < coordinateInfo.s2Outer)) {
        return -1;
    }
 
    return (coordinateInfo.batchId * n1 + coordinateInfo.n2Idx * this->constInfo.commonConstInfo.gSize +
            coordinateInfo.gIdx) *
               this->constInfo.s1Outer * this->constInfo.s2Outer +
           coordinateInfo.s2Idx * this->constInfo.s1Outer + coordinateInfo.s1Idx;
}
 
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::CalDeterMaxLoopNum()
{
    int64_t b = this->constInfo.bSize * this->constInfo.n2Size;
    int64_t m = this->constInfo.s1Outer;
    int64_t n = this->constInfo.s2Outer;
    int64_t k = static_cast<int64_t>(this->tilingData->s1s2BNGS1S2BaseParams.coreNum / NUM_TWO);
 
    InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, 0, 0, this->coordinateInfos[0]);
    InitCoordinateInfo(this->constInfo.s1Outer, this->constInfo.s2Outer, 0, 0, this->coordinateInfos[1]);
 
    if (n == 1) {
        return Max(Ceil<int64_t>(m * b, k), m);
    } else {
        return Ceil<int64_t>(n * b, Min(k, m * b)) * m;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::CalDeterIndex(
        uint32_t roundId, uint32_t maxLoopNum, int64_t &nextValidRoundId, int64_t &nextValidIndex, int64_t taskId,
        CoordinateInfo &coordinateInfo, FagRunInfo &runInfo)
{
    coordinateInfo.sparseMode = this->constInfo.sparseMode;
    for (uint32_t currentRoundId = roundId; currentRoundId < maxLoopNum; currentRoundId++) {
        nextValidIndex = CalDenseDeterIndex(currentRoundId, maxLoopNum, coordinateInfo);
 
        bool isValidBlock = (nextValidIndex >= 0);
        if (isValidBlock) { 
            nextValidRoundId = currentRoundId;
            return;
        }
    }
    coordinateInfo.batchId = -1;
    nextValidIndex = -1;
    nextValidRoundId = maxLoopNum;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::GetDeqScaleQOffset(FagRunInfo &runInfo)
{
    int64_t scaleOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    int64_t scaleNumPerS1 = Ceil<int64_t>(this->constInfo.commonConstInfo.s1Size, CUBE_BASEM);
    bOffset = runInfo.commonRunInfo.boIdx * this->constInfo.commonConstInfo.n2G * scaleNumPerS1;
    n2Offset = runInfo.commonRunInfo.n2oIdx * this->constInfo.commonConstInfo.gSize * scaleNumPerS1;
    gOffset = runInfo.commonRunInfo.goIdx * scaleNumPerS1;
    s1Offset = runInfo.commonRunInfo.s1oIdx;
    scaleOffset = bOffset + n2Offset + gOffset + s1Offset;
    return scaleOffset;
}
 
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t
FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::GetDeqScaleKOffset(FagRunInfo &runInfo)
{
    int64_t scaleOffset = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    int64_t scaleNumPerS2 = Ceil<int64_t>(this->constInfo.commonConstInfo.s2Size, CUBE_BASEN);
    bOffset = runInfo.commonRunInfo.boIdx * this->constInfo.n2Size * scaleNumPerS2;
    n2Offset = runInfo.commonRunInfo.n2oIdx * scaleNumPerS2;
    s2Offset = runInfo.s2oIdx;
    scaleOffset = bOffset + n2Offset + s2Offset;
    return scaleOffset;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::SetRunInfo(FagRunInfo &runInfo, FagRunInfo &lastRunInfo, int64_t taskId, CoordinateInfo &coordinateInfo, CoordinateInfo &nextCoordinateInfo)
{
    runInfo.isKeyReuse = ((nextCoordinateInfo.batchId == coordinateInfo.batchId) 
                        && (nextCoordinateInfo.n2Idx == coordinateInfo.n2Idx) 
                        && (nextCoordinateInfo.s2Idx == coordinateInfo.s2Idx)) || (nextCoordinateInfo.batchId == -1);
    runInfo.isLastProcessBlock = (nextCoordinateInfo.batchId == -1);
    runInfo.isFirstProcessBlock = taskId == 0;
    lastRunInfo.isNextKeyReuse = runInfo.isKeyReuse;
    runInfo.isValueReuse = (lastS2Idx == coordinateInfo.s2Idx && lastBatchIdx == coordinateInfo.batchId && lastN2Idx == coordinateInfo.n2Idx);
    lastBatchIdx = coordinateInfo.batchId;
    lastN2Idx = coordinateInfo.n2Idx;
    lastS2Idx = coordinateInfo.s2Idx;

    runInfo.commonRunInfo.boIdx = coordinateInfo.batchId;
    runInfo.commonRunInfo.n2oIdx = coordinateInfo.n2Idx;
    runInfo.commonRunInfo.goIdx = coordinateInfo.gIdx;
    runInfo.s2oIdx = coordinateInfo.s2Idx;
    runInfo.commonRunInfo.s1oIdx = coordinateInfo.s1Idx;
    runInfo.s2CvBegin = runInfo.s2oIdx * CUBE_BASEN;

    runInfo.commonRunInfo.taskId = taskId;
    runInfo.commonRunInfo.taskIdMod2 = taskId & 1;

    runInfo.commonRunInfo.s1RealSize =
        (runInfo.commonRunInfo.s1oIdx == this->constInfo.s1Outer - 1) ? this->constInfo.s1CvTail : CUBE_BASEM;
    runInfo.commonRunInfo.s2RealSize = (runInfo.s2oIdx == this->constInfo.s2Outer - 1) ? this->constInfo.s2Tail : CUBE_BASEN;

    runInfo.quantRunInfo.innerS1LoopNum = Ceil<int64_t>(runInfo.commonRunInfo.s1RealSize, 128);
    runInfo.quantRunInfo.innerS2LoopNum = Ceil<int64_t>(runInfo.commonRunInfo.s2RealSize, 128);

    if ASCEND_IS_AIV {
        int64_t deqScaleQGmOffset = GetDeqScaleQOffset(runInfo);
        int64_t deqScaleKGmOffset = GetDeqScaleKOffset(runInfo);
        runInfo.quantScaleInfo.deqScaleQValue = deqScaleQGm.GetValue(deqScaleQGmOffset);
        runInfo.quantScaleInfo.deqScaleKValue = deqScaleKGm.GetValue(deqScaleKGmOffset);
        runInfo.quantScaleInfo.deqScaleVValue = deqScaleVGm.GetValue(deqScaleKGmOffset);
        runInfo.quantScaleInfo.deqScaleDyValue = deqScaleDyGm.GetValue(deqScaleQGmOffset);

        runInfo.quantRunInfo.qkDScale = runInfo.quantScaleInfo.deqScaleQValue * runInfo.quantScaleInfo.deqScaleKValue;
        runInfo.quantRunInfo.vdyDScale = runInfo.quantScaleInfo.deqScaleDyValue * runInfo.quantScaleInfo.deqScaleVValue;

        runInfo.quantRunInfo.dsScaleDMulDeqScaleK = this->constInfo.dsScaleD * runInfo.quantScaleInfo.deqScaleKValue;
        runInfo.quantRunInfo.pScaleDMulDeqScaleDy = this->constInfo.pScaleD * runInfo.quantScaleInfo.deqScaleDyValue;
        runInfo.quantRunInfo.dsScaleDMulDeqScaleQ = this->constInfo.dsScaleD * runInfo.quantScaleInfo.deqScaleQValue;

        runInfo.maxsumOffset = ((runInfo.commonRunInfo.boIdx * this->constInfo.n2Size + runInfo.commonRunInfo.n2oIdx) *
                               this->constInfo.commonConstInfo.gSize +
                           runInfo.commonRunInfo.goIdx) *
                              this->constInfo.commonConstInfo.s1Size +
                          runInfo.commonRunInfo.s1oIdx * CUBE_BASEM;
    }

    runInfo.commonRunInfo.queryOffset = this->GetQueryOffset(runInfo);
    runInfo.dyOffset = runInfo.commonRunInfo.queryOffset;
    runInfo.commonRunInfo.keyOffset = this->GetKeyOffset(runInfo);
    runInfo.commonRunInfo.valueOffset = runInfo.commonRunInfo.keyOffset;

    int64_t innerS1TailSize =  (runInfo.commonRunInfo.s1RealSize % 128) == 0 ? 128 : runInfo.commonRunInfo.s1RealSize % 128;
    int64_t innerS2TailSize =  (runInfo.commonRunInfo.s2RealSize % 128) == 0 ? 128 : runInfo.commonRunInfo.s2RealSize % 128;
    for (int64_t i = 0; i < 4; i++) {
        runInfo.quantRunInfo.innerS1RealSize[i] = 
            (i == runInfo.quantRunInfo.innerS1LoopNum - 1) ? innerS1TailSize : 128;
        runInfo.quantRunInfo.innerS2RealSize[i] = 
            (i == runInfo.quantRunInfo.innerS2LoopNum - 1) ? innerS2TailSize : 128;
        
        runInfo.quantRunInfo.qInnerOffset[i] = runInfo.commonRunInfo.queryOffset + i * this->innerN2GD;
        runInfo.quantRunInfo.kvInnerOffset[i] = runInfo.commonRunInfo.keyOffset + i * this->innerN2GD;
    }
}

template <typename CubeBlockType, typename VecBlockType>
template<bool isSetSize>
__aicore__ inline void
FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::SetQuantRunInfo(FagRunInfo &runInfo, int64_t taskId, int64_t s1Idx, int64_t s2Idx) 
{
    runInfo.quantRunInfo.s1Idx = s1Idx;
    runInfo.quantRunInfo.s2Idx = s2Idx;
}
 
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::AllocEventID()
{
    if ASCEND_IS_AIV {
        // sdp
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(0);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(1);

        // process dqkv 
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_TRANSFER_DKV_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_TRANSFER_DQ_FLAG);
        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_DETER_FLAG);
    }

    if ASCEND_IS_AIC {
        for (int64_t i = 0; i<2;i++) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_UB2L1_P_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_UB2L1_P_FLAG);
        }
        for (int64_t i = 0; i<8;i++) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_UB2L1_DS_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_UB2L1_DS_FLAG);
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::FreeEventID()
{
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(0);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + 0);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(1);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + 1);

        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_TRANSFER_DKV_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_TRANSFER_DKV_FLAG);

        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_TRANSFER_DQ_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_TRANSFER_DQ_FLAG);
    }
    if ASCEND_IS_AIV {
        for (int64_t i = 0; i<2;i++) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_UB2L1_P_FLAG);
        }
        for (int64_t i = 0; i<8;i++) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_UB2L1_DS_FLAG);
        }
        CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_DETER_FLAG);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::Process()
{
    AllocEventID();
    if ASCEND_IS_AIV {
        this->vecBlock.AllocEventID();
    } else {
        this->cubeBlock.AllocEventID();
    }

    int64_t loopMax = CalDeterMaxLoopNum();
    int64_t blockInnerIdx = 0;
    int64_t taskId = 0;
    int64_t nextValidLoopIdx;
    int64_t nextblockIdx;
 
    FagRunInfo runInfos[2];
    CalDeterIndex(0, loopMax, nextValidLoopIdx, nextblockIdx, taskId,
        this->coordinateInfos[taskId & 1], runInfos[taskId & 1]);
 
    for (int64_t loopIdx = 0; loopIdx < loopMax; loopIdx++) {
        if (loopIdx >= nextValidLoopIdx) {
            blockInnerIdx = nextblockIdx;
            CalDeterIndex(loopIdx + 1, loopMax, nextValidLoopIdx, nextblockIdx, taskId,
                this->coordinateInfos[(taskId + 1) & 1], runInfos[(taskId + 1) & 1]);
            runInfos[taskId & 1].quantRunInfo.isDkvCompleted = false;
            runInfos[taskId & 1].quantRunInfo.isDqCompleted = false;
        } else {
            blockInnerIdx = -1;
        }
 
        bool isValidBlock = (blockInnerIdx >= 0);
        if (isValidBlock) {
            SetRunInfo(runInfos[taskId & 1], runInfos[(taskId + 1) & 1], taskId, this->coordinateInfos[taskId & 1], this->coordinateInfos[(taskId + 1) & 1]);
            for (int8_t kvInnerId = 0; kvInnerId < 4; kvInnerId++) {
                ProcessFirstS2(runInfos[taskId & 1], runInfos[(taskId + 1) & 1], kvInnerId);
            }
        } else {
            if ASCEND_IS_AIV {
                if (loopIdx > 0) {
                    for (uint8_t j = 0; j<4; j++) {
                        CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_DETER_FLAG);
                        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_DETER_FLAG);
                    }
                }
            }
        }
        
        if (isValidBlock) {
            taskId++;
        }
    }

    if (!runInfos[(taskId + 1) & 1].quantRunInfo.isDkvCompleted) {
        int64_t kv_iib = 3;
        if ASCEND_IS_AIV {
            SetQuantRunInfo<false>(runInfos[(taskId + 1) & 1], 0, 0, kv_iib - 1);
            if (runInfos[(taskId + 1) & 1].quantRunInfo.s2Idx < runInfos[(taskId + 1) & 1].quantRunInfo.innerS2LoopNum) {
                this->vecBlock.template CopyDqkv2GM<1>(runInfos[(taskId + 1) & 1], this->constInfo, dvWorkSpaceGm, dvTensor, dkWorkSpaceGm, dkTensor);
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_TRANSFER_DKV_FLAG);
        }

        SetQuantRunInfo<false>(runInfos[(taskId + 1) & 1], 0, 0, kv_iib);
        ProcessDKV<false>(runInfos[(taskId + 1) & 1], 0);
        SetQuantRunInfo<false>(runInfos[(taskId + 1) & 1], 0, 2, kv_iib);
        ProcessDKV<false>(runInfos[(taskId + 1) & 1], 2);

        SetQuantRunInfo<false>(runInfos[(taskId + 1) & 1], 0, 0, kv_iib);
        ProcessDKV<true>(runInfos[(taskId + 1) & 1], 0);
        SetQuantRunInfo<false>(runInfos[(taskId + 1) & 1], 0, 2, kv_iib);
        ProcessDKV<true>(runInfos[(taskId + 1) & 1], 2);

        if ASCEND_IS_AIV {
            SetQuantRunInfo<false>(runInfos[(taskId + 1) & 1], 0, 0, kv_iib);
            if (runInfos[(taskId + 1) & 1].quantRunInfo.s2Idx < runInfos[(taskId + 1) & 1].quantRunInfo.innerS2LoopNum) {
                this->vecBlock.template CopyDqkv2GM<1>(runInfos[(taskId + 1) & 1], this->constInfo, dvWorkSpaceGm, dvTensor, dkWorkSpaceGm, dkTensor);
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_TRANSFER_DKV_FLAG);
        }
    }

    if (!runInfos[(taskId + 1) & 1].quantRunInfo.isDqCompleted) {
        for (int64_t i = 0; i < 4; i++) {
            SetQuantRunInfo<false>(runInfos[(taskId + 1) & 1], 0, i, 0);
            ProcessDQ(runInfos[(taskId + 1) & 1], 0);
            SetQuantRunInfo<false>(runInfos[(taskId + 1) & 1], 0, i, 2);
            ProcessDQ(runInfos[(taskId + 1) & 1], 2);
            ComputeDQ(runInfos[(taskId + 1) & 1], 0);
            if ASCEND_IS_AIV{
                CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_DETER_FLAG);
                if (runInfos[(taskId + 1) & 1].quantRunInfo.s1Idx < runInfos[(taskId + 1) & 1].quantRunInfo.innerS1LoopNum) {
                    this->vecBlock.template CopyDqkv2GM<0>(runInfos[(taskId + 1) & 1], this->constInfo, dqWorkSpaceGm, dqTensor, dqWorkSpaceGm, dqTensor);
                }
                CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_DETER_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_TRANSFER_DQ_FLAG);
            }
        }
    } else {
        if ASCEND_IS_AIV{
            for (int64_t i = 0; i < 4; i++) {
                CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_DETER_FLAG);
                CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_DETER_FLAG);
            }
        }
    }
    if ASCEND_IS_AIV {
        this->vecBlock.FreeEventID();
    } else {
        this->cubeBlock.FreeEventID();
    }
    FreeEventID();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::ProcessFirstS2(FagRunInfo &runInfo, FagRunInfo &lastRunInfo, int8_t kvInnerId)
{
    SetQuantRunInfo<true>(runInfo, 0, 0, kvInnerId);
    ProcessSP(runInfo, 0);
    ProcessPDs(runInfo, 0);
    if ASCEND_IS_AIC {
        if (!lastRunInfo.quantRunInfo.isDqCompleted) {
            SetQuantRunInfo<false>(lastRunInfo, 0, kvInnerId, 0);
            ProcessDQ(lastRunInfo, 0);
        }
    }

    SetQuantRunInfo<true>(runInfo, 0, 1, kvInnerId);
    ProcessSP(runInfo, 1);
    ProcessPDs(runInfo, 1);
    if ASCEND_IS_AIC {
        if (!lastRunInfo.quantRunInfo.isDqCompleted) {
            SetQuantRunInfo<false>(lastRunInfo, 0, kvInnerId, 2);
            ProcessDQ(lastRunInfo, 2);
        }
    }

    if ASCEND_IS_AIV{
        if (kvInnerId > 1) {
            SetQuantRunInfo<false>(runInfo, 0, 0, kvInnerId - 2);
            if (runInfo.quantRunInfo.s2Idx < runInfo.quantRunInfo.innerS2LoopNum) {
                this->vecBlock.template CopyDqkv2GM<1>(runInfo, this->constInfo, dvWorkSpaceGm, dvTensor, dkWorkSpaceGm, dkTensor);
            }
            
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_TRANSFER_DKV_FLAG);
        } else if (!lastRunInfo.quantRunInfo.isDkvCompleted) {
            SetQuantRunInfo<false>(lastRunInfo, 0, 0, kvInnerId + 2);
            if (lastRunInfo.quantRunInfo.s2Idx < lastRunInfo.quantRunInfo.innerS2LoopNum) {
                this->vecBlock.template CopyDqkv2GM<1>(lastRunInfo, this->constInfo, dvWorkSpaceGm, dvTensor, dkWorkSpaceGm, dkTensor);
            }

            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_TRANSFER_DKV_FLAG);
            lastRunInfo.quantRunInfo.isDkvCompleted = (kvInnerId + 2) == 3;
        }

        if (!lastRunInfo.quantRunInfo.isDqCompleted) {
            SetQuantRunInfo<false>(lastRunInfo, 0, kvInnerId, 2);
            ComputeDQ(lastRunInfo, 0);
        }
    }

    if (kvInnerId > 0) {
        SetQuantRunInfo<false>(runInfo, 0, 0, kvInnerId - 1);
        ProcessDKV<false>(runInfo, 0);
    } else if (!lastRunInfo.quantRunInfo.isDkvCompleted && kvInnerId == 0) {
        SetQuantRunInfo<false>(lastRunInfo, 0, 0, 3);
        ProcessDKV<false>(lastRunInfo, 0);
    }

    SetQuantRunInfo<true>(runInfo, 0, 2, kvInnerId);
    ProcessSP(runInfo,0);
    ProcessPDs(runInfo, 0);
    if (kvInnerId > 0) {
        SetQuantRunInfo<false>(runInfo, 0, 2, kvInnerId - 1);
        ProcessDKV<false>(runInfo, 2);
    } else if (!lastRunInfo.quantRunInfo.isDkvCompleted && kvInnerId == 0) {
        SetQuantRunInfo<false>(lastRunInfo, 0, 2, 3);
        ProcessDKV<false>(lastRunInfo, 2);
    }

    SetQuantRunInfo<true>(runInfo, 0, 3, kvInnerId);
    ProcessSP(runInfo, 1);
    ProcessPDs(runInfo, 1);

    if ASCEND_IS_AIV{
        if (!lastRunInfo.quantRunInfo.isDqCompleted) {
            SetQuantRunInfo<false>(lastRunInfo, 0, kvInnerId, 2);
            CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_DETER_FLAG);
            if (lastRunInfo.quantRunInfo.s1Idx < lastRunInfo.quantRunInfo.innerS1LoopNum) {
                this->vecBlock.template CopyDqkv2GM<0>(lastRunInfo, this->constInfo, dqWorkSpaceGm, dqTensor, dqWorkSpaceGm, dqTensor);
            }
            CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_DETER_FLAG);

            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_TRANSFER_DQ_FLAG);
            lastRunInfo.quantRunInfo.isDqCompleted = kvInnerId == 3;
        }
    }

    if (kvInnerId > 0) {
        SetQuantRunInfo<false>(runInfo, 0, 0, kvInnerId - 1);
        ProcessDKV<true>(runInfo, 0);
        SetQuantRunInfo<false>(runInfo, 0, 2, kvInnerId - 1);
        ProcessDKV<true>(runInfo, 2);
    }  else if (!lastRunInfo.quantRunInfo.isDkvCompleted && kvInnerId == 0) {
        SetQuantRunInfo<false>(lastRunInfo, 0, 0, 3);
        ProcessDKV<true>(lastRunInfo, 0);
        SetQuantRunInfo<false>(lastRunInfo, 0, 2, 3);
        ProcessDKV<true>(lastRunInfo, 2);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::ProcessSP(FagRunInfo &runInfo, int8_t sdpId)
{
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(sdpId);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + sdpId);

        if (runInfo.quantRunInfo.s2Idx < runInfo.quantRunInfo.innerS2LoopNum && runInfo.quantRunInfo.s1Idx < runInfo.quantRunInfo.innerS1LoopNum) {
            this->cubeBlock.IterateMmDsP(spTensor[sdpId], dpdsTensor[sdpId], this->constInfo, runInfo);
        }
        
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(sdpId + 5);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + sdpId + 5);
    } 
}

#define L1DS(i, j, taskMod)   (((taskMod)  == 0)? dsL1Buf[i][j] : dsL1Buf[j][i])

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::ProcessPDs(FagRunInfo &runInfo, int8_t sdpId)
{
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(sdpId + 5);
        if ((sdpId & 1) == 0) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_UB2L1_P_FLAG);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_UB2L1_DS_FLAG);
        }
        LocalTensor<INPUT_TYPE> pL1Tensor = pL1Buf.Get().GetTensor<INPUT_TYPE>();
        if (runInfo.quantRunInfo.s2Idx < runInfo.quantRunInfo.innerS2LoopNum && runInfo.quantRunInfo.s1Idx < runInfo.quantRunInfo.innerS1LoopNum) {
            LocalTensor<INPUT_TYPE> dsL1Tensor = L1DS(runInfo.quantRunInfo.s1Idx, runInfo.quantRunInfo.s2Idx, runInfo.commonRunInfo.taskIdMod2).Get().GetTensor<INPUT_TYPE>();
            this->vecBlock.ProcessPDs(pL1Tensor, dsL1Tensor, spTensor[sdpId], dpdsTensor[sdpId], sdpId, this->constInfo, runInfo);
        }
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(sdpId);
        if (sdpId % 2 == 1) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_PDS_TO_DKV_FLAG);
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
template<bool isDK>
__aicore__ inline void FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::ProcessDKV(FagRunInfo &runInfo, int8_t sdpId)
{
    if ASCEND_IS_AIC {
        if constexpr(!isDK) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_PDS_TO_DKV_FLAG);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_PDS_TO_DKV_FLAG);
            LocalTensor<INPUT_TYPE> pL1Tensor0 = pL1Buf.Get().GetTensor<INPUT_TYPE>();
            LocalTensor<INPUT_TYPE> pL1Tensor1 = pL1Buf.Get().GetTensor<INPUT_TYPE>();
            if (runInfo.quantRunInfo.s2Idx < runInfo.quantRunInfo.innerS2LoopNum && runInfo.quantRunInfo.s1Idx < runInfo.quantRunInfo.innerS1LoopNum) {    
                runInfo.quantRunInfo.isDvFixOut = sdpId == 2;
                this->cubeBlock.IterateMmPDy(dvTensor, pL1Tensor0, pL1Tensor1, this->constInfo, runInfo);
            }

            // P反向同步
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_UB2L1_P_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_UB2L1_P_FLAG);
        } else {
            if (runInfo.quantRunInfo.s2Idx < runInfo.quantRunInfo.innerS2LoopNum && runInfo.quantRunInfo.s1Idx < runInfo.quantRunInfo.innerS1LoopNum) {
                runInfo.quantRunInfo.isDkFixOut = sdpId == 2;
                LocalTensor<INPUT_TYPE> dsL1Tensor0 = L1DS(runInfo.quantRunInfo.s1Idx, runInfo.quantRunInfo.s2Idx, runInfo.commonRunInfo.taskIdMod2).Get().GetTensor<INPUT_TYPE>();
                LocalTensor<INPUT_TYPE> dsL1Tensor1 = L1DS(runInfo.quantRunInfo.s1Idx + 1, runInfo.quantRunInfo.s2Idx, runInfo.commonRunInfo.taskIdMod2).Get().GetTensor<INPUT_TYPE>();
                this->cubeBlock.IterateMmDsQ(dkTensor, dsL1Tensor0, dsL1Tensor1, this->constInfo, runInfo);
            }
            if (sdpId == 0 && runInfo.quantRunInfo.s2Idx == 3) {
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_UB2L1_DS_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_UB2L1_DS_FLAG);
            }

            if (sdpId == 2) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_TRANSFER_DKV_FLAG);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_TRANSFER_DKV_FLAG);
                // 搬出dk dv
                if (runInfo.quantRunInfo.s2Idx < runInfo.quantRunInfo.innerS2LoopNum) {
                    this->cubeBlock.CopyOutDkDvResult(dvTensor, dkTensor, this->constInfo);
                }

                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_COMPUTE_DKV_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_COMPUTE_DKV_FLAG);
            }
        }
    }
    if ASCEND_IS_AIV {
        if (sdpId == 2 && isDK) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_COMPUTE_DKV_FLAG);
            if (runInfo.quantRunInfo.s2Idx < runInfo.quantRunInfo.innerS2LoopNum) {
                this->vecBlock.template DequantDqkv<1>(dvTensor, dkTensor, this->constInfo, runInfo, 0, 0);
            }
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::ProcessDQ(FagRunInfo &runInfo, int8_t sdpId)
{
    if ASCEND_IS_AIC {
        LocalTensor<INPUT_TYPE> dsL1Tensor0 = L1DS(runInfo.quantRunInfo.s1Idx, runInfo.quantRunInfo.s2Idx, runInfo.commonRunInfo.taskIdMod2).Get().GetTensor<INPUT_TYPE>();
        LocalTensor<INPUT_TYPE> dsL1Tensor1 = L1DS(runInfo.quantRunInfo.s1Idx, runInfo.quantRunInfo.s2Idx + 1, runInfo.commonRunInfo.taskIdMod2).Get().GetTensor<INPUT_TYPE>();
        runInfo.quantRunInfo.isDqFixOut = sdpId == 2;
        if (runInfo.quantRunInfo.s2Idx < runInfo.quantRunInfo.innerS2LoopNum && runInfo.quantRunInfo.s1Idx < runInfo.quantRunInfo.innerS1LoopNum) {
            this->cubeBlock.IterateMmDsK(dqTensor, dsL1Tensor0, dsL1Tensor1, this->constInfo, runInfo);
        }
        if (!(sdpId == 2 && runInfo.quantRunInfo.s1Idx == 0)) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_UB2L1_DS_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_UB2L1_DS_FLAG);
        }
        

        if (sdpId == 2) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_TRANSFER_DQ_FLAG);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_TRANSFER_DQ_FLAG);
            if (runInfo.quantRunInfo.s1Idx < runInfo.quantRunInfo.innerS1LoopNum) {
                this->cubeBlock.CopyOutDqResult(dqTensor, this->constInfo);
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_COMPUTE_DKV_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_COMPUTE_DKV_FLAG);
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernelQuant<CubeBlockType, VecBlockType>::ComputeDQ(FagRunInfo &runInfo, int8_t sdpId)
{
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_COMPUTE_DKV_FLAG);
        if (runInfo.quantRunInfo.s1Idx < runInfo.quantRunInfo.innerS1LoopNum) {
            this->vecBlock.template DequantDqkv<0>(dqTensor, dqTensor, this->constInfo, runInfo, 0, 0);
        }
    }
}

} // namespace FagBaseApi
 
 
#endif
