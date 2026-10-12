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
 * \file vec_op.h
 * \brief
 */

#pragma once
#include "kernel_operator.h"
#include "common_header.h"

namespace SFAG_BASIC {
struct StaticParams {
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint32_t sftBaseM = 0;
    uint32_t sftBaseN = 0;
};

template <typename SFAGT>
class VecOp {
public:
    using TILING_CLASS = typename SFAGT::tiling_class;
    using T1 = typename SFAGT::t1;
    static constexpr uint32_t ATTEN_ENABLE = SFAGT::atten_enable;
    static constexpr bool HAS_ROPE = SFAGT::has_rope;
    static constexpr bool IS_BSND = SFAGT::is_bsnd;

    __aicore__ inline VecOp(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                GM_ADDR key_rope, 
                                GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                const TILING_CLASS *__restrict ordTilingData, TPipe *pipe);
    __aicore__ inline void Process(const RunInfo &runInfo);
    __aicore__ inline void GatherKV(const int64_t n2Index, uint64_t currentS1Offset, const RunInfo &runInfo);
    __aicore__ inline void ScatterAdd(const RunInfo &runInfo);

protected:
    __aicore__ inline void InitParams(const TILING_CLASS *__restrict ordTilingData, GM_ADDR actual_seq_qlen,
                                      GM_ADDR actual_seq_kvlen);
    __aicore__ inline void InitGMBuffer(GM_ADDR key, GM_ADDR value, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR softmax_max,
                                        GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR key_rope, GM_ADDR workspace);
    __aicore__ inline void InitUB(TPipe *pipe);
    __aicore__ inline void AtomicClean();
    __aicore__ inline void DumpGmZero(GlobalTensor<float> &gm, int64_t num);
    __aicore__ inline void CalRowsumAndSftCopyIn(const int64_t dyGmOffset, const int64_t sumGmOffset, const int32_t processM);
    __aicore__ inline void CalAttenMsk(const int32_t processM, const int32_t actualSelS2Align, const RunInfo &runInfo);
    __aicore__ inline void CalSoftmax(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                      const int64_t mm345Addr, const RunInfo &runInfo);
    __aicore__ inline void CalSoftmaxGrad(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                          const int64_t mm345Addr, const RunInfo &runInfo);
    __aicore__ inline void CalSub(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, int32_t baseM,
                                  int32_t baseN);
    __aicore__ inline void ScatterAddUnDeter(const RunInfo &runInfo);
    __aicore__ inline void ScatterAddDeter(const RunInfo &runInfo);
protected:
    // core info
    int64_t usedCoreNum;
    int64_t formerCoreNum;
    uint32_t cubeBlockIdx;
    uint32_t vecBlockIdx;
    uint32_t subBlockIdx;
    StaticParams params;
    GlobalTensor<T1> attentionGm;
    GlobalTensor<T1> attentionGradGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<int32_t> topkIndicesGm;
    GlobalTensor<T1> keyGm;
    GlobalTensor<T1> valueGm;
    GlobalTensor<T1> keyRopeGm;

    // workspace
    GlobalTensor<T1> selectedKWorkspaceGm;
    GlobalTensor<T1> selectedVWorkspaceGm;
    GlobalTensor<float> mm1WorkspaceGm;
    GlobalTensor<float> mm2WorkspaceGm;
    GlobalTensor<T1> pWorkspaceGm;
    GlobalTensor<T1> dsWorkspaceGm;
    GlobalTensor<float> dqWorkspaceGm;
    GlobalTensor<float> dkWorkspaceGm;
    GlobalTensor<float> dvWorkspaceGm;
    GlobalTensor<float> mm4ResWorkspaceGm; // 24 * 2 * K * Dk
    GlobalTensor<float> mm5ResWorkspaceGm; // 24 * 2 * K * Dv

    TBuf<> vecQue;
    LocalTensor<uint8_t> helpTensor;
    LocalTensor<int32_t> topkIndicesTensor;
    LocalTensor<float> rowSumOutTensor;
    LocalTensor<float> maxTensor;
    LocalTensor<float> sumTensor;
    LocalTensor<float> maxTmp;
    LocalTensor<float> sumTmp;
    LocalTensor<T1> attentionT1Tensor;
    LocalTensor<float> attentionFP32Tensor;
    LocalTensor<T1> attentionGradT1Tensor;
    LocalTensor<float> attentionGradFP32Tensor;
    LocalTensor<float> pTensor;
    LocalTensor<float> dPTensor;
    LocalTensor<T1> sftOutT1Tensor;
    LocalTensor<T1> sftgOutT1Tensor;
    LocalTensor<T1> gatherTensorPing;
    LocalTensor<T1> gatherTensorPong;
    LocalTensor<T1> gatherRopePing;
    LocalTensor<T1> gatherRopePong;
    LocalTensor<float> scatterAddTensorK;
    LocalTensor<float> scatterAddTensorV;

    // 地址相关
    int64_t selectedKWspOffset{0};
    int64_t selectedVWspOffset{0};

    constexpr static const int32_t BLOCK = 32;
    constexpr static const int32_t BLOCK_T1 = BLOCK / sizeof(T1);
    constexpr static const int32_t BLOCK_FP32 = BLOCK / sizeof(float);
    constexpr static const int32_t BLOCK_INT32 = BLOCK / sizeof(int32_t);
    constexpr static const int32_t MSK_LEN = 64;
    constexpr static const uint32_t ATTEN_MASK_SCALE = 0xFF7FFFFF;
    constexpr static const uint32_t MAX_GATHER_SIZE = 64;
    constexpr static const uint32_t UB_ROW_SIZE = 16;

    const TILING_CLASS *__restrict tilingData;
    // Shape
    int64_t dimB;
    int64_t dimN2;
    int64_t dimS1;
    int64_t dimS2;
    int64_t dimG;
    int64_t dimD;
    int64_t dimD2;
    int64_t dimDqk; // Dqk = D
    int64_t dimDv;  // Dv = D2
    int64_t dimDAlign;
    int64_t dimD2Align;
    int64_t dimTkv{0};
    int64_t dimTq{0};
    int64_t selectedS2{0};
    int64_t curS1;
    int64_t curS2;
    int64_t lastBlock;
    int64_t selectedBlockSizeDqk;
    int64_t selectedBlockSizeDrope;
    int64_t selectedBlockSizeDimDAlign;
    int64_t selectedBlockSizeDimD2Align;
    int64_t ubRowSizeDAlign;
    int64_t ubRowSizeD2Align;
    float scaleValue;
    uint32_t selectedBlockCount;
    uint32_t selectedBlockSize;
    uint32_t selectedCountOffset;
    uint32_t actualSelectedCount;
    int32_t maxSelCnt;
    int64_t dimRope;
    bool deterministic = true;
    // workspace
    uint32_t mm12WorkspaceLen;
    int64_t dqWorkspaceLen;
    int64_t dkWorkspaceLen;
    int64_t dvWorkspaceLen;
    int64_t selectedKWorkspaceLen;
    int64_t selectedVWorkspaceLen;
    int64_t concatQWorkspaceLen;
    event_t vWaitMte2;
    event_t vWaitMte2Pong;
    event_t vWaitMte3;
    event_t mte3WaitMte2;
    event_t mte3WaitV;
    event_t sWaitMte2;
    event_t mte2WaitMte3;
    event_t mte2WaitMte3Pong;
    event_t mte3WaitMte2Pong;

    DataCopyPadExtParams<T1> padParams;
    DataCopyExtParams intriParamsKey;
    DataCopyExtParams intriParamsRope;
    DataCopyExtParams outParamK;
    DataCopyExtParams outParamRope;
    uint32_t mergePingPong{0};
    int32_t pingPongIdx = 0;
};

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                          GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                          GM_ADDR topk_indices, GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen,
                                          GM_ADDR key_rope, 
                                          GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                          const TILING_CLASS *__restrict ordTilingData, TPipe *pipe)
{
    InitParams(ordTilingData, actual_seq_qlen, actual_seq_kvlen);
    InitGMBuffer(key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices, key_rope, workspace);
    InitUB(pipe);
    AtomicClean();
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::InitParams(const TILING_CLASS *__restrict ordTilingData, GM_ADDR actual_seq_qlen,
                                                GM_ADDR actual_seq_kvlen)
{
    cubeBlockIdx = GetBlockIdx() / 2;
    vecBlockIdx = GetBlockIdx();
    subBlockIdx = GetSubBlockIdx();
    tilingData = ordTilingData;
    usedCoreNum = tilingData->opInfo.usedCoreNum;
    formerCoreNum = tilingData->opInfo.formerCoreNum;

    dimB = tilingData->opInfo.B;
    dimN2 = tilingData->opInfo.N2;
    dimS1 = tilingData->opInfo.S1;
    dimS2 = tilingData->opInfo.S2;
    dimG = tilingData->opInfo.G;
    dimD = tilingData->opInfo.D;
    dimD2 = tilingData->opInfo.D2;
    dimDqk = tilingData->opInfo.D;
    dimDv = tilingData->opInfo.D2;
    dimRope = tilingData->opInfo.ropeD;
    dimDAlign = (dimD + dimRope + BLOCK_T1 - 1) / BLOCK_T1 * BLOCK_T1;
    dimD2Align = (dimD2 + BLOCK_T1 - 1) / BLOCK_T1 * BLOCK_T1;
    deterministic = tilingData->opInfo.deterministic;

    scaleValue = tilingData->opInfo.scaleValue;
    selectedBlockCount = tilingData->opInfo.selectedBlockCount;
    selectedBlockSize = tilingData->opInfo.selectedBlockSize;
    mm12WorkspaceLen = tilingData->opInfo.mm12WorkspaceLen;
    dqWorkspaceLen = tilingData->opInfo.dqWorkspaceLen;
    dkWorkspaceLen = tilingData->opInfo.dkWorkspaceLen;
    dvWorkspaceLen = tilingData->opInfo.dvWorkspaceLen;
    selectedKWorkspaceLen = tilingData->opInfo.selectedKWorkspaceLen;
    selectedVWorkspaceLen = tilingData->opInfo.selectedVWorkspaceLen;
    selectedBlockSizeDqk = selectedBlockSize * dimDqk;
    selectedBlockSizeDrope = selectedBlockSize * dimRope;
    selectedBlockSizeDimDAlign = selectedBlockSize * dimDAlign;
    selectedBlockSizeDimD2Align = selectedBlockSize * dimD2Align;
    ubRowSizeDAlign = UB_ROW_SIZE * dimDAlign;
    ubRowSizeD2Align = UB_ROW_SIZE * dimD2Align;

    params.singleM = tilingData->splitCoreParams.singleM;
    params.singleN = tilingData->splitCoreParams.singleN;
    params.sftBaseM = tilingData->splitCoreParams.sftBaseM;
    params.sftBaseN = tilingData->splitCoreParams.sftBaseN;

    selectedS2 = selectedBlockCount * selectedBlockSize;
    selectedCountOffset = PER_LOOP_BLOCK_SIZE / selectedBlockSize;
    if (tilingData->opInfo.selectedBlockSize * tilingData->opInfo.selectedBlockCount <= PER_LOOP_BLOCK_SIZE) {
        selectedCountOffset = tilingData->opInfo.selectedBlockCount;
    }

    if constexpr (IS_BSND == false) {
        for (int64_t i = 0; i < dimB; i++) {
            int64_t seqS1Len = 0;
            int64_t seqS2Len = 0;
            if (unlikely(i == 0)) {
                seqS1Len = ((__gm__ int32_t *)actual_seq_qlen)[i];
                seqS2Len = ((__gm__ int32_t *)actual_seq_kvlen)[i];
            } else {
                seqS1Len = ((__gm__ int32_t *)actual_seq_qlen)[i] - ((__gm__ int32_t *)actual_seq_qlen)[i - 1];
                seqS2Len = ((__gm__ int32_t *)actual_seq_kvlen)[i] - ((__gm__ int32_t *)actual_seq_kvlen)[i - 1];
            }
            dimTq += (int64_t)seqS1Len;
            dimTkv += (int64_t)seqS2Len;
        }
    } else {
        dimTq = dimB * dimS1;
        dimTkv = dimB * dimS2;
    }

    intriParamsKey.blockLen = selectedBlockSize * dimDqk * sizeof(T1);
    intriParamsKey.dstStride = 0;
    intriParamsKey.blockCount = 2;

    intriParamsRope.blockLen = selectedBlockSize * dimRope * sizeof(T1);
    intriParamsRope.dstStride = 0;
    intriParamsRope.blockCount = 2;

    outParamK.blockCount = 2 * selectedBlockSize;
    outParamK.blockLen = dimDqk * sizeof(T1);
    outParamK.srcStride = 0;
    outParamK.dstStride = dimRope * sizeof(T1);

    outParamRope.blockCount = 2 * selectedBlockSize;
    outParamRope.blockLen = dimRope * sizeof(T1);
    outParamRope.srcStride = 0;
    outParamRope.dstStride = dimDqk * sizeof(T1);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::InitGMBuffer(GM_ADDR key, GM_ADDR value, GM_ADDR attention_out, GM_ADDR attention_out_grad,
                                                  GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR key_rope, GM_ADDR workspace)
{
    /*
     * 初始化输入
     */
    attentionGm.SetGlobalBuffer((__gm__ T1 *)attention_out);
    attentionGradGm.SetGlobalBuffer((__gm__ T1 *)attention_out_grad);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
    topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topk_indices);
    keyGm.SetGlobalBuffer((__gm__ T1 *)key);
    valueGm.SetGlobalBuffer((__gm__ T1 *)value);
    keyRopeGm.SetGlobalBuffer((__gm__ T1 *)key_rope);

    /*
     * 初始化workspace
     */
    int64_t usedWorkspaceLen = 0;
    // select
    int64_t selectedKAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * selectedKWorkspaceLen / sizeof(T1);
    usedWorkspaceLen += selectedKWorkspaceLen * usedCoreNum;
    int64_t selectedVAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * selectedVWorkspaceLen / sizeof(T1);
    usedWorkspaceLen += selectedVWorkspaceLen * usedCoreNum;

    int64_t mm1Addr = usedWorkspaceLen / sizeof(float) + cubeBlockIdx * mm12WorkspaceLen / sizeof(float);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    int64_t pAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;

    int64_t mm2Addr = usedWorkspaceLen / sizeof(float) + cubeBlockIdx * mm12WorkspaceLen / sizeof(float);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    int64_t dsAddr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * mm12WorkspaceLen / sizeof(T1);
    usedWorkspaceLen += mm12WorkspaceLen * usedCoreNum;
    // post
    int64_t dqAddr = usedWorkspaceLen / sizeof(float);
    int64_t dkAddr = dqAddr + dqWorkspaceLen / sizeof(float);
    int64_t dvAddr = dkAddr + dkWorkspaceLen / sizeof(float);
    usedWorkspaceLen += dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;

    // scatter add
    int64_t mm4ResAddr = usedWorkspaceLen / sizeof(float);
    int64_t mm5ResAddr = mm4ResAddr + MAX_CORE_NUM * selectedBlockCount * selectedBlockSizeDimDAlign * 2;
    usedWorkspaceLen += MAX_CORE_NUM * selectedBlockCount * selectedBlockSize * (dimDAlign + dimD2Align) * 2 * sizeof(float);

    mm1WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm1Addr);
    mm2WorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm2Addr);
    pWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + pAddr);
    dsWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + dsAddr);
    dqWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dqAddr);
    dkWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dkAddr);
    dvWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dvAddr);
    selectedKWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + selectedKAddr);
    selectedVWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + selectedVAddr);

    mm4ResWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm4ResAddr);
    mm5ResWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + mm5ResAddr);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::InitUB(TPipe *pipe)
{
    uint32_t dataSize = params.sftBaseM * dimDv;
    uint32_t ubOffset = 0;
    uint32_t rowsumUbOffset = 0;
    uint32_t totalUbSpace = 191 * 1024;

    pipe->InitBuffer(vecQue, totalUbSpace);

    // topk
    int32_t topkNumber = AlignTo<int32_t>(selectedCountOffset, BLOCK);
    topkIndicesTensor = vecQue.GetWithOffset<int32_t>(topkNumber, ubOffset);
    ubOffset += topkNumber * sizeof(int32_t);
    uint32_t topkUbOffset = ubOffset;

    // rowsum out
    rowSumOutTensor = vecQue.GetWithOffset<float>(params.sftBaseM * BLOCK_FP32, ubOffset);
    ubOffset += params.sftBaseM * BLOCK_FP32 * sizeof(float);
    maxTensor = vecQue.GetWithOffset<float>(params.sftBaseM * BLOCK_FP32, ubOffset);
    ubOffset += params.sftBaseM * BLOCK_FP32 * sizeof(float);
    sumTensor = vecQue.GetWithOffset<float>(params.sftBaseM * BLOCK_FP32, ubOffset);
    ubOffset += params.sftBaseM * BLOCK_FP32 * sizeof(float);

    int32_t maxTmpSize = AlignUp(params.sftBaseM, BLOCK_FP32);
    maxTmp = vecQue.GetWithOffset<float>(maxTmpSize, ubOffset);
    ubOffset += maxTmpSize * sizeof(float);
    sumTmp = vecQue.GetWithOffset<float>(maxTmpSize, ubOffset);
    ubOffset += maxTmpSize * sizeof(float);
    rowsumUbOffset = ubOffset;

    // rowsum cal
    attentionGradT1Tensor = vecQue.GetWithOffset<T1>(params.sftBaseM * dimDv, ubOffset);
    ubOffset += params.sftBaseM * dimDv * sizeof(T1); //
    attentionGradFP32Tensor = vecQue.GetWithOffset<float>(params.sftBaseM * dimDv, ubOffset);
    ubOffset += params.sftBaseM * dimDv * sizeof(float);
    attentionT1Tensor = vecQue.GetWithOffset<T1>(params.sftBaseM * dimDv, ubOffset);
    ubOffset += params.sftBaseM * dimDv * sizeof(T1);
    attentionFP32Tensor = vecQue.GetWithOffset<float>(params.sftBaseM * dimDv, ubOffset);
    ubOffset += params.sftBaseM * dimDv * sizeof(float);

    // sftGard
    int32_t sftDataSize = params.sftBaseM * params.sftBaseN;
    pTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(float);
    dPTensor = vecQue.GetWithOffset<float>(sftDataSize, ubOffset);
    sftgOutT1Tensor = vecQue.GetWithOffset<T1>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(float);

    sftOutT1Tensor = vecQue.GetWithOffset<T1>(sftDataSize, ubOffset);
    ubOffset += sftDataSize * sizeof(T1);

    helpTensor = vecQue.GetWithOffset<uint8_t>((totalUbSpace - rowsumUbOffset) / sizeof(uint8_t), rowsumUbOffset);
    scatterAddTensorK = vecQue.GetWithOffset<float>(16 * dimDAlign * 2, rowsumUbOffset);
    rowsumUbOffset += 16 * dimDAlign * 2 * sizeof(float);
    scatterAddTensorV = vecQue.GetWithOffset<float>(16 * dimD2Align * 2, rowsumUbOffset);
    rowsumUbOffset += 16 * dimD2Align * 2 * sizeof(float);

    maxSelCnt = selectedBlockSize <= 32 ? 2 : 1;
    gatherTensorPing = vecQue.GetWithOffset<T1>(MAX_GATHER_SIZE * dimDqk, topkUbOffset);
    topkUbOffset += MAX_GATHER_SIZE * dimDqk * sizeof(T1);
    gatherTensorPong = vecQue.GetWithOffset<T1>(MAX_GATHER_SIZE * dimDqk, topkUbOffset);
    topkUbOffset += MAX_GATHER_SIZE * dimDqk * sizeof(T1);
    gatherRopePing = vecQue.GetWithOffset<T1>(MAX_GATHER_SIZE * dimRope, topkUbOffset);
    topkUbOffset += MAX_GATHER_SIZE * dimRope * sizeof(T1);
    gatherRopePong = vecQue.GetWithOffset<T1>(MAX_GATHER_SIZE * dimRope, topkUbOffset);
    topkUbOffset += MAX_GATHER_SIZE * dimRope * sizeof(T1);

    uint32_t attentionShape[2] = {params.sftBaseM, static_cast<uint32_t>(dimDv)};
    uint32_t softmaxGradShape[2] = {params.sftBaseM, BLOCK_FP32};
    attentionGradFP32Tensor.SetShapeInfo(ShapeInfo(2, attentionShape, DataFormat::ND));
    attentionFP32Tensor.SetShapeInfo(ShapeInfo(2, attentionShape, DataFormat::ND));
    rowSumOutTensor.SetShapeInfo(ShapeInfo(2, softmaxGradShape, DataFormat::ND));
    rowSumOutTensor.SetSize(params.sftBaseM * BLOCK_FP32);
    attentionGradT1Tensor.SetSize(dataSize);
    attentionGradFP32Tensor.SetSize(dataSize);
    attentionT1Tensor.SetSize(dataSize);
    attentionFP32Tensor.SetSize(dataSize);
    // params
    sumTensor.SetSize(params.sftBaseM * 8);
    maxTensor.SetSize(params.sftBaseM * 8);

    // set shape
    uint32_t softmaxShape[2] = {static_cast<uint32_t>(params.sftBaseM), 8};
    sumTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));
    maxTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));

    uint32_t dstSoftShape[2] = {static_cast<uint32_t>(params.sftBaseM), static_cast<uint32_t>(params.sftBaseN)};
    pTensor.SetShapeInfo(ShapeInfo(2, dstSoftShape, DataFormat::ND));
    mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    mte3WaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
    mte2WaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    mte3WaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
    sWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
    vWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    vWaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    mte3WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    vWaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::AtomicClean()
{
    int64_t dqSize, dkSize, dvSize;
    dqSize = dimTq * dimN2 * dimG * dimDAlign;
    dkSize = dimTkv * dimN2 * dimDAlign;
    dvSize = dimTkv * dimN2 * dimD2Align;
    dqSize = (dqSize + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;
    dkSize = (dkSize + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;
    dvSize = (dvSize + BLOCK_INT32 - 1) / BLOCK_INT32 * BLOCK_INT32;

    DumpGmZero(dqWorkspaceGm, dqSize);
    DumpGmZero(dkWorkspaceGm, dkSize);
    DumpGmZero(dvWorkspaceGm, dvSize); // FP32 FUNC:T1 FP16
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::DumpGmZero(GlobalTensor<float> &gm, int64_t num)
{
    int64_t perSize = (num + tilingData->opInfo.castUsedCoreNum - 1) / tilingData->opInfo.castUsedCoreNum;
    int64_t coreNum = (num + perSize - 1) / perSize;
    int64_t tailSize = num - perSize * (coreNum - 1);
    int64_t initSize = perSize;

    if (vecBlockIdx == coreNum - 1) {
        initSize = tailSize;
    }

    if (vecBlockIdx < coreNum) {
        InitOutput<float>(gm[vecBlockIdx * perSize], initSize, 0);
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalSub(LocalTensor<float> &dstTensor, LocalTensor<float> &srcTensor, int32_t baseM,
                                            int32_t baseN)
{
    constexpr static int32_t subMsk = 64;
    int8_t s2Repeat = baseN / subMsk;
    int8_t s2RepeatTail = baseN % subMsk;

    for (int32_t i = 0; i < baseM; i++) {
        Sub(dstTensor[i * baseN], dstTensor[i * baseN], srcTensor[i * BLOCK_FP32], subMsk, s2Repeat,
            {1, 1, 0, 8, 8, 0});
    }

    if (s2RepeatTail) {
        for (int32_t i = 0; i < baseM; i++) {
            Sub(dstTensor[s2Repeat * subMsk + i * baseN], dstTensor[s2Repeat * subMsk + i * baseN],
                srcTensor[i * BLOCK_FP32], s2RepeatTail, 1, {1, 1, 0, 8, 8, 0});
        }
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalRowsumAndSftCopyIn(const int64_t dyGmOffset, const int64_t sumGmOffset, const int32_t processM)
{
    uint32_t dataSize = processM * dimDv;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(processM * dimDv * sizeof(T1)), 0, 0, 0};
    DataCopyExtParams copyParams2{1, static_cast<uint32_t>(processM * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    DataCopyPadExtParams<float> padParams2{false, 0, 0, 0};

    DataCopyPad(attentionGradT1Tensor, attentionGradGm[dyGmOffset], copyParams, padParams);
    DataCopyPad(attentionT1Tensor, attentionGm[dyGmOffset], copyParams, padParams);
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    Cast(attentionGradFP32Tensor, attentionGradT1Tensor, RoundMode::CAST_NONE, dataSize);
    Cast(attentionFP32Tensor, attentionT1Tensor, RoundMode::CAST_NONE, dataSize);
    PIPE_BARRIER(PIPE_V);
    SoftmaxGradFront<float, false>(rowSumOutTensor, attentionGradFP32Tensor, attentionFP32Tensor, helpTensor,
                                   tilingData->softmaxGradTilingData);

    // [N2, T1, G] or [B, N2, S1, G] --> [sftBaseM, 1] --> [sftBaseM, 8]
    DataCopyPad(maxTmp, softmaxMaxGm[sumGmOffset], copyParams2, padParams2);
    DataCopyPad(sumTmp, softmaxSumGm[sumGmOffset], copyParams2, padParams2);
    SET_FLAG(MTE2, V, vWaitMte2);
    
    WAIT_FLAG(MTE2, V, vWaitMte2);
    Brcb(maxTensor, maxTmp, CeilDiv(params.sftBaseM, BLOCK_FP32), {1, 8});
    Brcb(sumTensor, sumTmp, CeilDiv(params.sftBaseM, BLOCK_FP32), {1, 8});
    PIPE_BARRIER(PIPE_V);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalAttenMsk(const int32_t processM, const int32_t actualSelS2Align, const RunInfo &runInfo)
{ /*
   * 计算attenmask时对应blockSize需要保留的长度。
   * 例如：对于blockSize=64的块，需要保留0-9块，则attenMskRsvLen=10；
   */
    if (selectedBlockSize == 1 || !runInfo.isLastBasicBlock) {
        return;
    }
    int32_t attenMskRsv = 0;
    int32_t attenMskEnd = 0;
    int32_t attenMskStartIdx = -1; // 计算attenmask时attenmask开始计算的下标
    float scalar = *((float *)&ATTEN_MASK_SCALE);
    uint64_t mask[1];
    LocalTensor<float> tmpTensor;

    int64_t valid_col_end = runInfo.s1Index + runInfo.curS2 - runInfo.curS1;
    int32_t i = runInfo.blkCntOffset + runInfo.actualSelCntOffset - 1;
    int32_t topkIdx = topkIndicesGm[runInfo.indicesGmOffset].GetValue(i);
    // 处于对角线上的block
    if (topkIdx * selectedBlockSize <= valid_col_end && (topkIdx + 1) * selectedBlockSize > valid_col_end) {
        attenMskRsv = valid_col_end - (topkIdx * selectedBlockSize) + 1;
        attenMskEnd = (i == runInfo.blkCntOffset + runInfo.actualSelCntOffset - 1 && runInfo.isLastBasicBlock) ? runInfo.lastBlockSize : selectedBlockSize;
        attenMskStartIdx = i * selectedBlockSize + attenMskRsv - runInfo.blkCntOffset * selectedBlockSize;
    }

    if (attenMskStartIdx == -1) {
        return; // 默认mask掉的块为topk的无效值，不参与计算
    }

    int32_t selectIdx = (attenMskStartIdx - 1) / selectedBlockSize;
    int32_t selectBlkStartOffset = selectIdx * selectedBlockSize;
    int32_t selectBlkStartOffsetAlign = selectBlkStartOffset / 8 * 8;
    attenMskRsv += selectBlkStartOffset - selectBlkStartOffsetAlign;
    attenMskEnd += selectBlkStartOffset - selectBlkStartOffsetAlign;

    if (attenMskEnd - attenMskRsv > 0) {
        for (int32_t i = 0; i < processM; i++) { // selectedBlockSize need <= 64
            int32_t pOffset = i * actualSelS2Align;
            tmpTensor = pTensor[selectBlkStartOffsetAlign + pOffset];
            // attenMskRsv ~ attenMskEnd位置1
            uint64_t maskEnd = (attenMskEnd == 64) ? UINT64_MAX : (1ULL << attenMskEnd) - 1;
            uint64_t maskRsv = (attenMskRsv == 64) ? UINT64_MAX : (1ULL << attenMskRsv) - 1;
            mask[0] = maskEnd & ~maskRsv;
            Duplicate(tmpTensor, scalar, mask, 1, 0, 0);
        }
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalSoftmax(const int32_t loopIdx, const int32_t processM, const int64_t mm12Addr,
                                                const int64_t mm345Addr, const RunInfo &runInfo)
{
    int64_t actualSelS2 = runInfo.isLastBasicBlock ? 
                          (runInfo.actualSelCntOffset - 1) * selectedBlockSize + runInfo.lastBlockSize : 
                          runInfo.actualSelCntOffset * selectedBlockSize;
    int64_t actualSelS2Align = AlignUp(actualSelS2, 8);
    int64_t dataSize = processM * actualSelS2Align;

    DataCopyPad(pTensor, mm1WorkspaceGm[mm12Addr], 
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(float)), static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(float)), 0, 0},
                {false, 0, 0, 0});
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    Muls(pTensor, pTensor, scaleValue, dataSize);
    PIPE_BARRIER(PIPE_V);

    if constexpr (ATTEN_ENABLE) {
        CalAttenMsk(processM, actualSelS2Align, runInfo);
        PIPE_BARRIER(PIPE_V);
    }

    uint32_t dstSoftShape[2] = {static_cast<uint32_t>(processM), static_cast<uint32_t>(actualSelS2Align)};
    pTensor.SetShapeInfo(ShapeInfo(2, dstSoftShape, DataFormat::ND));
    SoftMaxShapeInfo softmaxShapeInfo{
            static_cast<uint32_t>(processM), static_cast<uint32_t>(actualSelS2Align),
            static_cast<uint32_t>(processM), static_cast<uint32_t>(actualSelS2)};
    SimpleSoftMax<float, true, false>(pTensor, sumTensor, maxTensor, pTensor, helpTensor,
                                      tilingData->softmaxTilingData, softmaxShapeInfo);
    PIPE_BARRIER(PIPE_V);

    auto castS2Loop = CeilDiv(actualSelS2Align, 64);
    for (int i = 0; i < castS2Loop; ++i) {
        uint64_t mask = i == (castS2Loop - 1) ? actualSelS2 - (i * 64) : 64;
        Cast(sftOutT1Tensor[i * 64], pTensor[i * 64], RoundMode::CAST_ROUND, mask, processM, {1, 1, static_cast<uint8_t>(CeilDiv(actualSelS2Align, 16)), static_cast<uint8_t>(CeilDiv(actualSelS2Align, 8))});
    }
    SET_FLAG(V, MTE3, mte3WaitV);
    WAIT_FLAG(V, MTE3, mte3WaitV);

    DataCopyPad(pWorkspaceGm[mm345Addr], sftOutT1Tensor, 
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(T1)), 0, static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(T1)), 0});
    SET_FLAG(MTE3, MTE2, mte2WaitMte3);
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::CalSoftmaxGrad(const int32_t loopIdx, const int32_t processM,
                                                    const int64_t mm12Addr, const int64_t mm345Addr, const RunInfo &runInfo)
{
    int64_t actualSelS2 = runInfo.isLastBasicBlock ? 
                          (runInfo.actualSelCntOffset - 1) * selectedBlockSize + runInfo.lastBlockSize : 
                          runInfo.actualSelCntOffset * selectedBlockSize;
    int64_t actualSelS2Align = AlignUp(actualSelS2, 8);
    int64_t dataSize = processM * actualSelS2Align;
    DataCopyPad(dPTensor, mm2WorkspaceGm[mm12Addr], 
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(float)), static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(float)), 0, 0},
                {false, 0, 0, 0});
    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    CalSub(dPTensor, rowSumOutTensor, processM, actualSelS2Align);
    PIPE_BARRIER(PIPE_V);

    Mul(pTensor, pTensor, dPTensor, dataSize);
    PIPE_BARRIER(PIPE_V);

    auto castS2Loop = CeilDiv(actualSelS2Align, 64);
    for (int i = 0; i < castS2Loop; ++i) {
        uint64_t mask = i == (castS2Loop - 1) ? actualSelS2 - (i * 64) : 64;
        Cast(sftgOutT1Tensor[i * 64], pTensor[i * 64], RoundMode::CAST_ROUND, mask, params.sftBaseM, {1, 1, static_cast<uint8_t>(CeilDiv(actualSelS2Align, 16)), static_cast<uint8_t>(CeilDiv(actualSelS2Align, 8))});
    }
    SET_FLAG(V, MTE3, mte3WaitV);
    WAIT_FLAG(V, MTE3, mte3WaitV);

    DataCopyPad(dsWorkspaceGm[mm345Addr], sftgOutT1Tensor, 
                {static_cast<uint16_t>(processM), static_cast<uint32_t>(actualSelS2 * sizeof(T1)), 0, static_cast<uint32_t>((params.sftBaseN - actualSelS2) * sizeof(T1)), 0});
    SET_FLAG(MTE3, MTE2, mte2WaitMte3);
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::GatherKV(const int64_t n2Index, uint64_t currentS1Offset, const RunInfo &runInfo)
{
    if(runInfo.isSmallS2) return;
    uint64_t kSelectedWsAddr = runInfo.selectedKGmOffset;
    uint64_t vSelectedWsAddr = runInfo.selectedVGmOffset;
    uint64_t s1Offset = IS_BSND ? currentS1Offset : currentS1Offset + runInfo.s1Index;
    uint64_t gmOffset = s1Offset * (dimN2 * selectedBlockCount) + n2Index * selectedBlockCount + runInfo.blkCntOffset;

    outParamK.blockCount = selectedBlockSize < 64 ? 2 * selectedBlockSize : selectedBlockSize;
    outParamRope.blockCount = selectedBlockSize < 64 ? 2 * selectedBlockSize : selectedBlockSize;

    uint64_t totalD = dimDqk + dimRope;
    event_t mte2WaitMte3EventId;
    event_t mte3WaitMte2EventId;

    SET_FLAG(MTE3, MTE2, mte2WaitMte3Pong);
    SET_FLAG(MTE3, MTE2, mte2WaitMte3);

    // ------------- MergeKv --------------
    uint32_t s2Pair = CeilDiv(runInfo.actualSelCntOffset, 2) * 2;
    uint32_t firstVecEnd = (s2Pair / 2);
    uint32_t curBlk = subBlockIdx == 0 ? 0 : firstVecEnd;
    uint32_t curActualSelCntEnd = subBlockIdx == 0 ? firstVecEnd : runInfo.actualSelCntOffset;
    uint32_t curActualSelCntOffset = curActualSelCntEnd - curBlk;
    uint64_t outWsOffset = subBlockIdx == 0 ? 0 : firstVecEnd * selectedBlockSize * totalD;
    uint64_t orgOutWsOffset = outWsOffset;
    uint32_t i;

    bool isLast = runInfo.isLastBasicBlock && subBlockIdx == 1;

    uint32_t curGatherSize = 0;
    for (i = curBlk; i < curBlk + curActualSelCntOffset / maxSelCnt * maxSelCnt; i += maxSelCnt) {
        int64_t keyOffset1 = topkIndicesGm.GetValue(gmOffset + i) * selectedBlockSize;
        int64_t keyOffset2 = topkIndicesGm.GetValue(gmOffset + i + 1) * selectedBlockSize;

        uint32_t s2OrgStride = keyOffset2 - keyOffset1 - selectedBlockSize;
        intriParamsKey.blockCount = 2;
        intriParamsKey.blockLen = selectedBlockSizeDqk * sizeof(T1);

        mte2WaitMte3EventId = mergePingPong ? mte2WaitMte3Pong : mte2WaitMte3;
        mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Pong : mte3WaitMte2;

        WAIT_FLAG(MTE3, MTE2, mte2WaitMte3EventId);

        // CopyIn
        intriParamsKey.srcStride = s2OrgStride * dimN2 * dimDqk * sizeof(T1);
        LocalTensor<T1> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;
        LocalTensor<T1> &gatherRopeTensor = mergePingPong ? gatherRopePing : gatherRopePong;

        bool isActualLast = isLast && i >= runInfo.actualSelCntOffset - 2;
        uint32_t curGatherSizeDqk = curGatherSize * dimDqk;
        if (keyOffset2 <= keyOffset1 || selectedBlockSize >= 64 || isActualLast) {
            intriParamsKey.blockCount = 1;
            DataCopyPad(gatherTensor[curGatherSizeDqk], keyGm[runInfo.keyGmOffset + keyOffset1 * dimN2 * dimDqk], intriParamsKey, padParams);
            if (selectedBlockSize < 64) {
                intriParamsKey.blockLen = isActualLast ? runInfo.lastBlockSize * dimDqk * sizeof(T1)
                                                       : selectedBlockSizeDqk * sizeof(T1);
                DataCopyPad(gatherTensor[curGatherSizeDqk + selectedBlockSizeDqk], keyGm[runInfo.keyGmOffset + keyOffset2 * dimN2 * dimDqk], intriParamsKey, padParams);
            }
        } else {
            DataCopyPad(gatherTensor[curGatherSizeDqk], keyGm[runInfo.keyGmOffset + keyOffset1 * dimN2 * dimDqk], intriParamsKey, padParams);
        }

        if constexpr (HAS_ROPE) {
            intriParamsRope.blockCount = 2;
            intriParamsRope.srcStride = s2OrgStride * dimN2 * dimRope * sizeof(T1);
            intriParamsRope.blockLen = selectedBlockSizeDrope * sizeof(T1);

            uint32_t curGatherSizeDrope = curGatherSize * dimRope;
            if (keyOffset2 <= keyOffset1 || selectedBlockSize >= 64 || isActualLast) {
                intriParamsRope.blockCount = 1;
                DataCopyPad(gatherRopeTensor[curGatherSizeDrope], keyRopeGm[runInfo.keyRopeGmOffset + keyOffset1 * dimN2 * dimRope], intriParamsRope, padParams);
                if (selectedBlockSize < 64) {
                    intriParamsRope.blockLen = isActualLast ? runInfo.lastBlockSize * dimRope * sizeof(T1)
                                                            : selectedBlockSizeDrope * sizeof(T1);
                    DataCopyPad(gatherRopeTensor[curGatherSizeDrope + selectedBlockSizeDrope], keyRopeGm[runInfo.keyRopeGmOffset + keyOffset2 * dimN2 * dimRope], intriParamsRope, padParams);  
                }
            } else {
                DataCopyPad(gatherRopeTensor[curGatherSizeDrope], keyRopeGm[runInfo.keyRopeGmOffset + keyOffset1 * dimN2 * dimRope], intriParamsRope, padParams);  
            }
        }
        curGatherSize += maxSelCnt * selectedBlockSize;
        
        if (curGatherSize == MAX_GATHER_SIZE) {
            SET_FLAG(MTE2, MTE3, mte3WaitMte2EventId);
            WAIT_FLAG(MTE2, MTE3, mte3WaitMte2EventId);

            outParamK.blockCount = curGatherSize;
            outParamRope.blockCount = curGatherSize;
            // CopyOut
            DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + outWsOffset], gatherTensor, outParamK);
            if constexpr (HAS_ROPE) {
                DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + dimDqk + outWsOffset], gatherRopeTensor, outParamRope);
            }

            SET_FLAG(MTE3, MTE2, mte2WaitMte3EventId);
            mergePingPong = 1 - mergePingPong;
            outWsOffset += curGatherSize * totalD;
            curGatherSize = 0;
        } else {
            SET_FLAG(MTE3, MTE2, mte2WaitMte3EventId);
        }
    }
    if (i < curActualSelCntEnd) {
        int64_t keyOffset1 = topkIndicesGm.GetValue(gmOffset + i) * selectedBlockSize;

        mte2WaitMte3EventId = mergePingPong ? mte2WaitMte3Pong : mte2WaitMte3;
        mte3WaitMte2EventId = mergePingPong ? mte3WaitMte2Pong : mte3WaitMte2;

        WAIT_FLAG(MTE3, MTE2, mte2WaitMte3EventId);

        // CopyIn
        LocalTensor<T1> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;
        LocalTensor<T1> &gatherRopeTensor = mergePingPong ? gatherRopePing : gatherRopePong;

        intriParamsKey.blockCount = 1;
        intriParamsRope.blockCount = 1;
        if (i == runInfo.actualSelCntOffset - 1 && isLast) {
            intriParamsKey.blockLen = runInfo.lastBlockSize * dimDqk * sizeof(T1);
            intriParamsRope.blockLen = runInfo.lastBlockSize * dimRope * sizeof(T1);
        }
        DataCopyPad(gatherTensor[curGatherSize * dimDqk], keyGm[runInfo.keyGmOffset + keyOffset1 * dimN2 * dimDqk], intriParamsKey, padParams);

        if constexpr (HAS_ROPE) {
            DataCopyPad(gatherRopeTensor[curGatherSize * dimRope], keyRopeGm[runInfo.keyRopeGmOffset + keyOffset1 * dimN2 * dimRope], intriParamsRope, padParams);
        }
        curGatherSize += selectedBlockSize;
        
        SET_FLAG(MTE3, MTE2, mte2WaitMte3EventId);
    }
        
    if (curGatherSize != 0) {
        LocalTensor<T1> &gatherTensor = mergePingPong ? gatherTensorPing : gatherTensorPong;
        LocalTensor<T1> &gatherRopeTensor = mergePingPong ? gatherRopePing : gatherRopePong;
        
        WAIT_FLAG(MTE3, MTE2, mte2WaitMte3EventId);

        SET_FLAG(MTE2, MTE3, mte3WaitMte2EventId);
        WAIT_FLAG(MTE2, MTE3, mte3WaitMte2EventId);

        outParamK.blockCount = curGatherSize;
        DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + outWsOffset], gatherTensor, outParamK);

        if constexpr (HAS_ROPE) {
            outParamRope.blockCount = curGatherSize;
            DataCopyPad(selectedKWorkspaceGm[kSelectedWsAddr + dimDqk + outWsOffset], gatherRopeTensor, outParamRope);
        }

        SET_FLAG(MTE3, MTE2, mte2WaitMte3EventId);

        mergePingPong = 1 - mergePingPong;
    }
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3Pong);
    WAIT_FLAG(MTE3, MTE2, mte2WaitMte3);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::Process(const RunInfo &runInfo)
{
    int32_t loopEnd = (params.singleM + params.sftBaseM - 1) / params.sftBaseM;
    int32_t firstLoopEnd = loopEnd / 2;
    int32_t substart = subBlockIdx == 0 ? 0 : firstLoopEnd;
    int32_t subLoopEnd = subBlockIdx == 0 ? firstLoopEnd : loopEnd;

    int32_t processM = params.sftBaseM;
    int32_t tailM = params.singleM % params.sftBaseM;
    int64_t dataSize = processM * params.sftBaseN;

    int64_t dyGmOffset = runInfo.dyGmOffset + subBlockIdx * (processM * dimDv * firstLoopEnd);
    int64_t sumGmOffset = runInfo.sumGmOffset + subBlockIdx * (processM * firstLoopEnd);
    int64_t mm12Addr = runInfo.mm12GmOffset + subBlockIdx * (dataSize * firstLoopEnd);
    int64_t mm345Addr = runInfo.mm345GmOffset + subBlockIdx * (dataSize * firstLoopEnd) ;

    for (int32_t i = substart; i < subLoopEnd; i++) {
        if (i == loopEnd - 1 && tailM != 0) {
            processM = tailM;
        }
        CalRowsumAndSftCopyIn(dyGmOffset, sumGmOffset, processM);
        dyGmOffset += (processM * dimDv);
        sumGmOffset += processM;

        CalSoftmax(i, processM, mm12Addr, mm345Addr, runInfo);
        CalSoftmaxGrad(i, processM, mm12Addr, mm345Addr, runInfo);
        mm12Addr += dataSize;
        mm345Addr += dataSize;
    }
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::ScatterAddUnDeter(const RunInfo &runInfo)
{
    LocalTensor<float> dkInUb;
    LocalTensor<float> dvInUb;

    GlobalTensor<float> dkOutGm = dkWorkspaceGm[runInfo.mm4OutGmOffset];
    GlobalTensor<float> dvOutGm = dvWorkspaceGm[runInfo.mm5OutGmOffset];
    int64_t s2RealSize = Min(selectedBlockCount, runInfo.actualSelectedBlockCount);
    int64_t firstCoreKSize = s2RealSize / 2;
    int64_t currentCoreKSize = subBlockIdx == 0 ? firstCoreKSize : s2RealSize - firstCoreKSize;

    if (currentCoreKSize == 0) {
        return;
    }
    SetAtomicAdd<float>();

    int64_t actTotalRows = subBlockIdx == 1 ? 
                           (currentCoreKSize - 1) * selectedBlockSize + runInfo.lastBlockSize : 
                           currentCoreKSize * selectedBlockSize;
    int64_t maxLoops = CeilDiv(actTotalRows, UB_ROW_SIZE);
    int64_t tailRows = actTotalRows - (maxLoops - 1) * UB_ROW_SIZE;

    int64_t currentDkSrcOffset = runInfo.scatterTaskId * MAX_CORE_NUM * selectedBlockCount * selectedBlockSizeDimDAlign + cubeBlockIdx * selectedBlockCount * selectedBlockSizeDimDAlign + subBlockIdx * firstCoreKSize * selectedBlockSizeDimDAlign;
    int64_t currentDvSrcOffset = runInfo.scatterTaskId * MAX_CORE_NUM * selectedBlockCount * selectedBlockSizeDimD2Align + cubeBlockIdx * selectedBlockCount * selectedBlockSizeDimD2Align + subBlockIdx * firstCoreKSize * selectedBlockSizeDimD2Align;
    int64_t currentIndicesOffset = runInfo.indicesGmOffset + subBlockIdx * firstCoreKSize;
    GlobalTensor<float> dkSrcGm = mm4ResWorkspaceGm[currentDkSrcOffset];
    GlobalTensor<float> dvSrcGm = mm5ResWorkspaceGm[currentDvSrcOffset];
    GlobalTensor<int32_t> indicesGm = topkIndicesGm[currentIndicesOffset];

    int64_t curSelBlk = 0;
    int64_t curRow = 0;
    int32_t s2Idx = 0;
    if (!runInfo.isSmallS2) {
        s2Idx = indicesGm.GetValue(curSelBlk);
    }
    int64_t curProcessRow = Min(UB_ROW_SIZE, selectedBlockSize);

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);
    for (int64_t loop = 0; loop < maxLoops - 1; loop++) {
        event_t backEvent = pingPongIdx == 0 ? mte2WaitMte3: mte2WaitMte3Pong;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
        dkInUb = scatterAddTensorK[pingPongIdx * ubRowSizeDAlign];
        dvInUb = scatterAddTensorV[pingPongIdx * ubRowSizeD2Align];
        DataCopy(dkInUb, dkSrcGm[loop * ubRowSizeDAlign], ubRowSizeDAlign);
        event_t event = pingPongIdx == 0 ? vWaitMte2: vWaitMte2Pong;
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        Muls(dkInUb, dkInUb, (float)tilingData->postTilingData.scaleValue, ubRowSizeDAlign);
        DataCopy(dvInUb, dvSrcGm[loop * ubRowSizeD2Align], ubRowSizeD2Align);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

        if (runInfo.isSmallS2) {
            DataCopy(dkOutGm[(loop * UB_ROW_SIZE + subBlockIdx * firstCoreKSize * selectedBlockSize) * dimDAlign], dkInUb, ubRowSizeDAlign);
            DataCopy(dvOutGm[(loop * UB_ROW_SIZE + subBlockIdx * firstCoreKSize * selectedBlockSize) * dimD2Align], dvInUb, ubRowSizeD2Align);
        } else {
            for (int64_t row = 0; row < UB_ROW_SIZE;) {
                if (curRow / selectedBlockSize > curSelBlk) {
                    curSelBlk += 1;
                    s2Idx = indicesGm.GetValue(curSelBlk);
                }
                if (s2Idx >= 0) {
                    DataCopy(dkOutGm[s2Idx * selectedBlockSize * dimDAlign + (curRow % selectedBlockSize) * dimDAlign], dkInUb[row * dimDAlign], curProcessRow * dimDAlign);
                    DataCopy(dvOutGm[s2Idx * selectedBlockSize * dimD2Align + (curRow % selectedBlockSize) * dimD2Align], dvInUb[row * dimD2Align], curProcessRow * dimD2Align);
                }
                row += curProcessRow;
                curRow += curProcessRow;
            }
        }

        SetFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
        pingPongIdx = 1 - pingPongIdx;
    }
    event_t backEvent = pingPongIdx == 0 ? mte2WaitMte3: mte2WaitMte3Pong;
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
    dkInUb = scatterAddTensorK[pingPongIdx * ubRowSizeDAlign];
    dvInUb = scatterAddTensorV[pingPongIdx * ubRowSizeD2Align];
    DataCopy(dkInUb, dkSrcGm[(maxLoops - 1) * ubRowSizeDAlign], tailRows * dimDAlign);
    event_t event = pingPongIdx == 0 ? vWaitMte2: vWaitMte2Pong;
    SetFlag<AscendC::HardEvent::MTE2_V>(event);
    WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    Muls(dkInUb, dkInUb, (float)tilingData->postTilingData.scaleValue, tailRows * dimDAlign);
    DataCopy(dvInUb, dvSrcGm[(maxLoops - 1) * ubRowSizeD2Align], tailRows * dimD2Align);
    SetFlag<AscendC::HardEvent::MTE2_V>(event);
    WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
    WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

    int64_t totalRound = CeilDiv(tailRows, curProcessRow);
    int64_t row = 0;
    if (runInfo.isSmallS2 && tailRows != 0) {
        DataCopy(dkOutGm[((maxLoops - 1) * UB_ROW_SIZE + subBlockIdx * firstCoreKSize * selectedBlockSize) * dimDAlign], dkInUb, tailRows * dimDAlign);
        DataCopy(dvOutGm[((maxLoops - 1) * UB_ROW_SIZE + subBlockIdx * firstCoreKSize * selectedBlockSize) * dimD2Align], dvInUb, tailRows * dimD2Align);
    } else {
        for (int64_t loop = 0; loop < totalRound; loop++) {
            if (curRow / selectedBlockSize > curSelBlk) {
                curSelBlk += 1;
                s2Idx = indicesGm.GetValue(curSelBlk);
            }
            if (s2Idx >= 0) {
                if (subBlockIdx == 1 && loop == totalRound - 1) {
                    curProcessRow = (runInfo.lastBlockSize % curProcessRow) ? 
                                    runInfo.lastBlockSize % curProcessRow : 
                                    curProcessRow;
                }
                DataCopy(dkOutGm[s2Idx * selectedBlockSize * dimDAlign + (curRow % selectedBlockSize) * dimDAlign], dkInUb[row * dimDAlign], curProcessRow * dimDAlign);
                DataCopy(dvOutGm[s2Idx * selectedBlockSize * dimD2Align + (curRow % selectedBlockSize) * dimD2Align], dvInUb[row * dimD2Align], curProcessRow * dimD2Align);
            }
            row += curProcessRow;
            curRow += curProcessRow;
        }
    }
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
    pingPongIdx = 1 - pingPongIdx;
    SetAtomicNone();

    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);

    SetFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
    WaitFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::ScatterAddDeter(const RunInfo &runInfo)
{
    LocalTensor<float> dkInUb;
    LocalTensor<float> dvInUb;

    GlobalTensor<float> dkOutGm = dkWorkspaceGm[runInfo.mm4OutGmOffset];
    GlobalTensor<float> dvOutGm = dvWorkspaceGm[runInfo.mm5OutGmOffset];
    int64_t s2RealSize = Min(selectedBlockCount, runInfo.actualSelectedBlockCount);

    int64_t totalVec = (runInfo.s1End - runInfo.s1Begin) * 2;

    int64_t firstCoreKSize = s2RealSize / totalVec;
    int64_t currentCoreKSize = (vecBlockIdx == totalVec - 1) ?
        (s2RealSize % totalVec + firstCoreKSize) : firstCoreKSize;

    if (currentCoreKSize == 0) {
        return;
    }
    SetAtomicAdd<float>();
    int64_t cubeBlockIdxCalculate = runInfo.s1Index - runInfo.s1Begin;
    int64_t actTotalRows = (vecBlockIdx == totalVec - 1) ? 
                           (currentCoreKSize - 1) * selectedBlockSize + runInfo.lastBlockSize : 
                           currentCoreKSize * selectedBlockSize;
    int64_t maxLoops = CeilDiv(actTotalRows, UB_ROW_SIZE);
    int64_t tailRows = actTotalRows - (maxLoops - 1) * UB_ROW_SIZE;

    int64_t currentDkSrcOffset = 
        runInfo.scatterTaskId * MAX_CORE_NUM * selectedBlockCount * selectedBlockSizeDimDAlign +
        cubeBlockIdxCalculate * selectedBlockCount * selectedBlockSizeDimDAlign +
        vecBlockIdx * firstCoreKSize * selectedBlockSizeDimDAlign;
    int64_t currentDvSrcOffset = 
        runInfo.scatterTaskId * MAX_CORE_NUM * selectedBlockCount * selectedBlockSizeDimD2Align +
        cubeBlockIdxCalculate * selectedBlockCount * selectedBlockSizeDimD2Align +
        vecBlockIdx * firstCoreKSize * selectedBlockSizeDimD2Align;
    int64_t currentIndicesOffset = runInfo.indicesGmOffset + vecBlockIdx * firstCoreKSize;
    GlobalTensor<float> dkSrcGm = mm4ResWorkspaceGm[currentDkSrcOffset];
    GlobalTensor<float> dvSrcGm = mm5ResWorkspaceGm[currentDvSrcOffset];
    GlobalTensor<int32_t> indicesGm = topkIndicesGm[currentIndicesOffset];

    int64_t curSelBlk = 0;
    int64_t curRow = 0;
    int32_t s2Idx = 0;
    if (!runInfo.isSmallS2) {
        s2Idx = indicesGm.GetValue(curSelBlk);
    }
    int64_t curProcessRow = Min(UB_ROW_SIZE, selectedBlockSize);

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);
    for (int64_t loop = 0; loop < maxLoops - 1; loop++) {
        event_t backEvent = pingPongIdx == 0 ? mte2WaitMte3: mte2WaitMte3Pong;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
        dkInUb = scatterAddTensorK[pingPongIdx * ubRowSizeDAlign];
        dvInUb = scatterAddTensorV[pingPongIdx * ubRowSizeD2Align];
        DataCopy(dkInUb, dkSrcGm[loop * ubRowSizeDAlign], ubRowSizeDAlign);
        event_t event = pingPongIdx == 0 ? vWaitMte2: vWaitMte2Pong;
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        Muls(dkInUb, dkInUb, (float)tilingData->postTilingData.scaleValue, ubRowSizeDAlign);
        DataCopy(dvInUb, dvSrcGm[loop * ubRowSizeD2Align], ubRowSizeD2Align);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

        if (runInfo.isSmallS2) {
            DataCopy(dkOutGm[(loop * UB_ROW_SIZE + vecBlockIdx * firstCoreKSize * selectedBlockSize) * dimDAlign], dkInUb, ubRowSizeDAlign);
            DataCopy(dvOutGm[(loop * UB_ROW_SIZE + vecBlockIdx * firstCoreKSize * selectedBlockSize) * dimD2Align], dvInUb, ubRowSizeD2Align);
        } else {
            for (int64_t row = 0; row < UB_ROW_SIZE;) {
                if (curRow / selectedBlockSize > curSelBlk) {
                    curSelBlk += 1;
                    s2Idx = indicesGm.GetValue(curSelBlk);
                }
                if (s2Idx >= 0) {
                    DataCopy(dkOutGm[s2Idx * selectedBlockSize * dimDAlign + (curRow % selectedBlockSize) * dimDAlign], dkInUb[row * dimDAlign], curProcessRow * dimDAlign);
                    DataCopy(dvOutGm[s2Idx * selectedBlockSize * dimD2Align + (curRow % selectedBlockSize) * dimD2Align], dvInUb[row * dimD2Align], curProcessRow * dimD2Align);
                }
                row += curProcessRow;
                curRow += curProcessRow;
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
        pingPongIdx = 1 - pingPongIdx;
    }
    event_t backEvent = pingPongIdx == 0 ? mte2WaitMte3: mte2WaitMte3Pong;
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
    dkInUb = scatterAddTensorK[pingPongIdx * ubRowSizeDAlign];
    dvInUb = scatterAddTensorV[pingPongIdx * ubRowSizeD2Align];
    DataCopy(dkInUb, dkSrcGm[(maxLoops - 1) * ubRowSizeDAlign], tailRows * dimDAlign);
    event_t event = pingPongIdx == 0 ? vWaitMte2: vWaitMte2Pong;
    SetFlag<AscendC::HardEvent::MTE2_V>(event);
    WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    Muls(dkInUb, dkInUb, (float)tilingData->postTilingData.scaleValue, tailRows * dimDAlign);
    DataCopy(dvInUb, dvSrcGm[(maxLoops - 1) * ubRowSizeD2Align], tailRows * dimD2Align);
    SetFlag<AscendC::HardEvent::MTE2_V>(event);
    WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
    WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

    int64_t totalRound = CeilDiv(tailRows, curProcessRow);
    int64_t row = 0;
    if (runInfo.isSmallS2) {
        DataCopy(dkOutGm[((maxLoops - 1) * UB_ROW_SIZE + vecBlockIdx * firstCoreKSize * selectedBlockSize) * dimDAlign], dkInUb, tailRows * dimDAlign);
        DataCopy(dvOutGm[((maxLoops - 1) * UB_ROW_SIZE + vecBlockIdx * firstCoreKSize * selectedBlockSize) * dimD2Align], dvInUb, tailRows * dimD2Align);
    } else {
        for (int64_t loop = 0; loop < totalRound; loop++) {
            if (curRow / selectedBlockSize > curSelBlk) {
                curSelBlk += 1;
                s2Idx = indicesGm.GetValue(curSelBlk);
            }
            if (s2Idx >= 0) {
                if (vecBlockIdx == totalVec - 1 && loop == totalRound - 1) {
                    curProcessRow = (runInfo.lastBlockSize % curProcessRow) ? 
                                    runInfo.lastBlockSize % curProcessRow : 
                                    curProcessRow;
                }
                DataCopy(dkOutGm[s2Idx * selectedBlockSize * dimDAlign + (curRow % selectedBlockSize) * dimDAlign], dkInUb[row * dimDAlign], curProcessRow * dimDAlign);
                DataCopy(dvOutGm[s2Idx * selectedBlockSize * dimD2Align + (curRow % selectedBlockSize) * dimD2Align], dvInUb[row * dimD2Align], curProcessRow * dimD2Align);
            }
            row += curProcessRow;
            curRow += curProcessRow;
        }
    }
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(backEvent);
    pingPongIdx = 1 - pingPongIdx;
    SetAtomicNone();

    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);

    SetFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
    WaitFlag<AscendC::HardEvent::MTE3_V>(vWaitMte3);
}

template <typename SFAGT>
__aicore__ inline void VecOp<SFAGT>::ScatterAdd(const RunInfo &runInfo)
{   
    if (deterministic) {
        ScatterAddDeter(runInfo);
    } else {
        ScatterAddUnDeter(runInfo);
    }
}

} // namespace SFAG_BASIC
