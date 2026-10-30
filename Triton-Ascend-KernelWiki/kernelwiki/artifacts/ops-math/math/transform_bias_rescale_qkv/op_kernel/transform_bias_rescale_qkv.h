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
 * \file transform_bias_rescale_qkv.h
 * \brief transform_bias_rescale_qkv head file
 */
#ifndef TRANSFORM_BIAS_RESCALE_QKV_H
#define TRANSFORM_BIAS_RESCALE_QKV_H

#include <type_traits>
#include "kernel_operator.h"

namespace TransformBiasRescaleQkv {

using namespace AscendC;

constexpr int32_t MAX_UB_SIZE = 192 * 1024;
constexpr int32_t ONE_REPEAT_ELE_NUM_FP32 = 64;
constexpr int32_t NUM_THREE = 3;
constexpr int32_t MAX_BLOCK_CNT = 4095;
// 32字节对齐
constexpr int32_t DATA_BLOCK = 32;

template <typename T>
class TransformBiasRescaleQkvND
{
public:
    TPipe pipe;
    __aicore__ inline TransformBiasRescaleQkvND(){};
    __aicore__ inline void Init(
        GM_ADDR qkv, GM_ADDR qkv_bias, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR workspace,
        const TransformBiasRescaleQkvTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void TilingLine();
    __aicore__ inline void ProcessFullLines(int64_t linesOneCoreOnce, int64_t fullLineOnUbCnt);
    __aicore__ inline void ProcessFullHeads(int64_t headsOneCoreOnce, int64_t fullHeadOnUbCnt);
    __aicore__ inline void ProcessPartHead();
    __aicore__ inline void CopyInQkv(int64_t qkvOffset, int64_t dataCount, int64_t qkvUbOffest, int64_t headCnt);
    __aicore__ inline void CopyInBias(
        int64_t qkvBiasOffset, int64_t dataCount, int64_t qkvBiasUbOffest, int64_t headCnt);
    __aicore__ inline void CastInQkv(int64_t dataCount);
    __aicore__ inline void CastInBias(int64_t dataCount);
    __aicore__ inline void ComputeBias(int64_t dataCount, int64_t qkvOffset = 0, int64_t biasOffset = 0);
    __aicore__ inline void ComputeRescale(int64_t qkvGmOffset, int64_t dataCount, int64_t qkvUbOffset);
    __aicore__ inline void CastOut(int64_t dataCount);
    __aicore__ inline void CopyOut(int64_t qkvGmOffset, int64_t headCnt, int64_t qkvUbOffset, int64_t dataCount);

private:
    TBuf<QuePosition::VECCALC> ubTBuf;
    LocalTensor<uint8_t> tmpTensor;

    LocalTensor<uint8_t> selMaskOne;
    LocalTensor<uint8_t> selMaskTwo;

    LocalTensor<T> xTmp;
    LocalTensor<T> xTensor;
    LocalTensor<float> xTensorFp32;

    LocalTensor<T> biasTmp;
    LocalTensor<T> biasTensor;
    LocalTensor<float> biasTensorFp32;

    GlobalTensor<T> qkvGm;
    GlobalTensor<T> qkvBiasGm;
    GlobalTensor<T> qOutputGm;
    GlobalTensor<T> kOutputGm;
    GlobalTensor<T> vOutputGm;

    int64_t qkvShapeSize;
    uint32_t needCoreNumber;
    int64_t batch;
    int64_t token;
    int64_t dim;
    int64_t numHeads;
    int64_t dimPerHead;

    int64_t inBatchStride;
    int64_t inTokenStride;

    int64_t outBatchStride;
    int64_t outHeadStride;

    int64_t blockIdx;
    int64_t maxEleNumUB;

    int64_t totalLines;
    int64_t linesNum;
    int64_t eachCoreStartLines;
    int64_t sizeOfType;

    float scale = 1.0;
    int64_t rPadPerHead = 0;

    event_t eventId = EVENT_ID0;
    int32_t pingPongFlag = 0;
};

template <typename T>
__aicore__ inline void TransformBiasRescaleQkvND<T>::Init(
    GM_ADDR qkv, GM_ADDR qkv_bias, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR workspace,
    const TransformBiasRescaleQkvTilingData* tilingData)
{
    qkvGm.SetGlobalBuffer((__gm__ T*)qkv);
    qkvBiasGm.SetGlobalBuffer((__gm__ T*)qkv_bias);
    qOutputGm.SetGlobalBuffer((__gm__ T*)q);
    kOutputGm.SetGlobalBuffer((__gm__ T*)k);
    vOutputGm.SetGlobalBuffer((__gm__ T*)v);

    qkvShapeSize = tilingData->qkvShapeSize;
    needCoreNumber = tilingData->needCoreNum;
    batch = tilingData->batch;
    token = tilingData->token;
    dim = tilingData->dimension;
    numHeads = tilingData->numHeads;
    dimPerHead = tilingData->dimPerHead;
    maxEleNumUB = tilingData->maxEleNumUB;

    inTokenStride = NUM_THREE * dim;
    inBatchStride = inTokenStride * token;

    outHeadStride = token * dimPerHead;
    outBatchStride = outHeadStride * numHeads;

    blockIdx = GetBlockIdx();
    pipe.InitBuffer(ubTBuf, MAX_UB_SIZE);
    tmpTensor = ubTBuf.Get<uint8_t>();

    sizeOfType = static_cast<int64_t>(sizeof(T));
    scale = static_cast<float>(1.0) / sqrt(static_cast<float>(dimPerHead));
    rPadPerHead = (DATA_BLOCK - dimPerHead * sizeOfType % DATA_BLOCK) % DATA_BLOCK;
}

template <typename T>
__aicore__ inline void TransformBiasRescaleQkvND<T>::Process()
{
    if (blockIdx >= needCoreNumber) {
        return;
    }

    // 按行分核
    TilingLine();

    // 为了能32位对齐，每个头需要右填充的大小
    int64_t rPadCnt = rPadPerHead / sizeOfType;

    // 填充后每头和每行的大小
    int64_t fullHeadOnUbCnt = dimPerHead + rPadCnt;
    int64_t fullLineOnUbCnt = numHeads * fullHeadOnUbCnt;

    int64_t linesOneCoreOnce = 1;
    int64_t headsOneCoreOnce = 1;

    if (NUM_THREE * fullLineOnUbCnt <= maxEleNumUB && numHeads <= MAX_BLOCK_CNT) {
        // 每个核能处理完整行
        linesOneCoreOnce = maxEleNumUB / fullLineOnUbCnt;
        ProcessFullLines(linesOneCoreOnce, fullLineOnUbCnt);
    } else if (fullHeadOnUbCnt <= maxEleNumUB) {
        // 每个核能处理完整头
        headsOneCoreOnce = maxEleNumUB / fullHeadOnUbCnt;
        headsOneCoreOnce = headsOneCoreOnce > MAX_BLOCK_CNT ? MAX_BLOCK_CNT : headsOneCoreOnce;
        ProcessFullHeads(headsOneCoreOnce, fullHeadOnUbCnt);
    } else {
        // 每个核要分多次处理一个头
        ProcessPartHead();
    }
}

template <typename T>
__aicore__ inline void TransformBiasRescaleQkvND<T>::TilingLine()
{
    totalLines = NUM_THREE * batch * token;
    linesNum = totalLines / needCoreNumber;

    int64_t linesRemain = totalLines % needCoreNumber;
    if (linesRemain > 0 && blockIdx < linesRemain) {
        linesNum++;
    }

    eachCoreStartLines = linesNum * blockIdx;
    if (linesRemain > 0) {
        if (blockIdx >= linesRemain) {
            eachCoreStartLines += linesRemain;
        }
    }
}

template <typename T>
__aicore__ inline void TransformBiasRescaleQkvND<T>::ProcessPartHead()
{
    pingPongFlag = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int64_t i = 0; i < linesNum; i++) {
        int64_t lineNum = eachCoreStartLines + i;
        // 处理第linNum行，每次处理headsOneCoreOnce头
        for (int64_t j = 0; j < numHeads; j++) {
            int64_t headNum = j;
            for (int64_t k = 0; k < dimPerHead; k += maxEleNumUB) {
                eventId = pingPongFlag ? EVENT_ID1 : EVENT_ID0;
                WaitFlag<HardEvent::MTE3_MTE2>(eventId);

                int64_t startOffset = k;
                int64_t endOffset = k + maxEleNumUB - 1;
                if (endOffset > dimPerHead - 1) {
                    endOffset = dimPerHead - 1;
                }
                int64_t calNum = endOffset - startOffset + 1;

                int64_t qkvBiasGmOffset =
                    (lineNum / (batch * token)) * (dimPerHead * numHeads) + headNum * dimPerHead + k;
                int64_t qkvGmOffset = (lineNum % (batch * token)) * inTokenStride + qkvBiasGmOffset;

                CopyInQkv(qkvGmOffset, calNum, 0, 0);

                CopyInBias(qkvBiasGmOffset, calNum, 0, 0);

                SetFlag<HardEvent::MTE2_V>(eventId);
                WaitFlag<HardEvent::MTE2_V>(eventId);

                CastInQkv(calNum);

                CastInBias(calNum);

                ComputeBias(calNum);

                ComputeRescale(qkvGmOffset, calNum, 0);

                CastOut(calNum);

                SetFlag<HardEvent::V_MTE3>(eventId);
                WaitFlag<HardEvent::V_MTE3>(eventId);

                CopyOut(qkvGmOffset, 0, 0, calNum);

                SetFlag<HardEvent::MTE3_MTE2>(eventId);
                pingPongFlag = 1 - pingPongFlag;
            }
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename T>
__aicore__ inline void TransformBiasRescaleQkvND<T>::ProcessFullHeads(int64_t headsOneCoreOnce, int64_t fullHeadOnUbCnt)
{
    pingPongFlag = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int64_t i = 0; i < linesNum; i++) {
        int64_t lineNum = eachCoreStartLines + i;
        // 处理第linNum行，每次处理headsOneCoreOnce头
        for (int64_t j = 0; j < numHeads; j += headsOneCoreOnce) {
            eventId = pingPongFlag ? EVENT_ID1 : EVENT_ID0;
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);

            int64_t startHeadNum = j;
            int64_t endHeadNum = startHeadNum + headsOneCoreOnce - 1;
            if (endHeadNum > numHeads - 1) {
                endHeadNum = numHeads - 1;
            }
            int64_t headCnt = endHeadNum - startHeadNum + 1;
            int64_t calNum = dimPerHead * headCnt;

            int64_t qkvBiasGmOffset = (lineNum / (batch * token)) * (dimPerHead * numHeads) + startHeadNum * dimPerHead;
            int64_t qkvGmOffset = (lineNum % (batch * token)) * inTokenStride + qkvBiasGmOffset;

            CopyInQkv(qkvGmOffset, calNum, 0, headCnt);

            CopyInBias(qkvBiasGmOffset, calNum, 0, headCnt);

            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);

            CastInQkv(fullHeadOnUbCnt * headCnt);

            CastInBias(fullHeadOnUbCnt * headCnt);

            ComputeBias(fullHeadOnUbCnt * headCnt);

            ComputeRescale(qkvGmOffset, fullHeadOnUbCnt * headCnt, 0);

            CastOut(fullHeadOnUbCnt * headCnt);

            SetFlag<HardEvent::V_MTE3>(eventId);
            WaitFlag<HardEvent::V_MTE3>(eventId);

            CopyOut(qkvGmOffset, headCnt, 0, calNum);

            SetFlag<HardEvent::MTE3_MTE2>(eventId);
            pingPongFlag = 1 - pingPongFlag;
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename T>
__aicore__ inline void TransformBiasRescaleQkvND<T>::ProcessFullLines(int64_t linesOneCoreOnce, int64_t fullLineOnUbCnt)
{
    int64_t calNum = dimPerHead * numHeads;
    pingPongFlag = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    // 处理eachCoreStartLines ~ eachCoreStartLines + linesNum -1 行

    // 先把qkvBias处理好，避免重复搬运
    CopyInBias(0, NUM_THREE * calNum, 0, NUM_THREE * numHeads);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    CastInBias(fullLineOnUbCnt * NUM_THREE);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

    for (int64_t i = 0; i < linesNum; i += linesOneCoreOnce) {
        eventId = pingPongFlag ? EVENT_ID1 : EVENT_ID0;
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);

        // 开始处理linesOneCoreOnce行
        int64_t startLineNum = eachCoreStartLines + i;
        int64_t endLineNum = startLineNum + linesOneCoreOnce - 1;
        if (endLineNum > eachCoreStartLines + linesNum - 1) {
            endLineNum = eachCoreStartLines + linesNum - 1;
        }

        int64_t qkvUbOffest = 0;
        for (int64_t lineNum = startLineNum; lineNum <= endLineNum;
             lineNum++, qkvUbOffest += fullLineOnUbCnt * sizeOfType) {
            int64_t qkvGmOffset =
                (lineNum % (batch * token)) * inTokenStride + (lineNum / (batch * token)) * (dimPerHead * numHeads);
            CopyInQkv(qkvGmOffset, calNum, qkvUbOffest, numHeads);
        }

        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        CastInQkv(fullLineOnUbCnt * (endLineNum - startLineNum + 1));

        for (int64_t lineNum = startLineNum; lineNum <= endLineNum; lineNum++) {
            int64_t qkvOffset = (lineNum - startLineNum) * fullLineOnUbCnt;
            int64_t biasOffset = (lineNum / (batch * token)) * fullLineOnUbCnt;
            ComputeBias(fullLineOnUbCnt, qkvOffset, biasOffset);
        }

        qkvUbOffest = 0;
        for (int64_t lineNum = startLineNum; lineNum <= endLineNum; lineNum++, qkvUbOffest += fullLineOnUbCnt) {
            int64_t qkvGmOffset =
                (lineNum % (batch * token)) * inTokenStride + (lineNum / (batch * token)) * (dimPerHead * numHeads);
            ComputeRescale(qkvGmOffset, fullLineOnUbCnt, qkvUbOffest);
        }

        CastOut(fullLineOnUbCnt * (endLineNum - startLineNum + 1));

        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        qkvUbOffest = 0;
        for (int64_t lineNum = startLineNum; lineNum <= endLineNum; lineNum++, qkvUbOffest += fullLineOnUbCnt) {
            int64_t qkvGmOffset =
                (lineNum % (batch * token)) * inTokenStride + (lineNum / (batch * token)) * (dimPerHead * numHeads);
            CopyOut(qkvGmOffset, numHeads, qkvUbOffest, calNum);
        }

        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        pingPongFlag = 1 - pingPongFlag;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename T>
__aicore__ inline void TransformBiasRescaleQkvND<T>::CopyInQkv(
    int64_t inputOffset, int64_t dataCount, int64_t qkvUbOffest, int64_t headCnt)
{
    xTensor = pingPongFlag ? tmpTensor[(MAX_UB_SIZE / 4) * 3 + qkvUbOffest].ReinterpretCast<T>() :
                             tmpTensor[MAX_UB_SIZE / 4 + qkvUbOffest].ReinterpretCast<T>();

    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    if (rPadPerHead != 0 && headCnt > 0) {
        dataCopyParams = {static_cast<uint16_t>(headCnt), static_cast<uint32_t>(dimPerHead * sizeof(T)), 0, 0, 0};
        padParams = {true, 0, 0, 0};
    }

    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        int64_t elementByte = maxEleNumUB * sizeOfType;
        xTmp = pingPongFlag ? tmpTensor[elementByte + (MAX_UB_SIZE / 4) * 3 + qkvUbOffest].ReinterpretCast<T>() :
                              tmpTensor[elementByte + MAX_UB_SIZE / 4 + qkvUbOffest].ReinterpretCast<T>();
        DataCopyPad(xTmp, qkvGm[inputOffset], dataCopyParams, padParams);
    } else {
        DataCopyPad(xTensor, qkvGm[inputOffset], dataCopyParams, padParams);
    }
}

template <typename T>
__aicore__ inline void TransformBiasRescaleQkvND<T>::CastInQkv(int64_t dataCount)
{
    xTensor = pingPongFlag ? tmpTensor[(MAX_UB_SIZE / 4) * 3].ReinterpretCast<T>() :
                             tmpTensor[MAX_UB_SIZE / 4].ReinterpretCast<T>();

    xTensorFp32 = xTensor.template ReinterpretCast<float>();
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        int64_t elementByte = maxEleNumUB * sizeOfType;
        xTmp = pingPongFlag ? tmpTensor[elementByte + (MAX_UB_SIZE / 4) * 3].ReinterpretCast<T>() :
                              tmpTensor[elementByte + MAX_UB_SIZE / 4].ReinterpretCast<T>();
        Cast(xTensorFp32, xTmp, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void TransformBiasRescaleQkvND<T>::CopyInBias(
    int64_t inputOffset, int64_t dataCount, int64_t qkvBiasUbOffest, int64_t headCnt)
{
    biasTensor = pingPongFlag ? tmpTensor[MAX_UB_SIZE / 2 + qkvBiasUbOffest].ReinterpretCast<T>() :
                                tmpTensor[qkvBiasUbOffest].ReinterpretCast<T>();

    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    if (rPadPerHead != 0 && headCnt > 0) {
        dataCopyParams = {static_cast<uint16_t>(headCnt), static_cast<uint32_t>(dimPerHead * sizeof(T)), 0, 0, 0};
        padParams = {true, 0, 0, 0};
    }

    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        int64_t elementByte = maxEleNumUB * sizeOfType;
        biasTmp = pingPongFlag ? tmpTensor[elementByte + MAX_UB_SIZE / 2 + qkvBiasUbOffest].ReinterpretCast<T>() :
                                 tmpTensor[elementByte + qkvBiasUbOffest].ReinterpretCast<T>();
        DataCopyPad(biasTmp, qkvBiasGm[inputOffset], dataCopyParams, padParams);
    } else {
        DataCopyPad(biasTensor, qkvBiasGm[inputOffset], dataCopyParams, padParams);
    }
}

template <typename T>
__aicore__ inline void TransformBiasRescaleQkvND<T>::CastInBias(int64_t dataCount)
{
    biasTensor = pingPongFlag ? tmpTensor[MAX_UB_SIZE / 2].ReinterpretCast<T>() : tmpTensor[0].ReinterpretCast<T>();
    biasTensorFp32 = biasTensor.template ReinterpretCast<float>();
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        int64_t elementByte = maxEleNumUB * sizeOfType;
        biasTmp = pingPongFlag ? tmpTensor[elementByte + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                                 tmpTensor[elementByte].ReinterpretCast<T>();
        Cast(biasTensorFp32, biasTmp, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void TransformBiasRescaleQkvND<T>::ComputeBias(int64_t dataCount, int64_t qkvOffset, int64_t biasOffset) {
    Add(xTensorFp32[qkvOffset], xTensorFp32[qkvOffset], biasTensorFp32[biasOffset], dataCount);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void TransformBiasRescaleQkvND<T>::ComputeRescale(
    int64_t qkvGmOffset, int64_t dataCount, int64_t qkvUbOffset)
{
    if ((qkvGmOffset % (NUM_THREE * numHeads * dimPerHead)) < numHeads * dimPerHead) {
        Muls(xTensorFp32[qkvUbOffset], xTensorFp32[qkvUbOffset], scale, dataCount);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void TransformBiasRescaleQkvND<T>::CastOut(int64_t dataCount)
{
    if (std::is_same_v<T, half>) {
        Cast(xTensor, xTensorFp32, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    } else if (std::is_same_v<T, bfloat16_t>) {
        Cast(xTensor, xTensorFp32, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void TransformBiasRescaleQkvND<T>::CopyOut(
    int64_t inputOffset, int64_t headCnt, int64_t qkvUbOffset, int64_t dataCount)
{
    int64_t b = inputOffset / inBatchStride;
    int64_t t = (inputOffset % inBatchStride) / inTokenStride;
    int64_t n = (inputOffset % (numHeads * dimPerHead)) / dimPerHead;
    int64_t d = ((inputOffset % inBatchStride) % inTokenStride) % dimPerHead;
    int64_t outputOffset = b * outBatchStride + n * outHeadStride + t * dimPerHead + d;

    int64_t isQKV = (inputOffset % inTokenStride) / dim;
    GlobalTensor<T> outGm = qOutputGm;
    if (isQKV == 1) {
        outGm = kOutputGm;
    } else if (isQKV == 2) {
        outGm = vOutputGm;
    }

    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if (headCnt > 0) {
        dataCopyParams = {
            static_cast<uint16_t>(headCnt), static_cast<uint32_t>(dimPerHead * sizeof(T)), 0,
            static_cast<uint32_t>((token - 1) * dimPerHead * sizeof(T)), 0};
    }

    DataCopyPad(outGm[outputOffset], xTensor[qkvUbOffset], dataCopyParams);
}

} // namespace TransformBiasRescaleQkv
#endif