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
 * \file stack_ball_query.h
 * \brief
 */
#ifndef _SRC_STACK_BALL_QUERY_H_
#define _SRC_STACK_BALL_QUERY_H_
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t ALIGN_32 = 32;
constexpr int32_t ALIGN_NUM = 8;
constexpr int32_t ALIGN_16 = 16;
constexpr int32_t XYZ_NUM = 3;
constexpr int32_t XYZ_GM_OFFSET = 2;

template <typename INPUT_T>
class KernelStackBallQuery
{
public:
    __aicore__ inline KernelStackBallQuery(AscendC::TPipe* p) : pipe(p) {};

    __aicore__ inline void Init(
        GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR xyz_batch_cnt, GM_ADDR center_xyz_batch_cnt, GM_ADDR idx,
        StackBallQueryTilingData tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero !");
        this->batchSize = tilingData.batchSize;
        this->totalLengthCenterXyz = tilingData.totalLengthCenterXyz;
        this->totalLengthXyz = tilingData.totalLengthXyz;
        this->totalIdxLength = tilingData.totalIdxLength;
        this->coreNum = tilingData.coreNum;
        this->centerXyzPerCore = tilingData.centerXyzPerCore;
        this->tailCenterXyzPerCore = tilingData.tailCenterXyzPerCore;
        this->maxRadius = tilingData.maxRadius * tilingData.maxRadius;
        this->sampleNum = tilingData.sampleNum;
        this->typeXyzBlockSize = ALIGN_32 / (sizeof(INPUT_T));
        this->typeIntBlockSize = ALIGN_NUM;
        this->centerXyzEachSegmentLength = this->centerXyzEachSegmentLength / BUFFER_NUM;
        this->xyzEachSegmentLength = this->xyzEachSegmentLength / BUFFER_NUM;
        this->idxEachSegmentLength = this->idxEachSegmentLength / BUFFER_NUM;

        int centerXyzEachCoreLength = Ceil(this->totalLengthCenterXyz, coreNum);
        centerXyzGm.SetGlobalBuffer(
            (__gm__ INPUT_T*)center_xyz + 3 * this->centerXyzPerCore * GetBlockIdx(), 3 * centerXyzEachCoreLength);
        xyzGm.SetGlobalBuffer((__gm__ INPUT_T*)xyz, 3 * this->totalLengthXyz);
        idxGm.SetGlobalBuffer((__gm__ int32_t*)idx, this->totalIdxLength);
        centerXyzBatchCntGm.SetGlobalBuffer((__gm__ int32_t*)center_xyz_batch_cnt, this->batchSize);
        xyzBatchCntGm.SetGlobalBuffer((__gm__ int32_t*)xyz_batch_cnt, this->batchSize);

        pipe->InitBuffer(inQueueCenterXyz, BUFFER_NUM, XYZ_NUM * this->centerXyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(inQueueX, BUFFER_NUM, this->xyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(inQueueY, BUFFER_NUM, this->xyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(inQueueZ, BUFFER_NUM, this->xyzEachSegmentLength * sizeof(INPUT_T));

        pipe->InitBuffer(resultBuf, this->idxEachSegmentLength * sizeof(int32_t));
        this->resultOut = resultBuf.Get<int32_t>();
        pipe->InitBuffer(resultAlignBuf, ALIGN_32);
        this->resultOutAlign = resultAlignBuf.Get<int32_t>();
        pipe->InitBuffer(calcBufCenterX, this->xyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(calcBufCenterY, this->xyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(calcBufCenterZ, this->xyzEachSegmentLength * sizeof(INPUT_T));

        pipe->InitBuffer(ubDstLt, ALIGN_16 * sizeof(uint16_t));
        this->ubDstLtLocal = ubDstLt.Get<uint16_t>();
        pipe->InitBuffer(ubMaxRadius, this->xyzEachSegmentLength * sizeof(INPUT_T));
        this->ubMaxRadiusLocal = ubMaxRadius.Get<INPUT_T>();
        Duplicate<INPUT_T>(ubMaxRadiusLocal, this->maxRadius, this->xyzEachSegmentLength);
        pipe->InitBuffer(ubResultLt, this->selMaxElements * sizeof(float));
        this->ubResultLtLocal = ubResultLt.Get<float>();

        pipe->InitBuffer(ubOneFloat32, this->selMaxElements * sizeof(float));
        this->ubOneFloat32Local = ubOneFloat32.Get<float>();
        Duplicate<float>(ubOneFloat32Local, 1.0, this->selMaxElements);
        pipe->InitBuffer(ubZeroFloat32, this->selMaxElements * sizeof(float));
        this->ubZeroFloat32Local = ubZeroFloat32.Get<float>();
        Duplicate<float>(ubZeroFloat32Local, 0.0, this->selMaxElements);
        pipe->InitBuffer(calcBufDistanceResult, this->xyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(calcBufCenterDistanceX, this->xyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(calcBufCenterDistanceY, this->xyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(calcBufCenterDistanceZ, this->xyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(xyzBatchValue, this->GetAlignValue(this->batchSize, ALIGN_NUM) * sizeof(int32_t));
        pipe->InitBuffer(centerXyzBatchValue, this->GetAlignValue(this->batchSize, ALIGN_NUM) * sizeof(int32_t));
        PipeBarrier<PIPE_ALL>();;
    }

    __aicore__ inline void Process()
    {
        this->CopyInBatchCnt();
        int currentCoreCenterXyz = this->centerXyzPerCore;
        if (this->tailCenterXyzPerCore != 0 && GetBlockIdx() == this->coreNum - 1) {
            currentCoreCenterXyz = this->tailCenterXyzPerCore;
        }

        int32_t centerXyzLoopCount = currentCoreCenterXyz / this->centerXyzEachSegmentLength;
        int32_t centerXyzLoopTail = currentCoreCenterXyz % this->centerXyzEachSegmentLength;

        this->offsetCenterXyzStart = this->centerXyzPerCore * GetBlockIdx();

        for (int32_t i = 0; i < centerXyzLoopCount; i++) {
            CopyInCenterXyz(i, this->centerXyzEachSegmentLength);

            this->centerXyzLocal = inQueueCenterXyz.DeQue<INPUT_T>();
            for (int j = 0; j < this->centerXyzEachSegmentLength; j++) {
                RunPerCluster(i, j);
            }
            PipeBarrier<PIPE_ALL>();;
            inQueueCenterXyz.FreeTensor(this->centerXyzLocal);
        }
        PipeBarrier<PIPE_ALL>();;

        if (centerXyzLoopTail != 0) {
            CopyInCenterXyz(centerXyzLoopCount, centerXyzLoopTail);

            this->centerXyzLocal = inQueueCenterXyz.DeQue<INPUT_T>();

            for (int j = 0; j < centerXyzLoopTail; j++) {
                RunPerCluster(centerXyzLoopCount, j);
            }
            PipeBarrier<PIPE_ALL>();;
            inQueueCenterXyz.FreeTensor(this->centerXyzLocal);
        }
        PipeBarrier<PIPE_ALL>();;
        this->SendResultToGm(true);
    }

private:
    __aicore__ inline int GetAlignValue(const uint32_t calCount, const uint32_t blockSize)
    {
        if (blockSize == 0) {
            return calCount;
        }
        uint32_t tail = calCount % blockSize;
        if (tail == 0) {
            return calCount;
        }
        uint32_t alignVal = blockSize - tail;
        return calCount + alignVal;
    }

    __aicore__ inline int Ceil(int a, int b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    }

    template <typename Type>
    __aicore__ inline void DataCopyGm2UbAlign32(
        const LocalTensor<Type>& dstLocal, const GlobalTensor<Type>& srcGlobal, const uint32_t calCount,
        const uint32_t blockSize)
    {
        uint32_t tail = calCount % blockSize;
        if (tail != 0) {
            uint32_t alignVal = blockSize - tail;
            if (g_coreType == AIC) {
                return;
            }
            DataCopy(dstLocal, srcGlobal, calCount + alignVal);
        } else {
            if (g_coreType == AIC) {
                return;
            }
            DataCopy(dstLocal, srcGlobal, calCount);
        }
    }

    __aicore__ inline void CopyInBatchCnt()
    {
        this->centerXyzBatchLocal = centerXyzBatchValue.Get<int32_t>();
        DataCopyGm2UbAlign32(this->centerXyzBatchLocal, centerXyzBatchCntGm, this->batchSize, typeIntBlockSize);

        this->xyzBatchLocal = xyzBatchValue.Get<int32_t>();
        DataCopyGm2UbAlign32(this->xyzBatchLocal, xyzBatchCntGm, this->batchSize, typeIntBlockSize);
    }

    __aicore__ inline void CopyInCenterXyz(int32_t segmentLoopIndex, int lenCenterXyzSegment)
    {
        int64_t offset = segmentLoopIndex * this->centerXyzEachSegmentLength * 3;
        this->centerXyzLocal = inQueueCenterXyz.AllocTensor<INPUT_T>();
        DataCopyGm2UbAlign32(centerXyzLocal, centerXyzGm[offset], XYZ_NUM * lenCenterXyzSegment, typeXyzBlockSize);
        inQueueCenterXyz.EnQue(centerXyzLocal);
    }

    __aicore__ inline void CopyInXyz(int offsetXyzStart, int xyzSegmentLoopIndex, int xyzSegmentLen)
    {
        xLocal = inQueueX.AllocTensor<INPUT_T>();
        yLocal = inQueueY.AllocTensor<INPUT_T>();
        zLocal = inQueueZ.AllocTensor<INPUT_T>();

        DataCopyGm2UbAlign32(
            xLocal, xyzGm[offsetXyzStart + xyzSegmentLoopIndex * this->xyzEachSegmentLength], xyzSegmentLen,
            typeXyzBlockSize);
        DataCopyGm2UbAlign32(
            yLocal, xyzGm[totalLengthXyz + offsetXyzStart + xyzSegmentLoopIndex * this->xyzEachSegmentLength],
            xyzSegmentLen, typeXyzBlockSize);
        DataCopyGm2UbAlign32(
            zLocal,
            xyzGm[XYZ_GM_OFFSET * totalLengthXyz + offsetXyzStart + xyzSegmentLoopIndex * this->xyzEachSegmentLength],
            xyzSegmentLen, typeXyzBlockSize);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        inQueueZ.EnQue(zLocal);
    }

    __aicore__ inline void SendResultToGm(bool forceSend)
    {
        PipeBarrier<PIPE_ALL>();;

        int tailLen = this->resultOffset % this->idxEachSegmentLength;
        if (forceSend or tailLen == 0) {
            int lenToSend = this->idxEachSegmentLength;
            if (tailLen != 0) {
                lenToSend = tailLen;
            }

            int sendTail = lenToSend % 8;
            int64_t gmOffset = this->sampleNum * this->offsetCenterXyzStart + this->resultOffset - lenToSend;
            if (sendTail == 0) {
                if (g_coreType == AIC) {
                    return;
                }
                DataCopy(idxGm[gmOffset], resultOut, lenToSend);
            } else {
                if (lenToSend >= ALIGN_NUM) {
                    if (lenToSend - sendTail > 0) {
                        if (g_coreType == AIC) {
                            return;
                        }
                        DataCopy(idxGm[gmOffset], resultOut, lenToSend - sendTail);
                    }
                    PipeBarrier<PIPE_ALL>();;
                    for (int k = 0; k < ALIGN_NUM; k++) {
                        this->resultOutAlign.SetValue(k, this->resultOut.GetValue(lenToSend - ALIGN_NUM + k));
                    }
                    if (g_coreType == AIC) {
                        return;
                    }
                    DataCopy(idxGm[gmOffset + lenToSend - ALIGN_NUM], resultOutAlign, ALIGN_NUM);
                } else {
                    if (g_coreType == AIC) {
                        return;
                    }
                    DataCopy(this->resultOutAlign, idxGm[gmOffset + lenToSend - ALIGN_NUM], ALIGN_NUM);
                    PipeBarrier<PIPE_ALL>();;
                    for (int k = 0; k < lenToSend; k++) {
                        this->resultOutAlign.SetValue(ALIGN_NUM - lenToSend + k, this->resultOut.GetValue(k));
                    }
                    if (g_coreType == AIC) {
                        return;
                    }
                    DataCopy(idxGm[gmOffset + lenToSend - ALIGN_NUM], this->resultOutAlign, ALIGN_NUM);
                }
            }
            PipeBarrier<PIPE_ALL>();;
        }
    }

    __aicore__ inline void SetResultAndTrySend(int currentN)
    {
        this->resultOut.SetValue(this->resultOffset % this->idxEachSegmentLength, currentN);
        this->resultOffset += 1;
        SendResultToGm(false);
    }

    __aicore__ inline void ComputeBallQueryFp16(int currentNStart, int currentSegmentLen)
    {
        int selLoopNum = Ceil(currentSegmentLen, this->selMaxElements);
        for (int selIdx = 0; selIdx < selLoopNum; selIdx++) {
            if (this->resultNum >= this->sampleNum) {
                break;
            }
            Compare(
                ubDstLtLocal, this->distanceEachSegment[selIdx * this->selMaxElements],
                this->ubMaxRadiusLocal[selIdx * this->selMaxElements], CMPMODE::LT, this->xyzEachSegmentLength);

            Select(
                ubResultLtLocal, ubDstLtLocal, ubOneFloat32Local, ubZeroFloat32Local, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                this->selMaxElements);

            for (int internalSelIdx = 0; internalSelIdx < this->selMaxElements; ++internalSelIdx) {
                auto currentCalNum = internalSelIdx + selIdx * this->selMaxElements;
                if (currentCalNum < currentSegmentLen && this->resultNum < this->sampleNum) {
                    int currentN = currentNStart + currentCalNum;
                    auto eachResult = ubResultLtLocal.GetValue(internalSelIdx);
                    if (eachResult == float(1.0)) {
                        if (this->resultNum == 0) {
                            this->firstResult = currentN;
                        }
                        this->resultNum += 1;
                        SetResultAndTrySend(currentN);
                    }
                }
            }
        }
    }

    __aicore__ inline void ComputeBallQueryFp32(int currentNStart, int currentSegmentLen)
    {
        for (int i = 0; i < currentSegmentLen; i++) {
            if (this->resultNum >= this->sampleNum) {
                break;
            }
            auto currentDistance = this->distanceEachSegment.GetValue(i);
            if (float(currentDistance) < this->maxRadius) {
                if (this->resultNum == 0) {
                    this->firstResult = i;
                }
                this->resultNum += 1;

                int currentN = currentNStart + i;
                SetResultAndTrySend(currentN);
            }
        }
    }

    __aicore__ inline void CalculateDistance()
    {
        this->xLocal = inQueueX.DeQue<INPUT_T>();
        this->yLocal = inQueueY.DeQue<INPUT_T>();
        this->zLocal = inQueueZ.DeQue<INPUT_T>();

        LocalTensor<INPUT_T> centerXList = calcBufCenterX.Get<INPUT_T>(this->xyzEachSegmentLength);
        LocalTensor<INPUT_T> centerYList = calcBufCenterY.Get<INPUT_T>(this->xyzEachSegmentLength);
        LocalTensor<INPUT_T> centerZList = calcBufCenterZ.Get<INPUT_T>(this->xyzEachSegmentLength);

        LocalTensor<INPUT_T> distanceX = calcBufCenterDistanceX.Get<INPUT_T>(this->xyzEachSegmentLength);
        LocalTensor<INPUT_T> distanceY = calcBufCenterDistanceY.Get<INPUT_T>(this->xyzEachSegmentLength);
        LocalTensor<INPUT_T> distanceZ = calcBufCenterDistanceZ.Get<INPUT_T>(this->xyzEachSegmentLength);
        distanceEachSegment = calcBufDistanceResult.Get<INPUT_T>(this->xyzEachSegmentLength);

        Duplicate(centerXList, this->centerX, this->xyzEachSegmentLength);
        Duplicate(centerYList, this->centerY, this->xyzEachSegmentLength);
        Duplicate(centerZList, this->centerZ, this->xyzEachSegmentLength);

        Sub(distanceX, centerXList, xLocal, this->xyzEachSegmentLength);
        Mul(distanceX, distanceX, distanceX, this->xyzEachSegmentLength);
        Sub(distanceY, centerYList, yLocal, this->xyzEachSegmentLength);
        Mul(distanceY, distanceY, distanceY, this->xyzEachSegmentLength);
        Sub(distanceZ, centerZList, zLocal, this->xyzEachSegmentLength);
        Mul(distanceZ, distanceZ, distanceZ, this->xyzEachSegmentLength);

        Add(distanceEachSegment, distanceX, distanceY, this->xyzEachSegmentLength);
        Add(distanceEachSegment, distanceEachSegment, distanceZ, this->xyzEachSegmentLength);
    }

    __aicore__ inline void GetXyzSliceAndCalDis(int currentBIndex)
    {
        this->resultNum = 0;
        this->firstResult = 0;

        int currentN = this->xyzBatchLocal.GetValue(currentBIndex);

        int xyzSegmentLoop = currentN / this->xyzEachSegmentLength;
        int xyzSegmentTail = currentN % this->xyzEachSegmentLength;

        int offsetXyzStart = 0;
        for (int i = 0; i < currentBIndex; i++) {
            offsetXyzStart += this->xyzBatchLocal.GetValue(i);
        }

        for (int i = 0; i < xyzSegmentLoop; i++) {
            if (this->resultNum >= this->sampleNum) {
                break;
            }
            int segmentLen = this->xyzEachSegmentLength;
            int currentNStart = i * this->xyzEachSegmentLength;

            CopyInXyz(offsetXyzStart, i, segmentLen);
            PipeBarrier<PIPE_ALL>();;
            this->CalculateDistance();
            ComputeBallQueryFp32(currentNStart, this->xyzEachSegmentLength);

            inQueueX.FreeTensor(this->xLocal);
            inQueueY.FreeTensor(this->yLocal);
            inQueueZ.FreeTensor(this->zLocal);
        }

        if (xyzSegmentTail != 0 && this->resultNum < this->sampleNum) {
            int segmentLen = xyzSegmentTail;
            int currentNStart = xyzSegmentLoop * this->xyzEachSegmentLength;

            CopyInXyz(offsetXyzStart, xyzSegmentLoop, segmentLen);
            PipeBarrier<PIPE_ALL>();;
            this->CalculateDistance();
            ComputeBallQueryFp32(currentNStart, xyzSegmentTail);
            inQueueX.FreeTensor(this->xLocal);
            inQueueY.FreeTensor(this->yLocal);
            inQueueZ.FreeTensor(this->zLocal);
        }

        if (resultNum == 0) {
            this->resultNum += 1;
            this->SetResultAndTrySend(-1);
        }

        for (int i = resultNum; i < sampleNum; i++) {
            this->SetResultAndTrySend(this->firstResult);
            PipeBarrier<PIPE_ALL>();;
        }
    }

    __aicore__ inline void RunPerCluster(int segmentLoopIndex, int clusterIndex)
    {
        this->centerX = centerXyzLocal.GetValue(XYZ_NUM * clusterIndex + 0);
        this->centerY = centerXyzLocal.GetValue(XYZ_NUM * clusterIndex + 1);
        this->centerZ = centerXyzLocal.GetValue(XYZ_NUM * clusterIndex + XYZ_GM_OFFSET);

        int currentIdx = segmentLoopIndex * this->centerXyzEachSegmentLength + clusterIndex + offsetCenterXyzStart;
        int currentBIndex = 0;
        int tmpB = 0;

        for (int i = 0; i < this->batchSize; i++) {
            tmpB += centerXyzBatchLocal.GetValue(i);
            if (tmpB > currentIdx) {
                currentBIndex = i;
                break;
            }
        }

        GetXyzSliceAndCalDis(currentBIndex);
    }

    TPipe* pipe{nullptr};
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCenterXyz;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueZ;

    LocalTensor<int32_t> centerXyzBatchLocal;
    LocalTensor<int32_t> xyzBatchLocal;
    LocalTensor<int32_t> resultOut;
    LocalTensor<int32_t> resultOutAlign;
    LocalTensor<uint16_t> ubDstLtLocal;
    LocalTensor<INPUT_T> distanceEachSegment;
    LocalTensor<INPUT_T> centerXyzLocal;
    LocalTensor<INPUT_T> xLocal, yLocal, zLocal;
    LocalTensor<INPUT_T> ubMaxRadiusLocal;
    LocalTensor<float> ubOneFloat32Local, ubZeroFloat32Local, ubResultLtLocal;

    GlobalTensor<INPUT_T> centerXyzGm, xyzGm;
    GlobalTensor<int32_t> idxGm, xyzBatchCntGm, centerXyzBatchCntGm;

    TBuf<TPosition::VECCALC> calcBufCenterX, calcBufCenterY, calcBufCenterZ, calcBufDistanceResult;
    TBuf<TPosition::VECCALC> calcBufCenterDistanceX, calcBufCenterDistanceY, calcBufCenterDistanceZ;
    TBuf<TPosition::VECCALC> ubDstLt, ubMaxRadius, ubResultLt, ubOneFloat32, ubZeroFloat32;
    TBuf<TPosition::VECCALC> resultBuf, resultAlignBuf;
    TBuf<TPosition::VECCALC> xyzBatchValue, centerXyzBatchValue;

    INPUT_T centerX, centerY, centerZ;

    int resultNum;
    int offsetCenterXyzStart;
    int resultOffset{0};
    int firstResult;

    int32_t batchSize;
    int32_t totalLengthCenterXyz;
    int32_t totalLengthXyz;
    int32_t totalIdxLength;

    int32_t coreNum;
    int32_t centerXyzPerCore;
    int32_t tailCenterXyzPerCore;

    int centerXyzEachSegmentLength = 2048;
    int xyzEachSegmentLength = 2048;
    int idxEachSegmentLength = 2048;

    float maxRadius{};
    int32_t sampleNum{};

    int typeXyzBlockSize = 8;
    int typeIntBlockSize = 8;
    int selMaxElements = 64;
};

#endif // _SRC_STACK_BALL_QUERY_H_