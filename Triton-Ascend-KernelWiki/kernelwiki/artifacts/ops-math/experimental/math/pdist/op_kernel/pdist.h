/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PDIST_H
#define PDIST_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "pdist_tiling_data.h"
#include "pdist_tiling_key.h"
#include "pdist_constants.h"

namespace NsPdist {

template <typename T>
class Pdist {
public:
    __aicore__ inline Pdist() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                 const PdistTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void GetIJFromIndex(uint32_t idx, uint32_t& rowI, uint32_t& rowJ);
    __aicore__ inline void ProcessBlock(uint64_t startIdx, uint64_t blockSize);
    __aicore__ inline void LoadDiff(uint32_t rowI, uint32_t rowJ, uint32_t colOffset,
                                     uint32_t length, uint32_t alignLen,
                                     AscendC::LocalTensor<float>& src1, AscendC::LocalTensor<float>& src2);
    __aicore__ inline void ComputeChunkDistance(AscendC::LocalTensor<float>& src1, AscendC::LocalTensor<float>& src2,
                                                uint32_t length,
                                                AscendC::LocalTensor<float>& workLocal, AscendC::LocalTensor<float>& tempLocal);
    __aicore__ inline void ApplyInvP(AscendC::LocalTensor<float>& outLocal, uint32_t blockSize);
    __aicore__ inline void WriteOutput(AscendC::LocalTensor<float>& outLocal, uint64_t startIdx, uint32_t blockSize);

    AscendC::GlobalTensor<T> inputGM;
    AscendC::GlobalTensor<T> outputGM;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueSrc1;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueSrc2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> workBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> cmpBuf;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> fp16Queue;

    uint32_t rows_ = 0;
    uint32_t cols_ = 0;
    float pValue_ = 0.0f;
    uint64_t computeNum_ = 0;
    uint32_t ubTensorEachLoop_ = 0;
    uint32_t coreNumVar_ = 0;
    uint32_t tilingKey_ = 1;
    uint32_t reduceBufSize_ = 0;
    uint64_t numBlockEachCore_ = 0;
    uint64_t lastNumsBlocks_ = 0;
    uint64_t lastNumsNoneFullBlock_ = 0;
};

template <typename T>
__aicore__ inline void Pdist<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const PdistTilingData* tilingData)
{
    rows_ = tilingData->rows;
    cols_ = tilingData->cols;
    pValue_ = tilingData->pValue;
    computeNum_ = tilingData->computeNum;
    ubTensorEachLoop_ = tilingData->ubTensorEachLoop;
    coreNumVar_ = tilingData->coreNumVar;
    tilingKey_ = tilingData->tilingKey;
    reduceBufSize_ = tilingData->reduceBufSize;
    numBlockEachCore_ = tilingData->numBlockEachCore;
    lastNumsBlocks_ = tilingData->lastNumsBlocks;
    lastNumsNoneFullBlock_ = tilingData->lastNumsNoneFullBlock;

    int64_t totalInputElements = static_cast<int64_t>(rows_) * cols_;
    inputGM.SetGlobalBuffer((__gm__ T*)x, totalInputElements);
    outputGM.SetGlobalBuffer((__gm__ T*)y, computeNum_);

    pipe.InitBuffer(inQueueSrc1, 1, ubTensorEachLoop_ * sizeof(float));
    pipe.InitBuffer(inQueueSrc2, 1, ubTensorEachLoop_ * sizeof(float));
    pipe.InitBuffer(outQueue, 1, 8 * sizeof(float));
    pipe.InitBuffer(workBuf, reduceBufSize_ + PDIST_SUM_TENSOR_SIZE * sizeof(float));

    uint32_t cmpBufSize = ((ubTensorEachLoop_ + 63) / 64 * 64) / 8;
    cmpBufSize = ((cmpBufSize + 31) / 32) * 32;
    pipe.InitBuffer(cmpBuf, cmpBufSize);

    if constexpr (std::is_same_v<T, half>) {
        pipe.InitBuffer(fp16Queue, 1, ubTensorEachLoop_ * sizeof(half));
    }
}

template <typename T>
__aicore__ inline void Pdist<T>::GetIJFromIndex(
    uint32_t idx, uint32_t& rowI, uint32_t& rowJ)
{
    uint32_t n = rows_;
    uint32_t total = n * (n - 1) / 2;
    uint32_t revIdx = total - 1 - idx;
    uint32_t disc = 1 + 8 * revIdx;

    uint32_t y = disc;
    for (uint32_t iter = 0; iter < 16; iter++) {
        y = (y + disc / y) / 2;
    }
    if (y * y > disc) {
        y = y - 1;
    }

    uint32_t i = n - 2 - (y - 1) / 2;

    uint32_t cumI = i * (2 * n - i - 1) / 2;
    if (idx < cumI) {
        i = i - 1;
        cumI = i * (2 * n - i - 1) / 2;
    } else if (idx >= cumI + (n - 1 - i)) {
        cumI = cumI + (n - 1 - i);
        i = i + 1;
    }

    rowI = i;
    rowJ = idx - cumI + i + 1;
}

template <typename T>
__aicore__ inline void Pdist<T>::Process()
{
    uint64_t dataEachBlock = 8;
    uint32_t coreId = AscendC::GetBlockIdx();

    uint64_t baseIdx = coreId * numBlockEachCore_ * dataEachBlock;

    for (uint64_t blockId = 0; blockId < numBlockEachCore_; blockId++) {
        ProcessBlock(baseIdx + blockId * dataEachBlock, dataEachBlock);
    }

    uint64_t lastBaseIdx = coreNumVar_ * numBlockEachCore_ * dataEachBlock;
    if (coreId < lastNumsBlocks_) {
        ProcessBlock(lastBaseIdx + coreId * dataEachBlock, dataEachBlock);
    }

    if (lastNumsNoneFullBlock_ > 0 && coreId == lastNumsBlocks_) {
        ProcessBlock(lastBaseIdx + lastNumsBlocks_ * dataEachBlock, lastNumsNoneFullBlock_);
    }
}

template <typename T>
__aicore__ inline void Pdist<T>::LoadDiff(
    uint32_t rowI, uint32_t rowJ, uint32_t colOffset,
    uint32_t length, uint32_t alignLen,
    AscendC::LocalTensor<float>& src1, AscendC::LocalTensor<float>& src2)
{
    int64_t gmOffI = static_cast<int64_t>(rowI) * cols_ + colOffset;
    int64_t gmOffJ = static_cast<int64_t>(rowJ) * cols_ + colOffset;

    AscendC::DataCopyParams cpParams;
    cpParams.blockCount = 1;
    cpParams.blockLen = length * sizeof(T);
    cpParams.srcStride = 0;
    cpParams.dstStride = 0;

    if constexpr (std::is_same_v<T, half>) {
        AscendC::LocalTensor<half> fp16Tmp = fp16Queue.AllocTensor<half>();
        AscendC::DataCopyPad(fp16Tmp, inputGM[gmOffI], cpParams, {false, 0, 0, 0});
        fp16Queue.EnQue(fp16Tmp);
        AscendC::LocalTensor<half> fp16In = fp16Queue.DeQue<half>();
        AscendC::Cast(src1, fp16In, AscendC::RoundMode::CAST_NONE, alignLen);
        fp16Queue.FreeTensor(fp16In);

        fp16Tmp = fp16Queue.AllocTensor<half>();
        AscendC::DataCopyPad(fp16Tmp, inputGM[gmOffJ], cpParams, {false, 0, 0, 0});
        fp16Queue.EnQue(fp16Tmp);
        fp16In = fp16Queue.DeQue<half>();
        AscendC::Cast(src2, fp16In, AscendC::RoundMode::CAST_NONE, alignLen);
        fp16Queue.FreeTensor(fp16In);
    } else {
        AscendC::DataCopyPad(src1, inputGM[gmOffI], cpParams, {false, 0, 0, 0});
        AscendC::DataCopyPad(src2, inputGM[gmOffJ], cpParams, {false, 0, 0, 0});
        inQueueSrc1.EnQue(src1);
        inQueueSrc2.EnQue(src2);
        src1 = inQueueSrc1.DeQue<float>();
        src2 = inQueueSrc2.DeQue<float>();
    }

    AscendC::Sub(src1, src1, src2, length);
}

template <typename T>
__aicore__ inline void Pdist<T>::ComputeChunkDistance(
    AscendC::LocalTensor<float>& src1, AscendC::LocalTensor<float>& src2,
    uint32_t length,
    AscendC::LocalTensor<float>& workLocal, AscendC::LocalTensor<float>& tempLocal)
{
    if (tilingKey_ == 0) {
        uint32_t align256Len = ((length + 63) / 64) * 64;
        AscendC::LocalTensor<uint8_t> cmpLocal = cmpBuf.Get<uint8_t>();
        AscendC::CompareScalar(cmpLocal, src1, 0.0f, AscendC::CMPMODE::NE, align256Len);
        AscendC::Duplicate(src2, 1.0f, align256Len);
        AscendC::Select(src1, cmpLocal, src2, 0.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, align256Len);
        AscendC::ReduceSum(tempLocal, src1, workLocal, length);
    } else if (tilingKey_ == 1) {
        AscendC::Abs(src1, src1, length);
        if (pValue_ == 1.0f) {
        } else if (pValue_ == 2.0f) {
            AscendC::Mul(src1, src1, src1, length);
        } else {
            AscendC::Ln(src1, src1, length);
            AscendC::Muls(src1, src1, pValue_, length);
            AscendC::Exp(src1, src1, length);
        }
        AscendC::ReduceSum(tempLocal, src1, workLocal, length);
    } else {
        AscendC::Abs(src1, src1, length);
        AscendC::ReduceMax(tempLocal, src1, workLocal, length, false);
    }
}

template <typename T>
__aicore__ inline void Pdist<T>::ApplyInvP(
    AscendC::LocalTensor<float>& outLocal, uint32_t blockSize)
{
    if (tilingKey_ != 1) return;
    float invP = 1.0f / pValue_;

    uint32_t tempOffset = reduceBufSize_ / sizeof(float);
    AscendC::LocalTensor<float> tmp = workBuf.Get<float>()[tempOffset];
    AscendC::LocalTensor<uint8_t> cmpLocal = cmpBuf.Get<uint8_t>();

    AscendC::CompareScalar(cmpLocal, outLocal, 0.0f, AscendC::CMPMODE::GT,
                  (uint64_t)8, (uint8_t)1, {1, 1, 8, 8});

    AscendC::Select(tmp, cmpLocal, outLocal, 1.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 8);

    if (pValue_ == 1.0f) {
    } else if (pValue_ == 2.0f) {
        AscendC::Sqrt(tmp, tmp, 8);
    } else {
        AscendC::Ln(tmp, tmp, 8);
        AscendC::Muls(tmp, tmp, invP, 8);
        AscendC::Exp(tmp, tmp, 8);
    }

    AscendC::Select(outLocal, cmpLocal, tmp, 0.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 8);
}

template <typename T>
__aicore__ inline void Pdist<T>::WriteOutput(
    AscendC::LocalTensor<float>& outLocal, uint64_t startIdx, uint32_t blockSize)
{
    outQueue.EnQue(outLocal);
    outLocal = outQueue.DeQue<float>();

    AscendC::DataCopyParams outParams;
    outParams.blockCount = 1;
    outParams.blockLen = blockSize * sizeof(T);
    outParams.srcStride = 0;
    outParams.dstStride = 0;

    if constexpr (std::is_same_v<T, half>) {
        AscendC::LocalTensor<half> outFp16 = workBuf.Get<float>().ReinterpretCast<half>();
        AscendC::Cast(outFp16, outLocal, AscendC::RoundMode::CAST_RINT, 8);
        AscendC::DataCopyPad(outputGM[startIdx], outFp16, outParams);
    } else {
        AscendC::DataCopyPad(outputGM[startIdx], outLocal, outParams);
    }

    outQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void Pdist<T>::ProcessBlock(
    uint64_t startIdx, uint64_t blockSize)
{
    AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
    AscendC::Duplicate(outLocal, 0.0f, 8);

    AscendC::LocalTensor<float> wBuf = workBuf.Get<float>();
    AscendC::LocalTensor<float> workLocal = wBuf[0];
    uint32_t tempOffset = reduceBufSize_ / sizeof(float);
    AscendC::LocalTensor<float> tempLocal = wBuf[tempOffset];

    for (uint64_t k = 0; k < blockSize; k++) {
        uint64_t globalIdx = startIdx + k;
        if (globalIdx >= computeNum_) break;

        uint32_t rowI, rowJ;
        GetIJFromIndex(static_cast<uint32_t>(globalIdx), rowI, rowJ);

        float accum = 0.0f;
        uint32_t remaining = cols_;
        uint32_t colOffset = 0;
        while (remaining > 0) {
            uint32_t length = (remaining > ubTensorEachLoop_) ?
                               ubTensorEachLoop_ : remaining;
            uint32_t alignLen = ((length + 7) / 8) * 8;

            AscendC::LocalTensor<float> src1 = inQueueSrc1.AllocTensor<float>();
            AscendC::LocalTensor<float> src2 = inQueueSrc2.AllocTensor<float>();

            LoadDiff(rowI, rowJ, colOffset, length, alignLen, src1, src2);
            ComputeChunkDistance(src1, src2, length, workLocal, tempLocal);

            float chunkVal = tempLocal.GetValue(0);
            if (tilingKey_ == 2) {
                accum = (chunkVal > accum) ? chunkVal : accum;
            } else {
                accum += chunkVal;
            }

            inQueueSrc1.FreeTensor(src1);
            inQueueSrc2.FreeTensor(src2);

            colOffset += length;
            remaining -= length;
        }

        outLocal.SetValue(k, accum);
    }

    ApplyInvP(outLocal, blockSize);
    WriteOutput(outLocal, startIdx, blockSize);
}

} // namespace NsPdist
#endif // PDIST_H
