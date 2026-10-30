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
 * \file unfold_grad.h
 * \brief
 */

#ifndef UNFOLD_GRAD_H
#define UNFOLD_GRAD_H

#include "kernel_operator.h"

constexpr int32_t MAX_UB_SIZE = 192 * 1024;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t TRANS_BLOCK = 16;
constexpr int32_t FP32_TYPESIZE = 4;
constexpr int32_t MAX_REPEATTIME = 255;
constexpr int32_t DOUBLE = 2;
constexpr uint32_t SIZE_LEVEL_NORMAL = 0;
constexpr uint32_t SIZE_LEVEL_BIG = 1;

struct sDataCopyExtParams {
    AscendC::DataCopyExtParams paramsIn = {0, 0, 0, 0, 0};
    AscendC::DataCopyExtParams paramsOut = {0, 0, 0, 0, 0};
};

struct CopyParams {
    __aicore__ inline CopyParams(){};
    __aicore__ inline CopyParams(int64_t offset, int64_t strideLoop, AscendC::DataCopyExtParams dcParams)
        : offset(offset), strideLoop(strideLoop), dcParams(dcParams){};
    int64_t offset{0};
    int64_t strideLoop{0};
    int64_t stridePage{0};
    AscendC::DataCopyExtParams dcParams = {0, 0, 0, 0, 0};
};

struct TransDataTo5HDParams {
    __aicore__ inline TransDataTo5HDParams(){};
    __aicore__ inline TransDataTo5HDParams(
        int rowLengthSrc, int rowLengthDst, int idxH, int idxW, AscendC::TransDataTo5HDParams transDataParams)
        : rowLengthSrc(rowLengthSrc),
          rowLengthDst(rowLengthDst),
          idxH(idxH),
          idxW(idxW),
          transDataParams(transDataParams){};
    int rowLengthSrc{0};
    int rowLengthDst{0};
    int idxH{0};
    int idxW{0};
    AscendC::TransDataTo5HDParams transDataParams = {false, false, 0, 0, 0};
};

template <typename T1, typename T2, bool ISCAST = false>
class UnfoldGrad
{
public:
    __aicore__ inline UnfoldGrad(AscendC::TPipe* p) : pipe(p){};

    __aicore__ inline void Init(const UnfoldGradTilingData* tiling_data)
    {
        ParseTilingData(tiling_data);

        workspaceLen_ = outputNumPerCore;
        outputLen_ = outputNumPerCore;
        tasksOnce = tasksOnceMaxPerCore;
        T2SrcDataSize = ubSizeT2;
        T2DstDataSize = ubSizeT2;

        if (ISCAST) {
            pipe->InitBuffer(inQueueSrc, 1, ubSizeT1);
        }
        if (sizeLevel == SIZE_LEVEL_NORMAL) {
            pipe->InitBuffer(computeInQueueSrc, 1, T2SrcDataSize);
        }
        pipe->InitBuffer(computeOutQueueDst, 1, T2DstDataSize);
    }

    __aicore__ inline void CalculateOutParms(sDataCopyExtParams& params)
    {
        uint16_t blockCount = 1;
        uint32_t blockLenIn = 0;
        uint32_t blockLenOut = 0;

        params.paramsIn = {blockCount, blockLenIn, 0, 0, 0};
        params.paramsOut = {blockCount, blockLenOut, 0, 0, 0};
    }

    template <AscendC::HardEvent hardEvent>
    __aicore__ inline void PipeSync()
    {
        int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(hardEvent));
        AscendC::SetFlag<hardEvent>(eventID);
        AscendC::WaitFlag<hardEvent>(eventID);
    }

    __aicore__ inline void CopyToOutBigShapeOnePage(int64_t inPageIdx, int64_t outPageIdx, sDataCopyExtParams& params)
    {
        int64_t inOffset = inPageIdx * workspaceLen_;
        int64_t outOffset = outPageIdx * outputLen_;
        CopyParams inCopyParams(inOffset, 0, params.paramsIn);
        CopyParams outCopyParams(outOffset, 0, params.paramsOut);
        CopyGMToGMLinesOnce(outputNumPerCore, inCopyParams, outCopyParams);
    }

    __aicore__ inline void SetGMtoZero(int64_t len, int64_t dstGlobalStart)
    {
        int64_t loop = len / (T2DstDataSize / typeSizeT1);
        int64_t tail = len % (T2DstDataSize / typeSizeT1);
        int64_t zeroNum = loop > 0 ? (T2DstDataSize / typeSizeT1) : tail;
        auto zeroLocal = computeOutQueueDst.AllocTensor<T1>(); // 此时T1与T2相等
        AscendC::Duplicate<T1>(zeroLocal, 0, zeroNum);
        uint32_t blockLen = static_cast<uint32_t>(T2DstDataSize);
        AscendC::DataCopyExtParams copyParams = {1, blockLen, 0, 0, 0};
        PipeSync<AscendC::HardEvent::V_MTE3>();
        for (int i = 0; i < loop; i++) {
            AscendC::DataCopyPad(dstGlobal[dstGlobalStart + i * zeroNum], zeroLocal, copyParams);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
        if (tail > 0) {
            copyParams.blockLen = static_cast<uint32_t>(tail * typeSizeT1);
            AscendC::DataCopyPad(dstGlobal[dstGlobalStart + loop * zeroNum], zeroLocal, copyParams);
        }
        PipeSync<AscendC::HardEvent::MTE3_S>();
        computeOutQueueDst.FreeTensor(zeroLocal);
    }

    __aicore__ inline void SetMIDGMtoZero(int64_t len, int64_t dstGlobalStart)
    {
        int64_t loop = len / (T2DstDataSize / typeSizeT2);
        int64_t tail = len % (T2DstDataSize / typeSizeT2);
        int64_t zeroNum = loop > 0 ? (T2DstDataSize / typeSizeT2) : tail;
        auto zeroLocal = computeOutQueueDst.AllocTensor<T2>();
        AscendC::Duplicate<T2>(zeroLocal, 0, zeroNum);
        uint32_t blockLen = static_cast<uint32_t>(T2DstDataSize);
        AscendC::DataCopyExtParams copyParams = {1, blockLen, 0, 0, 0};
        PipeSync<AscendC::HardEvent::V_MTE3>();
        for (int i = 0; i < loop; i++) {
            AscendC::DataCopyPad(workspaceT2SumRes[dstGlobalStart + i * zeroNum], zeroLocal, copyParams);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
        if (tail > 0) {
            copyParams.blockLen = static_cast<uint32_t>(tail * typeSizeT2);
            AscendC::DataCopyPad(workspaceT2SumRes[dstGlobalStart + loop * zeroNum], zeroLocal, copyParams);
        }
        PipeSync<AscendC::HardEvent::MTE3_S>();
        computeOutQueueDst.FreeTensor(zeroLocal);
    }

    __aicore__ inline void CopyGMToGMLinesOnce(int64_t len, CopyParams& inCopyParams, CopyParams& outCopyParams)
    {
        uint32_t rowsNum = ubSizeT1 / typeSizeT1;
        int64_t loop = len / rowsNum;
        int64_t tail = len % rowsNum;
        inCopyParams.dcParams.blockLen = static_cast<uint32_t>(rowsNum * typeSizeT2);
        outCopyParams.dcParams.blockLen = static_cast<uint32_t>(rowsNum * typeSizeT1);
        inCopyParams.strideLoop = rowsNum;
        outCopyParams.strideLoop = rowsNum;
        for (int i = 0; i < loop; i++) {
            CopyGMToGMOnceCastFromFP32(workspaceT2SumRes, dstGlobal, inCopyParams, outCopyParams);
            inCopyParams.offset += inCopyParams.strideLoop;
            outCopyParams.offset += outCopyParams.strideLoop;
        }
        inCopyParams.dcParams.blockLen = static_cast<uint32_t>(tail * typeSizeT2);
        outCopyParams.dcParams.blockLen = static_cast<uint32_t>(tail * typeSizeT1);
        if (tail > 0) {
            CopyGMToGMOnceCastFromFP32(workspaceT2SumRes, dstGlobal, inCopyParams, outCopyParams);
        }
    }

    __aicore__ inline void CopyGMToGMOnceCastFromFP32(
        AscendC::GlobalTensor<T2> srcGM, AscendC::GlobalTensor<T1> dstGM, CopyParams& inCopyParams,
        CopyParams& outCopyParams)
    {
        AscendC::DataCopyPadExtParams<T2> padParmsT2{false, 0, 0, 0};
        auto inLocalT2 = computeOutQueueDst.AllocTensor<T2>();
        PipeSync<AscendC::HardEvent::S_MTE2>();
        PipeSync<AscendC::HardEvent::V_MTE2>();
        PipeSync<AscendC::HardEvent::MTE3_MTE2>();
        AscendC::DataCopyPad(inLocalT2, srcGM[inCopyParams.offset], inCopyParams.dcParams, padParmsT2);
        PipeSync<AscendC::HardEvent::MTE2_S>();
        computeOutQueueDst.EnQue(inLocalT2);

        inLocalT2 = computeOutQueueDst.DeQue<T2>();
        auto inLocal = inQueueSrc.AllocTensor<T1>();
        PipeSync<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(inLocal, inLocalT2, AscendC::RoundMode::CAST_RINT, inCopyParams.dcParams.blockLen / typeSizeT2);
        inQueueSrc.EnQue(inLocal);
        computeOutQueueDst.FreeTensor(inLocalT2);

        inLocal = inQueueSrc.DeQue<T1>();
        PipeSync<AscendC::HardEvent::V_MTE3>();
        PipeSync<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyPad(dstGM[outCopyParams.offset], inLocal, outCopyParams.dcParams);
        inQueueSrc.FreeTensor(inLocal);
    }

    __aicore__ inline void ParseTilingData(const UnfoldGradTilingData* tilingData)
    {
        batchNum = tilingData->batchNum;
        batchNumPerCore = tilingData->batchNumPerCore;
        batchNumTailCore = tilingData->batchNumTailCore;
        maxBatchNum4Ub = tilingData->maxBatchNum4Ub;
        useCoreNum = tilingData->useCoreNum;
        ubSizeT1 = tilingData->ubSizeT1;
        ubSizeT2 = tilingData->ubSizeT2;
        outputNumPerCore = tilingData->outputNumPerCore;
        inputNumPerCore = tilingData->inputNumPerCore;
        iterationNumPerCore = tilingData->iterationNumPerCore;
        handleNUMOnceIterationPerCore = tilingData->handleNUMOnceIterationPerCore;
        tasksOnceMaxPerCore = tilingData->tasksOnceMaxPerCore;
        inputSizeLength = tilingData->inputSizeLength;
        rowAvailableLengthSrc = tilingData->rowAvailableLengthSrc;
        lowestCommonMultiple = tilingData->lowestCommonMultiple;
        colOnceMaxPerUB = tilingData->colOnceMaxPerUB;
        tailColLength = tilingData->tailColLength;
        typeSizeT1 = tilingData->typeSizeT1;
        typeSizeT2 = tilingData->typeSizeT2;
        width = tilingData->width;
        gradOutSizeDim = tilingData->gradOutSizeDim;
        dim = tilingData->dim;
        inputSizeLastDim = tilingData->inputSizeLastDim;
        size = tilingData->size;
        step = tilingData->step;
        loop = tilingData->loop;
        tail = tilingData->tail;
        sizeLevel = tilingData->sizeLevel;   
    }

protected:
    AscendC::TPipe* pipe{nullptr};
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> computeInQueueSrc;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> computeOutQueueDst;
    AscendC::GlobalTensor<T1> srcGlobal;
    AscendC::GlobalTensor<T1> dstGlobal;
    AscendC::GlobalTensor<T2> workspaceT2SumRes;

    uint32_t useCoreNum;
    int T2SrcDataSize = 0;
    int T2DstDataSize = 0;
    int width = 8;
    int gradOutSizeDim = 0;
    int iterationNumPerCore = 0;
    int handleNUMOnceIterationPerCore = 0;
    int tasksOnceMaxPerCore = 0;
    int tasksOnce = 0;
    int loop = 0;
    int tail = 0;
    int sizeLevel = 0;
    int ubSizeT1 = 0;
    int ubSizeT2 = 0;
    int workspaceLen_ = 0;
    int outputLen_ = 0;
    int dim = 0;
    int step = 0;
    int size = 0;
    int inputSizeLastDim = 0;
    int64_t inputSizeLength{0};
    int64_t rowAvailableLengthSrc{0};
    int64_t lowestCommonMultiple{1};
    int64_t colOnceMaxPerUB{0};
    int64_t tailColLength{0};

    int64_t batchNum{0};
    int64_t inputNumPerCore{0};
    int64_t outputNumPerCore{0};
    int64_t batchNumPerCore{0};
    int64_t batchNumTailCore{0};
    int64_t maxBatchNum4Ub{0};
    int32_t typeSizeT1{2};
    int32_t typeSizeT2{4};
};
#endif // UNFOLD_GRAD_H