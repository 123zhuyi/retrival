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
 * \file unfold_grad_final_axe.h
 * \brief
 */

#ifndef UNFOLD_GRAD_FINAL_AXE_H
#define UNFOLD_GRAD_FINAL_AXE_H

#include "unfold_grad.h"

template <typename T1, typename T2, bool ISCAST = false>
class UnfoldGradFinalAxe : public UnfoldGrad<T1, T2, ISCAST>
{
public:
    __aicore__ inline UnfoldGradFinalAxe(AscendC::TPipe* pipe) : UnfoldGrad<T1, T2, ISCAST>(pipe){};

    __aicore__ inline void InitFinalAxe(
        GM_ADDR srcGm, GM_ADDR dstGm, GM_ADDR workspace, const UnfoldGradTilingData* tiling_data)
    {
        this->Init(tiling_data);

        blockIdx = AscendC::GetBlockIdx();
        gradOutBlockOffset = this->batchNumPerCore * blockIdx * this->inputNumPerCore;
        gradInBlockOffset = this->batchNumPerCore * blockIdx * this->outputNumPerCore;
        if (blockIdx < this->useCoreNum - 1) {
            curCoreBatchNum = this->batchNumPerCore;
        } else { // blockIdx == useCoreNum - 1
            curCoreBatchNum = this->batchNumTailCore;
        }

        this->srcGlobal.SetGlobalBuffer((__gm__ T1*)srcGm + gradOutBlockOffset);
        this->dstGlobal.SetGlobalBuffer((__gm__ T1*)dstGm + gradInBlockOffset);
        this->workspaceT2SumRes.SetGlobalBuffer(reinterpret_cast<__gm__ T2*>(workspace) + gradInBlockOffset);
    }

    template <AscendC::HardEvent hardEvent>
    __aicore__ inline void PipeSync()
    {
        int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(hardEvent));
        AscendC::SetFlag<hardEvent>(eventID);
        AscendC::WaitFlag<hardEvent>(eventID);
    }

    __aicore__ inline void ProcessFinalAxes(int curSrcStart, int curDstStart)
    {
        this->tasksOnce = this->tasksOnceMaxPerCore;
        for (int i = 0; i < this->loop; i++) {
            CopyInFinalAxes(curSrcStart, i, this->tasksOnce);
            ComputeFinalAxes(this->tasksOnce);
            if constexpr (ISCAST) {
                SumFP32ResInGMFinalAxes<T2>(curDstStart, i, this->tasksOnce, this->workspaceT2SumRes);
            } else {
                SumFP32ResInGMFinalAxes<T1>(curDstStart, i, this->tasksOnce, this->dstGlobal);
            }
        }
        if (this->tail > 0) {
            this->tasksOnce = this->tail;
            CopyInFinalAxes(curSrcStart, this->loop, this->tasksOnce);
            ComputeFinalAxes(this->tasksOnce);
            if constexpr (ISCAST) {
                SumFP32ResInGMFinalAxes<T2>(curDstStart, this->loop, this->tasksOnce, this->workspaceT2SumRes);
            } else {
                SumFP32ResInGMFinalAxes<T1>(curDstStart, this->loop, this->tasksOnce, this->dstGlobal);
            }
        }
    }

    __aicore__ inline void Process()
    {
        for (int batchIdx = 0; batchIdx < curCoreBatchNum; batchIdx++) {
            srcStart = this->inputNumPerCore * batchIdx;
            dstStart = this->outputNumPerCore * batchIdx;
            if constexpr (ISCAST) {
                this->SetMIDGMtoZero(this->outputNumPerCore, dstStart);
            } else {
                this->SetGMtoZero(this->outputNumPerCore, dstStart);
            }

            ProcessFinalAxes(srcStart, dstStart);

            if constexpr (ISCAST) {
                sDataCopyExtParams params;
                this->CalculateOutParms(params);
                this->CopyToOutBigShapeOnePage(batchIdx, batchIdx, params);
            }
            PipeSync<AscendC::HardEvent::MTE3_V>();
            PipeSync<AscendC::HardEvent::S_V>();
        }
    }

private:
    __aicore__ inline void CopyInFinalAxes(int64_t curSrcStart, int64_t index, int64_t curHandleNum)
    {
        AscendC::LocalTensor<T1> srcLocal =
            ISCAST ? this->inQueueSrc.template AllocTensor<T1>() : this->computeInQueueSrc.template AllocTensor<T1>();
        AscendC::DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
        AscendC::PipeBarrier<PIPE_V>();
        PipeSync<AscendC::HardEvent::MTE3_V>();
        PipeSync<AscendC::HardEvent::S_V>();
        T1 zeroVal(0.0);
        int srcDataSize = ISCAST ? this->ubSizeT1 : this->T2SrcDataSize;
        AscendC::Duplicate<T1>(srcLocal, zeroVal, srcDataSize / this->typeSizeT1);

        int64_t realSize = this->step < this->size ? this->size : this->step;
        int64_t colHandleNum = this->ubSizeT2 / (TRANS_BLOCK * FP32_TYPESIZE) / realSize * this->size;
        int64_t colNumSpace = srcDataSize / TRANS_BLOCK;
        uint16_t blockCount = (curHandleNum + colHandleNum - 1) / colHandleNum;
        uint32_t blockLen = colHandleNum * this->typeSizeT1;
        uint32_t srcStride = 0;
        uint32_t dstStride =
            (colNumSpace - (colHandleNum * this->typeSizeT1 + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE) / BLOCK_SIZE;
        PipeSync<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopyExtParams copyParamsIn{
            static_cast<uint16_t>(blockCount - 1), blockLen, srcStride, dstStride, 0};
        if (blockCount > 1) {
            padParams.isPad = true;
            padParams.rightPadding =
                ((copyParamsIn.blockLen + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE - copyParamsIn.blockLen) /
                this->typeSizeT1;
            padParams.paddingValue = 0;
            AscendC::DataCopyPad(
                srcLocal, this->srcGlobal[curSrcStart + index * this->tasksOnceMaxPerCore], copyParamsIn, padParams);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        copyParamsIn.blockCount = 1;
        copyParamsIn.blockLen = (curHandleNum - (blockCount - 1) * colHandleNum) * this->typeSizeT1;
        padParams.rightPadding =
            ((copyParamsIn.blockLen + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE - copyParamsIn.blockLen) /
            this->typeSizeT1;
        AscendC::DataCopyPad(
            srcLocal[(blockCount - 1) * (colNumSpace / this->typeSizeT1)],
            this->srcGlobal[curSrcStart + index * this->tasksOnceMaxPerCore + (blockCount - 1) * colHandleNum],
            copyParamsIn, padParams);

        if (ISCAST) {
            this->inQueueSrc.template EnQue(srcLocal);

            // fp16转fp32
            srcLocal = this->inQueueSrc.template DeQue<T1>();
            AscendC::LocalTensor<T2> computeSrcLocal = this->computeInQueueSrc.template AllocTensor<T2>();
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                computeSrcLocal, srcLocal, AscendC::RoundMode::CAST_NONE, colNumSpace * blockCount / this->typeSizeT1);
            this->computeInQueueSrc.template EnQue(computeSrcLocal);
            this->inQueueSrc.template FreeTensor(srcLocal);
        } else { // T1与T2为相同类型
            this->computeInQueueSrc.template EnQue(srcLocal);
        }
    }

    __aicore__ inline void TransDataTo5HDInTranspose1(
        AscendC::LocalTensor<T2>& srcLocal, AscendC::LocalTensor<T2>& dstLocal,
        const TransDataTo5HDParams& transDataTo5HDParams)
    {
        uint64_t dstLocalList[TRANS_BLOCK];
        uint64_t srcLocalList[TRANS_BLOCK];
        for (int i = 0; i < TRANS_BLOCK; i++) {
            if (i % DOUBLE == 0) {
                dstLocalList[i] = reinterpret_cast<uint64_t>(
                    dstLocal
                        [transDataTo5HDParams.idxH * MAX_REPEATTIME * TRANS_BLOCK * this->width +
                         TRANS_BLOCK * (i / DOUBLE)]
                            .GetPhyAddr());
            } else {
                dstLocalList[i] = reinterpret_cast<uint64_t>(
                    dstLocal
                        [transDataTo5HDParams.idxH * MAX_REPEATTIME * TRANS_BLOCK * this->width +
                         TRANS_BLOCK * (i / DOUBLE) + this->width]
                            .GetPhyAddr());
            }
        }
        for (int i = 0; i < TRANS_BLOCK; i++) {
            srcLocalList[i] = reinterpret_cast<uint64_t>(
                srcLocal
                    [transDataTo5HDParams.idxH * MAX_REPEATTIME * this->width + transDataTo5HDParams.rowLengthSrc * i]
                        .GetPhyAddr());
        }
        AscendC::TransDataTo5HD<T2>(dstLocalList, srcLocalList, transDataTo5HDParams.transDataParams);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TransDataForUnfold1(
        AscendC::LocalTensor<T2>& srcLocal, AscendC::LocalTensor<T2>& dstLocal, int64_t curHandleNum)
    {
        uint16_t dstStride = TRANS_BLOCK;
        uint16_t srcStride = 1;
        AscendC::TransDataTo5HDParams transDataParams{false, false, MAX_REPEATTIME, dstStride, srcStride};
        int realTransNum = 0;
        int64_t realSize = this->step < this->size ? this->size : this->step;
        int64_t colHandleNum = this->ubSizeT2 / (TRANS_BLOCK * FP32_TYPESIZE) / realSize * this->size;
        uint16_t blockCount = (curHandleNum + colHandleNum - 1) / colHandleNum;
        realTransNum = curHandleNum > colHandleNum ? colHandleNum : curHandleNum;

        int transLoop = (realTransNum + this->width - 1) / this->width;
        int tmpSize2 = this->ubSizeT2 / TRANS_BLOCK / this->typeSizeT2;
        int loop = transLoop / MAX_REPEATTIME;
        int tail = transLoop % MAX_REPEATTIME;
        TransDataTo5HDParams transDataTo5HDParams(tmpSize2, 0, 0, 0, transDataParams);
        for (int j = 0; j < loop; j++) {
            transDataTo5HDParams.idxH = j;
            TransDataTo5HDInTranspose1(srcLocal, dstLocal, transDataTo5HDParams);
        }
        if (tail > 0) {
            transDataTo5HDParams.transDataParams.repeatTimes = tail;
            if (tail == 1) {
                transDataTo5HDParams.transDataParams.dstRepStride = 0;
                transDataTo5HDParams.transDataParams.srcRepStride = 0;
            }
            transDataTo5HDParams.idxH = loop;
            TransDataTo5HDInTranspose1(srcLocal, dstLocal, transDataTo5HDParams);
        }
    }

    __aicore__ inline void TransDataTo5HDInTranspose2(
        AscendC::LocalTensor<T2>& srcLocal, AscendC::LocalTensor<T2>& dstLocal,
        const TransDataTo5HDParams& transDataTo5HDParams)
    {
        uint64_t dstLocalList[TRANS_BLOCK];
        uint64_t srcLocalList[TRANS_BLOCK];
        for (int i = 0; i < TRANS_BLOCK; i++) {
            int row = transDataTo5HDParams.rowLengthDst * (i / DOUBLE);
            if (i % DOUBLE == 0) {
                dstLocalList[i] = reinterpret_cast<uint64_t>(
                    dstLocal
                        [transDataTo5HDParams.idxW * transDataTo5HDParams.rowLengthDst * this->width +
                         transDataTo5HDParams.idxH * MAX_REPEATTIME * DOUBLE * this->width + row]
                            .GetPhyAddr());
            } else {
                dstLocalList[i] = reinterpret_cast<uint64_t>(
                    dstLocal
                        [transDataTo5HDParams.idxW * transDataTo5HDParams.rowLengthDst * this->width +
                         transDataTo5HDParams.idxH * MAX_REPEATTIME * DOUBLE * this->width + row + this->width]
                            .GetPhyAddr());
            }
        }
        for (int i = 0; i < TRANS_BLOCK; i++) {
            srcLocalList[i] = reinterpret_cast<uint64_t>(
                srcLocal
                    [transDataTo5HDParams.idxW * this->width +
                     transDataTo5HDParams.idxH * MAX_REPEATTIME * DOUBLE * this->width * TRANS_BLOCK +
                     DOUBLE * this->width * i]
                        .GetPhyAddr());
        }
        AscendC::TransDataTo5HD<T2>(dstLocalList, srcLocalList, transDataTo5HDParams.transDataParams);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TransDataForUnfold2(
        AscendC::LocalTensor<T2>& srcLocal, AscendC::LocalTensor<T2>& dstLocal, int64_t curHandleNum)
    {
        uint16_t dstStride = DOUBLE;
        uint16_t srcStride = DOUBLE * TRANS_BLOCK;
        AscendC::TransDataTo5HDParams transDataParams{false, false, MAX_REPEATTIME, dstStride, srcStride};
        int realTransNum = 0;

        if (this->step >= this->size) {
            int64_t colHandleNum = this->ubSizeT2 / (TRANS_BLOCK * FP32_TYPESIZE) / this->step * this->size;
            realTransNum = curHandleNum > colHandleNum ?
                               (colHandleNum - this->size) / this->size * this->step + this->size :
                               (curHandleNum - this->size) / this->size * this->step + this->size;
        } else {
            int64_t colHandleNum = this->ubSizeT2 / (TRANS_BLOCK * FP32_TYPESIZE) / this->size * this->size;
            realTransNum = curHandleNum > colHandleNum ? colHandleNum : curHandleNum;
        }

        int transLoop = (realTransNum + TRANS_BLOCK - 1) / TRANS_BLOCK;
        int tmpSize2 = this->ubSizeT2 / TRANS_BLOCK / this->typeSizeT2;
        int loopw = DOUBLE;
        int loop = transLoop / MAX_REPEATTIME;
        int tail = transLoop % MAX_REPEATTIME;

        TransDataTo5HDParams transDataTo5HDParams(0, tmpSize2, 0, 0, transDataParams);
        for (int k = 0; k < loopw; k++) {
            transDataTo5HDParams.transDataParams.repeatTimes = MAX_REPEATTIME;
            transDataTo5HDParams.transDataParams.dstRepStride = dstStride;
            transDataTo5HDParams.transDataParams.srcRepStride = srcStride;
            transDataTo5HDParams.idxW = k;
            for (int j = 0; j < loop; j++) {
                transDataTo5HDParams.idxH = j;
                TransDataTo5HDInTranspose2(srcLocal, dstLocal, transDataTo5HDParams);
            }
            if (tail > 0) {
                transDataTo5HDParams.transDataParams.repeatTimes = tail;
                if (tail == 1) {
                    transDataTo5HDParams.transDataParams.dstRepStride = 0;
                    transDataTo5HDParams.transDataParams.srcRepStride = 0;
                }
                transDataTo5HDParams.idxH = loop;
                TransDataTo5HDInTranspose2(srcLocal, dstLocal, transDataTo5HDParams);
            }
        }
    }

    __aicore__ inline void AccumulateFinalAxes(
        AscendC::LocalTensor<T2>& srcLocal, AscendC::LocalTensor<T2>& dstLocal, int64_t curHandleNum)
    {
        int64_t realSize = this->step < this->size ? this->size : this->step;
        int64_t colHandleNum = this->ubSizeT2 / (TRANS_BLOCK * FP32_TYPESIZE) / realSize * this->size;
        int loopSize = curHandleNum > colHandleNum ? colHandleNum / this->size : curHandleNum / this->size;

        uint64_t srcStart = 0;
        uint64_t dstStart = 0;
        uint64_t srcOffset = 0;
        uint64_t dstOffset = 0;
        for (int j = 0; j < loopSize; j++) {
            srcOffset = srcStart * TRANS_BLOCK;
            dstOffset = dstStart * TRANS_BLOCK;
            AscendC::Add<T2>(srcLocal[dstOffset], srcLocal[dstOffset], dstLocal[srcOffset], TRANS_BLOCK * this->size);
            AscendC::PipeBarrier<PIPE_V>();
            srcStart += this->size;
            dstStart += this->step;
        }
    }

    __aicore__ inline void ComputeFinalAxes(int64_t curHandleNum)
    {
        AscendC::LocalTensor<T2> computeSrcLocal = this->computeInQueueSrc.template DeQue<T2>();
        AscendC::LocalTensor<T2> computeDstLocal = this->computeOutQueueDst.template AllocTensor<T2>();

        // 转置
        TransDataForUnfold1(computeSrcLocal, computeDstLocal, curHandleNum);

        // UB置0
        AscendC::PipeBarrier<PIPE_V>();
        T2 zeroVal(0.0);
        AscendC::Duplicate<T2>(computeSrcLocal, zeroVal, this->T2SrcDataSize / this->typeSizeT2);
        AscendC::PipeBarrier<PIPE_V>();

        // 累加计算
        AccumulateFinalAxes(computeSrcLocal, computeDstLocal, curHandleNum);

        // 转置
        TransDataForUnfold2(computeSrcLocal, computeDstLocal, curHandleNum);

        this->computeOutQueueDst.template EnQue<T2>(computeDstLocal);
        this->computeInQueueSrc.template FreeTensor(computeSrcLocal);
    }

    template <typename T>
    __aicore__ inline void SumFP32ResInGMFinalAxes(
        int64_t curDstStart, int64_t index, int64_t curHandleNum, AscendC::GlobalTensor<T> dstGlobal)
    {
        AscendC::LocalTensor<T> dstLocal = this->computeOutQueueDst.template DeQue<T>();
        uint64_t stepNumOnceMax = this->tasksOnceMaxPerCore / TRANS_BLOCK / this->size;
        int64_t realSize = this->step < this->size ? this->size : this->step;
        int64_t rowHandleNum = this->ubSizeT2 / (TRANS_BLOCK * FP32_TYPESIZE) / realSize * this->size;
        int64_t rowNumSpace = this->ubSizeT2 / TRANS_BLOCK;
        uint16_t blockCount = (curHandleNum + rowHandleNum - 1) / rowHandleNum;
        int64_t tailBlockLen = curHandleNum - (blockCount - 1) * rowHandleNum;
        uint64_t blockDstNum = this->step * (rowHandleNum / this->size - 1) + this->size;
        uint64_t tailBlockDstNum = this->step * (tailBlockLen / this->size - 1) + this->size;
        uint32_t srcStride =
            (rowNumSpace - (blockDstNum * FP32_TYPESIZE + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE) / BLOCK_SIZE;
        uint32_t dstStride = (this->step - this->size) * FP32_TYPESIZE;

        if (this->step >= this->size) {
            if (blockCount > 1) {
                AscendC::DataCopyExtParams copyParamsOut{
                    static_cast<uint16_t>(blockCount - 1), static_cast<uint32_t>(blockDstNum * FP32_TYPESIZE),
                    srcStride, dstStride, 0};
                AscendC::SetAtomicAdd<T>();
                AscendC::DataCopyPad(
                    dstGlobal[curDstStart + index * TRANS_BLOCK * stepNumOnceMax * this->step], dstLocal,
                    copyParamsOut);
                AscendC::SetAtomicNone();
            }
            AscendC::DataCopyExtParams copyParamsOut{
                1, static_cast<uint32_t>(tailBlockDstNum * FP32_TYPESIZE), 0, 0, 0};
            AscendC::SetAtomicAdd<T>();
            AscendC::DataCopyPad(
                dstGlobal
                    [curDstStart + index * TRANS_BLOCK * stepNumOnceMax * this->step +
                     (blockCount - 1) * stepNumOnceMax * this->step],
                dstLocal[rowNumSpace / FP32_TYPESIZE * (blockCount - 1)], copyParamsOut);
            AscendC::SetAtomicNone();
        } else {
            AscendC::DataCopyExtParams copyParamsOut{1, static_cast<uint32_t>(blockDstNum * FP32_TYPESIZE), 0, 0, 0};
            for (int i = 0; i < blockCount - 1; i++) {
                AscendC::SetAtomicAdd<T>();
                AscendC::DataCopyPad(
                    dstGlobal
                        [curDstStart + index * TRANS_BLOCK * stepNumOnceMax * this->step +
                         i * stepNumOnceMax * this->step],
                    dstLocal[rowNumSpace / FP32_TYPESIZE * i], copyParamsOut);
                AscendC::SetAtomicNone();
            }
            copyParamsOut.blockLen = static_cast<uint32_t>(tailBlockDstNum * FP32_TYPESIZE);
            AscendC::SetAtomicAdd<T>();
            AscendC::DataCopyPad(
                dstGlobal
                    [curDstStart + index * TRANS_BLOCK * stepNumOnceMax * this->step +
                     (blockCount - 1) * stepNumOnceMax * this->step],
                dstLocal[rowNumSpace / FP32_TYPESIZE * (blockCount - 1)], copyParamsOut);
            AscendC::SetAtomicNone();
        }

        this->computeOutQueueDst.template FreeTensor(dstLocal);
    }

private:
    uint32_t blockIdx;
    uint32_t gradOutBlockOffset;
    uint32_t gradInBlockOffset;
    uint32_t curCoreBatchNum{0};
    int srcStart = 0;
    int dstStart = 0;
};
#endif // UNFOLD_GRAD_FINAL_AXE_H