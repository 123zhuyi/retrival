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
 * \file unfold_grad_final_second_axe.h
 * \brief
 */

#ifndef UNFOLD_GRAD_FINAL_SECOND_AXE_H
#define UNFOLD_GRAD_FINAL_SECOND_AXE_H

#include "unfold_grad.h"

template <typename T1, typename T2, bool ISCAST = false>
class UnfoldGradFinalSecondAxe : public UnfoldGrad<T1, T2, ISCAST>
{
public:
    __aicore__ inline UnfoldGradFinalSecondAxe(AscendC::TPipe* pipe) : UnfoldGrad<T1, T2, ISCAST>(pipe){};

    __aicore__ inline void InitFinalSecondAxe(
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

    __aicore__ inline void ProcessFinalSecondAxes(int curSrcStart, int curDstStart)
    {
        for (int k = 0; k < this->iterationNumPerCore; k++) {
            this->tasksOnce = this->tasksOnceMaxPerCore;
            preTreatmentTasksOnce = this->colOnceMaxPerUB;
            for (int i = 0; i < this->loop; i++) {
                CopyInFinalSecondAxes(curSrcStart, i, this->tasksOnce);
                ComputeFinalSecondAxes(this->tasksOnce, preTreatmentTasksOnce);
                if constexpr (ISCAST) {
                    SumFP32ResInGMFinalSecondAxes<T2>(curDstStart, i, this->tasksOnce, this->workspaceT2SumRes);
                } else {
                    SumFP32ResInGMFinalSecondAxes<T1>(curDstStart, i, this->tasksOnce, this->dstGlobal);
                }
            }
            if (this->tail > 0) {
                this->tasksOnce = this->tail;
                preTreatmentTasksOnce = this->tailColLength;
                this->CopyInFinalSecondAxes(curSrcStart, this->loop, this->tasksOnce);
                this->ComputeFinalSecondAxes(this->tasksOnce, preTreatmentTasksOnce);
                if constexpr (ISCAST) {
                    SumFP32ResInGMFinalSecondAxes<T2>(
                        curDstStart, this->loop, this->tasksOnce, this->workspaceT2SumRes);
                } else {
                    SumFP32ResInGMFinalSecondAxes<T1>(curDstStart, this->loop, this->tasksOnce, this->dstGlobal);
                }
            }

            curSrcStart += this->size * this->inputSizeLastDim;
            curDstStart += this->step * this->inputSizeLastDim;
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

            ProcessFinalSecondAxes(srcStart, dstStart);

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
    __aicore__ inline void CopyInFinalSecondAxes(int64_t curSrcStart, int64_t index, int64_t curHandleNum)
    {
        AscendC::LocalTensor<T1> srcLocal =
            ISCAST ? this->inQueueSrc.template AllocTensor<T1>() : this->computeInQueueSrc.template AllocTensor<T1>();
        AscendC::DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
        uint32_t blockLen = curHandleNum * this->size * this->typeSizeT1;
        AscendC::DataCopyExtParams copyParamsIn{
            1, blockLen, 0, 0, 0}; // 处理tasksOnce个数需要从srcGM中取((tasksOnce-1) * size + 1)个数
        PipeSync<AscendC::HardEvent::MTE3_MTE2>();
        PipeSync<AscendC::HardEvent::S_MTE2>();
        PipeSync<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopyPad(
            srcLocal, this->srcGlobal[curSrcStart + index * this->tasksOnceMaxPerCore * this->size], copyParamsIn,
            padParams); // k * size * inputSizeLastDim + j + index * tasksOnceMaxPerCore * size

        if (ISCAST) {
            this->inQueueSrc.template EnQue(srcLocal);

            // fp16、bf16转fp32
            srcLocal = this->inQueueSrc.template DeQue<T1>();
            AscendC::LocalTensor<T2> computeSrcLocal = this->computeInQueueSrc.template AllocTensor<T2>();
            AscendC::Cast(computeSrcLocal, srcLocal, AscendC::RoundMode::CAST_NONE, curHandleNum * this->size);
            this->computeInQueueSrc.template EnQue(computeSrcLocal);
            this->inQueueSrc.template FreeTensor(srcLocal);
        } else { // T1与T2为相同类型
            this->computeInQueueSrc.template EnQue(srcLocal);
        }
    }

    __aicore__ inline void TransDataTo5HDInTranspose3(
        AscendC::LocalTensor<T2>& srcLocal, AscendC::LocalTensor<T2>& dstLocal,
        const TransDataTo5HDParams& transDataTo5HDParams)
    {
        uint64_t dstLocalList[TRANS_BLOCK];
        uint64_t srcLocalList[TRANS_BLOCK];
        for (int i = 0; i < TRANS_BLOCK; i++) {
            if (i % DOUBLE == 0) {
                dstLocalList[i] = reinterpret_cast<uint64_t>(
                    dstLocal
                        [transDataTo5HDParams.idxH * MAX_REPEATTIME * this->width +
                         transDataTo5HDParams.rowLengthDst * TRANS_BLOCK * transDataTo5HDParams.idxW +
                         transDataTo5HDParams.rowLengthDst * (i / DOUBLE)]
                            .GetPhyAddr());
            } else {
                dstLocalList[i] = reinterpret_cast<uint64_t>(
                    dstLocal
                        [transDataTo5HDParams.idxH * MAX_REPEATTIME * this->width +
                         transDataTo5HDParams.rowLengthDst * TRANS_BLOCK * transDataTo5HDParams.idxW +
                         transDataTo5HDParams.rowLengthDst * (i / DOUBLE + this->width)]
                            .GetPhyAddr());
            }
        }

        for (int i = 0; i < this->width; i++) {
            srcLocalList[i] = reinterpret_cast<uint64_t>(
                srcLocal
                    [transDataTo5HDParams.idxH * MAX_REPEATTIME * this->lowestCommonMultiple * this->width +
                     TRANS_BLOCK * transDataTo5HDParams.idxW + this->lowestCommonMultiple * i]
                        .GetPhyAddr());
        }
        for (int i = this->width; i < TRANS_BLOCK; i++) {
            srcLocalList[i] = reinterpret_cast<uint64_t>(
                srcLocal
                    [transDataTo5HDParams.idxH * MAX_REPEATTIME * this->lowestCommonMultiple * this->width +
                     TRANS_BLOCK * transDataTo5HDParams.idxW + this->lowestCommonMultiple * (i - this->width) +
                     this->width]
                        .GetPhyAddr());
        }
        AscendC::TransDataTo5HD<T2>(dstLocalList, srcLocalList, transDataTo5HDParams.transDataParams);
    }

    __aicore__ inline void TransDataForUnfold3(
        AscendC::LocalTensor<T2>& srcLocal, AscendC::LocalTensor<T2>& dstLocal, int64_t curHandleNum)
    {
        uint16_t dstStride = 1;
        uint16_t srcStride = this->lowestCommonMultiple;
        AscendC::TransDataTo5HDParams transDataParams{false, false, MAX_REPEATTIME, dstStride, srcStride};

        int loopw = this->lowestCommonMultiple / TRANS_BLOCK;
        int transLoop = (curHandleNum + this->width - 1) / this->width;
        int tmpSize2 = (curHandleNum + this->width - 1) / this->width * this->width;
        int looph = transLoop / MAX_REPEATTIME;
        int tailh = transLoop % MAX_REPEATTIME;

        uint64_t dstLocalList[TRANS_BLOCK];
        uint64_t srcLocalList[TRANS_BLOCK];
        TransDataTo5HDParams transDataTo5HDParams(0, tmpSize2, 0, 0, transDataParams);
        for (int k = 0; k < loopw; k++) {
            transDataTo5HDParams.transDataParams.repeatTimes = MAX_REPEATTIME;
            transDataTo5HDParams.transDataParams.dstRepStride = dstStride;
            transDataTo5HDParams.transDataParams.srcRepStride = srcStride;
            transDataTo5HDParams.idxW = k;
            for (int j = 0; j < looph; j++) {
                transDataTo5HDParams.idxH = j;
                TransDataTo5HDInTranspose3(srcLocal, dstLocal, transDataTo5HDParams);
            }
            if (tailh > 0) {
                transDataTo5HDParams.transDataParams.repeatTimes = tailh;
                if (tailh == 1) {
                    transDataTo5HDParams.transDataParams.dstRepStride = 0;
                    transDataTo5HDParams.transDataParams.srcRepStride = 0;
                }
                transDataTo5HDParams.idxH = looph;
                TransDataTo5HDInTranspose3(srcLocal, dstLocal, transDataTo5HDParams);
            }
        }
    }

    __aicore__ inline void TransDataTo5HDInTranspose4and5(
        AscendC::LocalTensor<T2>& srcLocal, AscendC::LocalTensor<T2>& dstLocal,
        const TransDataTo5HDParams& transDataTo5HDParams)
    {
        uint64_t dstLocalList[TRANS_BLOCK];
        uint64_t srcLocalList[TRANS_BLOCK];
        for (int i = 0; i < TRANS_BLOCK; i++) {
            if (i % DOUBLE == 0) {
                dstLocalList[i] = reinterpret_cast<uint64_t>(
                    dstLocal
                        [transDataTo5HDParams.idxH * MAX_REPEATTIME * TRANS_BLOCK +
                         transDataTo5HDParams.rowLengthDst * this->width * transDataTo5HDParams.idxW +
                         transDataTo5HDParams.rowLengthDst * (i / DOUBLE)]
                            .GetPhyAddr());
            } else {
                dstLocalList[i] = reinterpret_cast<uint64_t>(
                    dstLocal
                        [transDataTo5HDParams.idxH * MAX_REPEATTIME * TRANS_BLOCK +
                         transDataTo5HDParams.rowLengthDst * this->width * transDataTo5HDParams.idxW +
                         transDataTo5HDParams.rowLengthDst * (i / DOUBLE) + this->width]
                            .GetPhyAddr());
            }
        }
        for (int i = 0; i < TRANS_BLOCK; i++) {
            srcLocalList[i] = reinterpret_cast<uint64_t>(
                srcLocal
                    [transDataTo5HDParams.idxH * MAX_REPEATTIME * transDataTo5HDParams.rowLengthSrc * TRANS_BLOCK +
                     this->width * transDataTo5HDParams.idxW + transDataTo5HDParams.rowLengthSrc * i]
                        .GetPhyAddr());
        }
        AscendC::TransDataTo5HD<T2>(dstLocalList, srcLocalList, transDataTo5HDParams.transDataParams);
    }

    __aicore__ inline void TransDataForUnfold4and5(
        AscendC::LocalTensor<T2>& srcLocal, AscendC::LocalTensor<T2>& dstLocal, int64_t curHandleNum, int loopw,
        int transLoop)
    {
        int rowLengthDst = transLoop * TRANS_BLOCK;
        int rowLengthSrc = loopw * this->width;
        int looph = transLoop / MAX_REPEATTIME;
        int tailh = transLoop % MAX_REPEATTIME;

        uint16_t dstStride = DOUBLE;
        uint16_t srcStride = rowLengthSrc * DOUBLE;
        AscendC::TransDataTo5HDParams transDataParams{false, false, MAX_REPEATTIME, dstStride, srcStride};

        TransDataTo5HDParams transDataTo5HDParams(rowLengthSrc, rowLengthDst, 0, 0, transDataParams);
        for (int k = 0; k < loopw; k++) {
            transDataTo5HDParams.transDataParams.repeatTimes = MAX_REPEATTIME;
            transDataTo5HDParams.transDataParams.dstRepStride = dstStride;
            transDataTo5HDParams.transDataParams.srcRepStride = srcStride;
            transDataTo5HDParams.idxW = k;
            for (int j = 0; j < looph; j++) {
                transDataTo5HDParams.idxH = j;
                TransDataTo5HDInTranspose4and5(srcLocal, dstLocal, transDataTo5HDParams);
            }
            if (tailh > 0) {
                transDataTo5HDParams.transDataParams.repeatTimes = tailh;
                if (tailh == 1) {
                    transDataTo5HDParams.transDataParams.dstRepStride = 0;
                    transDataTo5HDParams.transDataParams.srcRepStride = 0;
                }
                transDataTo5HDParams.idxH = looph;
                TransDataTo5HDInTranspose4and5(srcLocal, dstLocal, transDataTo5HDParams);
            }
        }
    }

    // 重排来生成冗余空间
    __aicore__ inline void rearrangement4Redundancy(
        AscendC::LocalTensor<T2>& srcLocal, AscendC::LocalTensor<T2>& dstLocal, int64_t curHandleNum)
    {
        uint16_t blockCount = this->lowestCommonMultiple / this->size;
        uint16_t rowLength = (curHandleNum + this->width - 1) / this->width * this->width;
        uint16_t blockLen = this->size * rowLength * this->typeSizeT2 / BLOCK_SIZE;
        uint16_t srcStride = 0;
        uint16_t dstStride = (this->rowAvailableLengthSrc - this->size) * rowLength * this->typeSizeT2 / BLOCK_SIZE;
        AscendC::DataCopyParams repeatParams{blockCount, blockLen, srcStride, dstStride};
        AscendC::DataCopy(srcLocal, dstLocal, repeatParams);
    }

    __aicore__ inline void ComputeFinalSecondAxes(int64_t curHandleNum, int64_t preTreatmentCurHandleNum)
    {
        AscendC::LocalTensor<T2> computeSrcLocal = this->computeInQueueSrc.template DeQue<T2>();
        AscendC::LocalTensor<T2> computeDstLocal = this->computeOutQueueDst.template AllocTensor<T2>();

        // 为每个size增加冗余值
        //      转置，将行转为列
        TransDataForUnfold3(computeSrcLocal, computeDstLocal, preTreatmentCurHandleNum);

        //      UB置0
        AscendC::PipeBarrier<PIPE_V>();
        T2 zeroVal(0.0);
        AscendC::Duplicate<T2>(computeSrcLocal, zeroVal, this->T2SrcDataSize / this->typeSizeT2);

        //      重排，添加冗余值
        rearrangement4Redundancy(computeSrcLocal, computeDstLocal, preTreatmentCurHandleNum);

        //      转置，将列转回行，得到带冗余值的结果
        TransDataForUnfold4and5(
            computeSrcLocal, computeDstLocal, preTreatmentCurHandleNum,
            (preTreatmentCurHandleNum + this->width - 1) / this->width,
            this->lowestCommonMultiple / this->size * this->rowAvailableLengthSrc / TRANS_BLOCK);

        AscendC::DataCopy(computeSrcLocal, computeDstLocal, this->T2SrcDataSize / this->typeSizeT2);

        // 转置，得到当前迭代的目标结果
        TransDataForUnfold4and5(
            computeSrcLocal, computeDstLocal, curHandleNum, this->rowAvailableLengthSrc / this->width,
            (curHandleNum + TRANS_BLOCK - 1) / TRANS_BLOCK);

        this->computeOutQueueDst.template EnQue<T2>(computeDstLocal);
        this->computeInQueueSrc.template FreeTensor(computeSrcLocal);
    }

    template <typename T>
    __aicore__ inline void SumFP32ResInGMFinalSecondAxes(
        int64_t curDstStart, int64_t index, int64_t curHandleNum, AscendC::GlobalTensor<T> dstGlobal)
    {
        AscendC::LocalTensor<T> dstLocal = this->computeOutQueueDst.template DeQue<T>();
        int64_t rowHandleNum = curHandleNum;
        int64_t rowNumSpace = (curHandleNum + TRANS_BLOCK - 1) / TRANS_BLOCK * TRANS_BLOCK * FP32_TYPESIZE;
        uint16_t blockCount = this->size;
        uint32_t blockLen = rowHandleNum * FP32_TYPESIZE;
        uint32_t srcStride =
            (rowNumSpace - (rowHandleNum * FP32_TYPESIZE + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE) / BLOCK_SIZE;
        uint32_t dstStride = (this->inputSizeLastDim - rowHandleNum) * FP32_TYPESIZE;

        AscendC::DataCopyExtParams copyParamsOut{blockCount, blockLen, srcStride, dstStride, 0};
        AscendC::SetAtomicAdd<T>();
        AscendC::DataCopyPad(dstGlobal[curDstStart + index * this->tasksOnceMaxPerCore], dstLocal, copyParamsOut);
        AscendC::SetAtomicNone();

        this->computeOutQueueDst.template FreeTensor(dstLocal);
    }

private:
    uint32_t blockIdx;
    uint32_t gradOutBlockOffset;
    uint32_t gradInBlockOffset;
    uint32_t curCoreBatchNum{0};
    int srcStart = 0;
    int dstStart = 0;
    int64_t preTreatmentTasksOnce{0};
};
#endif // UNFOLD_GRAD_FINAL_SECOND_AXE_H