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
 * \file unfold_grad_final_axe_big_size.h
 * \brief
 */

#ifndef UNFOLD_GRAD_FINAL_AXE_BIG_SIZE_H
#define UNFOLD_GRAD_FINAL_AXE_BIG_SIZE_H

#include "unfold_grad.h"

template <typename T1, typename T2, bool ISCAST = false>
class UnfoldGradFinalAxeBigSize : public UnfoldGrad<T1, T2, ISCAST>
{
public:
    __aicore__ inline UnfoldGradFinalAxeBigSize(AscendC::TPipe* pipe) : UnfoldGrad<T1, T2, ISCAST>(pipe){};

    __aicore__ inline void InitFinalAxeBigSize(
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

    __aicore__ inline void ProcessFinalAxeBigSize(int curSrcStart, int curDstStart)
    {
        this->tasksOnce = this->tasksOnceMaxPerCore;
        for (int i = 0; i < this->loop; i++) {
            CopyInFinalAxeBigSize(curSrcStart, i, this->tasksOnce);

            if (ISCAST){
                SumFP32ResInGMFinalAxeBigSize<T2>(curDstStart, i, this->tasksOnce, this->workspaceT2SumRes); 
            } else {
                SumFP32ResInGMFinalAxeBigSize<T1>(curDstStart, i, this->tasksOnce, this->dstGlobal);
            }
        }
        if (this->tail > 0) {
            this->tasksOnce = this->tail;
            CopyInFinalAxeBigSize(curSrcStart, this->loop, this->tasksOnce);

            if (ISCAST){
                SumFP32ResInGMFinalAxeBigSize<T2>(curDstStart, this->loop, this->tasksOnce, this->workspaceT2SumRes); 
            } else {
                SumFP32ResInGMFinalAxeBigSize<T1>(curDstStart, this->loop, this->tasksOnce, this->dstGlobal);
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

            ProcessFinalAxeBigSize(srcStart, dstStart);

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
    __aicore__ inline void CopyInFinalAxeBigSize(int64_t curSrcStart, int64_t index, int64_t curHandleNum)
    {
        AscendC::LocalTensor<T1> srcLocal =
            ISCAST ? this->inQueueSrc.template AllocTensor<T1>() : this->computeOutQueueDst.template AllocTensor<T1>();
        AscendC::DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
        AscendC::PipeBarrier<PIPE_V>();
        PipeSync<AscendC::HardEvent::MTE3_V>();
        PipeSync<AscendC::HardEvent::S_V>();
        T1 zeroVal(0.0);
        int srcDataSize = ISCAST ? this->ubSizeT1 : this->T2SrcDataSize;
        AscendC::Duplicate<T1>(srcLocal, zeroVal, srcDataSize / this->typeSizeT1);
        PipeSync<AscendC::HardEvent::V_MTE2>();
        // 搬运次数，一定能整除，tiling里分核已对齐size的倍数
        uint16_t blockCount = curHandleNum / this->size; 

        uint32_t blockLen = this->size * this->typeSizeT1;
        uint32_t srcStride = 0; //连续搬
        uint32_t dstStride = 0; 

        AscendC::DataCopyExtParams copyParamsIn{blockCount, blockLen, srcStride, dstStride, 0};
        padParams.isPad = true;
        padParams.rightPadding = 
            ((copyParamsIn.blockLen + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE - copyParamsIn.blockLen) 
            / this->typeSizeT1; 
        
        padParams.paddingValue = 0;
        AscendC::DataCopyPad(
            srcLocal, this->srcGlobal[curSrcStart + index * this->tasksOnceMaxPerCore], copyParamsIn, padParams
        );
        AscendC::PipeBarrier<PIPE_MTE2>();
        if (ISCAST) {
            this->inQueueSrc.template EnQue(srcLocal);
            // fp16转fp32
            srcLocal = this->inQueueSrc.template DeQue<T1>();
            AscendC::LocalTensor<T2> computeDstLocal = this->computeOutQueueDst.template AllocTensor<T2>();
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                computeDstLocal, srcLocal, AscendC::RoundMode::CAST_NONE, srcDataSize / this->typeSizeT1);
            this->computeOutQueueDst.template EnQue(computeDstLocal);
            this->inQueueSrc.template FreeTensor(srcLocal);
        } else { // T1与T2为相同类型
            this->computeOutQueueDst.template EnQue(srcLocal);
        }
    }

    template <typename T>
    __aicore__ inline void SumFP32ResInGMFinalAxeBigSize(
        int64_t curDstStart, int64_t index, int64_t curHandleNum, AscendC::GlobalTensor<T> dstGlobal)
    {
        AscendC::LocalTensor<T> computeDstLocal = this->computeOutQueueDst.template DeQue<T>();
        // 每核处理数据数量包含的step数量
        uint64_t stepNumOnceMax = this->tasksOnceMaxPerCore / this->size;
        int64_t neededSpaceForOnePaddedSize = (this->size * FP32_TYPESIZE + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE; 
        if (ISCAST){
            // 按照CopyIn时对齐长度搬运
            neededSpaceForOnePaddedSize = (this->size * this->typeSizeT1 + BLOCK_SIZE - 1) 
                                            / BLOCK_SIZE * BLOCK_SIZE // paddedSize when CopyIn
                                            / this->typeSizeT1 * FP32_TYPESIZE; 
        }
        
        uint16_t blockCount = curHandleNum / this->size;
        PipeSync<AscendC::HardEvent::MTE2_MTE3>();
        //与padded后的size无重叠：non-overlap DataCopyPad自动连续搬
        if (this->step >= neededSpaceForOnePaddedSize / FP32_TYPESIZE){ 
            uint32_t srcStride = 0;
            uint32_t dstStride = this->step * FP32_TYPESIZE - neededSpaceForOnePaddedSize;
            AscendC::DataCopyExtParams copyParamsOut{blockCount, static_cast<uint32_t>(neededSpaceForOnePaddedSize), srcStride, dstStride, 0};
            AscendC::SetAtomicAdd<T>();
            AscendC::DataCopyPad(
                dstGlobal[curDstStart + index * stepNumOnceMax * this->step], 
                computeDstLocal, copyParamsOut);
            AscendC::SetAtomicNone();
            
        }else{
            // 有重叠 按size对齐后大小手动依次搬（因为dstStride是无符号数）
            AscendC::DataCopyExtParams copyParamsOut{1,  static_cast<uint32_t>(neededSpaceForOnePaddedSize), 0, 0, 0};
            for (int i = 0; i < blockCount; i++){
                AscendC::SetAtomicAdd<T>();
                AscendC::DataCopyPad(
                    dstGlobal[curDstStart + index * stepNumOnceMax * this->step + i * this->step],
                    computeDstLocal[neededSpaceForOnePaddedSize / FP32_TYPESIZE * i], copyParamsOut);
                AscendC::SetAtomicNone();
            }
        }
        PipeSync<AscendC::HardEvent::MTE3_S>();
        this->computeOutQueueDst.template FreeTensor(computeDstLocal);
    }

private:
    uint32_t blockIdx;
    uint32_t gradOutBlockOffset;
    uint32_t gradInBlockOffset;
    uint32_t curCoreBatchNum{0};
    int srcStart = 0;
    int dstStart = 0;
};
#endif // UNFOLD_GRAD_FINAL_AXE_BIG_SIZE_H