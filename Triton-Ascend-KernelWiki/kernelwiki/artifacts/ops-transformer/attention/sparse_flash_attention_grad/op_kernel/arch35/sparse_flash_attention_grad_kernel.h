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
 * \file sparse_flash_attention_grad_kernel.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_GRAD_KERNEL_H
#define SPARSE_FLASH_ATTENTION_GRAD_KERNEL_H
 
#include "sparse_flash_attention_grad_common.h"
#include "sparse_flash_attention_grad_kernel_base.h"
#include "sparse_flash_attention_grad_tiling_data_regbase.h"
 
namespace SfagBaseApi {
 
template <typename CubeBlockType, typename VecBlockType>
 
class FlashAttentionScoreGradKernel
    : public FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>,
                                               CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    using BaseClass = FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>,
                                                        CubeBlockType, VecBlockType>;
    __aicore__ inline void SetUniqueRunInfo(FagRunInfo &runInfo);
    __aicore__ inline void SetUniqueConstInfo(FagConstInfo &constInfo);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPreload(FagRunInfo &runInfo, int64_t taskId);
    __aicore__ inline void ProcessNotFirst(FagRunInfo &runInfo);
};
 
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::SetUniqueRunInfo(FagRunInfo &runInfo)
{
}
 
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::SetUniqueConstInfo(FagConstInfo &constInfo)
{
}
 
 template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessPreload(FagRunInfo &runInfo, int64_t taskId) {
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }
    this->vecBlock.GatherKV(this->selectedKWorkSpaceGm, this->constInfo, runInfo);
    
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V0_TO_C1_FLAG[runInfo.commonRunInfo.taskIdMod2]); // dqk must wait ds copy completely
    } else {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V0_TO_C1_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(16 + SYNC_V0_TO_C1_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }

    LocalTensor<CALC_TYPE> mm1ResTensor = this->mm1ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    this->cubeBlock.IterateMmDyV(mm1ResTensor, this->selectedKWorkSpaceGm, this->constInfo, runInfo);
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C1_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }

    LocalTensor<CALC_TYPE> mm2ResTensor = this->mm2ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    this->cubeBlock.IterateMmQK(mm2ResTensor, this->selectedKWorkSpaceGm, this->constInfo, runInfo);
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C2_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }

    this->vecBlock.CopyMaxSum(this->constInfo, runInfo, taskId); // copy in max and sum double buffer
    runInfo.taskStep = TASK_C1C2;
 }


template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessNotFirst(FagRunInfo &runInfo) {
    this->vecBlock.ProcessVec1(this->constInfo, runInfo);
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C1_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]); // dqk must wait ds copy completely
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C2_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]); // dqk must wait ds copy completely
    }

    LocalTensor<CALC_TYPE> mm1ResTensor = this->mm1ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    LocalTensor<CALC_TYPE> mm2ResTensor = this->mm2ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    this->vecBlock.ProcessVec2(mm2ResTensor, this->constInfo, runInfo); // v2: pse + attenMask + simpleSoftmax

    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C4_TO_V3_FLAG);
    }
    Buffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = this->dSL1Buf.Get();
    Buffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = this->pL1Buf.Get();
    this->vecBlock.ProcessVec3(dSL1Buffer, mm1ResTensor, mm2ResTensor, this->constInfo,
                    runInfo); // v3: dropout + cast + nd2nz
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V3_TO_C3_FLAG); // dqk must wait ds copy completely
    }

    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C5_TO_V4_FLAG);
    }
    this->vecBlock.ProcessVec4(pL1Buffer, mm2ResTensor, this->constInfo, runInfo); // v4: cast + nd2nz
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V4_TO_C5_FLAG); // dv must wait ds copy completely
    }

    if ASCEND_IS_AIC {
        // wait ds in ub copy to l1
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V3_TO_C3_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V3_TO_C3_FLAG);
    }
    // compute dq
    this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(
        this->dqWorkSpaceGm, this->selectedKWorkSpaceGm, this->dSL1Buf, this->constInfo, runInfo); // c3
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(16 + SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_V5_TO_C4_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_V5_TO_C4_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }

    // compute dk
    this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(
        this->mm4ResWorkSpaceGm, this->dSL1Buf, this->constInfo, runInfo); // c4
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
    }

    if ASCEND_IS_AIC {
        // wait p in ub copy to l1
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V4_TO_C5_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V4_TO_C5_FLAG);
    }
    // compute dv
    this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(
        this->mm5ResWorkSpaceGm, this->pL1Buf, this->constInfo, runInfo); // c5
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C5_TO_V5_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C5_TO_V5_FLAG);
    }
    runInfo.taskStep = TASK_C3C4C5;
    if ASCEND_IS_AIV {
        // wait mm4 mm5 result
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C5_TO_V5_FLAG);
    }
    this->vecBlock.ScatterAdd(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, this->dkWorkSpaceGm,
                              mm1ResTensor, mm2ResTensor, this->constInfo, runInfo);
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V5_TO_C4_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }
}


template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::Process()
{
    this->AllocEventID();
    int64_t taskId = 0;
    FagRunInfo runInfos[2]; // for cv ping pong
    for (int32_t i = 0; i < this->processBS1ByCore; i++) {
        this->t1Index = this->cBlockIdx + this->usedCoreNum * i;
        this->GetTndSeqLen(this->t1Index, this->bIndex);
        for (this->n2Index = 0; this->n2Index < this->tilingData->baseParams.n2; this->n2Index++) {
            this->GetActualSelCount(this->t1Index, this->n2Index, this->actualSelectedBlockCount);
            for (this->blkCntOffset = 0; this->blkCntOffset < this->actualSelectedBlockCount; this->blkCntOffset += this->constInfo.selectedCountOffset) {
                if (taskId >= 0) {
                    this->SetRunInfo(runInfos[taskId & 1], taskId);
                    ProcessPreload(runInfos[taskId & 1], taskId);
                } 
                
                if (taskId > 0) {
                    ProcessNotFirst(runInfos[(taskId + 1) & 1]);
                }
                taskId++;
            }
        }
    }

    if (runInfos[(taskId + 1) & 1].taskStep == TASK_C1C2) {
        ProcessNotFirst(runInfos[(taskId + 1) & 1]);
    }
    this->FreeEventID();
}
} // namespace SfagBaseApi
 
 
#endif