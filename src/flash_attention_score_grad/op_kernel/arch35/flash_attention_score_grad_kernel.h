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
 * \file flash_attention_score_grad_kernel.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_KERNEL_H
#define FLASH_ATTENTION_SCORE_GRAD_KERNEL_H
 
#include "flash_attention_score_grad_common.h"
#include "flash_attention_score_grad_kernel_base.h"
#include "flash_attention_score_grad_tiling_data_regbase.h"
 
namespace FagBaseApi {
 
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
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::Process()
{
    if (this->tilingData->s1s2BNGS1S2BlockNumList.blockEnds[this->cBlockIdx] == 0) {
        return;
    }
    int64_t taskId = 0;
    FagRunInfo runInfos[2]; // for cv ping pong
    int64_t nextValidBlockInnerIdx = 0;
    int64_t blockInnerIdx = 0;
    int64_t curLoopIdx = 0; // just for continuous split core
    nextValidBlockInnerIdx = this->GetNextValidIdx(
        runInfos[0], taskId, this->tilingData->s1s2BNGS1S2BlockNumList.blockStarts[this->cBlockIdx], curLoopIdx);
    blockInnerIdx = nextValidBlockInnerIdx;
    
    LocalTensor<CALC_TYPE> mm1ResTensor;
    LocalTensor<CALC_TYPE> mm2ResTensor;
    FagRunInfo prevRunInfo;
    bool needSyncDkMM = false;
    bool needSyncDkDvFixUb = false;
    while (true) {
        this->isLastLoop = (blockInnerIdx == -1);
        if (taskId > 0) {
            prevRunInfo = runInfos[(taskId + 1) & 1];
            this->vecBlock.ProcessVec1(this->constInfo, prevRunInfo); // v1: softmaxGrad
            // wait mm1 and mm2 result
            if ASCEND_IS_AIV {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C1_TO_V2_FLAG[(taskId + 1) & 1]);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C2_TO_V2_FLAG[(taskId + 1) & 1]);
            }
        }
        if (!this->isLastLoop) {
            // get mm1 mm2 next valid block index and next s2 begin end
            nextValidBlockInnerIdx = this->GetNextValidIdx(runInfos[(taskId + 1) & 1], taskId + 1, blockInnerIdx + 1, curLoopIdx + 1);
            this->SetRunInfo(runInfos[taskId & 1], runInfos[(taskId + 1) & 1], taskId, blockInnerIdx, nextValidBlockInnerIdx);
            if (this->tilingData->s1s2BNGS1S2BaseParams.isSplitByBlockIdx || IS_TND_SWIZZLE) {
                curLoopIdx++;
            } else {
                blockInnerIdx++;
            }
 
            mm1ResTensor = this->mm1ResBuf[runInfos[taskId & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            if constexpr (SPLIT_AXIS == BN2 && BaseClass::IS_DK_WRITE_UB) {
                if ASCEND_IS_AIC {
                    if (needSyncDkMM) {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_DETER_FIX_FLAG);
                    }
                }
            }
            if constexpr (SPLIT_AXIS == BN2S2 && BaseClass::IS_DK_WRITE_UB) {
                if ASCEND_IS_AIC {
                    if (needSyncDkDvFixUb) {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_DETER_FIX_FLAG);
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_DETER_FIX_FLAG);
                    }
                }
                needSyncDkDvFixUb = false; // BN2S2存在l0c累加，不一定每一次mm345的结果都需要搬到mm1，mm2resbuf上
            }
            this->cubeBlock.IterateMmDyV(mm1ResTensor, this->constInfo, runInfos[taskId & 1], this->preloadArgs);
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V2_FLAG[taskId & 1]);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C1_TO_V2_FLAG[taskId & 1]);
            }
 
            mm2ResTensor = this->mm2ResBuf[runInfos[taskId & 1].commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            this->cubeBlock.IterateMmQK(mm2ResTensor, this->constInfo, runInfos[taskId & 1], this->preloadArgs);
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_TO_V2_FLAG[taskId & 1]);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C2_TO_V2_FLAG[taskId & 1]);
            }
 
            this->vecBlock.CopyMaxSum(this->constInfo, runInfos[taskId & 1],
                                      taskId); // copy in max and sum double buffer
        }
        if (taskId > 0) {
            mm1ResTensor =
                this->mm1ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            mm2ResTensor =
                this->mm2ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
            this->vecBlock.ProcessVec2(mm2ResTensor, this->constInfo, prevRunInfo); // v2: pse + attenMask + simpleSoftmax
            if ASCEND_IS_AIV {
                if (needSyncDkMM) {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C4_TO_V3_FLAG);
                }
            }
            Buffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = this->dSL1Buf.Get();
            Buffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = this->pL1Buf.Get();
            this->vecBlock.ProcessVec3(dSL1Buffer, mm1ResTensor, mm2ResTensor, this->constInfo,
                                       prevRunInfo); // v3: dropout + cast + nd2nz
            if ASCEND_IS_AIV {
                if (needSyncDkMM) {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C5_TO_V4_FLAG);
                }
            }
            this->vecBlock.ProcessVec4(pL1Buffer, mm2ResTensor, this->constInfo, prevRunInfo); // v4: cast + nd2nz
            if ASCEND_IS_AIV {
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V3_TO_C3_FLAG); // dqk must wait ds copy completely
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V4_TO_C5_FLAG); // dv must wait ds copy completely
            }
 
            if ASCEND_IS_AIC {
                // wait ds in ub copy to l1
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V3_TO_C3_FLAG);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V3_TO_C3_FLAG);
                // wait p in ub copy to l1
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V4_TO_C5_FLAG);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V4_TO_C5_FLAG);
            }
 
            if constexpr (SPLIT_AXIS == BN2 && !IS_BN2_MULTIBLK) {
                // compute dq
                if constexpr (BaseClass::IS_DQ_WRITE_UB) {
                    mm1ResTensor =
                        this->mm1ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
                    this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(
                        mm1ResTensor, this->dSL1Buf, this->constInfo,
                        prevRunInfo); // c3
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V5_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C3_TO_V5_FLAG);
                    } else {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C3_TO_V5_FLAG);
                    }
                    this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB, DQ_IDX>(
                        mm1ResTensor, this->constInfo, prevRunInfo); // v5: dq muls + cast
                } else {
                    this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(
                        this->dqWorkSpaceGm, this->dSL1Buf, this->constInfo,
                        prevRunInfo); // c3
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V5_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C3_TO_V5_FLAG);
                    } else {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V5_FLAG);
                    }
                    this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB, DQ_IDX>(
                        this->dqWorkSpaceGm, this->constInfo, prevRunInfo); // v5: dq muls + cast
                }
                // compute dk
                if constexpr (BaseClass::IS_DK_WRITE_UB) {
                    mm2ResTensor =
                        this->mm2ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
                    this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(
                        mm2ResTensor, this->dSL1Buf, this->constInfo,
                        prevRunInfo); // c4
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C4_TO_V6_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C4_TO_V6_FLAG);
                    } else {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C4_TO_V6_FLAG);
                    }
                    this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DK_WRITE_UB, DK_IDX>(
                        mm2ResTensor, this->constInfo, prevRunInfo); // v6: dk muls + cast
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
                    }
                    if ASCEND_IS_AIV {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(SYNC_DETER_FIX_FLAG);
                    }
                } else {
                    this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(
                        this->dkWorkSpaceGm, this->dSL1Buf, this->constInfo,
                        prevRunInfo); // c4
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C4_TO_V6_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C4_TO_V6_FLAG);
                    } else {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C4_TO_V6_FLAG);
                    }                        
                    this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DK_WRITE_UB, DK_IDX>(
                            this->dkWorkSpaceGm, this->constInfo, prevRunInfo); // v6: dk muls + cast
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
                    }
                }
                // compute dv
                this->cubeBlock.template IterateMmPDy<OUTDTYPE, BaseClass::IS_DV_WRITE_UB>(
                    this->dvGm, this->pL1Buf, this->constInfo, prevRunInfo); // c5
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
                }
                needSyncDkMM = true;
            } else if constexpr (IS_BN2_MULTIBLK) {
                // compute dq
                this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(
                    this->dqWorkSpaceGm, this->dSL1Buf, this->constInfo,
                    prevRunInfo); // c3
                if (prevRunInfo.isLastS1Outer) {
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V5_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C3_TO_V5_FLAG);
                    } else {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V5_FLAG);
                    }
                    this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB, DQ_IDX>(
                        this->dqWorkSpaceGm, this->constInfo, prevRunInfo); // v5: dq muls + cast
                }

                // compute dk
                this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(
                    this->dkWorkSpaceGm, this->dSL1Buf, this->constInfo,
                    prevRunInfo); // c4
                if (!prevRunInfo.isNextS2IdxNoChange) {
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C4_TO_V6_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C4_TO_V6_FLAG);
                    } else {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C4_TO_V6_FLAG);
                    }
                    this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DK_WRITE_UB, DK_IDX>(
                        this->dkWorkSpaceGm, this->constInfo, prevRunInfo); // v6: dk muls + cast
                }
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
                }

                // compute dv
                this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(
                    this->dvWorkSpaceGm, this->pL1Buf, this->constInfo, prevRunInfo); // c5
                if (!prevRunInfo.isNextS2IdxNoChange) {
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V5_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C3_TO_V5_FLAG);
                    } else {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V5_FLAG);
                    }
                    this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DV_WRITE_UB, DV_IDX>(
                        this->dvWorkSpaceGm, this->constInfo, prevRunInfo); // v6: dv muls + cast
                }
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
                }
                needSyncDkMM = true;
            } else if constexpr (SPLIT_AXIS == BN2S2) {
                // compute dq
                this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(
                    this->dqWorkSpaceGm, this->dSL1Buf, this->constInfo,
                    prevRunInfo); // c3
                // compute dk
                if constexpr (BaseClass::IS_DK_WRITE_UB) {
                    mm1ResTensor =
                        this->mm1ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
                    this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(
                        mm1ResTensor, this->dSL1Buf, this->constInfo,
                        prevRunInfo); // c4
                    if (!prevRunInfo.isNextS2IdxNoChange) {
                        if ASCEND_IS_AIC {
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C4_TO_V6_FLAG);
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C4_TO_V6_FLAG);
                        } else {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C4_TO_V6_FLAG);
                        }
                        this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DK_WRITE_UB, DK_IDX>(
                            mm1ResTensor, this->constInfo, prevRunInfo); // v6: dk muls + cast
                    }
                } else {
                    this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(
                        this->dkWorkSpaceGm, this->dSL1Buf, this->constInfo,
                        prevRunInfo); // c4
                    if (!prevRunInfo.isNextS2IdxNoChange) {
                        if ASCEND_IS_AIC {
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C4_TO_V6_FLAG);
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C4_TO_V6_FLAG);
                        } else {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C4_TO_V6_FLAG);
                        }
                        this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DK_WRITE_UB, DK_IDX>(
                            this->dkWorkSpaceGm, this->constInfo, prevRunInfo); // v5: dk muls + cast
                    }
                }
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
                }
                // compute dv
                if constexpr (BaseClass::IS_DV_WRITE_UB) {
                    mm2ResTensor =
                        this->mm2ResBuf[prevRunInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
                    this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(
                        mm2ResTensor, this->pL1Buf, this->constInfo,
                        prevRunInfo); // c3
                    if (!prevRunInfo.isNextS2IdxNoChange) {
                        if ASCEND_IS_AIC {
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V5_FLAG);
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C3_TO_V5_FLAG);
                        } else {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C3_TO_V5_FLAG);
                        }
                        this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DV_WRITE_UB, DV_IDX>(
                            mm2ResTensor, this->constInfo, prevRunInfo); // v6: dv muls + cast
                        if ASCEND_IS_AIV {
                            CrossCoreSetFlag<SYNC_MODE, PIPE_V>(SYNC_DETER_FIX_FLAG);
                        }
                        needSyncDkDvFixUb = true;
                    }
                } else {
                    this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(
                        this->dvWorkSpaceGm, this->pL1Buf, this->constInfo, prevRunInfo); // c5
                    if (!prevRunInfo.isNextS2IdxNoChange) {
                        if ASCEND_IS_AIC {
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V5_FLAG);
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C3_TO_V5_FLAG);
                        } else {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V5_FLAG);
                        }
                        this->vecBlock.template ProcessMulsAndCast<CALC_TYPE, BaseClass::IS_DV_WRITE_UB, DV_IDX>(
                            this->dvWorkSpaceGm, this->constInfo, prevRunInfo); // v6: dv muls + cast
                    }
                }
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
                }
                needSyncDkMM = true;
            } else {
                // compute dq
                this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(
                    this->dqWorkSpaceGm, this->dSL1Buf, this->constInfo,
                    prevRunInfo); // c3
                // compute dk
                this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(
                    this->dkWorkSpaceGm, this->dSL1Buf, this->constInfo,
                    prevRunInfo); // c4
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
                }
 
                // compute dv
                this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(
                    this->dvWorkSpaceGm, this->pL1Buf, this->constInfo, prevRunInfo); // c5
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
                }
                needSyncDkMM = true;
            }
        }
        if (blockInnerIdx == -1) {
            break;
        }
        taskId++;
        blockInnerIdx = nextValidBlockInnerIdx;
    }
} 
} // namespace FagBaseApi
 
 
#endif