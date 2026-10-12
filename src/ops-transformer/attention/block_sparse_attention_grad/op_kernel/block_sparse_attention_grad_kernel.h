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
 * \file block_sparse_attention_grad_kernel.h
 * \brief Block Sparse Attention Grad Kernel Implementation
 */

#ifndef BLOCK_SPARSE_ATTENTION_GRAD_KERNEL_H
#define BLOCK_SPARSE_ATTENTION_GRAD_KERNEL_H

#include "attn_infra/base_defs.hpp"
#include "attn_infra/arch/arch.hpp"
#include "attn_infra/arch/cross_core_sync.hpp"
#include "attn_infra/arch/resource.hpp"
#include "attn_infra/layout/layout.hpp"

#include "attn_infra/gemm/block/block_mmad.hpp"
#include "attn_infra/gemm/dispatch_policy.hpp"
#include "attn_infra/gemm/gemm_type.hpp"
#include "attn_infra/epilogue/block/block_epilogue.hpp"
#include "attn_infra/epilogue/dispatch_policy.hpp"
#include "attn_infra/epilogue/block/block_epilogue_fag_pre.hpp"
#include "attn_infra/epilogue/block/block_epilogue_post.hpp"
#include "attn_infra/epilogue/block/block_epilogue_softmaxgrad.hpp"
#include "attn_infra/epilogue/block/block_epilogue_simply_softmax.hpp"

using namespace NpuArch;

namespace BSA {

    constexpr uint32_t CUBE2VEC = 7;
    constexpr uint32_t VEC2CUBE = 8;
    constexpr uint32_t CUBE2POST = 9;

    constexpr int32_t PRE_LAUNCH = 2;
    constexpr uint64_t WORKSPACE_BLOCK_SIZE = 128 * 128 * 2;
    constexpr uint64_t WORKSPACE_P16_OFFSET_ELEMENT = 128 * 128 * 2; // dp, P 的在workspace相对起始地址后偏移元素个数
    constexpr uint64_t WORKSPACE_P16_OFFSET = 128 * 128 * 4; // dp, P 的在workspace相对起始地址后偏移字节数
    constexpr uint64_t WORKSPACE_BLOCK_SIZE_DB = 128 * 128 * 2 * 2;
    constexpr uint64_t L1_SIZE_OFFSET = 131072;
    constexpr uint32_t PINGPONG_OFFSET_2 = 2;
    constexpr uint32_t PINGPONG_OFFSET_4 = 4;

    template <
        class BlockMmadBSAG1_,
        class BlockMmadBSAG2_,
        class BlockMmadBSAG3_,
        class EpilogueFAGPre_,
        class EpilogueFAGSfmg_,
        class EpilogueFAGOp_,
        class EpilogueFAGPost_,
        uint32_t INPUT_LAYOUT>
    class BlockSparseAttentionGradKernel {
    public:
        using BlockMmadBSAG1 = BlockMmadBSAG1_;
        using BlockMmadBSAG2 = BlockMmadBSAG2_;
        using BlockMmadBSAG3 = BlockMmadBSAG3_;
        using EpilogueFAGPre = EpilogueFAGPre_;
        using EpilogueFAGSfmg = EpilogueFAGSfmg_;
        using EpilogueFAGOp = EpilogueFAGOp_;
        using EpilogueFAGPost = EpilogueFAGPost_;
        using PreParams = typename EpilogueFAGPre::Params;
        using PostParams = typename EpilogueFAGPost::Params;
        using SfmgParams = typename EpilogueFAGSfmg_::Params;
        using SfmParams = typename EpilogueFAGOp_::Params;
        using ArchTag = typename BlockMmadBSAG1_::ArchTag;
        using ElementInput = typename BlockMmadBSAG1::ElementA;

        using L1TileShape = typename BlockMmadBSAG1::L1TileShape;
        using ElementA1 = typename BlockMmadBSAG1::ElementA;
        using LayoutA1 = typename BlockMmadBSAG1::LayoutA;
        using ElementB1 = typename BlockMmadBSAG1::ElementB;
        using LayoutB1 = typename BlockMmadBSAG1::LayoutB;
        using ElementC1 = typename BlockMmadBSAG1::ElementC;
        using LayoutC1 = typename BlockMmadBSAG1::LayoutC;

        using ElementA2 = typename BlockMmadBSAG2::ElementA;
        using LayoutA2 = typename BlockMmadBSAG2::LayoutA;
        using ElementB2 = typename BlockMmadBSAG2::ElementB;
        using LayoutB2 = typename BlockMmadBSAG2::LayoutB;
        using ElementC2 = typename BlockMmadBSAG2::ElementC;
        using LayoutC2 = typename BlockMmadBSAG2::LayoutC;

        using ElementA3 = typename BlockMmadBSAG3::ElementA;
        using LayoutA3 = typename BlockMmadBSAG3::LayoutA;
        using ElementB3 = typename BlockMmadBSAG3::ElementB;
        using LayoutB3 = typename BlockMmadBSAG3::LayoutB;
        using ElementC3 = typename BlockMmadBSAG3::ElementC;
        using LayoutC3 = typename BlockMmadBSAG3::LayoutC;

        /// Parameters structure
        struct Params {
            // Data members
            GM_ADDR dout;
            GM_ADDR q;
            GM_ADDR k;
            GM_ADDR v;
            GM_ADDR out;
            GM_ADDR softmaxLse;
            GM_ADDR blockSparseMask; 
            GM_ADDR blockShape;
            GM_ADDR attentionMask;
            GM_ADDR actualQseqlen; 
            GM_ADDR actualKvseqlen;
            GM_ADDR dq;
            GM_ADDR dk;
            GM_ADDR dv;
            GM_ADDR workspace;
            GM_ADDR tiling;

            // Methods
            __aicore__ inline
            Params() {}

            __aicore__ inline
            Params(
                GM_ADDR dout_, GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR out_, GM_ADDR softmaxLse_, GM_ADDR blockSparseMask_,
                GM_ADDR blockShape_, GM_ADDR attentionMask_, GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_,
                GM_ADDR dq_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR workspace_, GM_ADDR tiling_data_
            ) : dout(dout_), q(q_), k(k_), v(v_), out(out_), softmaxLse(softmaxLse_), blockSparseMask(blockSparseMask_),
                blockShape(blockShape_), attentionMask(attentionMask_), actualQseqlen(actualQseqlen_), actualKvseqlen(actualKvseqlen_),
                dq(dq_), dk(dk_), dv(dv_), workspace(workspace_), tiling(tiling_data_)
            {
            }    
        };

        struct TaskInfo {
            uint32_t curBatchIdx;
            uint32_t curHeadIdx;
            uint32_t curQSeqIdx;
            uint32_t curQBlcokIdx;
            uint32_t curCalQSize;
            uint32_t curCalKVSize;
            uint32_t qSeqlen;
            uint32_t kvSeqlen;
            uint64_t qOffset; //Q, dout, dq
            uint64_t kvOffset;//K, V, dk, dv
            uint64_t sOffset; //workspace : S, P, dp, ds
        };

        __aicore__ inline void SetFlag() {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        }

        __aicore__ inline void WaitFlag() {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        }

        __aicore__ inline void UpdateTaskInfoCalQSize(uint32_t blockShapeX, uint32_t basicQBlockSize, TaskInfo &taskInfo) {
            taskInfo.curQBlcokIdx = taskInfo.curQSeqIdx / blockShapeX;
            uint32_t curQBlcokSeqIdx = taskInfo.curQBlcokIdx * blockShapeX;
            if (curQBlcokSeqIdx + blockShapeX <= taskInfo.qSeqlen) {
                if (taskInfo.curQSeqIdx + basicQBlockSize <= curQBlcokSeqIdx + blockShapeX) {
                    taskInfo.curCalQSize = basicQBlockSize;
                } else {
                    taskInfo.curCalQSize = curQBlcokSeqIdx + blockShapeX - taskInfo.curQSeqIdx;
                }
            } else {
                if (taskInfo.curQSeqIdx + basicQBlockSize <= taskInfo.qSeqlen) {
                    taskInfo.curCalQSize = basicQBlockSize;
                } else {
                    taskInfo.curCalQSize = taskInfo.qSeqlen - taskInfo.curQSeqIdx;
                }
            }
        }

        __aicore__ inline void initTaskInfo(AscendC::GlobalTensor<int64_t> gActualQseqlen, AscendC::GlobalTensor<int64_t> gActualKvseqlen,
                                            __gm__ BlockSparseAttentionGradTilingData *tilingData,
                                            uint32_t numHeads, uint32_t kvHeads, uint32_t groupSize, uint32_t headDim,
                                            uint32_t maxQSeqlen, uint32_t maxKvSeqlen, uint32_t blockShapeX,
                                            uint32_t basicQBlockSize, uint32_t inputLayout, uint32_t coreIdx,
                                            TaskInfo &taskInfo) {
            //BNSD:curBatch * numHeads * maxQSeqlen + curQSeqOffset; TND:cusum(gActualQseqlen[0:curBatch-1]) + curQSeqOffset
            uint32_t preQSeqLengths = tilingData->preQSeqLengths[coreIdx];
            //BNSD:curBatch * kvHeads * maxKvSeqlen; TND:cusum(gActualKvseqlen[0:curBatch-1])
            uint32_t preKVSeqLengths = tilingData->preKVSeqLengths[coreIdx];
            taskInfo.curBatchIdx =  tilingData->beginBatch[coreIdx];
            taskInfo.curHeadIdx = tilingData->beginHead[coreIdx];
            taskInfo.curQSeqIdx = tilingData->beginQSeqOffset[coreIdx];
            if (inputLayout == 0) {
                taskInfo.qOffset = preQSeqLengths * numHeads * headDim + taskInfo.curHeadIdx * headDim;
                taskInfo.kvOffset = preKVSeqLengths * kvHeads * headDim + taskInfo.curHeadIdx / groupSize * headDim;
                taskInfo.qSeqlen = static_cast<uint32_t>(static_cast<int64_t>(gActualQseqlen.GetValue(taskInfo.curBatchIdx)));
                taskInfo.kvSeqlen = static_cast<uint32_t>(static_cast<int64_t>(gActualKvseqlen.GetValue(taskInfo.curBatchIdx)));
            } else {
                taskInfo.qOffset = preQSeqLengths * headDim;
                taskInfo.kvOffset = preKVSeqLengths * headDim;
                taskInfo.qSeqlen = maxQSeqlen;
                taskInfo.kvSeqlen = maxKvSeqlen;
            }
            UpdateTaskInfoCalQSize(blockShapeX, basicQBlockSize, taskInfo);
        }

        __aicore__ inline void updateNextTaskInfo(AscendC::GlobalTensor<int64_t> gActualQseqlen, AscendC::GlobalTensor<int64_t> gActualKvseqlen,
                                                  uint32_t numHeads, uint32_t kvHeads, uint32_t groupSize, uint32_t headDim,
                                                  uint32_t blockShapeX, uint32_t basicQBlockSize, uint32_t inputLayout,
                                                  const TaskInfo &taskInfo, TaskInfo &nextTask) {
            nextTask = taskInfo;
            if (inputLayout == 0) { // TND, Traverse N-axis first
                if (taskInfo.curHeadIdx == numHeads - 1) {
                    nextTask.qOffset = taskInfo.qOffset + (taskInfo.curCalQSize - 1) * numHeads * headDim;
                    nextTask.kvOffset = taskInfo.kvOffset + (taskInfo.kvSeqlen - 1) * kvHeads * headDim;
                    if (taskInfo.curQSeqIdx + taskInfo.curCalQSize == taskInfo.qSeqlen) {
                        nextTask.curBatchIdx = taskInfo.curBatchIdx + 1;
                        nextTask.curHeadIdx = 0;
                        nextTask.curQSeqIdx = 0;
                        nextTask.qSeqlen = static_cast<uint32_t>(static_cast<int64_t>(gActualQseqlen.GetValue(nextTask.curBatchIdx)));
                        nextTask.kvSeqlen = static_cast<uint32_t>(static_cast<int64_t>(gActualKvseqlen.GetValue(nextTask.curBatchIdx)));
                    } else { // batch 不变
                        nextTask.curHeadIdx = 0;
                        nextTask.curQSeqIdx = taskInfo.curQSeqIdx + taskInfo.curCalQSize;
                    }
                } else { // batch/QSeqIdx 不变
                    nextTask.qOffset = taskInfo.qOffset + headDim;
                    nextTask.curHeadIdx = taskInfo.curHeadIdx + 1;
                    if (nextTask.curHeadIdx % groupSize == 0) {
                        nextTask.kvOffset = taskInfo.kvOffset + headDim;
                    }
                }
            } else { // BNSD, Traverse S-axis first
                nextTask.qOffset = taskInfo.qOffset + taskInfo.curCalQSize * headDim;
                if (taskInfo.curQSeqIdx + taskInfo.curCalQSize == taskInfo.qSeqlen) {
                    nextTask.curQSeqIdx = 0;
                    if (taskInfo.curHeadIdx == numHeads - 1) {
                        nextTask.curBatchIdx = taskInfo.curBatchIdx + 1;
                        nextTask.curHeadIdx = 0;
                        nextTask.kvOffset = taskInfo.kvOffset + taskInfo.kvSeqlen * headDim;
                    } else {
                        nextTask.curHeadIdx = taskInfo.curHeadIdx + 1;
                        if (nextTask.curHeadIdx % groupSize == 0) {
                            nextTask.kvOffset = taskInfo.kvOffset + taskInfo.kvSeqlen * headDim;
                        }
                    }
                } else {
                    nextTask.curQSeqIdx = taskInfo.curQSeqIdx + taskInfo.curCalQSize;
                }
            }
            UpdateTaskInfoCalQSize(blockShapeX, basicQBlockSize, nextTask);
        }

        // Methods
        __aicore__ inline
        BlockSparseAttentionGradKernel() {}

        template <int32_t CORE_TYPE = g_coreType>
        __aicore__ inline
        void operator()(Params const &params);

        template <>
        __aicore__ inline
        void operator()<AscendC::AIC>(Params const &params)
        {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

            __gm__ BlockSparseAttentionGradTilingData *tilingData = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);
            uint32_t batch = tilingData->batch;
            uint32_t numHeads = tilingData->numHeads;
            uint32_t kvHeads = tilingData->kvHeads;
            uint32_t groupSize = numHeads / kvHeads;
            uint32_t headDim = tilingData->headDim;
            uint32_t maxQSeqlen = tilingData->maxQSeqlen;
            uint32_t maxKvSeqlen = tilingData->maxKvSeqlen;
            uint32_t inputLayout = tilingData->inputLayout;
            uint32_t blockShapeX = tilingData->blockShapeX;
            uint32_t blockShapeY = tilingData->blockShapeY;

            uint64_t sOutSize = tilingData->sOutSize;
            uint64_t dPOutSize = tilingData->dPOutSize;
            uint64_t dQOutSize = tilingData->dQOutSize;
            uint64_t dKOutSize = tilingData->dKOutSize;
            uint64_t dVOutSize = tilingData->dVOutSize;

            uint32_t basicQBlockSize = tilingData->basicQBlockSize;
            uint32_t basicKVBlockSize = tilingData->basicKVBlockSize;
            uint32_t taskNumPerCore = tilingData->taskNumPerCore;
            uint32_t tailTaskNum = tilingData->tailTaskNum;
            uint32_t taskLength = tailTaskNum > coreIdx ? taskNumPerCore + 1 : taskNumPerCore;

            if (groupSize == 0 || blockShapeX == 0 || blockShapeY == 0) {
                return;
            }
            // Initialize global tensors
            AscendC::GlobalTensor<ElementA1> gDout;
            gDout.SetGlobalBuffer((__gm__ ElementInput *)params.dout);
            AscendC::GlobalTensor<ElementInput> gQ;
            gQ.SetGlobalBuffer((__gm__ ElementInput *)params.q);
            AscendC::GlobalTensor<ElementInput> gK;
            gK.SetGlobalBuffer((__gm__ ElementInput *)params.k);
            AscendC::GlobalTensor<ElementInput> gV;
            gV.SetGlobalBuffer((__gm__ ElementInput *)params.v);
            AscendC::GlobalTensor<bool> gBlcokSpaseMask;
            gBlcokSpaseMask.SetGlobalBuffer((__gm__ bool *)params.blockSparseMask);
            AscendC::GlobalTensor<int64_t> gActualQseqlen;
            gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
            AscendC::GlobalTensor<int64_t> gActualKvseqlen;
            gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);

            AscendC::GlobalTensor<float> gS;
            gS.SetGlobalBuffer((__gm__ float *)params.workspace);
            AscendC::GlobalTensor<ElementInput> gP;
            gP.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + WORKSPACE_P16_OFFSET));

            AscendC::GlobalTensor<float> gDp;
            gDp.SetGlobalBuffer((__gm__ float *)(params.workspace + sOutSize));
            AscendC::GlobalTensor<ElementInput> gDs;
            gDs.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + sOutSize + WORKSPACE_P16_OFFSET));
            AscendC::GlobalTensor<float> gDq;
            gDq.SetGlobalBuffer((__gm__ float *)(params.workspace + sOutSize + dPOutSize));
            AscendC::GlobalTensor<float> gDk;
            gDk.SetGlobalBuffer((__gm__ float *)(params.workspace + sOutSize + dPOutSize + dQOutSize));
            AscendC::GlobalTensor<float> gDv;
            gDv.SetGlobalBuffer((__gm__ float *)(params.workspace + sOutSize + dPOutSize + dQOutSize + dKOutSize));

            TaskInfo taskInfo[2];
            TaskInfo preTaskInfo;
            initTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData, numHeads, kvHeads, groupSize, headDim,
                maxQSeqlen, maxKvSeqlen, blockShapeX, basicQBlockSize, inputLayout, coreIdx, taskInfo[0]);
            uint32_t qBlockNum = (maxQSeqlen + blockShapeX - 1) / blockShapeX;
            uint32_t kvBlockNum = (maxKvSeqlen + blockShapeY - 1) / blockShapeY;
            uint32_t batchBlocks = numHeads * qBlockNum * kvBlockNum;
            uint32_t headBlocks = qBlockNum * kvBlockNum;

            uint64_t actualStrideQ = headDim;
            uint64_t actualStrideKV = headDim;
            if (inputLayout == 0) {
                actualStrideQ = numHeads * headDim;
                actualStrideKV = kvHeads * headDim;
            }

            BlockMmadBSAG1 blockMmad1(resource);
            BlockMmadBSAG2 blockMmad2(resource, L1_SIZE_OFFSET, PINGPONG_OFFSET_2, true);
            BlockMmadBSAG3 blockMmad3(resource, L1_SIZE_OFFSET * 2, PINGPONG_OFFSET_4, true);
            uint32_t count = 0;
            uint32_t pingpongFlag = 0;
            uint64_t gSOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB;
            uint32_t mmadFlag = 0;

            SetFlag();
            for (uint32_t i = 0; i < taskLength; i++) {
                TaskInfo curInfo = taskInfo[i % 2];
                uint64_t kvBlockOffset = 0;
                uint64_t beginKVOffset = curInfo.kvOffset;
                for (uint32_t idx = 0; idx < kvBlockNum; idx++) {
                    // BlcokSpaseMask shape : [batch, numhead, CeilDiv(maxQSeqlen, blockShapeX), CeilDiv(maxKvSeqlen, blockShapeY)]
                    uint64_t maskOffset = curInfo.curBatchIdx * batchBlocks + curInfo.curHeadIdx * headBlocks + curInfo.curQBlcokIdx * kvBlockNum + idx;
                    if (gBlcokSpaseMask.GetValue(maskOffset)) {
                        uint64_t kvBlockBasicOffset = 0;
                        uint32_t kvBlockSize = (idx != kvBlockNum - 1) ? blockShapeY : maxKvSeqlen - blockShapeY * idx;
                        uint32_t kvLoop = (kvBlockSize + basicKVBlockSize - 1) / basicKVBlockSize;
                        for (uint32_t loop = 0; loop < kvLoop; loop++) {
                            curInfo.curCalKVSize = (loop != kvLoop - 1) ? basicKVBlockSize : kvBlockSize - basicKVBlockSize * loop;
                            if (inputLayout == 0) {
                                curInfo.kvOffset = beginKVOffset + (kvBlockOffset + kvBlockBasicOffset) * kvHeads * headDim;
                            } else {
                                curInfo.kvOffset = beginKVOffset + (kvBlockOffset + kvBlockBasicOffset) * headDim;
                            }
                            curInfo.sOffset = gSOffset + WORKSPACE_BLOCK_SIZE * pingpongFlag;
                            LayoutA1 layoutA1(curInfo.curCalQSize, headDim);
                            LayoutB1 layoutB1(headDim, curInfo.curCalKVSize);
                            LayoutC1 layoutC1(curInfo.curCalQSize, curInfo.curCalKVSize);
                            GemmCoord actualShape1{curInfo.curCalQSize, curInfo.curCalKVSize, headDim};
                            blockMmad1(gQ[curInfo.qOffset],
                                gK[curInfo.kvOffset],
                                gS[curInfo.sOffset],
                                layoutA1,
                                layoutB1,
                                layoutC1,
                                actualShape1,
                                mmadFlag);

                            blockMmad1(gDout[curInfo.qOffset],
                                gV[curInfo.kvOffset],
                                gDp[curInfo.sOffset],
                                layoutA1,
                                layoutB1,
                                layoutC1,
                                actualShape1,
                                mmadFlag);
                            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2VEC);
                            if (count > 0) {
                                AscendC::WaitEvent(VEC2CUBE);
                                LayoutA2 layoutA2(preTaskInfo.curCalQSize, preTaskInfo.curCalKVSize);
                                LayoutB2 layoutB2(preTaskInfo.curCalKVSize, headDim);
                                LayoutC2 layoutC2(preTaskInfo.curCalQSize, headDim);
                                GemmCoord actualShape2{preTaskInfo.curCalQSize, headDim, preTaskInfo.curCalKVSize};

                                blockMmad2(gDs[preTaskInfo.sOffset],
                                    gK[preTaskInfo.kvOffset],
                                    gDq[preTaskInfo.qOffset],
                                    layoutA2,
                                    layoutB2,
                                    layoutC2,
                                    actualShape2,
                                    mmadFlag);

                                LayoutA3 layoutA3(preTaskInfo.curCalKVSize, preTaskInfo.curCalQSize);
                                LayoutB3 layoutB3(preTaskInfo.curCalQSize, headDim);
                                LayoutC3 layoutC3(preTaskInfo.curCalKVSize, headDim);
                                GemmCoord actualShape3{preTaskInfo.curCalKVSize, headDim, preTaskInfo.curCalQSize};

                                blockMmad3(gP[preTaskInfo.sOffset],
                                    gDout[preTaskInfo.qOffset],
                                    gDv[preTaskInfo.kvOffset],
                                    layoutA3,
                                    layoutB3,
                                    layoutC3,
                                    actualShape3,
                                    mmadFlag);

                                blockMmad3(gDs[preTaskInfo.sOffset],
                                    gQ[preTaskInfo.qOffset],
                                    gDk[preTaskInfo.kvOffset],
                                    layoutA3,
                                    layoutB3,
                                    layoutC3,
                                    actualShape3,
                                    mmadFlag);
                            }
                            preTaskInfo = curInfo;
                            preTaskInfo.sOffset = curInfo.sOffset * 2; // float32偏移转成bf16/half偏移
                            pingpongFlag = 1 - pingpongFlag;
                            count++;
                            kvBlockBasicOffset += basicKVBlockSize;
                        }
                    }
                    kvBlockOffset += blockShapeY;
                }
                if (i != taskLength - 1) {
                    updateNextTaskInfo(gActualQseqlen, gActualKvseqlen, numHeads, kvHeads, groupSize, headDim,
                        blockShapeX, basicQBlockSize, inputLayout, taskInfo[i % 2], taskInfo[(i + 1) % 2]);
                }
            }
            AscendC::WaitEvent(VEC2CUBE);
            LayoutA2 layoutA2(preTaskInfo.curCalQSize, preTaskInfo.curCalKVSize);
            LayoutB2 layoutB2(preTaskInfo.curCalKVSize, headDim);
            LayoutC2 layoutC2(preTaskInfo.curCalQSize, headDim);
            GemmCoord actualShape2{preTaskInfo.curCalQSize, headDim, preTaskInfo.curCalKVSize};
            LayoutA3 layoutA3(preTaskInfo.curCalKVSize, preTaskInfo.curCalQSize);
            LayoutB3 layoutB3(preTaskInfo.curCalQSize, headDim);
            LayoutC3 layoutC3(preTaskInfo.curCalKVSize, headDim);
            GemmCoord actualShape3{preTaskInfo.curCalKVSize, headDim, preTaskInfo.curCalQSize};

            blockMmad2(gDs[preTaskInfo.sOffset],
                gK[preTaskInfo.kvOffset],
                gDq[preTaskInfo.qOffset],
                layoutA2,
                layoutB2,
                layoutC2,
                actualShape2,
                mmadFlag);
            blockMmad3(gP[preTaskInfo.sOffset],
                gDout[preTaskInfo.qOffset],
                gDv[preTaskInfo.kvOffset],
                layoutA3,
                layoutB3,
                layoutC3,
                actualShape3,
                mmadFlag);
            blockMmad3(gDs[preTaskInfo.sOffset],
                gQ[preTaskInfo.qOffset],
                gDk[preTaskInfo.kvOffset],
                layoutA3,
                layoutB3,
                layoutC3,
                actualShape3,
                mmadFlag);

            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2POST);
            WaitFlag();
        }

  template <>
        __aicore__ inline
        void operator()<AscendC::AIV>(Params const &params)
        {
            __gm__ BlockSparseAttentionGradTilingData *tilingData = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);
            // pre
            VecPre(params);

            // simply softmax
            VecOp(params);

            AscendC::WaitEvent(CUBE2POST);
            AscendC::SyncAll();

            // post
            VecPost(params);
        }

        __aicore__ inline
        void VecOp(Params const &params)
        {
            uint32_t vecCoreIdx = AscendC::GetBlockIdx(); // vecore 核数idx
            uint32_t coreIdx = vecCoreIdx / 2; // cube 核数idx

            __gm__ BlockSparseAttentionGradTilingData *tilingData = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);
            uint32_t batch = tilingData->batch;
            uint32_t numHeads = tilingData->numHeads;
            uint32_t kvHeads = tilingData->kvHeads;
            uint32_t groupSize = numHeads / kvHeads;
            uint32_t inputLayout = tilingData->inputLayout;
            uint32_t blockShapeX = tilingData->blockShapeX;
            uint32_t blockShapeY = tilingData->blockShapeY;
            uint32_t headDim = tilingData->headDim;
            uint32_t maxQSeqlen = tilingData->maxQSeqlen;
            uint32_t maxKvSeqlen = tilingData->maxKvSeqlen;

            uint32_t basicQBlockSize = tilingData->basicQBlockSize;
            uint32_t basicKVBlockSize = tilingData->basicKVBlockSize;
            uint32_t taskNumPerCore = tilingData->taskNumPerCore;
            uint32_t tailTaskNum = tilingData->tailTaskNum;
            uint32_t taskLengthVec = tailTaskNum > coreIdx ? taskNumPerCore + 1 : taskNumPerCore;

            uint64_t sOutSize = tilingData->sOutSize;
            uint64_t dPOutSize = tilingData->dPOutSize;
            uint64_t dQOutSize = tilingData->dQOutSize;
            uint64_t dKOutSize = tilingData->dKOutSize;
            uint64_t dVOutSize = tilingData->dVOutSize;

            // Initialize global tensors
            AscendC::GlobalTensor<bool> gBlcokSpaseMask;
            gBlcokSpaseMask.SetGlobalBuffer((__gm__ bool *)params.blockSparseMask);
            AscendC::GlobalTensor<int64_t> gActualQseqlen;
            gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
            AscendC::GlobalTensor<int64_t> gActualKvseqlen;
            gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);

            TaskInfo taskInfoVec[2]; // 索引以及shape信息
            TaskInfo preTaskInfo;
            initTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData, numHeads, kvHeads, groupSize, headDim,
                maxQSeqlen, maxKvSeqlen, blockShapeX, basicQBlockSize, inputLayout, coreIdx, taskInfoVec[0]);
            uint32_t qBlockNum = (maxQSeqlen + blockShapeX - 1) / blockShapeX;
            uint32_t kvBlockNum = (maxKvSeqlen + blockShapeY - 1) / blockShapeY;
            uint32_t batchBlocks = numHeads * qBlockNum * kvBlockNum;
            uint32_t headBlocks = qBlockNum * kvBlockNum;

            // uint32_t count = 0;
            uint32_t pingpongFlag = 0;
            uint64_t gSOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB;

            GM_ADDR sftmgGm = params.workspace + sOutSize + dPOutSize + dQOutSize + dKOutSize + dVOutSize;
            SfmgParams SfmgParams(params.dout, params.out, params.actualQseqlen, sftmgGm, params.tiling);
            EpilogueFAGSfmg vecSftmg(SfmgParams);
            EpilogueFAGOp sStmOp;

            PipeBarrier<PIPE_ALL>(); // 等待上一步vec完成
            for (uint32_t i = 0; i < taskLengthVec; i++) {
                TaskInfo curInfo = taskInfoVec[i % 2];
                uint64_t beginKVOffset = curInfo.kvOffset;

                // idx, 计算的行数
                // 一个ai core里的aiv 0执行的行数：actualRow / 2 + actualRow % 2
                // 一个ai core里的aiv 1执行的行数：actualRow / 2
                if (vecCoreIdx % 2 != 0) {
                    vecSftmg(curInfo.qOffset / headDim + curInfo.curCalQSize / 2 + curInfo.curCalQSize % 2,
                        curInfo.curCalQSize / 2);
                } else {
                    vecSftmg(curInfo.qOffset / headDim, curInfo.curCalQSize / 2 + curInfo.curCalQSize % 2);
                }

                for (uint32_t idx = 0; idx < kvBlockNum; idx++) {
                    // BlcokSpaseMask shape : [batch, numhead, CeilDiv(maxQSeqlen, blockShapeX), CeilDiv(maxKvSeqlen, blockShapeY)]
                    uint64_t maskOffset = curInfo.curBatchIdx * batchBlocks + curInfo.curHeadIdx * headBlocks + curInfo.curQBlcokIdx * kvBlockNum + idx;
                    if (gBlcokSpaseMask.GetValue(maskOffset)) {
                        uint32_t kvBlockSize = (idx != kvBlockNum - 1) ? blockShapeY : maxKvSeqlen - blockShapeY * idx;
                        uint32_t kvLoop = (kvBlockSize + basicKVBlockSize - 1) / basicKVBlockSize;
                        for (uint32_t loop = 0; loop < kvLoop; loop++) {
                            curInfo.curCalKVSize = (loop != kvLoop - 1) ? basicKVBlockSize : kvBlockSize - basicKVBlockSize * loop;
                            curInfo.sOffset = gSOffset + WORKSPACE_BLOCK_SIZE * pingpongFlag;

                            uint64_t actualRow = curInfo.curCalQSize;
                            uint64_t actualCol = curInfo.curCalKVSize;
                            uint64_t processNums = curInfo.curCalQSize * curInfo.curCalKVSize;
                            uint64_t curCoreBatch = curInfo.curBatchIdx;
                            uint64_t curCoreN1Idx = curInfo.curHeadIdx; // 当前q的N1的idx
                            uint64_t curCoreS1Idx = curInfo.curQSeqIdx; // 当前q的s1的idx
                            uint64_t curT1Idx = 0;
                            uint64_t sOutSize = tilingData->sOutSize;
                            uint64_t dPOutSize = tilingData->dPOutSize;
                            uint64_t dQOutSize = tilingData->dQOutSize;
                            uint64_t dKOutSize = tilingData->dKOutSize;
                            uint64_t dVOutSize = tilingData->dVOutSize;

                            // 默认一个ai core里的aiv 0的实际计算的行数和偏移地址
                            uint64_t executeRow = actualRow / 2 + actualRow % 2;
                            uint64_t coreOffset = 0; // ai core 的每个vector core 的偏移
                            uint64_t curVecCoreS1Idx = curCoreS1Idx;

                            // 一个ai core里的aiv 0执行的行数：actualRow / 2 + actualRow % 2
                            // 一个ai core里的aiv 1执行的行数：actualRow / 2
                            if (vecCoreIdx % 2 != 0) {
                                coreOffset += (actualRow / 2 + actualRow % 2) * actualCol;
                                curVecCoreS1Idx += (actualRow / 2 + actualRow % 2);
                                executeRow = actualRow / 2;
                            }

                            uint64_t vector16Soffset =
                                (curInfo.sOffset * 2 + WORKSPACE_P16_OFFSET_ELEMENT + coreOffset) *
                                sizeof(ElementInput);
                            uint64_t vector32Soffset = (curInfo.sOffset + coreOffset) * sizeof(float);

                            GM_ADDR s = params.workspace + vector32Soffset;
                            GM_ADDR softmaxLse = params.softmaxLse;
                            GM_ADDR dp = params.workspace + sOutSize + vector32Soffset;
                            GM_ADDR actualSeqQlen = params.actualQseqlen;
                            GM_ADDR actualSeqKvlen = params.actualKvseqlen;
                            GM_ADDR sftmgGm = params.workspace + sOutSize + dPOutSize + dQOutSize + dKOutSize + dVOutSize;
                            GM_ADDR pWorkspace = params.workspace + vector16Soffset; // 连续
                            GM_ADDR dsWorkspace = params.workspace + sOutSize + vector16Soffset; // 连续
                            GM_ADDR tiling = params.tiling;

                            AscendC::WaitEvent(CUBE2VEC);

                            SfmParams sfmParams(s,
                                softmaxLse,
                                dp,
                                actualSeqQlen,
                                actualSeqKvlen,
                                sftmgGm,
                                pWorkspace,
                                dsWorkspace,
                                tiling,
                                executeRow,
                                actualCol,
                                executeRow * actualCol,
                                curCoreBatch,
                                curCoreN1Idx,
                                curVecCoreS1Idx,
                                curT1Idx);
                            sStmOp(sfmParams);

                            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(VEC2CUBE);

                            preTaskInfo = curInfo;
                            pingpongFlag = 1 - pingpongFlag;
                            preTaskInfo.sOffset = curInfo.sOffset * 2; // float32偏移转成bf16/half偏移
                        }
                    }
                }

                if (i != taskLengthVec - 1) {
                    updateNextTaskInfo(gActualQseqlen, gActualKvseqlen, numHeads, kvHeads, groupSize, headDim,
                        blockShapeX, basicQBlockSize, inputLayout, taskInfoVec[i % 2], taskInfoVec[(i + 1) % 2]);
                }
            }
        }

        __aicore__ inline
        void VecPost(Params const &params)
        {
            __gm__ BlockSparseAttentionGradTilingData *tilingData = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);

            uint64_t sOutSize = tilingData->sOutSize;
            uint64_t dPOutSize = tilingData->dPOutSize;
            uint64_t dQOutSize = tilingData->dQOutSize;
            uint64_t dKOutSize = tilingData->dKOutSize;
            uint64_t dVOutSize = tilingData->dVOutSize;

            GM_ADDR gDqGm = params.workspace + sOutSize + dPOutSize;
            GM_ADDR gDkGm = params.workspace + sOutSize + dPOutSize + dQOutSize;
            GM_ADDR gDvGm = params.workspace + sOutSize + dPOutSize + dQOutSize + dKOutSize;
            GM_ADDR actualSeqQlen = params.actualQseqlen;
            GM_ADDR actualSeqKvlen = params.actualKvseqlen;

            PostParams postParams(params.dq, params.dk, params.dv, gDqGm, gDkGm, gDvGm, params.tiling, actualSeqQlen, actualSeqKvlen);
            EpilogueFAGPost vecPost(postParams);
            vecPost();
        }

        __aicore__ inline
        void VecPre(Params const &params)
        {
            __gm__ BlockSparseAttentionGradTilingData *tilingData = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);

            uint64_t sOutSize = tilingData->sOutSize;
            uint64_t dPOutSize = tilingData->dPOutSize;
            uint64_t dQOutSize = tilingData->dQOutSize;
            uint64_t dKOutSize = tilingData->dKOutSize;
            uint64_t dVOutSize = tilingData->dVOutSize;


            GM_ADDR gDqWrkGm = params.workspace + sOutSize + dPOutSize;
            GM_ADDR gDkWrkGm = params.workspace + sOutSize + dPOutSize + dQOutSize;
            GM_ADDR gDvWrkGm = params.workspace + sOutSize + dPOutSize + dQOutSize + dKOutSize;

            PreParams preParms(gDqWrkGm, gDkWrkGm, gDvWrkGm, params.tiling);
            EpilogueFAGPre vecPre(preParms);
            vecPre();
        }

    private:
        NpuArch::Arch::Resource<ArchTag> resource;
    };

} // namespace BSA

#endif // BLOCK_SPARSE_ATTENTION_GRAD_KERNEL_H

