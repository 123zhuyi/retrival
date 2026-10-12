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
 * \file prompt_flash_attention_kvcache.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_KVCACHE_H
#define PROMPT_FLASH_ATTENTION_KVCACHE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "prompt_flash_attention_comm.h"
#include "prompt_flash_attention_sparse.h"
#include "../prompt_flash_attention_tiling_regbase.h"

using namespace matmul;

template<typename PFAT>
__aicore__ inline void InitConstParam(ConstParam &constParam, 
    const optiling::PromptFlashAttentionTilingDataV2* tilingData)
{
    constParam.tmpBlockIdx = GetBlockIdx();
    constParam.subBlockIdx = constParam.tmpBlockIdx % 2; // 2: One numBlocks has 2 vectorCore

    constParam.preTokens = tilingData->promptAttentionBaseParams.preTokens;
    constParam.nextTokens = tilingData->promptAttentionBaseParams.nextTokens;

    constParam.isRowInvalid = tilingData->promptAttentionBaseParams.isRowInvalid;

    constParam.isIFA = tilingData->promptAttentionBaseParams.isIFA;

    constParam.seqSize = tilingData->promptAttentionBaseParams.seqSize;
    constParam.seqInnerSize = tilingData->promptAttentionBaseParams.seqInnerSize;

    constParam.singleProcessSInnerSize = tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize;
    constParam.singleProcessSOuterSizeWhole = tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize;
    constParam.singleProcessCubeSOuterSizeWhole = tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize * 2; // CubeSouterSize = 2 * SouterSize

    constParam.headSize = tilingData->promptAttentionBaseParams.headSize;
    constParam.ropeHeadSize = tilingData->promptAttentionBaseParams.ropeHeadSize;
    constParam.qkHeadSize = tilingData->promptAttentionBaseParams.qkHeadSize;
    constParam.vHeadSize = tilingData->promptAttentionBaseParams.vHeadSize;
    constParam.headNumSize = tilingData->promptAttentionBaseParams.headNumSize;
    constParam.headNumRatio = tilingData->promptAttentionBaseParams.headNumRatio;
    constParam.gOfMla = tilingData->promptAttentionBaseParams.gOfMla;
    constParam.kvHeadNumSize = constParam.headNumSize / constParam.headNumRatio;
    if constexpr (PFAT::IFA_MLA) {
        constParam.multiHeadQRope = constParam.ropeHeadSize * constParam.headNumSize;
        constParam.multiHeadKRope = constParam.multiHeadQRope / constParam.headNumRatio;
        constParam.multiHeadQ = constParam.headSize * constParam.headNumSize;
        constParam.rStride = (PFAT::layout == PFALayout::BNSD) ? constParam.ropeHeadSize :
                             constParam.ropeHeadSize * constParam.kvHeadNumSize;
    } else {
        constParam.multiHeadQ = constParam.qkHeadSize * constParam.headNumSize;
    }
    constParam.multiHeadK = constParam.multiHeadQ / constParam.headNumRatio;
    constParam.multiHeadV = constParam.kvHeadNumSize * constParam.vHeadSize;
    constParam.multiHeadOut = constParam.headNumSize * constParam.vHeadSize;
    constParam.isKvContinuous = tilingData->promptAttentionBaseParams.isKvContinuous;
    constParam.actualSeqLenSize = tilingData->promptAttentionBaseParams.actualSeqLengthsSize;
    constParam.actualSeqLenKVSize = tilingData->promptAttentionBaseParams.actualSeqLengthsKVSize;
    constParam.isActualLenDimsNull = tilingData->promptAttentionBaseParams.isActualSeqLengthsNull ? true : false;
    constParam.isActualLenDimsKVNull = tilingData->promptAttentionBaseParams.isActualSeqLengthsKVNull ? true : false;

    constParam.attentionMaskType = tilingData->promptAttentionBaseParams.sparseMode;
    constParam.attentionMaskStride = tilingData->promptAttentionBaseParams.maskKVsSize;
    constParam.maskQsSize = tilingData->promptAttentionBaseParams.maskQsSize;
    constParam.attenMaskBatch = tilingData->promptAttentionSingleCoreParams.attenMaskBatch;
    constParam.maskTypeByteNum = tilingData->promptAttentionBaseParams.maskTypeByteNum;

    constParam.pseShiftS1Size = tilingData->promptAttentionBaseParams.pseShiftS1Size;
    constParam.pseShiftS2Size = tilingData->promptAttentionBaseParams.pseShiftS2Size;
    constParam.pseShiftBatch = tilingData->promptAttentionSingleCoreParams.pseShiftBatch;
    constParam.pseShiftTypeByteNum = tilingData->promptAttentionBaseParams.pseShiftTypeByteNum;
    constParam.pseShiftStride = tilingData->promptAttentionBaseParams.pseShiftS2Size;

    constParam.isQHasLeftPadding = (tilingData->promptAttentionBaseParams.isQHasLeftPadding == 1) ? true : false;
    constParam.isKVHasLeftPadding = (tilingData->promptAttentionBaseParams.isKVHasLeftPadding == 1) ? true : false;


    // service mm1
    constParam.bmm1TilingDataRectM = tilingData->bmm1TilingDataRect.M;
    constParam.bmm1TilingDataRectN = tilingData->bmm1TilingDataRect.N;
    constParam.bmm1TilingDataRectKa = tilingData->bmm1TilingDataRect.Ka;
    constParam.bmm1TilingDataRectKb =  tilingData->bmm1TilingDataRect.Kb;

    // service mm2
    constParam.bmm2TilingDataRectN = tilingData->bmm2TilingDataRect.N;                
    constParam.bmm2TilingDataRectKb = tilingData->bmm2TilingDataRect.Kb;

    // pageAttention
    constParam.blockTableDim2 = tilingData->promptAttentionBaseParams.blockTableDim2;
    constParam.blockSize = tilingData->promptAttentionBaseParams.blockSize;
    constParam.paLayoutType = tilingData->promptAttentionBaseParams.PAlayoutType;
    constParam.paBlockNumSum = tilingData->promptAttentionBaseParams.PABlockNumSum;

    // service vector1
    constParam.scaleValue = static_cast<float>(tilingData->promptAttentionBaseParams.scaleValue);
    constParam.softmaxFlashTilingDataSrcM = tilingData->softmaxFlashTilingDataRect.srcM;
    constParam.softmaxFlashTilingDataSrcK = tilingData->softmaxFlashTilingDataRect.srcK;
    constParam.softmaxFlashTilingDataSrcSize = tilingData->softmaxFlashTilingDataRect.srcSize;
    constParam.typeByteNum = tilingData->promptAttentionBaseParams.typeByteNum;

    // service vector2
    constParam.outputTypeByteNum = tilingData->promptAttentionBaseParams.outputTypeByteNum;
    constParam.softmaxTypeByteNum = tilingData->promptAttentionBaseParams.softmaxTypeByteNum;
    constParam.isBSNDOut = tilingData->promptAttentionBaseParams.isBSNDOut;

    // lse output
    constParam.isSoftmaxLseEnable = tilingData->promptAttentionBaseParams.isSoftMaxLseEnable;
    constParam.totalSoftmaxLseOutputSize = tilingData->promptAttentionInitOutputParams.totalSoftMaxLseOutputSize;
}

template<typename PFAT>
__aicore__ inline void ComputeParamCore(RunParam& runParam, ConstParam& constParam,
    const optiling::PromptFlashAttentionTilingDataV2* tilingData, uint32_t coreIdx)
{
    constParam.sNum = tilingData->promptAttentionBaseParams.dimNumOfseq;
    uint32_t splitCoreIdx = coreIdx;
    if constexpr (PFAT::isSplitCoreByCube) {
        // cube视角分核时，按cube核的index来获取切块信息
        splitCoreIdx = coreIdx / 2;
    }
    constParam.sIdStart = tilingData->promptAttentionSeqParams.actualCoreNums[splitCoreIdx];
    constParam.sIdEnd = tilingData->promptAttentionSeqParams.singleCoreHeadNumSize[splitCoreIdx];
    constParam.outerLoopStart = tilingData->promptAttentionSeqParams.coreSeqPosStart[splitCoreIdx];
    constParam.outerLoopEnd = tilingData->promptAttentionSeqParams.coreSeqPosEnd[splitCoreIdx];
    constParam.nLoopStart = tilingData->promptAttentionSeqParams.CoreHeadNumTail[splitCoreIdx];
    constParam.nLoopEnd = tilingData->promptAttentionSeqParams.actualS1[splitCoreIdx];

    runParam.isLast = false;
    runParam.actualSeqLengthsIdx = 0;
}

template<typename PFAT>
__aicore__ inline void ComputeParamN(RunParam &runParam, const ConstParam &constParam, int32_t sIdx)
{
    if (sIdx != constParam.sIdEnd - 1) {
        runParam.tmpNLoopEnd = constParam.headNumSize;
    } else {
        runParam.tmpNLoopEnd = constParam.nLoopEnd;
        runParam.isLast = true;
    }
}

__aicore__ inline void InitQueryLeftPaddingSize(RunParam &runParam, const ConstParam &constParam, int64_t actualSeqLengthPerBatch)
{
    if (!constParam.isQHasLeftPadding) {
        runParam.queryLeftPaddingSize = 0;
    } else {
        int64_t qLeftPaddingSize = constParam.seqSize - actualSeqLengthPerBatch - constParam.queryRightPaddingSize;
        runParam.queryLeftPaddingSize = qLeftPaddingSize > 0 ? qLeftPaddingSize : 0;
    }
}

__aicore__ inline void InitKVLeftPaddingSize(RunParam &runParam, const ConstParam &constParam, int64_t actualSeqLengthKVPerBatch)
{
    if (!constParam.isKVHasLeftPadding) {
        runParam.kvLeftPaddingSize = 0;
    } else {
        int64_t kvLeftPaddingSize = constParam.seqInnerSize - actualSeqLengthKVPerBatch - constParam.kvRightPaddingSize;
        runParam.kvLeftPaddingSize = kvLeftPaddingSize > 0 ? kvLeftPaddingSize : 0;
    }
}

template<typename PFAT>
__aicore__ inline void GetSingleCoreParam(RunParam &runParam, const ConstParam &constParam, int32_t sIdx,
    GlobalTensor<typename PFAT::kvInputType>& keyGm, GlobalTensor<int64_t>& actualSeqLengthsGm,
    GlobalTensor<int64_t>& actualSeqLengthsKVGm)
{
    // TensorList场景获取不同batch的KvSeq长度
    if (constParam.isKvContinuous == 0) {
        ListTensorDesc keyListTensorDesc((__gm__ void*)keyGm.GetPhyAddr());
        AscendC::TensorDesc<__gm__ uint8_t> kvTensorDesc;
        uint64_t dimInfo[4];
        kvTensorDesc.SetShapeAddr(&dimInfo[0]);
        keyListTensorDesc.GetDesc(kvTensorDesc, sIdx);
        if constexpr (PFAT::layout == PFALayout::BNSD) {
            runParam.s2InCurrentBatch = kvTensorDesc.GetShape(2);
        } else {
            runParam.s2InCurrentBatch = kvTensorDesc.GetShape(1);
        }
    }

    int64_t actualSeqLengthPerBatch = 0;
    int64_t actualSeqLengthKVPerBatch = 0;
    int64_t actualSeqMin = 1;
    int64_t actualSeqKVMin = 1;
    if (constParam.isActualLenDimsNull) {
        actualSeqLengthPerBatch = constParam.seqSize;
        if constexpr (PFAT::IFA_MLA) {
            runParam.actualSeqLengthOfMlaPerBatch = actualSeqLengthPerBatch / constParam.gOfMla;
        }
    } else {
        if constexpr (PFAT::IFA_MLA && PFAT::layout == PFALayout::BSH) {
            runParam.actualSeqLengthOfMlaPerBatch = ((constParam.actualSeqLenSize == actualSeqMin) ?
                actualSeqLengthsGm.GetValue(0) : actualSeqLengthsGm.GetValue(sIdx));
            actualSeqLengthPerBatch = runParam.actualSeqLengthOfMlaPerBatch * constParam.gOfMla;
        } else if constexpr (PFAT::IFA_MLA && PFAT::layout == PFALayout::TND) {
            runParam.actualSeqLengthOfMlaPerBatch = ((sIdx == 0) ? actualSeqLengthsGm.GetValue(0) :
                actualSeqLengthsGm.GetValue(sIdx) - actualSeqLengthsGm.GetValue(sIdx - 1));
            actualSeqLengthPerBatch = runParam.actualSeqLengthOfMlaPerBatch * constParam.gOfMla;
        } else if constexpr (PFAT::IFA_MLA && PFAT::layout == PFALayout::BNSD) {
            actualSeqLengthPerBatch = constParam.seqSize;
            runParam.actualSeqLengthOfMlaPerBatch = (constParam.actualSeqLenSize == actualSeqMin) ?
                actualSeqLengthsGm.GetValue(0) : actualSeqLengthsGm.GetValue(sIdx);
        } else if constexpr (PFAT::layout == PFALayout::TND) {
            actualSeqLengthPerBatch = (sIdx == 0) ? actualSeqLengthsGm.GetValue(0) :
                actualSeqLengthsGm.GetValue(sIdx) - actualSeqLengthsGm.GetValue(sIdx - 1);
        } else {
            actualSeqLengthPerBatch = (constParam.actualSeqLenSize == actualSeqMin) ? actualSeqLengthsGm.GetValue(0) :
                actualSeqLengthsGm.GetValue(sIdx);
        }
    }
    if (constParam.isActualLenDimsKVNull) {
        actualSeqLengthKVPerBatch = (constParam.isKvContinuous == 1) ? constParam.seqInnerSize :
            runParam.s2InCurrentBatch;
    } else {
        if constexpr (PFAT::layout == PFALayout::TND) {
            actualSeqLengthKVPerBatch = actualSeqLengthsKVGm.GetValue(sIdx);
            if ((sIdx > 0) && !(PFAT::MM_TYPE == PFAMatMulType::MM_PA ||
                PFAT::MM_TYPE == PFAMatMulType::MM_PA_D512 || PFAT::MM_TYPE == PFAMatMulType::MM_IFA_MLA_PA)) {
                actualSeqLengthKVPerBatch -= actualSeqLengthsKVGm.GetValue(sIdx - 1);
            }
        } else {
            actualSeqLengthKVPerBatch = (constParam.actualSeqLenKVSize == actualSeqKVMin) ? 
                actualSeqLengthsKVGm.GetValue(0) : actualSeqLengthsKVGm.GetValue(sIdx);
        }
    }

    InitQueryLeftPaddingSize(runParam, constParam, actualSeqLengthPerBatch);
    InitKVLeftPaddingSize(runParam, constParam, actualSeqLengthKVPerBatch);

    runParam.actualSeqLengthPerBatch = actualSeqLengthPerBatch;
    runParam.actualSeqLengthKVPerBatch = actualSeqLengthKVPerBatch;
    GetSparseParam<PFAT>(runParam, constParam);

    if (constParam.isKvContinuous == 1) {
        runParam.actualSeqLengthPerBatch = 
            (runParam.actualSeqLengthPerBatch > runParam.actualSeqLengthKVPerBatch + runParam.preTokensPerBatch) ?
            runParam.actualSeqLengthKVPerBatch + runParam.preTokensPerBatch : runParam.actualSeqLengthPerBatch;
    } else {
        runParam.actualSeqLengthPerBatch = ((int64_t)runParam.actualSeqLengthPerBatch >
            runParam.s2InCurrentBatch + (int64_t)runParam.preTokensPerBatch) ?
            runParam.s2InCurrentBatch + runParam.preTokensPerBatch : runParam.actualSeqLengthPerBatch;
    }

    // 计算S1的尾块大小，非对齐
    runParam.actualSeqLengthPerBatch = (runParam.nextTokensPerBatch >= 0) ? runParam.actualSeqLengthPerBatch :
        (runParam.actualSeqLengthPerBatch + runParam.nextTokensPerBatch);
    runParam.singleProcessSOuterSizeTail =
        (runParam.actualSeqLengthPerBatch % constParam.singleProcessSOuterSizeWhole != 0) ? 
        runParam.actualSeqLengthPerBatch % constParam.singleProcessSOuterSizeWhole :
        constParam.singleProcessSOuterSizeWhole;

    runParam.singleProcessCubeSOuterSizeTail =
        (runParam.actualSeqLengthPerBatch % constParam.singleProcessCubeSOuterSizeWhole != 0) ?
        runParam.actualSeqLengthPerBatch % constParam.singleProcessCubeSOuterSizeWhole :
        constParam.singleProcessCubeSOuterSizeWhole;

    // 计算S2的尾块大小，非对齐
    runParam.unalignSInner = (runParam.actualSeqLengthKVPerBatch % constParam.singleProcessSInnerSize != 0) ?
        runParam.actualSeqLengthKVPerBatch % constParam.singleProcessSInnerSize :
        constParam.singleProcessSInnerSize;
    // s2方向尾块长度，按64对齐
    runParam.singleProcessSInnerSizeTail = (runParam.unalignSInner + 64 - 1) / 64 * 64;

    // 计算S2的切块个数，用于判断是否为尾块
    runParam.maxInnerLoopTimes = (runParam.actualSeqLengthKVPerBatch + constParam.singleProcessSInnerSize - 1) /
        constParam.singleProcessSInnerSize;

    // s2方向尾块长度，按mask数据类型对齐(maskTypeByteNum)
    runParam.maskInnerTailAlign = (runParam.unalignSInner + constParam.maskTypeByteNum - 1) /
        constParam.maskTypeByteNum * constParam.maskTypeByteNum;
    // s2方向尾块时，mask的pad大小
    runParam.padSize = runParam.maskInnerTailAlign - runParam.unalignSInner;

    if (constParam.pseShiftTypeByteNum != 0) {
        // s2方向尾块长度，按pse数据类型对齐(pseShiftTypeByteNum)
        runParam.pseShiftInnerTailAlign = (runParam.unalignSInner + constParam.pseShiftTypeByteNum - 1) /
            constParam.pseShiftTypeByteNum * constParam.pseShiftTypeByteNum;
        // s2方向尾块时，mask的pad大小                                        
        runParam.pseShiftPadSize = runParam.pseShiftInnerTailAlign - runParam.unalignSInner;
    }

    if constexpr (PFAT::layout == PFALayout::BSH) {
        runParam.multiSeqOffset = sIdx * constParam.seqSize * constParam.multiHeadQ + runParam.queryLeftPaddingSize * constParam.multiHeadQ;
    } else if constexpr (PFAT::layout == PFALayout::TND) {
        runParam.multiSeqOffset = (sIdx == 0) ? 0 : actualSeqLengthsGm.GetValue(sIdx - 1) * constParam.multiHeadQ;
        if constexpr (PFAT::IFA_MLA) {
            runParam.multiSeqOffset *= constParam.gOfMla;
        }
    } else {
        runParam.multiSeqOffset = sIdx * constParam.seqSize * constParam.multiHeadQ + runParam.queryLeftPaddingSize * constParam.qkHeadSize;
    }
    if constexpr (PFAT::IFA_MLA) {
        if constexpr (PFAT::layout == PFALayout::TND) {
            runParam.qRopeBOffset = (sIdx == 0) ? 0 : actualSeqLengthsGm.GetValue(sIdx - 1) * constParam.multiHeadQRope;
            runParam.qRopeBOffset *= constParam.gOfMla;
        } else {
            runParam.qRopeBOffset = sIdx * constParam.seqSize * constParam.multiHeadQRope;
        }
    }

    // PageAttention时mm需要使用
    runParam.taskBatch = sIdx;
}

template<typename PFAT>
__aicore__ inline void GetKeyCoreOffsetParam(RunParam &runParam, const ConstParam &constParam, int32_t sIdx, GlobalTensor<int64_t>& actualSeqLengthsKVGm)
{
    uint64_t keyInnerOffsetSize = 0;
    if constexpr (PFAT::layout == PFALayout::BSH) {
        if (constParam.isKvContinuous == 1) {
            // 这是从KV的GM 到 每一个batch的开始地址 所需要的偏移量，即每一个batch需要偏移前面一整个batch的长度
            keyInnerOffsetSize = sIdx * constParam.seqInnerSize * constParam.multiHeadK +
                runParam.kvLeftPaddingSize * constParam.multiHeadK;
        } else {
            //KV tensorlist场景下，我们能直接将KV的GM设置成当前batch的开始地址，所以偏移量总是0
            keyInnerOffsetSize = 0;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.batchNOffset /
            constParam.headNumRatio * constParam.qkHeadSize;
    } else if constexpr (PFAT::layout == PFALayout::TND) {
        if (!(PFAT::MM_TYPE == PFAMatMulType::MM_PA || PFAT::MM_TYPE == PFAMatMulType::MM_PA_D512)) {
            keyInnerOffsetSize = (sIdx == 0) ? 0 : actualSeqLengthsKVGm.GetValue(sIdx - 1) * constParam.multiHeadK;
        } else {
            keyInnerOffsetSize = sIdx * constParam.seqInnerSize * constParam.multiHeadK;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.batchNOffset /
            constParam.headNumRatio * constParam.qkHeadSize;
    } else {
        uint64_t headStrideK = 0;
        if (constParam.isKvContinuous == 1) {
            headStrideK = constParam.qkHeadSize * constParam.seqInnerSize;
            keyInnerOffsetSize = sIdx * constParam.kvHeadNumSize * headStrideK +
                runParam.kvLeftPaddingSize * constParam.qkHeadSize;
        } else {
            headStrideK = constParam.qkHeadSize * runParam.s2InCurrentBatch;
            keyInnerOffsetSize = 0;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.batchNOffset /
            constParam.headNumRatio * headStrideK;
    }
}

template<typename PFAT>
__aicore__ inline void GetValueCoreOffsetParam(RunParam &runParam, const ConstParam &constParam, int32_t sIdx, GlobalTensor<int64_t>& actualSeqLengthsKVGm)
{
    uint64_t valueInnerOffsetSize = 0;
    if constexpr (PFAT::layout == PFALayout::BSH) {
        if (constParam.isKvContinuous == 1) {
            // 这是从KV的GM 到 每一个batch的开始地址 所需要的偏移量，即每一个batch需要偏移前面一整个batch的长度
            valueInnerOffsetSize = sIdx * constParam.seqInnerSize * constParam.multiHeadV +
                runParam.kvLeftPaddingSize * constParam.multiHeadV;
        } else {
            //KV tensorlist场景下，我们能直接将KV的GM设置成当前batch的开始地址，所以偏移量总是0
            valueInnerOffsetSize = 0;
        }
        runParam.valueCoreOffset = valueInnerOffsetSize + runParam.batchNOffset /
            constParam.headNumRatio * constParam.vHeadSize;
    } else if constexpr (PFAT::layout == PFALayout::TND) {
        if (!(PFAT::MM_TYPE == PFAMatMulType::MM_PA || PFAT::MM_TYPE == PFAMatMulType::MM_PA_D512)) {
            valueInnerOffsetSize = (sIdx == 0) ? 0 : actualSeqLengthsKVGm.GetValue(sIdx - 1) * constParam.multiHeadV;
        } else {
            valueInnerOffsetSize = sIdx * constParam.seqInnerSize * constParam.multiHeadV;
        }
        runParam.valueCoreOffset = valueInnerOffsetSize + runParam.batchNOffset /
            constParam.headNumRatio * constParam.vHeadSize;
    } else {
        uint64_t headStrideV = 0;
        if (constParam.isKvContinuous == 1) {
            headStrideV = constParam.vHeadSize * constParam.seqInnerSize;
            valueInnerOffsetSize = sIdx * constParam.kvHeadNumSize * headStrideV +
                runParam.kvLeftPaddingSize * constParam.vHeadSize;
        } else {
            headStrideV = constParam.vHeadSize * runParam.s2InCurrentBatch;
            valueInnerOffsetSize = 0;
        }
        runParam.valueCoreOffset = valueInnerOffsetSize + runParam.batchNOffset /
            constParam.headNumRatio * headStrideV;
    }
    if constexpr (PFAT::PFA_MLA) {
        GetKeyCoreOffsetParam<PFAT>(runParam, constParam, sIdx, actualSeqLengthsKVGm);
    } else {
        runParam.keyCoreOffset = runParam.valueCoreOffset;
    }
}

template<typename PFAT>
__aicore__ inline void GetKeyRopeCoreOffsetParam(RunParam &runParam, const ConstParam &constParam, int32_t sIdx, GlobalTensor<int64_t>& actualSeqLengthsKVGm)
{
    uint64_t kRopeInnerOffsetSize = 0;
    if constexpr (PFAT::layout == PFALayout::BSH) {
        kRopeInnerOffsetSize = sIdx * constParam.seqInnerSize * constParam.multiHeadKRope;
        runParam.kRopeNBGOffset = kRopeInnerOffsetSize + runParam.batchNOffset /
            constParam.headNumRatio * constParam.ropeHeadSize;
    } else if constexpr (PFAT::layout == PFALayout::TND) {
        if (!(PFAT::MM_TYPE == PFAMatMulType::MM_PA || PFAT::MM_TYPE == PFAMatMulType::MM_PA_D512)) {
            kRopeInnerOffsetSize = (sIdx == 0)? 0 : actualSeqLengthsKVGm.GetValue(sIdx - 1) * constParam.multiHeadKRope;
        } else {
            kRopeInnerOffsetSize = sIdx * constParam.seqInnerSize * constParam.multiHeadKRope;
        }
        runParam.kRopeNBGOffset = kRopeInnerOffsetSize + runParam.batchNOffset /
            constParam.headNumRatio * constParam.ropeHeadSize;
    } else {
        uint64_t headStrideKV = 0;
        headStrideKV = constParam.ropeHeadSize * constParam.seqInnerSize;
        kRopeInnerOffsetSize = sIdx * constParam.kvHeadNumSize * headStrideKV;
        runParam.kRopeNBGOffset = kRopeInnerOffsetSize + runParam.batchNOffset /
            constParam.headNumRatio * headStrideKV;
    }
}

template<typename PFAT>
__aicore__ inline void ComputeParamBatch(RunParam &runParam, const ConstParam &constParam, int32_t sIdx,
    GlobalTensor<typename PFAT::kvInputType>& keyGm, GlobalTensor<int64_t>& actualSeqLengthsGm,
    GlobalTensor<int64_t>& actualSeqLengthsKVGm)
{
    GetSingleCoreParam<PFAT>(runParam, constParam, sIdx, keyGm, actualSeqLengthsGm, actualSeqLengthsKVGm);
    GetValueCoreOffsetParam<PFAT>(runParam, constParam, sIdx, actualSeqLengthsKVGm);
    if constexpr (PFAT::IFA_MLA) {
        GetKeyRopeCoreOffsetParam<PFAT>(runParam, constParam, sIdx, actualSeqLengthsKVGm);
    }
}

template<typename PFAT>
__aicore__ inline void ComputeS1LoopInfo(RunParam &runParam, const ConstParam &constParam, int32_t loopNIdx)
{
    int32_t sOuterSize = constParam.singleProcessSOuterSizeWhole;
    if constexpr (PFAT::isSplitCoreByCube) {
        sOuterSize = constParam.singleProcessCubeSOuterSizeWhole;
    }
    runParam.sOuterBlockNum = (runParam.actualSeqLengthPerBatch + sOuterSize - 1) / sOuterSize;
    // 不是最后一个N, 也不是batch循环的结束, 将 S1方向循环的结束配置为 outerLoopEnd
    if (runParam.isLast && (loopNIdx == (runParam.tmpNLoopEnd - 1))) {
        runParam.tmpOuterLoopEnd = constParam.outerLoopEnd;
    } else { // 最后一个N或者, 或者batch循环的结束, 将 S1方向循环的结束配置为 sOuterBlockNum
        runParam.tmpOuterLoopEnd = runParam.sOuterBlockNum;
    }
}

template<typename PFAT>
__aicore__ inline void ComputeSouterParam(RunParam &runParam, const ConstParam &constParam,
    int sIdx, uint32_t sOuterLoopIdx)
{
    if constexpr (PFAT::isSplitCoreByCube) {        
        int64_t cubeSOuterOffset = sOuterLoopIdx * constParam.singleProcessCubeSOuterSizeWhole;
        cubeSOuterOffset += (runParam.nextTokensPerBatch < 0) ? -runParam.nextTokensPerBatch : 0;
        if (sOuterLoopIdx == runParam.sOuterBlockNum - 1) {
            uint32_t singleProcessCubeSOuterSizeTailV1;
            if constexpr (PFAT::useDN) { // DN场景下bmm1 split N轴，此时S1方向尾块会向上对齐32再分到vec核上，且优先分满vec0核
                singleProcessCubeSOuterSizeTailV1 = ((runParam.singleProcessCubeSOuterSizeTail + 31) >> 5 << 5) / 2;
                uint32_t vec0SOuterSizeTail = (runParam.singleProcessCubeSOuterSizeTail > 16) ?
                    singleProcessCubeSOuterSizeTailV1 : runParam.singleProcessCubeSOuterSizeTail; // 当S1方向尾块小于16时，有效数据全部分布在vec0核
                uint32_t vec1SOuterSizeTail = (runParam.singleProcessCubeSOuterSizeTail < singleProcessCubeSOuterSizeTailV1) ?
                    0 : (runParam.singleProcessCubeSOuterSizeTail - singleProcessCubeSOuterSizeTailV1); // 判断vec1核是否有数据
                runParam.singleProcessSOuterSizeTail = (constParam.subBlockIdx == 0) ?
                    vec0SOuterSizeTail : vec1SOuterSizeTail;
            } else {
                // 奇数情况下, v0比v1多算1块, v1向上取整除
                singleProcessCubeSOuterSizeTailV1 = CeilDiv(runParam.singleProcessCubeSOuterSizeTail, 2);
                runParam.singleProcessSOuterSizeTail = (constParam.subBlockIdx == 0) ?
                    singleProcessCubeSOuterSizeTailV1 :
                    (runParam.singleProcessCubeSOuterSizeTail / 2); // 2 for div factor
            }
            runParam.sOuterOffset = (constParam.subBlockIdx == 0) ? cubeSOuterOffset :
                cubeSOuterOffset + singleProcessCubeSOuterSizeTailV1;// 2 for div factor
            runParam.singleProcessSOuterSize = runParam.singleProcessSOuterSizeTail;
            runParam.cubeSOuterSize = runParam.singleProcessCubeSOuterSizeTail;
        } else {
            runParam.singleProcessSOuterSize = constParam.singleProcessSOuterSizeWhole;
            runParam.sOuterOffset = (constParam.subBlockIdx == 0) ? 
                cubeSOuterOffset : cubeSOuterOffset + runParam.singleProcessSOuterSize;
            runParam.cubeSOuterSize = constParam.singleProcessCubeSOuterSizeWhole;
        }
        runParam.cubeSOuterOffset = cubeSOuterOffset;
    } else {
        if (sOuterLoopIdx == runParam.sOuterBlockNum - 1) {
            runParam.singleProcessSOuterSize = runParam.singleProcessSOuterSizeTail;
        } else {
            runParam.singleProcessSOuterSize = constParam.singleProcessSOuterSizeWhole;
        }
        runParam.sOuterOffset = sOuterLoopIdx * constParam.singleProcessSOuterSizeWhole;
        runParam.cubeSOuterSize = runParam.singleProcessSOuterSize;
        runParam.cubeSOuterOffset = runParam.sOuterOffset;
    }
}

template<typename PFAT>
__aicore__ inline void LoopSOuterOffsetInit(RunParam &runParam, const ConstParam &constParam, int32_t sIdx, GlobalTensor<int64_t>& actualSeqLengthsGm)
{
    if constexpr (PFAT::isHasAtten) {
        CalAttenMasktCoreOffset<PFAT>(runParam, constParam, sIdx, runParam.sOuterOffset);
    }
    if constexpr (PFAT::isHasPse) {
        CalPseShiftCoreOffset<PFAT>(runParam, constParam, sIdx, runParam.sOuterOffset);
    }
    if constexpr (PFAT::IFA_MLA) {
        if constexpr (PFAT::layout == PFALayout::TND) {
            int64_t actualSeqOfTND = (sIdx == 0) ? actualSeqLengthsGm.GetValue(0) :
                actualSeqLengthsGm.GetValue(sIdx) - actualSeqLengthsGm.GetValue(sIdx - 1);
            actualSeqOfTND *= constParam.gOfMla;
            runParam.tensorAOffset = runParam.multiSeqOffset + runParam.batchNOffset * constParam.headSize *
                actualSeqOfTND + runParam.cubeSOuterOffset * constParam.headSize;
            runParam.qRopeNBGOffset = runParam.qRopeBOffset + runParam.batchNOffset * constParam.ropeHeadSize *
                actualSeqOfTND + runParam.cubeSOuterOffset * constParam.ropeHeadSize;
        } else {
            runParam.tensorAOffset = runParam.multiSeqOffset + runParam.batchNOffset * constParam.headSize *
                constParam.seqSize + runParam.cubeSOuterOffset * constParam.headSize; // IFA MLA场景, BSH与BNSD一致
            runParam.qRopeNBGOffset = runParam.qRopeBOffset + runParam.batchNOffset * constParam.ropeHeadSize *
                constParam.seqSize + runParam.cubeSOuterOffset * constParam.ropeHeadSize;
        }
    } else {
        if (constParam.isIFA) {
            runParam.tensorAOffset = runParam.multiSeqOffset + runParam.batchNOffset * constParam.qkHeadSize *
                constParam.seqSize + runParam.cubeSOuterOffset * constParam.qkHeadSize;
        } else {
            if constexpr (PFAT::layout == PFALayout::BSH || PFAT::layout == PFALayout::TND) {
                runParam.tensorAOffset = runParam.multiSeqOffset + runParam.cubeSOuterOffset * constParam.multiHeadQ +
                    runParam.batchNOffset * constParam.qkHeadSize;
            } else {
                runParam.tensorAOffset = runParam.multiSeqOffset + runParam.batchNOffset * constParam.qkHeadSize *
                    constParam.seqSize + runParam.cubeSOuterOffset * constParam.qkHeadSize;
            }
        }
    }

    int64_t attentionOutSeqOffset;
    if constexpr (PFAT::layout == PFALayout::TND) {
        attentionOutSeqOffset = (sIdx == 0)? 0 : actualSeqLengthsGm.GetValue(sIdx - 1) * constParam.multiHeadOut;
        if constexpr (PFAT::IFA_MLA) {
            attentionOutSeqOffset *= constParam.gOfMla;
        }
    } else {
        attentionOutSeqOffset = sIdx * constParam.seqSize * constParam.multiHeadOut;
    }
    if constexpr (PFAT::IFA_MLA) {
        if constexpr (PFAT::layout == PFALayout::TND) {
            int64_t actualSeqOfTND = (sIdx == 0) ? actualSeqLengthsGm.GetValue(0) :
                actualSeqLengthsGm.GetValue(sIdx) - actualSeqLengthsGm.GetValue(sIdx - 1);
            actualSeqOfTND *= constParam.gOfMla;
            runParam.attentionOutOffset = attentionOutSeqOffset + runParam.batchNOffset * constParam.vHeadSize *
                actualSeqOfTND + runParam.sOuterOffset * constParam.vHeadSize;
        } else {
            runParam.attentionOutOffset = attentionOutSeqOffset + runParam.batchNOffset * constParam.vHeadSize *
                constParam.seqSize + runParam.sOuterOffset * constParam.vHeadSize;
        }
    } else {
        if (constParam.isIFA) {
            runParam.attentionOutOffset = attentionOutSeqOffset + runParam.batchNOffset * constParam.vHeadSize *
                constParam.seqSize + runParam.sOuterOffset * constParam.vHeadSize;
        } else {
            if (constParam.isBSNDOut == 1 || PFAT::layout == PFALayout::BSH || PFAT::layout == PFALayout::TND) {
                int64_t multiSeqOffsetBeforeCopyout = attentionOutSeqOffset + runParam.queryLeftPaddingSize * constParam.multiHeadOut;
                runParam.attentionOutOffset = multiSeqOffsetBeforeCopyout + runParam.sOuterOffset * constParam.multiHeadOut +
                    runParam.batchNOffset * constParam.vHeadSize;
            } else {
                runParam.attentionOutOffset = attentionOutSeqOffset + runParam.batchNOffset * constParam.vHeadSize *
                    constParam.seqSize + runParam.sOuterOffset * constParam.vHeadSize +
                    runParam.queryLeftPaddingSize * constParam.vHeadSize;
            }
        }
    }

    if constexpr (PFAT::layout == PFALayout::TND) {
        int64_t totalSOfTND = (sIdx == 0) ? 0 : actualSeqLengthsGm.GetValue(sIdx - 1);
        if constexpr (PFAT::IFA_MLA) {
            runParam.softmaxLseOffset = totalSOfTND * constParam.headNumSize * constParam.gOfMla +
                runParam.sOuterOffset * constParam.headNumSize + runParam.batchNOffset * constParam.gOfMla;
        } else {
            runParam.softmaxLseOffset = totalSOfTND * constParam.headNumSize + runParam.sOuterOffset * constParam.headNumSize +
                runParam.batchNOffset + runParam.queryLeftPaddingSize;
        }
    } else {
        if constexpr (PFAT::layout == PFALayout::BSH && PFAT::IFA_MLA) {
            int64_t nsOffset = runParam.batchNOffset * constParam.seqSize + runParam.sOuterOffset;
            int64_t sOffset = nsOffset / constParam.gOfMla;
            runParam.softmaxLseOffset = sIdx * constParam.seqSize * constParam.headNumSize +
                (nsOffset % constParam.gOfMla) * (constParam.seqSize / constParam.gOfMla) + sOffset;
        } else {
            runParam.softmaxLseOffset = sIdx * constParam.seqSize * constParam.headNumSize + runParam.batchNOffset *
                constParam.seqSize + runParam.sOuterOffset + runParam.queryLeftPaddingSize;
        }
    }
}

template<typename PFAT>
__aicore__ inline bool ComputeParamS1(RunParam &runParam, const ConstParam &constParam,
    int32_t sIdx, uint32_t sOuterLoopIdx, GlobalTensor<int64_t>& actualSeqLengthsGm)
{
    bool s1NeedCalc = true;
    // 后续的函数依赖 sOuterOffset
    ComputeSouterParam<PFAT>(runParam, constParam, sIdx, sOuterLoopIdx);

    // 使用转换后的左上角的pretoken nexttoken
    if (runParam.nextTokensPerBatch < 0 && runParam.sOuterOffset < ((runParam.nextTokensPerBatch * (-1)) /
        runParam.singleProcessSOuterSize * runParam.singleProcessSOuterSize)) {
        s1NeedCalc = false;
        return s1NeedCalc;
    }

    LoopSOuterOffsetInit<PFAT>(runParam, constParam, sIdx, actualSeqLengthsGm);
    return s1NeedCalc;
}

template<typename PFAT>
__aicore__ inline int64_t ClipSInnerTokenCube(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

template<typename PFAT>
__aicore__ inline bool ComputeS2LoopInfo(RunParam &runParam, const ConstParam &constParam)
{
    bool s2NeedCalc = true;
    int64_t sInnerFirstToken = ClipSInnerTokenCube<PFAT>(runParam.cubeSOuterOffset - runParam.preTokensPerBatch,
        0, runParam.actualSeqLengthKVPerBatch);
    int64_t sInnerLastToken = ClipSInnerTokenCube<PFAT>(runParam.cubeSOuterOffset + runParam.nextTokensPerBatch +
        runParam.cubeSOuterSize, 0, runParam.actualSeqLengthKVPerBatch);

    runParam.startIndex = sInnerFirstToken / (int32_t)constParam.singleProcessSInnerSize;
    runParam.endIndex = (sInnerLastToken + (int32_t)constParam.singleProcessSInnerSize - 1) /
        (int32_t)constParam.singleProcessSInnerSize;

    if (runParam.startIndex <= 0) {
        runParam.startIndex = 0;
    }
    if (runParam.endIndex > runParam.maxInnerLoopTimes) {
        runParam.endIndex = runParam.maxInnerLoopTimes;
    }
    if (runParam.endIndex - runParam.startIndex <= 0) {
        s2NeedCalc = false;
        return s2NeedCalc;
    }
    return s2NeedCalc;
}

template<typename PFAT>
__aicore__ inline void ComputeOffset(TaskParam &taskParam, RunParam &runParam,
    const ConstParam &constParam, uint32_t sInnerLoopIdx)
{
    int64_t sInnerOffsetDataSize = sInnerLoopIdx * constParam.singleProcessSInnerSize;
    if constexpr (PFAT::isHasPse) {
        taskParam.pseShiftOffset = ComputePseShiftOffset<PFAT>(runParam, sInnerOffsetDataSize);
    } else {
        taskParam.pseShiftOffset = 0;
    }
    if constexpr (PFAT::isHasAtten) {
        taskParam.attenMaskOffset = ComputeAttenMaskOffset<PFAT>(runParam, constParam, sInnerOffsetDataSize);
        taskParam.attenMaskOffsetPre = ComputeAttenMaskOffsetPre<PFAT>(runParam, constParam, sInnerOffsetDataSize);
    } else {
        taskParam.attenMaskOffset = 0;
        taskParam.attenMaskOffsetPre = 0;
    }

    if constexpr (PFAT::layout == PFALayout::BSH || PFAT::layout == PFALayout::TND) {
        taskParam.valueOffset = runParam.valueCoreOffset + sInnerOffsetDataSize * constParam.multiHeadV;
        if constexpr (PFAT::PFA_MLA) {
            taskParam.tensorBOffset = runParam.keyCoreOffset + sInnerOffsetDataSize * constParam.multiHeadK;
        } else {
            taskParam.tensorBOffset = taskParam.valueOffset;
        }
        if constexpr (PFAT::IFA_MLA) {
            taskParam.kRopeOffset = runParam.kRopeNBGOffset + sInnerOffsetDataSize * constParam.multiHeadKRope;
        }
    } else {
        taskParam.valueOffset = runParam.valueCoreOffset + sInnerOffsetDataSize * constParam.vHeadSize;
        if constexpr (PFAT::PFA_MLA) {
            taskParam.tensorBOffset = runParam.keyCoreOffset + sInnerOffsetDataSize * constParam.qkHeadSize;
        } else {
            taskParam.tensorBOffset = taskParam.valueOffset;
        }
        if constexpr (PFAT::IFA_MLA) {
            taskParam.kRopeOffset = runParam.kRopeNBGOffset + sInnerOffsetDataSize * constParam.ropeHeadSize;
        }
    }

    taskParam.sInnerOffsetDataSize = sInnerOffsetDataSize;
}

template<typename PFAT>
__aicore__ inline void ComputeParamS2(TaskParam &taskParam, RunParam &runParam,
    const ConstParam &constParam, uint32_t sInnerLoopIdx)
{
    taskParam.isFirstInnerIter = (sInnerLoopIdx == runParam.startIndex);
    taskParam.isSecondInnerIter = (sInnerLoopIdx == (runParam.startIndex + 1));
    taskParam.isLastInnerIter = (sInnerLoopIdx == runParam.endIndex - 1);

    // 是否为S2全载 isS2Load
    if (runParam.maxInnerLoopTimes == 1) {
        taskParam.isInnerTail = true;
    } else {
        taskParam.isInnerTail = (sInnerLoopIdx == runParam.maxInnerLoopTimes - 1);
    }
    if (taskParam.isInnerTail) {
        taskParam.mm1SingleCoreN = runParam.singleProcessSInnerSizeTail;
        taskParam.singleProcessSInnerSizeNow = runParam.singleProcessSInnerSizeTail;
        taskParam.singleProcessSInnerBmmTail = runParam.unalignSInner;
        taskParam.maskCopyInCol = runParam.maskInnerTailAlign;
        taskParam.pseShiftCopyInCol = runParam.pseShiftInnerTailAlign;
    } else {
        taskParam.mm1SingleCoreN = constParam.singleProcessSInnerSize;
        taskParam.singleProcessSInnerSizeNow = constParam.singleProcessSInnerSize;
        taskParam.singleProcessSInnerBmmTail = constParam.singleProcessSInnerSize;
        taskParam.maskCopyInCol = constParam.singleProcessSInnerSize;
        taskParam.pseShiftCopyInCol = constParam.singleProcessSInnerSize;
    }

    ComputeOffset<PFAT>(taskParam, runParam, constParam, sInnerLoopIdx);
}

template<typename PFAT>
__aicore__ inline void InitTaskParamByRun(TaskParam &taskParam, RunParam &runParam)
{
    taskParam.tensorAOffset = runParam.tensorAOffset;
    taskParam.qRopeOffset = runParam.qRopeNBGOffset;
    taskParam.taskBatch = runParam.taskBatch;
    taskParam.batchNOffset = runParam.batchNOffset;
    taskParam.cubeSOuterSize = runParam.cubeSOuterSize;
    taskParam.singleProcessSOuterSize = runParam.singleProcessSOuterSize;
    taskParam.padSize = runParam.padSize;
    taskParam.pseShiftPadSize = runParam.pseShiftPadSize;
    taskParam.preTokensPerBatch = runParam.preTokensPerBatch;
    taskParam.nextTokensPerBatch = runParam.nextTokensPerBatch;
    taskParam.nextTokensOfMlaPerBatch = runParam.nextTokensOfMlaPerBatch;
    taskParam.actualSeqLengthPerBatch = runParam.actualSeqLengthPerBatch;
    taskParam.actualSeqLengthOfMlaPerBatch = runParam.actualSeqLengthOfMlaPerBatch;
    taskParam.actualSeqLengthKVPerBatch = runParam.actualSeqLengthKVPerBatch;
    taskParam.sOuterOffset = runParam.sOuterOffset;
    taskParam.attentionOutOffset = runParam.attentionOutOffset;
    taskParam.softmaxLseOffset = runParam.softmaxLseOffset;
}

template<typename PFAT>
__aicore__ inline void IterateAllPreProcess(const TaskParam &taskParam, const ConstParam &constParam,
    GlobalTensor<typename PFAT::kvInputType>& keyValueGm, GlobalTensor<typename PFAT::kvInputType>& tempKeyValueGm)
{
    if (constParam.isKvContinuous != 0) {
        return;
    }
    ListTensorDesc keyValueListTensorDesc((__gm__ void*)keyValueGm.GetPhyAddr());
    __gm__ uint8_t* tempKeyValueGmPtr =
        (__gm__ uint8_t*)keyValueListTensorDesc.GetDataPtr<__gm__ uint8_t>(taskParam.taskBatch);
    tempKeyValueGm.SetGlobalBuffer((__gm__ typename PFAT::kvInputType*)tempKeyValueGmPtr);
}

#endif  // PROMPT_FLASH_ATTENTION_KVCACHE_H