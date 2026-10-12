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
 * \file prompt_flash_attention_normal_bns1_preload.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_NORMAL_BNS1_PRELOAD_H
#define PROMPT_FLASH_ATTENTION_NORMAL_BNS1_PRELOAD_H

#include "../comm/prompt_flash_attention_comm.h"
#include "../comm/prompt_flash_attention_kvcache.h"
#include "../comm/prompt_flash_attention_sparse.h"
#include "../service/prompt_flash_attention_mm1.h"
#include "../service/prompt_flash_attention_mm2.h"
#include "../service/prompt_flash_attention_vector1.h"
#include "../service/prompt_flash_attention_vector2.h"
#include "../matmul_modules/pfa_matmul_policy.h"
#include "../prompt_flash_attention_tiling_regbase.h"

const uint32_t PFA_PRELOAD_TASK_CACHE_SIZE = 3;

template <typename PFAT>
class PromptFlashAttentionNormalBNS1Preload {
public:
    using T = typename PFAT::inputType;
    using KV_T = typename PFAT::kvInputType;
    using U = typename PFAT::maskType;
    using O = typename PFAT::outputType;
    using mmBiasType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::mmBiasType;
    using mmOutputType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::mmOutputType;
    using computeType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::softmaxType;
    using pseShiftType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::pseShiftType;
    using pseShiftCastType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::pseShiftCastType;

    __aicore__ inline PromptFlashAttentionNormalBNS1Preload() {};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                __gm__ uint8_t* blocktable, __gm__ uint8_t* queryPaddingSize,
                                __gm__ uint8_t* kvPaddingSize, __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope,
                                __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                                const optiling::PromptFlashAttentionTilingDataV2* __restrict tiling, TPipe* tPipe);
    __aicore__ inline void InitQuant(__gm__ uint8_t* deq_scale1, __gm__ uint8_t* scale1, __gm__ uint8_t* deq_scale2,
                                     __gm__ uint8_t* scale2, __gm__ uint8_t* offset2);
    __aicore__ inline void Process();

    // define matmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false, LayoutMode::NONE, PFAT::isSplitCoreByCube>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, T, true, LayoutMode::NONE, PFAT::isSplitCoreByCube>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, mmBiasType>;
    using c1Type = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, mmOutputType>;
    constexpr static MatmulConfig mm1cfg = PFAT::GetMM1Config();
    constexpr static auto stcMm1cfg = matmul::GetMatmulApiTiling<a1Type, b1Type, c1Type, bias1Type>(mm1cfg);
    using mm1Type = typename AscendC::Conditional<PFAT::isUseNormMatmul,
        Matmul<a1Type, b1Type, c1Type, bias1Type, PFAT::GetMM1Config()>,
        Matmul<a1Type, b1Type, c1Type, bias1Type, stcMm1cfg, matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
        ConstPolicySelector<PFAT::MM_TYPE>::template Result>>::type;

    // define batchmatmul
    // DN场景的a2是[S2,S1]，需要转置
    using a2Type = typename AscendC::Conditional<PFAT::isBmm2Concat,
        MatmulType<TPosition::TSCM, CubeFormat::NZ, T, PFAT::useDN, LayoutMode::NONE, true, TPosition::VECOUT>,
        MatmulType<TPosition::VECCALC, CubeFormat::NZ, T, false, LayoutMode::NONE, PFAT::isBmm2Concat>>::type;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false, LayoutMode::NONE, PFAT::isBmm2Concat>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, mmBiasType>;
    using c2Type = MatmulType<TPosition::VECCALC, CubeFormat::ND, mmOutputType>;
    constexpr static MatmulConfig mm2cfg = PFAT::GetMM2Config();
    constexpr static auto stcMm2cfg = matmul::GetMatmulApiTiling<a2Type, b2Type, c2Type, bias2Type>(mm2cfg);
    using mm2Type = typename AscendC::Conditional<PFAT::isUseNormMatmul,
        Matmul<a2Type, b2Type, c2Type, bias2Type, PFAT::GetMM2Config()>,
        Matmul<a2Type, b2Type, c2Type, bias2Type, stcMm2cfg, matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
        PFABmm2ConstPolicySelector<PFAT::MM_TYPE,
        ((PFAT::sInner == SINNER_CONST_256) && (PFAT::useDN))>::template Result>>::type;

    PromptFlashAttentionNormalMM1<PFAT, mm1Type> mm1;
    PromptFlashAttentionNormalMM2<PFAT, mm2Type> mm2;
    PromptFlashAttentionNormalVector1<PFAT> vector1;
    PromptFlashAttentionNormalVector2<PFAT> vector2;
protected:
    __aicore__ inline void InitOutputSingleCore();
    __aicore__ inline void InitLseOutputSingleCore();
    __aicore__ inline void AllocGlobalResources();
    __aicore__ inline void FreeGlobalResources();
    __aicore__ inline void ComputeEachCore(uint32_t coreIdx);
    __aicore__ inline void LaunchTask(uint64_t taskIdx);
    __aicore__ inline void IterateAllMm1(TaskParam& taskParam);
    __aicore__ inline void WaitIterateAllMm1(TaskParam& taskParam);
    __aicore__ inline void IterateAllMm2(TaskParam& taskParam);
    __aicore__ inline void WaitIterateAllMm2(TaskParam& taskParam);
    __aicore__ inline void ProcVector1(TaskParam& taskParam);
    __aicore__ inline void ProcVector2(TaskParam& taskParam);

protected:
    TPipe* pipe;
    const optiling::PromptFlashAttentionTilingDataV2* __restrict tilingData;

    // GM
    __gm__ uint8_t* currentKey;    // pageattention需要   // PFATODO mm1直接从GlobalTensor.GetPhyAddr
    __gm__ uint8_t* currentValue;  // pageattention需要
    __gm__ uint8_t* blocktablePtr; // pageattention需要

    GlobalTensor<T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<T> queryRopeGm;
    GlobalTensor<KV_T> keyRopeGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<O> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;

    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;
    GlobalTensor<int64_t> queryPaddingSizeGm;
    GlobalTensor<int64_t> kvPaddingSizeGm;

    GlobalTensor<float> deqScale1Fp32Gm;  // 全量化参数
    GlobalTensor<float> deqScale2Fp32Gm;  // 全量化参数
    
    GlobalTensor<bfloat16_t> quantScale2BF16Gm;    // 后quant融合
    GlobalTensor<bfloat16_t> quantOffset2BF16Gm;   // 后quant融合

    // define the que
    TQue<QuePosition::VECIN, 1> tempBmm2Queue;
    TQue<QuePosition::VECIN, 1> Bmm2Queue;
    TQue<QuePosition::VECOUT, 1> Bmm1Queue;
    TQue<QuePosition::VECIN, 1> maskQueue;
    TQue<QuePosition::VECIN, 1> pseQueue;
    TQue<QuePosition::VECOUT, 2> Bmm1CastUb;      // que depth is 2.
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;

    TSCM<QuePosition::VECIN, 1, 0x4> bmm2Scm[2];

    TBuf<> softmaxApiBuffer;
    TBuf<> softmaxMaxBuffer[2];
    TBuf<> softmaxSumBuffer[2];
    TBuf<> softmaxExpBuffer[2]; // 开启DB
    TBuf<> tempBmm2Ub;
    TBuf<> quantScale2BF16Ub;  // 初始化和 vector2中使用   后quant融合
    TBuf<> quantOffset2BF16Ub; // 初始化和 vector2中使用   后quant融合
    TBuf<> quantScale2FloatUb; // 初始化和 vector2中使用   后quant融合
    TBuf<> quantOffset2FloatUb; // 初始化和 vector2中使用   后quant融合

    LocalTensor<computeType> mmResUb[2];
    LocalTensor<T> tmpSoftmaxResUb[2];
    LocalTensor<computeType> mm2ResPQUb[2];
    LocalTensor<uint8_t> softmaxApiUb;
    LocalTensor<float> softmaxMaxUb[2];
    LocalTensor<float> softmaxSumUb[2];
    LocalTensor<computeType> softmaxExpUb[2];
    LocalTensor<mmOutputType> mmQuantResUb[2];
    LocalTensor<mmOutputType> mm2QuantResUb[2];

    float dequantScale1;  // 全量
    float quantScale1;       // 全量
    float dequantScale2;  // 全量
    float quantScale2;       // 后 quant融合
    float quantOffset2;      // 后 quant融合
    bool isQuant2PerChn = false;        //  后quant融合  quant bf16 per-channel
    bool isQuantOffset2Exit = false;    //  后quant融合 quant bf16 per-channel
    uint32_t perChannelQuantUBSize = 0; //  后quant融合 quant bf16 per-channel

    ConstParam constParam;
    TaskManager<PFA_PRELOAD_TASK_CACHE_SIZE> taskManager;
};

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalBNS1Preload<PFAT>::Init(__gm__ uint8_t* query, __gm__ uint8_t* key,
    __gm__ uint8_t* value, __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengths,
    __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable, __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
    __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope, __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse,
    __gm__ uint8_t* workspace, const optiling::PromptFlashAttentionTilingDataV2* __restrict tiling, TPipe* tPipe)
{
    pipe = tPipe;

    // init const param
    tilingData = tiling;
    InitConstParam<PFAT>(constParam, tilingData);

    // init GM
    queryGm.SetGlobalBuffer((__gm__ T*)query);
    queryRopeGm.SetGlobalBuffer((__gm__ T*)queryRope);
    keyRopeGm.SetGlobalBuffer((__gm__ T*)keyRope);
    attentionOutGm.SetGlobalBuffer((__gm__ O*)attentionOut);
    if (constParam.isSoftmaxLseEnable) {
        softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmaxLse);
    }
    if (tilingData->promptAttentionBaseParams.fromFused) {
        ListTensorDesc keyListTensorDescInit((__gm__ void*)key);
        ListTensorDesc valueListTensorDescInit((__gm__ void*)value);
        currentKey = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
        currentValue = (__gm__ uint8_t*)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
        if constexpr (PFAT::MM_TYPE == PFAMatMulType::MM_PA || PFAT::MM_TYPE == PFAMatMulType::MM_PA_D512 ||
                      PFAT::MM_TYPE == PFAMatMulType::MM_IFA_MLA_PA) {
            blocktablePtr = blocktable;
        }
        if (constParam.isKvContinuous == 1) {
            keyGm.SetGlobalBuffer((__gm__ KV_T*)currentKey);
            valueGm.SetGlobalBuffer((__gm__ KV_T*)currentValue);
        } else {
            keyGm.SetGlobalBuffer((__gm__ KV_T*)key);
            valueGm.SetGlobalBuffer((__gm__ KV_T*)value);
        }
    } else {
        keyGm.SetGlobalBuffer((__gm__ KV_T*)key);
        valueGm.SetGlobalBuffer((__gm__ KV_T*)value);
    }

    if (!constParam.isActualLenDimsNull) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths,
                                           tilingData->promptAttentionBaseParams.batchSize);
    }
    if (!constParam.isActualLenDimsKVNull) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV,
                                             tilingData->promptAttentionBaseParams.batchSize);
    }

    if (constParam.isQHasLeftPadding) {
        queryPaddingSizeGm.SetGlobalBuffer((__gm__ int64_t*)queryPaddingSize);
        constParam.queryRightPaddingSize = (queryPaddingSizeGm.GetValue(0) > 0) ? queryPaddingSizeGm.GetValue(0) : 0;
    }
    if (constParam.isKVHasLeftPadding) {
        kvPaddingSizeGm.SetGlobalBuffer((__gm__ int64_t*)kvPaddingSize);
        constParam.kvRightPaddingSize = (kvPaddingSizeGm.GetValue(0) > 0) ? kvPaddingSizeGm.GetValue(0) : 0;
    }

    // init UB
    pipe->InitBuffer(tempBmm2Ub, PFAT::vsOuter * PFAT::vDSize * sizeof(computeType));
    pipe->InitBuffer(Bmm1CastUb, 2, (PFAT::vsOuter + 1) * PFAT::sInner * sizeof(T)); // ping pong形式避免读写冲突
    pipe->InitBuffer(softmaxApiBuffer, 2 * PFAT::vsOuter * SOFTMAX_COLUMN_SIZE * sizeof(computeType));
    pipe->InitBuffer(softmaxMaxBuffer[0], PFAT::vsOuter * SOFTMAX_COLUMN_SIZE * sizeof(float));
    pipe->InitBuffer(softmaxMaxBuffer[1], PFAT::vsOuter * SOFTMAX_COLUMN_SIZE * sizeof(float));
    pipe->InitBuffer(softmaxSumBuffer[0], PFAT::vsOuter * SOFTMAX_COLUMN_SIZE * sizeof(float));
    pipe->InitBuffer(softmaxSumBuffer[1], PFAT::vsOuter * SOFTMAX_COLUMN_SIZE * sizeof(float));
    pipe->InitBuffer(softmaxExpBuffer[0], PFAT::vsOuter * SOFTMAX_COLUMN_SIZE * sizeof(computeType));
    pipe->InitBuffer(softmaxExpBuffer[1], PFAT::vsOuter * SOFTMAX_COLUMN_SIZE * sizeof(computeType));
    pipe->InitBuffer(Bmm2Queue, 2, PFAT::vsOuter * PFAT::vDSize * sizeof(computeType));
    pipe->InitBuffer(Bmm1Queue, 2, PFAT::vsOuter * PFAT::sInner * sizeof(computeType));
    if (constParam.isSoftmaxLseEnable) {
        pipe->InitBuffer(softmaxLseQueue, 1, PFAT::vsOuter * sizeof(float) * 8); // 8: 适配TND, 每行的结果存为8个重复lse元素（32B对齐）
    }
    if constexpr (PFAT::isBand) {
        pipe->InitBuffer(maskQueue, 1, PFAT::vsOuter * PFAT::sInner * sizeof(U) * 2); // 2: band mode needs 2 mask
    } else {
        pipe->InitBuffer(maskQueue, 1, PFAT::vsOuter * PFAT::sInner * sizeof(U));
    }
    if constexpr (PFAT::isHasPse) {
        pipe->InitBuffer(pseQueue, 1, PFAT::vsOuter * PFAT::sInner * sizeof(T));
    }

    softmaxApiUb = softmaxApiBuffer.Get<uint8_t>();
    softmaxMaxUb[0] = softmaxMaxBuffer[0].Get<float>(PFAT::vsOuter * SOFTMAX_COLUMN_SIZE);
    softmaxMaxUb[1] = softmaxMaxBuffer[1].Get<float>(PFAT::vsOuter * SOFTMAX_COLUMN_SIZE);
    softmaxSumUb[0] = softmaxSumBuffer[0].Get<float>(PFAT::vsOuter * SOFTMAX_COLUMN_SIZE);
    softmaxSumUb[1] = softmaxSumBuffer[1].Get<float>(PFAT::vsOuter * SOFTMAX_COLUMN_SIZE);
    softmaxExpUb[0] = softmaxExpBuffer[0].Get<computeType>(PFAT::vsOuter * SOFTMAX_COLUMN_SIZE);
    softmaxExpUb[1] = softmaxExpBuffer[1].Get<computeType>(PFAT::vsOuter * SOFTMAX_COLUMN_SIZE);

    // init L1
    if constexpr (PFAT::isBmm2Concat) {
        pipe->InitBuffer(bmm2Scm[0], 1, 128 * 256 * sizeof(T)); // 128：sOuter, 256：sInner
        pipe->InitBuffer(bmm2Scm[1], 1, 128 * 256 * sizeof(T)); // 128：sOuter, 256：sInner
    }

    if (tilingData->promptAttentionInitOutputParams.needInit == 1) {
        InitOutputSingleCore();
        InitLseOutputSingleCore();
    }

    // init service
    mm2.Init();
    vector1.Init(attenMask, pseShift);
    vector2.Init(attentionOut, softmaxLse);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalBNS1Preload<PFAT>::InitQuant(__gm__ uint8_t* deq_scale1,
    __gm__ uint8_t* scale1, __gm__ uint8_t* deq_scale2, __gm__ uint8_t* scale2, __gm__ uint8_t* offset2)
{
    if (deq_scale1 != nullptr) {
        if(tilingData->promptAttentionBaseParams.deqScale2Flag == 1){
            deqScale1Fp32Gm.SetGlobalBuffer((__gm__ float*)deq_scale1);
            dequantScale1 = deqScale1Fp32Gm(0);
        } else {
            dequantScale1 = *(reinterpret_cast<__gm__ float*>(deq_scale1));
        }
    }
    if (scale1 != nullptr) { quantScale1 = *(reinterpret_cast<__gm__ float*>(scale1));}
    if (deq_scale2 != nullptr) {
        if(tilingData->promptAttentionBaseParams.deqScaleFlag == 1){
            deqScale2Fp32Gm.SetGlobalBuffer((__gm__ float*)deq_scale2);
            dequantScale2 = deqScale2Fp32Gm(0);
        } else {
            dequantScale2 = *(reinterpret_cast<__gm__ float*>(deq_scale2));
        }
    }
    isQuant2PerChn = tilingData->promptAttentionBaseParams.isQuant2Perchannel == 0 ? false : true;
    bool isQuant2BF16 = tilingData->promptAttentionBaseParams.isQuant2BF16 == 0 ? false : true;
    isQuantOffset2Exit = offset2 == nullptr ? false : true;
    if (scale2 != nullptr && !isQuant2PerChn && !isQuant2BF16) { quantScale2 = *(reinterpret_cast<__gm__ float*>(scale2));}
    if (offset2 != nullptr && !isQuant2PerChn && !isQuant2BF16) { quantOffset2 = *(reinterpret_cast<__gm__ float*>(offset2));}
    if (scale2 != nullptr && !isQuant2PerChn && isQuant2BF16) {
        quantScale2BF16Gm.SetGlobalBuffer((__gm__ bfloat16_t*)(scale2));
        quantScale2 = ToFloat(quantScale2BF16Gm.GetValue(0));
    }
    if (offset2 != nullptr && !isQuant2PerChn && isQuant2BF16) {
        quantOffset2BF16Gm.SetGlobalBuffer((__gm__ bfloat16_t*)(offset2));
        quantOffset2 = ToFloat(quantOffset2BF16Gm.GetValue(0));
    }
    if (scale2 != nullptr && isQuant2PerChn && isQuant2BF16) {
        perChannelQuantUBSize = this->tilingData->promptAttentionBaseParams.qkHeadSize;
        quantScale2BF16Gm.SetGlobalBuffer((__gm__ bfloat16_t*)(scale2));
        quantOffset2BF16Gm.SetGlobalBuffer((__gm__ bfloat16_t*)(offset2));
        pipe->InitBuffer(quantScale2BF16Ub, perChannelQuantUBSize * sizeof(bfloat16_t));
        pipe->InitBuffer(quantScale2FloatUb, perChannelQuantUBSize * sizeof(float));
        pipe->InitBuffer(quantOffset2BF16Ub, perChannelQuantUBSize * sizeof(bfloat16_t));
        pipe->InitBuffer(quantOffset2FloatUb, perChannelQuantUBSize * sizeof(float));
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalBNS1Preload<PFAT>::InitLseOutputSingleCore()
{
    if (constParam.isSoftmaxLseEnable) {
        int64_t tmpBlockIdx = constParam.tmpBlockIdx;
        int64_t coreNum = GetBlockNum() * GetTaskRation();
        if (coreNum != 0 && tmpBlockIdx < coreNum) {
            int64_t singleCoreLseSize = constParam.totalSoftmaxLseOutputSize / coreNum;
            if (tmpBlockIdx == coreNum - 1) {
                singleCoreLseSize += constParam.totalSoftmaxLseOutputSize % coreNum;
            }
            InitOutput<float>(softmaxLseGm[tmpBlockIdx * (constParam.totalSoftmaxLseOutputSize / coreNum)], singleCoreLseSize, 3e+99); // 3e+99:set the value of invalid batch to inf
            SyncAll();
        }
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalBNS1Preload<PFAT>::InitOutputSingleCore()
{
    auto &initParams = tilingData->promptAttentionInitOutputParams;
    uint32_t tailSize = initParams.totalOutputSize - constParam.tmpBlockIdx * initParams.singleCoreSize;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    InitOutput<O>(attentionOutGm[constParam.tmpBlockIdx * initParams.singleCoreSize], singleInitOutputSize, 0);
    SyncAll();
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalBNS1Preload<PFAT>::Process()
{
    AllocGlobalResources();
    ComputeEachCore(constParam.tmpBlockIdx);
    FreeGlobalResources();
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalBNS1Preload<PFAT>::ComputeEachCore(uint32_t coreIdx)
{
    int actualCoreNums = this->tilingData->promptAttentionSingleCoreParams.actualCoreNums;
    if (g_coreType == AIV && coreIdx >= actualCoreNums) { // PFATODO 是否可以删除？没有分满核是否
        return;
    }
    bool splitPingPong = false;          // 用于两轮sinner之间的ping pong
    bool taskPingPong = false;           // 用于sinner内每个task之间的ping pong
    uint64_t taskIdx = 0;
    RunParam runParam;
    ComputeParamCore<PFAT>(runParam, constParam, tilingData, coreIdx);
    for (int32_t sIdx = constParam.sIdStart; sIdx < constParam.sIdEnd; sIdx++) {
        ComputeParamN<PFAT>(runParam, constParam, sIdx);
        for (int32_t loopNIdx = constParam.nLoopStart; loopNIdx < runParam.tmpNLoopEnd; loopNIdx++) {
            runParam.batchNOffset = loopNIdx;
            ComputeParamBatch<PFAT>(runParam, constParam, sIdx, keyGm, actualSeqLengthsGm, actualSeqLengthsKVGm);
            ComputeS1LoopInfo<PFAT>(runParam, constParam, loopNIdx);
            for (int32_t sOuterLoopIdx = constParam.outerLoopStart;
                 sOuterLoopIdx < runParam.tmpOuterLoopEnd;
                 sOuterLoopIdx++) {
                bool s1NeedCalc = ComputeParamS1<PFAT>(runParam, constParam, sIdx, sOuterLoopIdx, actualSeqLengthsGm);
                bool s2NeedCalc = ComputeS2LoopInfo<PFAT>(runParam, constParam);
                bool lastLoopThisCore = (loopNIdx == runParam.tmpNLoopEnd - 1) && (sIdx == constParam.sIdEnd - 1) && \
                    (sOuterLoopIdx == runParam.tmpOuterLoopEnd - 1);
                // s1和s2有任意一个不需要算, 则continue, 如果时当前核最后一次循环，则补充计算taskIdx+2的部分
                if ((!s1NeedCalc || !s2NeedCalc) && !lastLoopThisCore) {
                    continue;
                }
                    
                int32_t sInnerEndIndex = runParam.endIndex;
                if (lastLoopThisCore) {
                    sInnerEndIndex += 2;
                }

                splitPingPong ^= 1;
                for (int32_t sInnerLoopIdx = runParam.startIndex;
                     sInnerLoopIdx < sInnerEndIndex;
                     sInnerLoopIdx++) {
                    taskIdx++;
                    taskPingPong ^= 1;
                    TaskParam &taskParam = taskManager.GetTaskRef(taskIdx);
                    taskParam.isValid = (sInnerLoopIdx < runParam.endIndex);
                    taskParam.splitPingPong = splitPingPong;
                    taskParam.taskPingPong = taskPingPong;
                    InitTaskParamByRun<PFAT>(taskParam, runParam);
                    ComputeParamS2<PFAT>(taskParam, runParam, constParam, sInnerLoopIdx);
                    LaunchTask(taskIdx);
                }
            }
            constParam.outerLoopStart = 0;
        }
        constParam.nLoopStart = 0;
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalBNS1Preload<PFAT>::LaunchTask(uint64_t taskIdx)
{
    TaskParam &taskParam2 = taskManager.GetTaskRef(taskIdx);
    if (taskParam2.isValid) {
        IterateAllMm1(taskParam2);
    }
    TaskParam &taskParam1 = taskManager.GetTaskRef(taskIdx - 1);
    if (taskParam1.isValid) {
        WaitIterateAllMm1(taskParam1);
        ProcVector1(taskParam1);
        IterateAllMm2(taskParam1);
    }
    TaskParam &taskParam0 = taskManager.GetTaskRef(taskIdx - 2);
    if (taskParam0.isValid) {
        WaitIterateAllMm2(taskParam0);
        ProcVector2(taskParam0);
        taskParam0.isValid = false;
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalBNS1Preload<PFAT>::IterateAllMm1(TaskParam& taskParam)
{
    GlobalTensor<typename PFAT::kvInputType> tempKeyGm = keyGm;
    IterateAllPreProcess<PFAT>(taskParam, constParam, keyGm, tempKeyGm);
    if constexpr (IsSameType<T, int8_t>::value) {
        mm1.IterateAll(mmQuantResUb[taskParam.taskPingPong], queryGm, tempKeyGm, queryRopeGm, keyRopeGm,
            blocktablePtr, taskParam, constParam);
    } else {
        mm1.IterateAll(mmResUb[taskParam.taskPingPong], queryGm, tempKeyGm, queryRopeGm, keyRopeGm,
            blocktablePtr, taskParam, constParam);
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalBNS1Preload<PFAT>::WaitIterateAllMm1(TaskParam& taskParam)
{
    mm1.WaitIterateAll();
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalBNS1Preload<PFAT>::IterateAllMm2(TaskParam& taskParam)
{
    GlobalTensor<typename PFAT::kvInputType> tempValueGm = valueGm;
    IterateAllPreProcess<PFAT>(taskParam, constParam, valueGm, tempValueGm);
    if constexpr (IsSameType<T, int8_t>::value) {
        mm2.IterateAll(mm2QuantResUb[taskParam.taskPingPong], tmpSoftmaxResUb[taskParam.taskPingPong],
            bmm2Scm[taskParam.taskPingPong], keyGm, tempValueGm, blocktablePtr, taskParam, constParam);
    } else {
        mm2.IterateAll(mm2ResPQUb[taskParam.taskPingPong], tmpSoftmaxResUb[taskParam.taskPingPong],
            bmm2Scm[taskParam.taskPingPong], keyGm, tempValueGm, blocktablePtr, taskParam, constParam);
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalBNS1Preload<PFAT>::WaitIterateAllMm2(TaskParam& taskParam)
{
    mm2.WaitIterateAll();
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalBNS1Preload<PFAT>::ProcVector1(TaskParam& taskParam)
{
    if constexpr (IsSameType<T, int8_t>::value) {
        vector1.ProVector(tmpSoftmaxResUb[taskParam.taskPingPong], maskQueue, pseQueue, mmQuantResUb[taskParam.taskPingPong],
            softmaxMaxUb[taskParam.splitPingPong], softmaxSumUb[taskParam.splitPingPong], softmaxExpUb[taskParam.taskPingPong],
            softmaxApiUb, taskParam, constParam, dequantScale1, quantScale1);
    } else {
        vector1.ProVector(tmpSoftmaxResUb[taskParam.taskPingPong], maskQueue, pseQueue, mmResUb[taskParam.taskPingPong],
            softmaxMaxUb[taskParam.splitPingPong], softmaxSumUb[taskParam.splitPingPong], softmaxExpUb[taskParam.taskPingPong],
            softmaxApiUb, taskParam, constParam, dequantScale1, quantScale1);
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalBNS1Preload<PFAT>::ProcVector2(TaskParam& taskParam)
{
    LocalTensor<computeType> bmm2ResPreUb = tempBmm2Ub.Get<computeType>(
        constParam.singleProcessSOuterSizeWhole * PFAT::vDSize);  // souter * d

    if constexpr (IsSameType<T, int8_t>::value) {
        vector2.ProVector(bmm2ResPreUb, softmaxMaxUb[taskParam.splitPingPong], softmaxSumUb[taskParam.splitPingPong],
            mm2QuantResUb[taskParam.taskPingPong], softmaxExpUb[taskParam.taskPingPong], softmaxLseQueue, taskParam, constParam, dequantScale2);
    } else {
        vector2.ProVector(bmm2ResPreUb, softmaxMaxUb[taskParam.splitPingPong], softmaxSumUb[taskParam.splitPingPong],
            mm2ResPQUb[taskParam.taskPingPong], softmaxExpUb[taskParam.taskPingPong], softmaxLseQueue, taskParam, constParam, dequantScale2);
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalBNS1Preload<PFAT>::AllocGlobalResources()
{
    for (int i = 0; i < 2; ++i) {
        this->mmResUb[i] = this->Bmm1Queue.template AllocTensor<computeType>();
        this->mm2ResPQUb[i] = this->Bmm2Queue.template AllocTensor<computeType>();
        this->tmpSoftmaxResUb[i] = this->Bmm1CastUb.template AllocTensor<T>();
    }

    // INT8全量化场景，将存放matmul结果的两块PingPong UB空间重新转换为int32_t类型
    if constexpr (IsSameType<T, int8_t>::value) {
        mmQuantResUb[0] = mmResUb[0].template ReinterpretCast<int32_t>();
        mmQuantResUb[0].SetSize(mmResUb[0].GetSize());
        mmQuantResUb[1] = mmResUb[1].template ReinterpretCast<int32_t>();
        mmQuantResUb[1].SetSize(mmResUb[1].GetSize());
        mm2QuantResUb[0] = mm2ResPQUb[0].template ReinterpretCast<int32_t>();
        mm2QuantResUb[0].SetSize(mm2ResPQUb[0].GetSize());
        mm2QuantResUb[1] = mm2ResPQUb[1].template ReinterpretCast<int32_t>();
        mm2QuantResUb[1].SetSize(mm2ResPQUb[1].GetSize());
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionNormalBNS1Preload<PFAT>::FreeGlobalResources()
{
    for (int i = 0; i < 2; ++i) {
        this->Bmm1Queue.FreeTensor(this->mmResUb[i]);
        this->Bmm2Queue.FreeTensor(this->mm2ResPQUb[i]);
        this->Bmm1CastUb.FreeTensor(this->tmpSoftmaxResUb[i]);
    }
}

#endif  // PROMPT_FLASH_ATTENTION_NORMAL_BNS1_PRELOAD_H