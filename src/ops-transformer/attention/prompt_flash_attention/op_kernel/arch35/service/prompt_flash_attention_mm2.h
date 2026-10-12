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
 * \file prompt_flash_attention_mm2.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_MM2_H
#define PROMPT_FLASH_ATTENTION_MM2_H

using namespace matmul;

template <typename PFAT, typename mmType>
class PromptFlashAttentionNormalMM2 {
public:
    using T = typename PFAT::inputType;
    using KV_T = typename PFAT::kvInputType;
    using computeType = typename PromptFlashAttentionTypeTraits<T, PFAT::calcMode>::softmaxType;
    using mmOutputType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::mmOutputType;
    __aicore__ inline PromptFlashAttentionNormalMM2() {};
    __aicore__ inline void Init();
    __aicore__ inline void WaitIterateAll();

    // 全量化、非量化、PA场景
    __aicore__ inline void IterateAll(LocalTensor<mmOutputType>& bmm2ResUb, LocalTensor<T>& tmpSoftmaxResUb,
        TSCM<QuePosition::VECIN, 1, 0x4>& bmm2Scm, GlobalTensor<KV_T>& keyGm, GlobalTensor<KV_T>& valueGm,
        __gm__ uint8_t* blocktablePtr, const TaskParam& taskParam, const ConstParam& constParam);
    // 伪量化场景
    __aicore__ inline void IterateAll(LocalTensor<mmOutputType>& bmm2ResUb, LocalTensor<T>& tmpSoftmaxResUb,
        TSCM<QuePosition::VECIN, 1, 0x4> &bmm2Scm, TSCM<QuePosition::VECIN, 1, 0x4>& valueScmQueue,
        const TaskParam& taskParam, const ConstParam& constParam);
public:
    mmType mm;
private:
    GlobalTensor<int8_t> quant1ResGmDb[2];
    event_t nd2NZEvent;
    LocalTensor<KV_T> valueScm;
};

template <typename PFAT, typename mmType>
__aicore__ inline void PromptFlashAttentionNormalMM2<PFAT, mmType>::Init()
{
    this->nd2NZEvent = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
}

template <typename PFAT, typename mmType>
__aicore__ inline void PromptFlashAttentionNormalMM2<PFAT, mmType>::WaitIterateAll()
{
    mm.WaitIterateAll();
}

template <typename PFAT, typename mmType>
__aicore__ inline void PromptFlashAttentionNormalMM2<PFAT, mmType>::IterateAll(
    LocalTensor<mmOutputType>& bmm2ResUb, LocalTensor<T>& tmpSoftmaxResUb, TSCM<QuePosition::VECIN, 1, 0x4>& bmm2Scm,
    GlobalTensor<KV_T>& keyGm, GlobalTensor<KV_T>& valueGm, __gm__ uint8_t* blocktablePtr,
    const TaskParam& taskParam, const ConstParam& constParam)
{
    // Matmul接口使用限制，传输自定义结构体需要放在最前面,否则matmul的消息传输不完善
    if constexpr (PFAT::isSplitCoreByCube && (PFAT::MM_TYPE == PFAMatMulType::MM_PFA ||
        PFAT::MM_TYPE == PFAMatMulType::MM_IFA_MLA || PFAT::MM_TYPE == PFAMatMulType::MM_DN)) {
        PFAMatmulPolicyData pData = {0};
        pData.rightBufIdx = 4 + taskParam.taskPingPong; // 4,5:value pingpong buf idx
        mm.SetSelfDefineData(pData);
    }
    if constexpr (PFAT::MM_TYPE == PFAMatMulType::MM_PA || PFAT::MM_TYPE == PFAMatMulType::MM_PA_D512) {
        FaPaPolicyData flag = {0};
        flag.bIdx = taskParam.taskBatch;
        flag.nIdx = taskParam.batchNOffset / constParam.headNumRatio;
        flag.s2SingleOffset = taskParam.sInnerOffsetDataSize;
        flag.tensorBAddr = reinterpret_cast<uint64_t>(valueGm.GetPhyAddr()); // currentValue
        flag.blockTableAddr = reinterpret_cast<uint64_t>(blocktablePtr);
        flag.blockTableDim2 = constParam.blockTableDim2;
        flag.blockSize = constParam.blockSize;
        flag.isLayoutBSH = constParam.paLayoutType;
        flag.kvHeadNum = constParam.headNumSize / constParam.headNumRatio;
        flag.kvD = constParam.vHeadSize;
        flag.paBlockNumSum = constParam.paBlockNumSum;
        flag.splitD = (PFAT::vDSize == DSIZE_CONST_512) ? 1 : 0;
        flag.rightBufIdx = 4 + taskParam.taskPingPong; // 4,5:value pingpong buf idx
        mm.SetSelfDefineData(flag);
    }
    if constexpr (PFAT::MM_TYPE == PFAMatMulType::MM_IFA_MLA_PA) {
        PAFlagData flag = {0};
        flag.bIdx = taskParam.taskBatch;
        flag.nIdx = taskParam.batchNOffset / constParam.headNumRatio;
        flag.s2SingleOffset = taskParam.sInnerOffsetDataSize;
        flag.tensorBAddr = reinterpret_cast<uint64_t>(valueGm.GetPhyAddr());   // currentValue
        flag.blockTableAddr = reinterpret_cast<uint64_t>(blocktablePtr);
        flag.blockTableDim2 = constParam.blockTableDim2;
        flag.blockSize = constParam.blockSize;
        flag.isLayoutBSH = constParam.paLayoutType;
        flag.kvHeadNum = constParam.headNumSize / constParam.headNumRatio;
        flag.kvD = constParam.vHeadSize;
        flag.paBlockNumSum = constParam.paBlockNumSum;
        flag.isbmm1 = false;
        mm.SetSelfDefineData(flag);
    }

    mm.SetOrgShape(taskParam.singleProcessSOuterSize,    // M stride for trans a
        constParam.bmm2TilingDataRectN, // N stride for b
        PFAT::sInner,   // Ka为S2尾块向上对齐到64
        constParam.bmm2TilingDataRectKb,    // Kb stride for trans b
        PFAT::vDSize);   // Kc

    if constexpr (PFAT::isBmm2Concat) { // SAMEB concat模式
        // 这里的dSize是原始的D大小，不是对齐之后的；singleM设置-1，按照mm常量化模板的原始配置计算
        mm.SetTail(-1, constParam.vHeadSize, taskParam.singleProcessSInnerBmmTail);
    } else {
        mm.SetTail(taskParam.singleProcessSOuterSize,
            constParam.vHeadSize, taskParam.singleProcessSInnerBmmTail);
    }

    // david只有高精度 if constexpr (PFAT::calcMode == RunMode::HighPrecision || IsSameType<T, bfloat16_t>::value) {
    if constexpr (PFAT::isSplitCoreByCube) {
        if constexpr (PFAT::isBmm2Concat) { // SAMEB concat模式
            // SAMEB dual dst模式时，v0和v1的softmax结果按souterAlign * 32B的大小以交织方式存到L1中
            struct DataCopyParams dataCopyParams1;
            LocalTensor<T> scmTensor = bmm2Scm.template AllocTensor<T>();
            if constexpr (PFAT::useDN) {
                uint32_t singleProcessSInnerAlign16 = (taskParam.singleProcessSInnerBmmTail + 15) >> 4 << 4;
                if (taskParam.singleProcessSInnerBmmTail > (PFAT::sInner / 2)) {
                    dataCopyParams1.blockCount = PFAT::sOuter / 32;
                    dataCopyParams1.blockLen = PFAT::sInner / 2;
                    dataCopyParams1.srcStride = 1;
                    dataCopyParams1.dstStride = static_cast<uint16_t>(singleProcessSInnerAlign16 - PFAT::sInner / 2);
                    struct DataCopyParams dataCopyParams2;
                    dataCopyParams2.blockCount = PFAT::sOuter / 32;
                    dataCopyParams2.blockLen = static_cast<uint16_t>(singleProcessSInnerAlign16 - PFAT::sInner / 2);
                    dataCopyParams2.srcStride = static_cast<uint16_t>(PFAT::sInner - singleProcessSInnerAlign16 + 1);
                    dataCopyParams2.dstStride = PFAT::sInner / 2;
                    SetFlag<HardEvent::V_MTE3>(this->nd2NZEvent);
                    WaitFlag<HardEvent::V_MTE3>(this->nd2NZEvent);
                    DataCopy(scmTensor[constParam.subBlockIdx * PFAT::sOuter * singleProcessSInnerAlign16 / 2], tmpSoftmaxResUb, dataCopyParams1);
                    DataCopy(scmTensor[PFAT::sInner * 8 + constParam.subBlockIdx * PFAT::sOuter * singleProcessSInnerAlign16 / 2],
                        tmpSoftmaxResUb[PFAT::sInner * 32 + 64], dataCopyParams2);
                } else {
                    dataCopyParams1.blockCount = PFAT::sOuter / 32;
                    dataCopyParams1.blockLen = static_cast<uint16_t>(singleProcessSInnerAlign16);
                    dataCopyParams1.srcStride = static_cast<uint16_t>((PFAT::sInner / 2) - singleProcessSInnerAlign16 + 1);
                    dataCopyParams1.dstStride = 0;
                    SetFlag<HardEvent::V_MTE3>(this->nd2NZEvent);
                    WaitFlag<HardEvent::V_MTE3>(this->nd2NZEvent);
                    DataCopy(scmTensor[constParam.subBlockIdx * PFAT::sOuter * singleProcessSInnerAlign16 / 2], tmpSoftmaxResUb, dataCopyParams1);
                }
                bmm2Scm.template EnQue(scmTensor);
                bmm2Scm.template DeQue<T>();
                mm.SetTensorA(scmTensor, true);
            } else {
                dataCopyParams1.blockCount = PFAT::sInner * sizeof(T) / 32; // 32: 按32B为一块
                dataCopyParams1.blockLen = taskParam.singleProcessSOuterSize;
                dataCopyParams1.srcStride = (PFAT::sOuter / 2 + 1) - taskParam.singleProcessSOuterSize;
                dataCopyParams1.dstStride = PFAT::sOuter - taskParam.singleProcessSOuterSize;
                SetFlag<HardEvent::V_MTE3>(this->nd2NZEvent);
                WaitFlag<HardEvent::V_MTE3>(this->nd2NZEvent);
                if (dataCopyParams1.blockLen) {
                    if constexpr (IsSameType<T, int8_t>::value ||
                        IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, hifloat8_t>::value) {
                        DataCopy(scmTensor[constParam.subBlockIdx * PFAT::sOuter * 16], tmpSoftmaxResUb, dataCopyParams1);
                    } else {
                        DataCopy(scmTensor[constParam.subBlockIdx * PFAT::sOuter * 8], tmpSoftmaxResUb, dataCopyParams1);
                    }
                }
                bmm2Scm.template EnQue(scmTensor);
                bmm2Scm.template DeQue<T>();
                mm.SetTensorA(scmTensor);  // SAMEB dual dst模式时，v0和v1设置相同的L1起始地址
            }
            bmm2Scm.template FreeTensor(scmTensor);
        } else { // SAMEB Single dest模式
            SetFlag<HardEvent::V_MTE3>(this->nd2NZEvent);
            WaitFlag<HardEvent::V_MTE3>(this->nd2NZEvent);
            mm.SetTensorA(tmpSoftmaxResUb);
        }
    } else { // NORMAL模式
        SetFlag<HardEvent::V_MTE3>(this->nd2NZEvent);
        WaitFlag<HardEvent::V_MTE3>(this->nd2NZEvent);
        mm.SetTensorA(tmpSoftmaxResUb);
    }

    if constexpr (PFAT::MM_TYPE == PFAMatMulType::MM_PA || PFAT::MM_TYPE == PFAMatMulType::MM_PA_D512 ||
                  PFAT::MM_TYPE == PFAMatMulType::MM_IFA_MLA_PA) {
        mm.SetTensorB(valueGm);
    } else {
        if constexpr (PFAT::IFA_MLA) {
            mm.SetTensorB(keyGm[taskParam.tensorBOffset]);
        } else {
            mm.SetTensorB(valueGm[taskParam.valueOffset]);
        }
    }

    mm.template IterateAll<false>(bmm2ResUb, false, false, true);
}

template <typename PFAT, typename mmType>
__aicore__ inline void PromptFlashAttentionNormalMM2<PFAT, mmType>::IterateAll(
    LocalTensor<mmOutputType>& bmm2ResUb, LocalTensor<T>& tmpSoftmaxResUb, TSCM<QuePosition::VECIN, 1, 0x4>& bmm2Scm,
    TSCM<QuePosition::VECIN, 1, 0x4>& valueScmQueue, const TaskParam& taskParam, const ConstParam& constParam)
{
    // 减少SetOrgShape调用次数，可以减少cv通信次数，判断性能提升
    mm.SetOrgShape(taskParam.singleProcessSOuterSize,  // M stride for trans a
        constParam.bmm2TilingDataRectN,     // N stride for b
        taskParam.mm1SingleCoreN,                     // Ka stride for a
        constParam.bmm2TilingDataRectKb,    // Kb stride for trans b
        constParam.vHeadSize);  // Kc

    mm.SetTail(taskParam.singleProcessSOuterSize,
        constParam.vHeadSize,
        taskParam.singleProcessSInnerBmmTail);

    SetFlag<HardEvent::V_MTE3>(this->nd2NZEvent);
    WaitFlag<HardEvent::V_MTE3>(this->nd2NZEvent);
    mm.SetTensorA(tmpSoftmaxResUb);

    LocalTensor<T> valueScmAntiquant = valueScmQueue.template DeQue<T>();
    mm.SetTensorB(valueScmAntiquant);
    mm.template IterateAll<false>(bmm2ResUb, false, false, true);
    valueScmQueue.template FreeTensor(valueScmAntiquant);
}
#endif  // PROMPT_FLASH_ATTENTION_MM2_H