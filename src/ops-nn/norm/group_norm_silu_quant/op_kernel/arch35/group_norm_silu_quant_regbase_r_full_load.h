/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file group_norm_silu_quant_regbase_r_full_load.h
 * \brief GroupNormSiluQuant arch35 调度: R(reduce)轴全载入(整行进 UB), two-pass 两遍算 mean/rstd。
 *        对偶为 r_partial_load(R 部分载入 + Welford 单遍)。IsGeneralized=false 为常规版; true 为大 channel
 *        通用版(gamma/quantScale 按组载入), 二者原为独立文件, 现合一, 差异走 if constexpr。
 */

#ifndef GROUP_NORM_SILU_QUANT_REGBASE_R_FULL_LOAD_H_
#define GROUP_NORM_SILU_QUANT_REGBASE_R_FULL_LOAD_H_

#include "group_norm_silu_quant_regbase_base.h"

namespace GroupNormSiluQuant {
using namespace AscendC;
template <typename T1, typename T2, bool IsGeneralized, int32_t BUFFER_NUM = 2>
class GroupNormSiluQuantRFullLoad
    : public GroupNormSiluQuantLoadBase<GroupNormSiluQuantRFullLoad<T1, T2, IsGeneralized, BUFFER_NUM>, T1, T2> {
public:
    __aicore__ inline GroupNormSiluQuantRFullLoad(){};

private:
    template <typename D, typename A, typename B>
    friend class GroupNormSiluQuantLoadBase;

    __aicore__ inline void CalNormalize(uint64_t offset, uint32_t numPerCoreTmp)
    {
        if constexpr (IsGeneralized) {
            int64_t numPerCoreExtent = CeilDiv(numPerCoreTmp, onceNumPerCore);
            uint32_t numPerCoreTail = numPerCoreTmp % onceNumPerCore == 0 ? onceNumPerCore :
                                                                            numPerCoreTmp % onceNumPerCore;
            uint32_t numPerCoreProcess = onceNumPerCore;
            uint64_t xGmBaseOffset = blockIdx * tiling->numPerCore * elemNum + offset * elemNum;
            auto evMte2ToVPingF = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            auto evMte2ToVPongF = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            auto evVToMte3PingF = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
            auto evVToMte3PongF = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
            auto evVToMte2F = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            auto evVToMte2PingF = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            auto evVToMte2PongF = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            auto evMte3ToVPingF = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
            auto evMte3ToVPongF = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
            __local_mem__ float* dichotomyAddLocal = (__local_mem__ float*)dichotomyAddTensor.GetPhyAddr();
            for (int64_t i = 0; i < numPerCoreExtent; i++) {
                ProcessTwoPassTileGen(i, numPerCoreExtent, numPerCoreProcess, numPerCoreTail, xGmBaseOffset, offset,
                                      dichotomyAddLocal, evMte2ToVPingF, evMte2ToVPongF, evVToMte3PingF, evVToMte3PongF,
                                      evVToMte2F, evVToMte2PingF, evVToMte2PongF, evMte3ToVPingF, evMte3ToVPongF);
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(evMte2ToVPingF);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(evMte2ToVPongF);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(evVToMte3PingF);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(evVToMte3PongF);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(evVToMte2F);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(evVToMte2PingF);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(evVToMte2PongF);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(evMte3ToVPingF);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(evMte3ToVPongF);
        } else {
            int64_t numPerCoreExtent = CeilDiv(numPerCoreTmp, onceNumPerCore);
            uint32_t numPerCoreTail = numPerCoreTmp % onceNumPerCore == 0 ? onceNumPerCore :
                                                                            numPerCoreTmp % onceNumPerCore;
            uint32_t numPerCoreProcess = onceNumPerCore;
            uint64_t xGmBaseOffset = blockIdx * tiling->numPerCore * elemNum + offset * elemNum;
            ProcessGammaAndBeta(offset);
            PingPongEvents ev = AllocPingPongEvents();
            __local_mem__ float* dichotomyAddLocal = (__local_mem__ float*)dichotomyAddTensor.GetPhyAddr();
            for (int64_t i = 0; i < numPerCoreExtent; i++) {
                ProcessTwoPassTile(i, numPerCoreExtent, numPerCoreProcess, numPerCoreTail, xGmBaseOffset, offset,
                                   dichotomyAddLocal, ev.mte2ToVPing, ev.mte2ToVPong, ev.vToMte3Ping, ev.vToMte3Pong,
                                   ev.vToMte2Ping, ev.vToMte2Pong, ev.mte3ToVPing, ev.mte3ToVPong);
            }
            ReleasePingPongEvents(ev);
        }
    }

    // 常规版 two-pass 全载双缓冲流水的单次迭代。逐行原样保留(CalMeanAndRstd 与 NormalizeAndSwish 均不动)。
    __aicore__ inline void ProcessTwoPassTile(int64_t i, int64_t numPerCoreExtent, uint32_t& numPerCoreProcess,
                                              uint32_t numPerCoreTail, uint64_t xGmBaseOffset, uint64_t offset,
                                              __local_mem__ float* dichotomyAddLocal, event_t eventIDMte2ToVPing,
                                              event_t eventIDMte2ToVPong, event_t eventIDVToMte3Ping,
                                              event_t eventIDVToMte3Pong, event_t eventIDVToMte2Ping,
                                              event_t eventIDVToMte2Pong, event_t eventIDMte3ToVPing,
                                              event_t eventIDMte3ToVPong)
    {
        if (i == numPerCoreExtent - 1) {
            numPerCoreProcess = numPerCoreTail;
        }
        bool isPing = (i % BUFFER_NUM) == 0;
        if (i > 1) {
            WaitFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2Ping : eventIDVToMte2Pong);
        }
        uint32_t xGmOffset = xGmBaseOffset + i * onceNumPerCore * elemNum;
        uint32_t xUbOffset = isPing * processSize;
        uint32_t yUbOffset = isPing * ySpanPerPing;
        CopyX2UB<T1>(xGm[xGmOffset], xTensor[xUbOffset], numPerCoreProcess, elemNum);
        SetFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
        WaitFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
        __local_mem__ T1* xLocal = (__local_mem__ T1*)xTensor[xUbOffset].GetPhyAddr();
        __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor[onceNumPerCore * i].GetPhyAddr();
        __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor[onceNumPerCore * i].GetPhyAddr();
        if (i > 1) {
            WaitFlag<HardEvent::MTE3_V>(isPing ? eventIDMte3ToVPing : eventIDMte3ToVPong);
        }
        CalMeanAndRstd<T1>(xLocal, meanLocal, rstdLocal, dichotomyAddLocal, numPerCoreProcess, dichotomyAddPower,
                           dichotomyAddK, dichotomyAddLastNum, powerOfTwoForReduce, elemNum, eps);
        NormalizeAndSwish(xUbOffset, yUbOffset, offset + i * onceNumPerCore, numPerCoreProcess, i);
        if (i < numPerCoreExtent - BUFFER_NUM) {
            SetFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2Ping : eventIDVToMte2Pong);
        }
        SetFlag<HardEvent::V_MTE3>(isPing ? eventIDVToMte3Ping : eventIDVToMte3Pong);
        WaitFlag<HardEvent::V_MTE3>(isPing ? eventIDVToMte3Ping : eventIDVToMte3Pong);
        CopySilu2Gm<int8_t>(siluGm[xGmOffset], yTensor[yUbOffset], numPerCoreProcess, elemNum);
        if (i < numPerCoreExtent - BUFFER_NUM) {
            SetFlag<HardEvent::MTE3_V>(isPing ? eventIDMte3ToVPing : eventIDMte3ToVPong);
        }
    }

    // 泛化版 two-pass 全载双缓冲流水的单次迭代。逐行原样保留(CalMeanAndRstd 与 NormalizeAndSwish 均不动)。
    __aicore__ inline void ProcessTwoPassTileGen(int64_t i, int64_t numPerCoreExtent, uint32_t& numPerCoreProcess,
                                                 uint32_t numPerCoreTail, uint64_t xGmBaseOffset, uint64_t offset,
                                                 __local_mem__ float* dichotomyAddLocal, event_t eventIDMte2ToVPing,
                                                 event_t eventIDMte2ToVPong, event_t eventIDVToMte3Ping,
                                                 event_t eventIDVToMte3Pong, event_t eventIDVToMte2,
                                                 event_t eventIDVToMte2Ping, event_t eventIDVToMte2Pong,
                                                 event_t eventIDMte3ToVPing, event_t eventIDMte3ToVPong)
    {
        if (i == numPerCoreExtent - 1) {
            numPerCoreProcess = numPerCoreTail;
        }
        bool isPingG = (i % BUFFER_NUM) == 0;
        if (i > 1) {
            WaitFlag<HardEvent::V_MTE2>(isPingG ? eventIDVToMte2Ping : eventIDVToMte2Pong);
        }
        uint32_t xGmOffsetG = xGmBaseOffset + i * onceNumPerCore * elemNum;
        uint32_t xUbOffsetG = isPingG * processSize;
        uint32_t yUbOffsetG = isPingG * ySpanPerPing;
        CopyX2UB<T1>(xGm[xGmOffsetG], xTensor[xUbOffsetG], numPerCoreProcess, elemNum);
        SetFlag<HardEvent::MTE2_V>(isPingG ? eventIDMte2ToVPing : eventIDMte2ToVPong);
        WaitFlag<HardEvent::MTE2_V>(isPingG ? eventIDMte2ToVPing : eventIDMte2ToVPong);
        __local_mem__ T1* xLocalG = (__local_mem__ T1*)xTensor[xUbOffsetG].GetPhyAddr();
        __local_mem__ float* meanLocalG = (__local_mem__ float*)meanTensor[onceNumPerCore * i].GetPhyAddr();
        __local_mem__ float* rstdLocalG = (__local_mem__ float*)rstdTensor[onceNumPerCore * i].GetPhyAddr();
        if (i > 1) {
            WaitFlag<HardEvent::MTE3_V>(isPingG ? eventIDMte3ToVPing : eventIDMte3ToVPong);
        }
        CalMeanAndRstd<T1>(xLocalG, meanLocalG, rstdLocalG, dichotomyAddLocal, numPerCoreProcess, dichotomyAddPower,
                           dichotomyAddK, dichotomyAddLastNum, powerOfTwoForReduce, elemNum, eps);
        if (i > 0) {
            WaitFlag<HardEvent::V_MTE2>(eventIDVToMte2);
        }
        NormalizeAndSwish(xUbOffsetG, yUbOffsetG, offset + i * onceNumPerCore, numPerCoreProcess, i);
        if (i < numPerCoreExtent - 1) {
            SetFlag<HardEvent::V_MTE2>(eventIDVToMte2);
        }
        SetFlag<HardEvent::V_MTE3>(isPingG ? eventIDVToMte3Ping : eventIDVToMte3Pong);
        WaitFlag<HardEvent::V_MTE3>(isPingG ? eventIDVToMte3Ping : eventIDVToMte3Pong);
        if (i < numPerCoreExtent - BUFFER_NUM) {
            SetFlag<HardEvent::V_MTE2>(isPingG ? eventIDVToMte2Ping : eventIDVToMte2Pong);
        }
        CopySilu2Gm<int8_t>(siluGm[xGmOffsetG], yTensor[yUbOffsetG], numPerCoreProcess, elemNum);
        if (i < numPerCoreExtent - BUFFER_NUM) {
            SetFlag<HardEvent::MTE3_V>(isPingG ? eventIDMte3ToVPing : eventIDMte3ToVPong);
        }
    }

    __aicore__ inline void NormalizeAndSwish(uint32_t xUbOffset, uint32_t yUbOffset, uint32_t numPerCoreoffset,
                                             int64_t numPerCoreProcess, uint32_t numPerCoreLoop)
    {
        if constexpr (IsGeneralized) {
            int64_t outputUbOffset = yUbOffset;
            auto eventIDMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            auto eventIDVToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            for (int64_t i = 0; i < numPerCoreProcess; i++) {
                uint64_t gammaOffset = ((blockIdx * tiling->numPerCore + numPerCoreoffset + i) % numGroups) * shapeD;
                uint64_t betaOffset = gammaOffset;
                __local_mem__ T1* xLocal = (__local_mem__ T1*)xTensor[xUbOffset + i * elemNumAlign].GetPhyAddr();
                __local_mem__ int8_t* yOutLocal = (__local_mem__ int8_t*)yTensor[outputUbOffset + i * yRowStride]
                                                      .GetPhyAddr();
                __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor[numPerCoreLoop * onceNumPerCore + i]
                                                     .GetPhyAddr();
                __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor[numPerCoreLoop * onceNumPerCore + i]
                                                     .GetPhyAddr();
                __local_mem__ T2* gammaLocal = hasGamma ? (__local_mem__ T2*)gammaTensor.GetPhyAddr() : nullptr;
                __local_mem__ T2* betaLocal = hasBeta ? (__local_mem__ T2*)betaTensor.GetPhyAddr() : nullptr;
                __local_mem__ float* quantScaleLocal = (__local_mem__ float*)quantScaleTensor.GetPhyAddr();
                if (i > 0) {
                    WaitFlag<HardEvent::V_MTE2>(eventIDVToMte2);
                }
                CopyGammaAndBeta2UB<T2>(gammaGm[gammaOffset], betaGm[betaOffset], gammaTensor, betaTensor, 1, shapeD,
                                        hasGamma, hasBeta);
                // per-channel: 当前组 shapeD 个通道 quantScale; per-tensor: [0]
                CopyQuantScale2UB(quantScaleGm[perChannelScale ? gammaOffset : 0], quantScaleTensor,
                                  perChannelScale ? shapeD : 1);
                SetFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
                VFNormalizeAndSwishQuantUnAlign<T1, T2>(xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal,
                                                        quantScaleLocal, yOutLocal, shapeD, hwNum, activateSilu,
                                                        hasGamma, hasBeta, perChannelScale);
                if (i < numPerCoreProcess - 1) {
                    SetFlag<HardEvent::V_MTE2>(eventIDVToMte2);
                }
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToV);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIDVToMte2);
        } else {
            if (isFold) {
                return NormalizeAndSwishFold(xUbOffset, yUbOffset, numPerCoreoffset, numPerCoreProcess, numPerCoreLoop);
            }
            return NormalizeAndSwishCommon(xUbOffset, yUbOffset, numPerCoreoffset, numPerCoreProcess, numPerCoreLoop);
        }
    }

    __aicore__ inline void NormalizeAndSwishCommon(uint32_t xUbOffset, uint32_t yUbOffset, uint32_t numPerCoreoffset,
                                                   int64_t numPerCoreProcess, uint32_t numPerCoreLoop)
    {
        for (int64_t i = 0; i < numPerCoreProcess; i++) {
            uint64_t gammaOffset = ((blockIdx * tiling->numPerCore + numPerCoreoffset + i) % numGroups) * shapeD;
            uint64_t betaOffset = gammaOffset;
            __local_mem__ T1* xLocal = (__local_mem__ T1*)xTensor[xUbOffset + i * elemNumAlign].GetPhyAddr();
            __local_mem__ int8_t* yOutLocal = (__local_mem__ int8_t*)yTensor[yUbOffset + i * yRowStride].GetPhyAddr();
            __local_mem__ T2* gammaLocal = hasGamma ? (__local_mem__ T2*)gammaTensor[gammaOffset].GetPhyAddr() :
                                                      nullptr;
            __local_mem__ T2* betaLocal = hasBeta ? (__local_mem__ T2*)betaTensor[betaOffset].GetPhyAddr() : nullptr;
            __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor[numPerCoreLoop * onceNumPerCore + i]
                                                 .GetPhyAddr();
            __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor[numPerCoreLoop * onceNumPerCore + i]
                                                 .GetPhyAddr();
            __local_mem__ float* quantScaleLocal = perChannelScale ?
                                                       (__local_mem__ float*)quantScaleTensor[gammaOffset]
                                                           .GetPhyAddr() :
                                                       (__local_mem__ float*)quantScaleTensor.GetPhyAddr();
            VFNormalizeAndSwishQuantUnAlign<T1, T2>(xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal,
                                                    quantScaleLocal, yOutLocal, shapeD, hwNum, activateSilu, hasGamma,
                                                    hasBeta, perChannelScale);
        }
    }

    // HW <= 32场景或者是gamma/beta均为空的场景，对VF做融合，融合后最少一个VF,最多3个VF
    // 假设当前需要处理的A轴范围为:[curNumPerCoreHead, curNumPerCoreTail),那么区间划分存在如下几种情况
    // case1: A轴范围不包含numGroups
    // [curNumPerCoreHead, curNumPerCoreTail)
    // case2: A轴范围包含numGroups
    // [curNumPerCoreHead, n * numGroups) [n * numGroups, m * numGroups) [m * numGroups, curNumPerCoreTail)
    // 其中：n * numGroups = curNumPerCoreHeadUpAlign, m * numGroups = curNumPerCoreTailDownAlign
    // 每个区间对应一个VF
    __aicore__ inline void NormalizeAndSwishFold(uint32_t xUbOffset, uint32_t yUbOffset, uint32_t numPerCoreoffset,
                                                 int64_t numPerCoreProcess, uint32_t numPerCoreLoop)
    {
        uint64_t curNumPerCoreHead = blockIdx * tiling->numPerCore + numPerCoreoffset;
        uint64_t curNumPerCoreTail = curNumPerCoreHead + numPerCoreProcess;
        uint64_t curNumPerCoreHeadUpAlign = CeilDiv(curNumPerCoreHead, numGroups) * numGroups;
        uint64_t curNumPerCoreTailDownAlign = (curNumPerCoreTail / numGroups) * numGroups;
        uint64_t gammaOffset = ((blockIdx * tiling->numPerCore + numPerCoreoffset) % numGroups) * elemNumAlign;
        uint64_t betaOffset = gammaOffset;
        __local_mem__ T1* xLocal = (__local_mem__ T1*)xTensor[xUbOffset].GetPhyAddr();
        __local_mem__ int8_t* yOutLocal = (__local_mem__ int8_t*)yTensor[yUbOffset].GetPhyAddr();
        __local_mem__ T2* gammaLocal = hasGamma ? (__local_mem__ T2*)gammaTensor[gammaOffset].GetPhyAddr() : nullptr;
        __local_mem__ T2* betaLocal = hasBeta ? (__local_mem__ T2*)betaTensor[betaOffset].GetPhyAddr() : nullptr;
        __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor[numPerCoreLoop * onceNumPerCore].GetPhyAddr();
        __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor[numPerCoreLoop * onceNumPerCore].GetPhyAddr();
        // per-tensor: 广播 quantScaleTensor[0]; per-channel: quantScale 已按 fold 布局 NDDMA 复制(同 gamma), 用
        // gammaOffset。
        __local_mem__ float* quantScaleLocal = perChannelScale ?
                                                   (__local_mem__ float*)quantScaleTensor[gammaOffset].GetPhyAddr() :
                                                   (__local_mem__ float*)quantScaleTensor.GetPhyAddr();
        // case1
        if (curNumPerCoreTailDownAlign < curNumPerCoreHeadUpAlign) {
            VFNormalizeAndSwishQuantFold(xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal,
                                         yOutLocal, 1, numPerCoreProcess, shapeD * hwNum, activateSilu, hasGamma,
                                         hasBeta, perChannelScale);
            return;
        }

        // case2
        // 区间1：[curNumPerCoreHead, curNumPerCoreHeadUpAlign)
        if (curNumPerCoreHeadUpAlign - curNumPerCoreHead > 0) {
            VFNormalizeAndSwishQuantFold(xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal,
                                         yOutLocal, 1, curNumPerCoreHeadUpAlign - curNumPerCoreHead, shapeD * hwNum,
                                         activateSilu, hasGamma, hasBeta, perChannelScale);
        }
        // 区间2+区间3: 对齐组 + 尾部
        NormalizeFoldAlignedTail(xLocal, yOutLocal, meanLocal, rstdLocal, curNumPerCoreHead, curNumPerCoreHeadUpAlign,
                                 curNumPerCoreTailDownAlign, curNumPerCoreTail);
    }

    // fold case2 的对齐组(区间2)+尾部(区间3): 指针在传入基础上前进后各调一次 VF。逐行原样保留。
    __aicore__ inline void NormalizeFoldAlignedTail(__local_mem__ T1* xLocal, __local_mem__ int8_t* yOutLocal,
                                                    __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal,
                                                    uint64_t curNumPerCoreHead, uint64_t curNumPerCoreHeadUpAlign,
                                                    uint64_t curNumPerCoreTailDownAlign, uint64_t curNumPerCoreTail)
    {
        // 区间2：[curNumPerCoreHeadUpAlign, curNumPerCoreTailDownAlign)
        uint32_t groupsCount = (curNumPerCoreTailDownAlign - curNumPerCoreHeadUpAlign) / numGroups;
        xLocal = xLocal + (curNumPerCoreHeadUpAlign - curNumPerCoreHead) * elemNumAlign;
        yOutLocal = yOutLocal + (curNumPerCoreHeadUpAlign - curNumPerCoreHead) * yRowStride;
        meanLocal = meanLocal + (curNumPerCoreHeadUpAlign - curNumPerCoreHead);
        rstdLocal = rstdLocal + (curNumPerCoreHeadUpAlign - curNumPerCoreHead);
        __local_mem__ T2* gammaLocal = hasGamma ? (__local_mem__ T2*)gammaTensor.GetPhyAddr() : nullptr;
        __local_mem__ T2* betaLocal = hasBeta ? (__local_mem__ T2*)betaTensor.GetPhyAddr() : nullptr;
        __local_mem__ float* quantScaleLocal = (__local_mem__ float*)
                                                   quantScaleTensor.GetPhyAddr(); // 对齐组: 同 gamma 回到 base
        if (groupsCount > 0) {
            VFNormalizeAndSwishQuantFold(xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal,
                                         yOutLocal, groupsCount, numGroups, shapeD * hwNum, activateSilu, hasGamma,
                                         hasBeta, perChannelScale);
        }
        // 区间3: [curNumPerCoreTailDownAlign, curNumPerCoreTail)
        if (curNumPerCoreTail - curNumPerCoreTailDownAlign > 0) {
            xLocal = xLocal + groupsCount * numGroups * elemNumAlign;
            yOutLocal = yOutLocal + groupsCount * numGroups * yRowStride;
            meanLocal = meanLocal + groupsCount * numGroups;
            rstdLocal = rstdLocal + groupsCount * numGroups;
            VFNormalizeAndSwishQuantFold(xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal,
                                         yOutLocal, 1, curNumPerCoreTail - curNumPerCoreTailDownAlign, shapeD * hwNum,
                                         activateSilu, hasGamma, hasBeta, perChannelScale);
        }
    }

    // 常规版: two-pass 首次迭代前一次性载入整段 gamma/beta/quantScale(fold 走 NDDMA 布局)。仅 IsGeneralized=false
    // 实例化。
    __aicore__ inline void ProcessGammaAndBeta(uint64_t curNumPerCore)
    {
        if (curNumPerCore != 0) {
            return;
        }
        if (isFold) {
            CopyGammaAndBeta2UBByNDDMA<T2>(gammaGm, betaGm, gammaTensor, betaTensor, numGroups, shapeD, hwNum,
                                           elemNumAlign, hasGamma, hasBeta);
            // per-channel+fold: quantScale 按 fold 布局复制(同 gamma); per-tensor: 单值即可
            if (perChannelScale) {
                CopyQuantScale2UBByNDDMA(quantScaleGm, quantScaleTensor, numGroups, shapeD, hwNum, elemNumAlign);
            } else {
                CopyQuantScale2UB(quantScaleGm, quantScaleTensor, shapeQuantScale);
            }
        } else {
            CopyGammaAndBeta2UB<T2>(gammaGm, betaGm, gammaTensor, betaTensor, 1, shapeC, hasGamma, hasBeta);
            CopyQuantScale2UB(quantScaleGm, quantScaleTensor, shapeQuantScale);
        }
        if (hasGamma || hasBeta) {
            auto evMte2ToVPgb = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            SetFlag<HardEvent::MTE2_V>(evMte2ToVPgb);
            WaitFlag<HardEvent::MTE2_V>(evMte2ToVPgb);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(evMte2ToVPgb);
        }
    }

    // 公共项走 base 的 ParseCommonTilingData; 此处补 full 专属(elemNumAlign/onceNumPerCore/yRowStride 等)。
    __aicore__ inline void ParseTilingData()
    {
        this->ParseCommonTilingData();
        elemNumAlign = RoundUp<T1>(elemNum);
        onceNumPerCore = processSize / elemNumAlign;
        // int8 y 每组行距按 32B(int8 block)对齐(DataCopyPad 以 32B 块读),独立 ping/pong 基址防与 x 的 processSize
        // 基址重叠。 x(half)仍用 elemNumAlign(=32B对齐),不动。Common/Fold 两路 y 均按 yRowStride 存,与 CopySilu2Gm 的
        // 32B 块读一致。
        yRowStride = RoundUp<int8_t>(elemNum);
        ySpanPerPing = onceNumPerCore * yRowStride;
        powerOfTwoForReduce = tiling->powerOfTwoForReduce;
        if constexpr (IsGeneralized) {
            hwNumAlign = RoundUp<T1>(hwNum);
        } else {
            isFold = hwNum <= NDDMA_SIZE || (!hasGamma && !hasBeta);
        }
    }

    // full 全载 UB 布局: x/y 双缓冲 + mean/rstd/dichotomy + (bf16/half)meanOut/rstdOut + gamma/beta/quantScale。
    __aicore__ inline void InitInnerBuffer()
    {
        pipe.InitBuffer(innerBuf, ubSize);
        LocalTensor<T1> ubTen = innerBuf.template Get<T1>();
        int32_t xSize = processSize * BUFFER_NUM;
        int32_t ySize = processSize * BUFFER_NUM;
        int32_t realNumPerCore = numPerCore > innerNumPerCore ? innerNumPerCore : numPerCore;
        int32_t meanSize = RoundUp<T1>(realNumPerCore * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t rstdSize = RoundUp<T1>(realNumPerCore * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t dichotomySize = RoundUp<T1>((dichotomyAddPower / FP32_ONE_REPEAT) * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t meanOutSz = 0;
        int32_t rstdOutSz = 0;
        if constexpr (IsSameType<T2, half>::value || IsSameType<T2, bfloat16_t>::value) {
            meanOutSz = RoundUp<T1>(realNumPerCore);
            rstdOutSz = RoundUp<T1>(realNumPerCore);
        }

        int32_t yOffset = xSize;
        int32_t meanOffset = yOffset + ySize;
        int32_t rstdOffset = meanOffset + meanSize;
        int32_t dichotomyAddOffset = rstdOffset + rstdSize;

        xTensor = ubTen;
        yTensor = ubTen[yOffset].template ReinterpretCast<int8_t>();
        meanTensor = ubTen[meanOffset].template ReinterpretCast<float>();
        rstdTensor = ubTen[rstdOffset].template ReinterpretCast<float>();
        dichotomyAddTensor = ubTen[dichotomyAddOffset].template ReinterpretCast<float>();
        int32_t curOff = dichotomyAddOffset + dichotomySize;

        if constexpr (IsSameType<T2, half>::value || IsSameType<T2, bfloat16_t>::value) {
            meanOutTensor = ubTen[curOff];
            curOff += meanOutSz;
            rstdOutTensor = ubTen[curOff];
            curOff += rstdOutSz;
        }

        int32_t optSize;
        if constexpr (IsGeneralized) {
            optSize = RoundUp<T1>(shapeD * (sizeof(T2) / sizeof(T1)));
        } else {
            optSize = isFold ? RoundUp<T1>(static_cast<unsigned long>(numGroups * elemNumAlign) *
                                           (sizeof(T2) / sizeof(T1))) :
                               RoundUp<T1>(static_cast<unsigned long>(shapeC) * (sizeof(T2) / sizeof(T1)));
        }
        if (hasGamma) {
            gammaTensor = ubTen[curOff].template ReinterpretCast<T2>();
            curOff += optSize;
        }
        if (hasBeta) {
            betaTensor = ubTen[curOff].template ReinterpretCast<T2>();
            curOff += optSize;
        }
        quantScaleTensor = ubTen[curOff].template ReinterpretCast<float>();
    }

private:
    const GroupNormSiluQuantRegbaseTilingData* tiling;
    TPipe pipe;
    // input GM tensors
    GlobalTensor<T1> xGm;
    GlobalTensor<T2> gammaGm;
    GlobalTensor<T2> betaGm;

    // output GM tensors
    GlobalTensor<T1> meanGm;
    GlobalTensor<T1> rstdGm;
    GlobalTensor<T2> meanT2Gm;
    GlobalTensor<T2> rstdT2Gm;
    GlobalTensor<int8_t> siluGm;
    GlobalTensor<float> quantScaleGm;

    TBuf<> innerBuf;

    // tiling parameters
    int64_t blockIdx;
    int64_t blockNum;
    bool hasGamma{false};
    bool hasBeta{false};
    int64_t numPerCore;
    int64_t elemNum;
    int64_t elemNumAlign;
    int64_t ubSize;
    int64_t shapeC;
    int64_t shapeD;
    int64_t shapeQuantScale;
    bool perChannelScale{false};
    int64_t hwNum;
    int64_t hwNumAlign;
    int64_t yRowStride;
    int64_t ySpanPerPing;
    int64_t processSize;
    int64_t numGroups;
    int64_t innerNumPerCore{MAX_ONCE_NUM_PER_CORE};
    int64_t onceNumPerCore;
    int64_t dichotomyAddPower;
    int64_t dichotomyAddK;
    int64_t dichotomyAddLastNum;
    int64_t powerOfTwoForReduce;
    float eps;
    bool activateSilu{true};
    bool isFold{false};

    LocalTensor<T1> xTensor;
    LocalTensor<T2> gammaTensor;
    LocalTensor<T2> betaTensor;
    LocalTensor<float> meanTensor;
    LocalTensor<float> rstdTensor;
    LocalTensor<float> dichotomyAddTensor;
    LocalTensor<T1> meanOutTensor;
    LocalTensor<T1> rstdOutTensor;
    LocalTensor<float> quantScaleTensor;
    // output ub tensor
    LocalTensor<int8_t> yTensor;
};

} // namespace GroupNormSiluQuant

#endif
