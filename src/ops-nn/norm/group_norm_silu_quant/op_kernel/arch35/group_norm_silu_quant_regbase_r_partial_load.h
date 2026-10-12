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

 * \file group_norm_silu_quant_regbase_r_partial_load.h
 * \brief GroupNormSiluQuant arch35 调度: R(reduce)轴部分载入(装不下整行), 流式分块 + Welford 单遍算 mean/rstd。
 *        对偶为 r_full_load(R 全载 + two-pass)。IsGeneralized=false 为常规版; true 为大 channel 通用版
 *        (gamma/quantScale 按通道范围流式载入), 二者原为独立文件, 现合一, 差异走 if constexpr。
*/
#ifndef GROUP_NORM_SILU_QUANT_REGBASE_R_PARTIAL_LOAD_H_
#define GROUP_NORM_SILU_QUANT_REGBASE_R_PARTIAL_LOAD_H_

#include "group_norm_silu_quant_regbase_base.h"
namespace GroupNormSiluQuant {
using namespace AscendC;
template <typename T1, typename T2, bool IsGeneralized, int32_t BUFFER_NUM = 2>
class GroupNormSiluQuantRPartialLoad
    : public GroupNormSiluQuantLoadBase<GroupNormSiluQuantRPartialLoad<T1, T2, IsGeneralized, BUFFER_NUM>, T1, T2> {
public:
    __aicore__ inline GroupNormSiluQuantRPartialLoad(){};

private:
    template <typename D, typename A, typename B>
    friend class GroupNormSiluQuantLoadBase;

    __aicore__ inline void CalNormalize(uint64_t offset, uint32_t curNumPerCore)
    {
        auto eventIDMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        for (uint64_t i = 0; i < curNumPerCore; i++) {
            if (i > 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
            }
            CalMeanAndRstdByWelford(offset + i, i);
            NormalizeAndSwish(offset + i, i);
            if (i < curNumPerCore - 1) {
                SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
            }
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
    }

    __aicore__ inline void CalMeanAndRstdByWelford(uint64_t curNumPerCore, uint64_t curInnerNumPerCore)
    {
        __local_mem__ float* tmpMeanLocal = (__local_mem__ float*)tMeanTensor.GetPhyAddr();
        __local_mem__ float* tmpVarLocal = (__local_mem__ float*)tVarTensor.GetPhyAddr();
        __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor.GetPhyAddr();
        __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor.GetPhyAddr();
        __local_mem__ float* dichotomyAddLocal = (__local_mem__ float*)dichotomyAddTensor.GetPhyAddr();
        uint64_t xGmOffset = blockIdx * tiling->numPerCore * elemNum;
        uint32_t welfordLen = parallelN;
        count = 0;
        auto eventIDMte2ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        auto eventIDMte2ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        auto eventIDVToMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        auto eventIDVToMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        for (int64_t i = 0; i < welfordLoopCount; i++) {
            if (i == welfordLoopCount - 1) {
                welfordLen = welfordLoopTail;
            }
            int64_t xPhase1Offset = (i % BUFFER_NUM) * parallelNAlign;
            bool isPing = (i % BUFFER_NUM) == 0;
            if (i > BUFFER_NUM - 1) {
                WaitFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2Ping : eventIDVToMte2Pong);
            }
            CopyX2UB<T1>(xGm[xGmOffset + curNumPerCore * elemNum + i * parallelN], xPhase1Tensor[xPhase1Offset], 1,
                         welfordLen);
            SetFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
            WaitFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
            __local_mem__ T1* x1Local = (__local_mem__ T1*)xPhase1Tensor[xPhase1Offset].GetPhyAddr();
            count = count + 1;
            float scale = static_cast<float>(1.0) / static_cast<float>(count);
            VFWelfordParallelUpdate<T1>(x1Local, tmpMeanLocal, tmpVarLocal, i, welfordLen, scale);
            if (i < welfordLoopCount - BUFFER_NUM) {
                SetFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2Ping : eventIDVToMte2Pong);
            }
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToVPing);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToVPong);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIDVToMte2Ping);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIDVToMte2Pong);
        if constexpr (!IsGeneralized) {
            ProcessGammaAndBeta(curNumPerCore);
        }
        float reduceScale = float(1.0) / static_cast<float>(elemNum);
        float scale = float(1.0) / static_cast<float>(parallelN);
        VFWelfordParallelFinalize(meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, parallelN,
                                  dichotomyAddPower, dichotomyAddK, dichotomyAddLastNum, curInnerNumPerCore,
                                  welfordLoopTail, reduceScale, scale, count, eps, welfordAlign);
        auto eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void NormalizeAndSwish(uint64_t curNumPerCore, uint64_t curInnerNumPerCore)
    {
        if constexpr (!IsGeneralized) {
            if (curNumPerCore == 0 && (hasGamma || hasBeta)) {
                auto eventIDMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
                SetFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToV);
            }
        }
        if (innerLoopNum == 1) {
            NormalizeAndSwishByChannel(curNumPerCore, curInnerNumPerCore);
        } else {
            NormalizeAndSwishByHW(curNumPerCore, curInnerNumPerCore);
        }
    }

    // 常规版: welford 结束时一次性把整段 gamma/beta/quantScale 载入 UB。仅 IsGeneralized=false 实例化。
    __aicore__ inline void ProcessGammaAndBeta(uint64_t curNumPerCore)
    {
        // 保证normalize阶段MTE2流水在welfordMTE2流水完成之后启动，避免不必要的带宽竞争
        PipeBarrier<PIPE_MTE2>();
        if (curNumPerCore != 0) {
            return;
        }
        CopyGammaAndBeta2UB<T2>(gammaGm, betaGm, gammaTensor, betaTensor, 1, shapeC, hasGamma, hasBeta);
        CopyQuantScale2UB(quantScaleGm, quantScaleTensor, shapeQuantScale);
    }

    __aicore__ inline void NormalizeAndSwishByChannel(int64_t curNumPerCore, uint64_t curInnerNumPerCore)
    {
        if constexpr (IsGeneralized) {
            uint64_t inputBaseOffset = blockIdx * tiling->numPerCore * elemNum;
            uint64_t outputBaseOffset = blockIdx * tiling->numPerCore * elemNum;
            // 行 stride 按读取粒度 VL_FP32 对齐(与 host processSize / Align compute reduceCountAlign 一致),否则小
            // hwNum 过读越界。
            int64_t hwNumAlignLocal = CeilDiv(hwNum, VL_FP32) * VL_FP32;
            int64_t rowsCount = processSize / hwNumAlignLocal;
            int32_t reduceCount = hwNum;
            uint64_t gammaBaseOffset = ((blockIdx * tiling->numPerCore + curNumPerCore) % numGroups) * shapeD;
            auto eventIDMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            auto eventIDMte2ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            auto eventIDMte2ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            auto eventIDVToMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
            auto eventIDVToMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
            auto eventIDVToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            auto eventIDVToMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            auto eventIDVToMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            auto eventIDMte3ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
            auto eventIDMte3ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());

            __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor[curInnerNumPerCore].GetPhyAddr();
            __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor[curInnerNumPerCore].GetPhyAddr();
            __local_mem__ T2* gammaLocal = (__local_mem__ T2*)gammaTensor.GetPhyAddr();
            __local_mem__ T2* betaLocal = (__local_mem__ T2*)betaTensor.GetPhyAddr();
            __local_mem__ float* quantScaleLocal = (__local_mem__ float*)quantScaleTensor.GetPhyAddr();
            for (int64_t i = 0; i < loopNum; i++) {
                ProcessChannelTileGen(i, inputBaseOffset, outputBaseOffset, rowsCount, reduceCount, gammaBaseOffset,
                                      curNumPerCore, meanLocal, rstdLocal, gammaLocal, betaLocal, quantScaleLocal,
                                      eventIDMte2ToV, eventIDMte2ToVPing, eventIDMte2ToVPong, eventIDVToMte3Ping,
                                      eventIDVToMte3Pong, eventIDVToMte2, eventIDVToMte2Ping, eventIDVToMte2Pong,
                                      eventIDMte3ToVPing, eventIDMte3ToVPong);
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToV);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToVPing);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToVPong);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIDVToMte3Ping);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIDVToMte3Pong);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDVToMte2);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIDVToMte2Ping);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIDVToMte2Pong);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIDMte3ToVPing);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIDMte3ToVPong);
        } else {
            int64_t inputBaseOffset = blockIdx * tiling->numPerCore * elemNum;
            int64_t outputBaseOffset = blockIdx * tiling->numPerCore * elemNum;
            // 行 stride 按读取粒度 VL_FP32 对齐(与 host processSize / Align compute reduceCountAlign 一致),否则小
            // hwNum 过读越界。
            int64_t hwNumAlignLocal = CeilDiv(hwNum, VL_FP32) * VL_FP32;
            if (hwNumAlignLocal == 0) {
                return;
            }
            int64_t rowsCount = processSize / hwNumAlignLocal;
            int32_t reduceCount = hwNum;
            uint64_t gammaBaseOffset = ((blockIdx * tiling->numPerCore + curNumPerCore) % numGroups) * shapeD;
            // normalize阶段，采用double-buffer流水排布
            // VECTOR流水启动依赖MTE2_V的正向同步和MTE3_V的反向同步
            // MTE2流水启动依赖V_MTE2的反向同步, MTE3流水启动依赖MTE3_V的正向同步
            PingPongEvents ev = AllocPingPongEvents();
            for (int16_t i = 0; i < loopNum; i++) { // for D
                ProcessChannelTile(i, inputBaseOffset, outputBaseOffset, hwNumAlignLocal, rowsCount, reduceCount,
                                   gammaBaseOffset, curNumPerCore, curInnerNumPerCore, ev.mte2ToVPing, ev.mte2ToVPong,
                                   ev.vToMte3Ping, ev.vToMte3Pong, ev.vToMte2Ping, ev.vToMte2Pong, ev.mte3ToVPing,
                                   ev.mte3ToVPong);
            }
            ReleasePingPongEvents(ev);
        }
    }

    // 常规版 ByChannel 双缓冲流水的单次迭代(for D)。offset/事件全由调用方传入,逐行原样保留。
    __aicore__ inline void ProcessChannelTile(
        int16_t i, int64_t inputBaseOffset, int64_t outputBaseOffset, int64_t hwNumAlign, int64_t& rowsCount,
        int32_t reduceCount, uint64_t gammaBaseOffset, int64_t curNumPerCore, uint64_t curInnerNumPerCore,
        event_t eventIDMte2ToVPing, event_t eventIDMte2ToVPong, event_t eventIDVToMte3Ping, event_t eventIDVToMte3Pong,
        event_t eventIDVToMte2Ping, event_t eventIDVToMte2Pong, event_t eventIDMte3ToVPing, event_t eventIDMte3ToVPong)
    {
        bool isPing = (i % BUFFER_NUM) == 0;
        int64_t inputGmOffset = inputBaseOffset + hwNum * rowsCount * i + elemNum * curNumPerCore;
        int64_t inputUbOffset = isPing ? 0 : processSize;
        if (i == loopNum - 1) {
            rowsCount = loopTail / hwNumAlign;
        }
        if (i > 1) {
            WaitFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2Ping : eventIDVToMte2Pong);
        }
        CopyX2UB(xGm[inputGmOffset], xPhase2Tensor[inputUbOffset], rowsCount, hwNum, hwNumAlign);
        SetFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
        WaitFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
        uint64_t gammaOffset = gammaBaseOffset + i * (processSize / hwNumAlign);
        uint64_t betaOffset = gammaOffset;
        __local_mem__ T1* xLocal = (__local_mem__ T1*)xPhase2Tensor[inputUbOffset].GetPhyAddr();
        __local_mem__ T2* gammaLocal = hasGamma ? (__local_mem__ T2*)gammaTensor[gammaOffset].GetPhyAddr() : nullptr;
        __local_mem__ T2* betaLocal = hasBeta ? (__local_mem__ T2*)betaTensor[betaOffset].GetPhyAddr() : nullptr;
        __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor[curInnerNumPerCore].GetPhyAddr();
        __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor[curInnerNumPerCore].GetPhyAddr();
        __local_mem__ float* quantScaleLocal = perChannelScale ?
                                                   (__local_mem__ float*)quantScaleTensor[gammaOffset].GetPhyAddr() :
                                                   (__local_mem__ float*)quantScaleTensor.GetPhyAddr();
        __local_mem__ int8_t* yOutLocal = (__local_mem__ int8_t*)yTensor[inputUbOffset].GetPhyAddr();
        if (i > 1) {
            WaitFlag<HardEvent::MTE3_V>(isPing ? eventIDMte3ToVPing : eventIDMte3ToVPong);
        }
        VFNormalizeAndSwishQuantAlign<T1, T2>(xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal,
                                              yOutLocal, rowsCount, reduceCount, activateSilu, hasGamma, hasBeta,
                                              perChannelScale);
        SetFlag<HardEvent::V_MTE3>(isPing ? eventIDVToMte3Ping : eventIDVToMte3Pong);
        WaitFlag<HardEvent::V_MTE3>(isPing ? eventIDVToMte3Ping : eventIDVToMte3Pong);
        if (i < loopNum - BUFFER_NUM) {
            SetFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2Ping : eventIDVToMte2Pong);
        }
        int32_t outputGmOffset = outputBaseOffset + hwNum * (processSize / hwNumAlign) * i + elemNum * curNumPerCore;
        CopySilu2Gm<int8_t>(siluGm[outputGmOffset], yTensor[inputUbOffset], rowsCount, hwNum, hwNumAlign);
        if (i < loopNum - BUFFER_NUM) {
            SetFlag<HardEvent::MTE3_V>(isPing ? eventIDMte3ToVPing : eventIDMte3ToVPong);
        }
    }

    // 常规版 ByHW 双缓冲流水的单次内层迭代(for HW)。offset/事件全由调用方传入,逐行原样保留。
    __aicore__ inline void ProcessHwTile(int64_t i, int64_t j, int64_t& copyLen, uint64_t gammaOffset,
                                         uint64_t betaOffset, uint64_t inputBaseOffset, uint64_t outputBaseOffset,
                                         int64_t totalSize, int64_t tailSize, int64_t curNumPerCore,
                                         uint64_t curInnerNumPerCore, event_t eventIDMte2ToVPing,
                                         event_t eventIDMte2ToVPong, event_t eventIDVToMte3Ping,
                                         event_t eventIDVToMte3Pong, event_t eventIDVToMte2Ping,
                                         event_t eventIDVToMte2Pong, event_t eventIDMte3ToVPing,
                                         event_t eventIDMte3ToVPong)
    {
        int64_t inputOffsetH = inputBaseOffset + totalSize * j + hwNum * i + elemNum * curNumPerCore;
        auto extentH = i * innerLoopNum + j;
        bool isPingH = (extentH % BUFFER_NUM) == 0;
        int64_t inputUbOffsetH = isPingH ? 0 : processSize;
        if (j == innerLoopNum - 1) {
            copyLen = tailSize;
        }
        if (extentH > 1) {
            WaitFlag<HardEvent::V_MTE2>(isPingH ? eventIDVToMte2Ping : eventIDVToMte2Pong);
        }
        CopyX2UB(xGm[inputOffsetH], xPhase2Tensor[inputUbOffsetH], 1, copyLen);
        SetFlag<HardEvent::MTE2_V>(isPingH ? eventIDMte2ToVPing : eventIDMte2ToVPong);
        WaitFlag<HardEvent::MTE2_V>(isPingH ? eventIDMte2ToVPing : eventIDMte2ToVPong);
        __local_mem__ T1* xLocalH = (__local_mem__ T1*)xPhase2Tensor[inputUbOffsetH].GetPhyAddr();
        __local_mem__ T2* gammaLocalH = hasGamma ? (__local_mem__ T2*)gammaTensor[gammaOffset].GetPhyAddr() : nullptr;
        __local_mem__ T2* betaLocalH = hasBeta ? (__local_mem__ T2*)betaTensor[betaOffset].GetPhyAddr() : nullptr;
        __local_mem__ float* meanLocalH = (__local_mem__ float*)meanTensor[curInnerNumPerCore].GetPhyAddr();
        __local_mem__ float* rstdLocalH = (__local_mem__ float*)rstdTensor[curInnerNumPerCore].GetPhyAddr();
        __local_mem__ float* quantScaleLocalH = perChannelScale ?
                                                    (__local_mem__ float*)quantScaleTensor[gammaOffset].GetPhyAddr() :
                                                    (__local_mem__ float*)quantScaleTensor.GetPhyAddr();
        __local_mem__ int8_t* yOutLocalH = (__local_mem__ int8_t*)yTensor[inputUbOffsetH].GetPhyAddr();
        if (extentH > 1) {
            WaitFlag<HardEvent::MTE3_V>(isPingH ? eventIDMte3ToVPing : eventIDMte3ToVPong);
        }
        VFNormalizeAndSwishQuantAlign<T1, T2>(xLocalH, gammaLocalH, betaLocalH, meanLocalH, rstdLocalH,
                                              quantScaleLocalH, yOutLocalH, 1, copyLen, activateSilu, hasGamma, hasBeta,
                                              perChannelScale);
        SetFlag<HardEvent::V_MTE3>(isPingH ? eventIDVToMte3Ping : eventIDVToMte3Pong);
        WaitFlag<HardEvent::V_MTE3>(isPingH ? eventIDVToMte3Ping : eventIDVToMte3Pong);
        if (extentH < loopNum * innerLoopNum - BUFFER_NUM) {
            SetFlag<HardEvent::V_MTE2>(isPingH ? eventIDVToMte2Ping : eventIDVToMte2Pong);
        }
        uint64_t outputGmOffsetH = outputBaseOffset + totalSize * j + hwNum * i + elemNum * curNumPerCore;
        CopySilu2Gm<int8_t>(siluGm[outputGmOffsetH], yTensor[inputUbOffsetH], 1, copyLen);
        if (extentH < loopNum * innerLoopNum - BUFFER_NUM) {
            SetFlag<HardEvent::MTE3_V>(isPingH ? eventIDMte3ToVPing : eventIDMte3ToVPong);
        }
    }

    // 泛化版 ByChannel 单次迭代的输入准备段: 载 x + stream gamma/beta/quantScale + 前置双缓冲同步。逐行原样保留。
    __aicore__ inline void PrepareChannelTile(int64_t i, int64_t rowsCount, uint64_t inputGmOffset,
                                              int64_t inputUbOffset, bool isPing, uint64_t gmOffset,
                                              event_t eventIDMte2ToV, event_t eventIDMte2ToVPing,
                                              event_t eventIDMte2ToVPong, event_t eventIDVToMte2,
                                              event_t eventIDVToMte2Ping, event_t eventIDVToMte2Pong,
                                              event_t eventIDMte3ToVPing, event_t eventIDMte3ToVPong)
    {
        if (i > 1) {
            WaitFlag<HardEvent::V_MTE2>(isPing ? eventIDVToMte2Ping : eventIDVToMte2Pong);
        }
        CopyX2UB<T1>(xGm[inputGmOffset], xPhase2Tensor[inputUbOffset], rowsCount, hwNum, hwNumAlign);
        if (i > 0) {
            WaitFlag<HardEvent::V_MTE2>(eventIDVToMte2);
        }
        CopyGammaAndBeta2UB<T2>(gammaGm[gmOffset], betaGm[gmOffset], gammaTensor, betaTensor, 1, rowsCount, hasGamma,
                                hasBeta);
        // per-channel: 按当前通道范围载 quantScale(同 gamma); per-tensor: 单值
        CopyQuantScale2UB(quantScaleGm[perChannelScale ? gmOffset : 0], quantScaleTensor,
                          perChannelScale ? rowsCount : 1);
        if (i > 1) {
            WaitFlag<HardEvent::MTE3_V>(isPing ? eventIDMte3ToVPing : eventIDMte3ToVPong);
        }
        SetFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
        WaitFlag<HardEvent::MTE2_V>(isPing ? eventIDMte2ToVPing : eventIDMte2ToVPong);
        SetFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
    }

    // 泛化版 ByChannel 双缓冲流水的单次迭代(for D): 逐行 stream gamma/beta/quantScale。逐行原样保留。
    __aicore__ inline void ProcessChannelTileGen(
        int64_t i, uint64_t inputBaseOffset, uint64_t outputBaseOffset, int64_t& rowsCount, int32_t reduceCount,
        uint64_t gammaBaseOffset, int64_t curNumPerCore, __local_mem__ float* meanLocalCg,
        __local_mem__ float* rstdLocalCg, __local_mem__ T2* gammaLocalCg, __local_mem__ T2* betaLocalCg,
        __local_mem__ float* quantScaleLocalCg, event_t eventIDMte2ToV, event_t eventIDMte2ToVPing,
        event_t eventIDMte2ToVPong, event_t eventIDVToMte3Ping, event_t eventIDVToMte3Pong, event_t eventIDVToMte2,
        event_t eventIDVToMte2Ping, event_t eventIDVToMte2Pong, event_t eventIDMte3ToVPing, event_t eventIDMte3ToVPong)
    {
        uint64_t inputGmOffsetCg = inputBaseOffset + hwNum * rowsCount * i + elemNum * curNumPerCore;
        bool isPingCg = (i % BUFFER_NUM) == 0;
        int64_t inputUbOffsetCg = isPingCg ? 0 : processSize;
        uint64_t gmOffsetCg = gammaBaseOffset + i * (processSize / hwNumAlign);
        if (i == loopNum - 1) {
            rowsCount = loopTail / hwNumAlign;
        }
        PrepareChannelTile(i, rowsCount, inputGmOffsetCg, inputUbOffsetCg, isPingCg, gmOffsetCg, eventIDMte2ToV,
                           eventIDMte2ToVPing, eventIDMte2ToVPong, eventIDVToMte2, eventIDVToMte2Ping,
                           eventIDVToMte2Pong, eventIDMte3ToVPing, eventIDMte3ToVPong);
        __local_mem__ T1* xLocalCg = (__local_mem__ T1*)xPhase2Tensor[inputUbOffsetCg].GetPhyAddr();
        __local_mem__ int8_t* yOutLocalCg = (__local_mem__ int8_t*)yTensor[inputUbOffsetCg].GetPhyAddr();
        VFNormalizeAndSwishQuantAlign<T1, T2>(xLocalCg, gammaLocalCg, betaLocalCg, meanLocalCg, rstdLocalCg,
                                              quantScaleLocalCg, yOutLocalCg, rowsCount, reduceCount, activateSilu,
                                              hasGamma, hasBeta, perChannelScale);
        SetFlag<HardEvent::V_MTE3>(isPingCg ? eventIDVToMte3Ping : eventIDVToMte3Pong);
        WaitFlag<HardEvent::V_MTE3>(isPingCg ? eventIDVToMte3Ping : eventIDVToMte3Pong);
        if (i < loopNum - 1) {
            SetFlag<HardEvent::V_MTE2>(eventIDVToMte2);
        }
        if (i < loopNum - BUFFER_NUM) {
            SetFlag<HardEvent::V_MTE2>(isPingCg ? eventIDVToMte2Ping : eventIDVToMte2Pong);
        }
        int32_t outputGmOffsetCg = outputBaseOffset + hwNum * (processSize / hwNumAlign) * i + elemNum * curNumPerCore;
        CopySilu2Gm<int8_t>(siluGm[outputGmOffsetCg], yTensor[inputUbOffsetCg], rowsCount, hwNum, hwNumAlign);
        if (i < loopNum - BUFFER_NUM) {
            SetFlag<HardEvent::MTE3_V>(isPingCg ? eventIDMte3ToVPing : eventIDMte3ToVPong);
        }
    }

    // 泛化版 ByHW 双缓冲流水的单次内层迭代(for HW)。逐行原样保留。
    __aicore__ inline void ProcessHwTileGen(int64_t i, int64_t j, int64_t& copyLen, uint64_t inputBaseOffset,
                                            uint64_t outputBaseOffset, int64_t totalSize, int64_t tailSize,
                                            int64_t curNumPerCore, uint64_t curInnerNumPerCore,
                                            event_t eventIDMte2ToVPing, event_t eventIDMte2ToVPong,
                                            event_t eventIDVToMte3Ping, event_t eventIDVToMte3Pong,
                                            event_t eventIDVToMte2Ping, event_t eventIDVToMte2Pong,
                                            event_t eventIDMte3ToVPing, event_t eventIDMte3ToVPong)
    {
        __local_mem__ float* meanLocalHg = (__local_mem__ float*)meanTensor[curInnerNumPerCore].GetPhyAddr();
        __local_mem__ float* rstdLocalHg = (__local_mem__ float*)rstdTensor[curInnerNumPerCore].GetPhyAddr();
        __local_mem__ T2* gammaLocalHg = (__local_mem__ T2*)gammaTensor.GetPhyAddr();
        __local_mem__ T2* betaLocalHg = (__local_mem__ T2*)betaTensor.GetPhyAddr();
        __local_mem__ float* quantScaleLocalHg = (__local_mem__ float*)quantScaleTensor.GetPhyAddr();
        int64_t inputGmOffsetHg = inputBaseOffset + totalSize * j + hwNum * i + elemNum * curNumPerCore;
        auto extentHg = i * innerLoopNum + j;
        bool isPingHg = (extentHg % BUFFER_NUM) == 0;
        int64_t inputUbOffsetHg = isPingHg ? 0 : processSize;
        if (j == innerLoopNum - 1) {
            copyLen = tailSize;
        }
        if (extentHg > 1) {
            WaitFlag<HardEvent::V_MTE2>(isPingHg ? eventIDVToMte2Ping : eventIDVToMte2Pong);
        }
        CopyX2UB<T1>(xGm[inputGmOffsetHg], xPhase2Tensor[inputUbOffsetHg], 1, copyLen);
        SetFlag<HardEvent::MTE2_V>(isPingHg ? eventIDMte2ToVPing : eventIDMte2ToVPong);
        WaitFlag<HardEvent::MTE2_V>(isPingHg ? eventIDMte2ToVPing : eventIDMte2ToVPong);
        if (extentHg > 1) {
            WaitFlag<HardEvent::MTE3_V>(isPingHg ? eventIDMte3ToVPing : eventIDMte3ToVPong);
        }
        __local_mem__ T1* xLocalHg = (__local_mem__ T1*)xPhase2Tensor[inputUbOffsetHg].GetPhyAddr();
        __local_mem__ int8_t* yOutLocalHg = (__local_mem__ int8_t*)yTensor[inputUbOffsetHg].GetPhyAddr();
        VFNormalizeAndSwishQuantAlign<T1, T2>(xLocalHg, gammaLocalHg, betaLocalHg, meanLocalHg, rstdLocalHg,
                                              quantScaleLocalHg, yOutLocalHg, 1, copyLen, activateSilu, hasGamma,
                                              hasBeta, perChannelScale);
        SetFlag<HardEvent::V_MTE3>(isPingHg ? eventIDVToMte3Ping : eventIDVToMte3Pong);
        WaitFlag<HardEvent::V_MTE3>(isPingHg ? eventIDVToMte3Ping : eventIDVToMte3Pong);
        if (extentHg < loopNum * innerLoopNum - BUFFER_NUM) {
            SetFlag<HardEvent::V_MTE2>(isPingHg ? eventIDVToMte2Ping : eventIDVToMte2Pong);
        }
        int32_t outputGmOffsetHg = outputBaseOffset + totalSize * j + hwNum * i + elemNum * curNumPerCore;
        CopySilu2Gm<int8_t>(siluGm[outputGmOffsetHg], yTensor[inputUbOffsetHg], 1, copyLen);
        if (extentHg < loopNum * innerLoopNum - BUFFER_NUM) {
            SetFlag<HardEvent::MTE3_V>(isPingHg ? eventIDMte3ToVPing : eventIDMte3ToVPong);
        }
    }

    __aicore__ inline void NormalizeAndSwishByHW(int64_t curNumPerCore, uint64_t curInnerNumPerCore)
    {
        if constexpr (IsGeneralized) {
            uint64_t inputBaseOffset = blockIdx * tiling->numPerCore * elemNum;
            uint64_t outputBaseOffset = blockIdx * tiling->numPerCore * elemNum;
            int64_t totalSize = processSize;
            int64_t tailSize = innerLoopTail;
            uint64_t gammaBaseOffset = ((blockIdx * tiling->numPerCore + curNumPerCore) % numGroups) * shapeD;
            // 正向同步
            auto eventIDMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            auto eventIDMte2ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            auto eventIDMte2ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            auto eventIDVToMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
            auto eventIDVToMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
            // 反向同步
            auto eventIDVToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            auto eventIDVToMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            auto eventIDVToMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            auto eventIDMte3ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
            auto eventIDMte3ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
            for (int64_t i = 0; i < loopNum; i++) { // for D
                int64_t copyLen = totalSize;
                uint64_t gammaOffset = gammaBaseOffset + i;
                uint64_t betaOffset = gammaOffset;
                if (i > 0) {
                    WaitFlag<HardEvent::V_MTE2>(eventIDVToMte2);
                }
                CopyGammaAndBeta2UB<T2>(gammaGm[gammaOffset], betaGm[betaOffset], gammaTensor, betaTensor, 1, 1,
                                        hasGamma, hasBeta);
                // per-channel: 当前通道单值 quantScale; per-tensor: [0]
                CopyQuantScale2UB(quantScaleGm[perChannelScale ? gammaOffset : 0], quantScaleTensor, 1);
                SetFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIDMte2ToV);
                for (int64_t j = 0; j < innerLoopNum; j++) { // for HW
                    ProcessHwTileGen(i, j, copyLen, inputBaseOffset, outputBaseOffset, totalSize, tailSize,
                                     curNumPerCore, curInnerNumPerCore, eventIDMte2ToVPing, eventIDMte2ToVPong,
                                     eventIDVToMte3Ping, eventIDVToMte3Pong, eventIDVToMte2Ping, eventIDVToMte2Pong,
                                     eventIDMte3ToVPing, eventIDMte3ToVPong);
                }
                if (i < loopNum - 1) {
                    SetFlag<HardEvent::V_MTE2>(eventIDVToMte2);
                }
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToV);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToVPing);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMte2ToVPong);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIDVToMte2);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIDVToMte3Ping);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIDVToMte3Pong);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIDVToMte2Ping);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIDVToMte2Pong);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIDMte3ToVPing);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIDMte3ToVPong);
        } else {
            uint64_t inputBaseOffset = blockIdx * tiling->numPerCore * elemNum;
            uint64_t outputBaseOffset = blockIdx * tiling->numPerCore * elemNum;
            int64_t totalSize = processSize;
            int64_t tailSize = innerLoopTail;
            uint64_t gammaBaseOffset = ((blockIdx * tiling->numPerCore + curNumPerCore) % numGroups) * shapeD;
            PingPongEvents ev = AllocPingPongEvents();
            for (int64_t i = 0; i < loopNum; i++) {
                int64_t copyLen = totalSize;
                uint64_t gammaOffset = gammaBaseOffset + i;
                uint64_t betaOffset = gammaOffset;
                for (int64_t j = 0; j < innerLoopNum; j++) {
                    ProcessHwTile(i, j, copyLen, gammaOffset, betaOffset, inputBaseOffset, outputBaseOffset, totalSize,
                                  tailSize, curNumPerCore, curInnerNumPerCore, ev.mte2ToVPing, ev.mte2ToVPong,
                                  ev.vToMte3Ping, ev.vToMte3Pong, ev.vToMte2Ping, ev.vToMte2Pong, ev.mte3ToVPing,
                                  ev.mte3ToVPong);
                }
            }
            ReleasePingPongEvents(ev);
        }
    }

    __aicore__ inline void ParseTilingData()
    {
        this->ParseCommonTilingData();
        ubElemNum = ubSize / sizeof(T1);
        parallelN = tiling->parallelN;
        parallelNAlign = RoundUp<T1>(parallelN);
        loopNum = tiling->loopNum;
        loopTail = tiling->loopTail;
        innerLoopNum = tiling->innerLoopNum;
        innerLoopTail = tiling->innerLoopTail;
        welfordLoopCount = CeilDiv(elemNum, parallelN);
        welfordAlign = elemNum % parallelN == 0;
        welfordLoopTail = welfordAlign ? parallelN : elemNum - parallelN * (welfordLoopCount - 1);
        if constexpr (IsGeneralized) {
            // byChannel 行对齐按 VL_FP32(读取粒度), 与 host / compute 一致
            hwNumAlign = CeilDiv(hwNum, VL_FP32) * VL_FP32;
        }
    }

    __aicore__ inline void InitInnerBuffer()
    {
        pipe.InitBuffer(innerBuf, ubSize);
        LocalTensor<T1> ubTensor = innerBuf.template Get<T1>();

        int32_t xPhase1Size = parallelNAlign * BUFFER_NUM;
        int32_t tMeanSize = RoundUp<T1>(parallelNAlign * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t tVarSize = RoundUp<T1>(parallelNAlign * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t tMeanOffset = xPhase1Size;
        int32_t tVarOffset = tMeanOffset + tMeanSize;
        int32_t dichotomyAddOffset = tVarOffset + tVarSize;
        int32_t realNumPerCore = numPerCore > innerNumPerCore ? innerNumPerCore : numPerCore;
        int32_t meanSize = RoundUp<T1>(realNumPerCore * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t rstdSize = RoundUp<T1>(realNumPerCore * (FLOAT_BYTE_SIZE / sizeof(T1)));
        int32_t gammaSize = 0;
        int32_t betaSize = 0;
        int32_t quantScaleSize = 0;
        if constexpr (IsGeneralized) {
            int32_t rows = processSize / hwNumAlign;
            int32_t baseSize = (rows == 0) ? BLOCK_SIZE / sizeof(T1) : RoundUp<T1>(rows * (sizeof(T2) / sizeof(T1)));
            gammaSize = hasGamma ? baseSize : 0;
            betaSize = hasBeta ? baseSize : 0;
            // quantScale per-range: 与 gamma 同 rows 个(fp32), 覆盖 per-channel; per-tensor 亦够
            quantScaleSize = (rows == 0) ? BLOCK_SIZE / sizeof(T1) : RoundUp<T1>(rows * (FLOAT_BYTE_SIZE / sizeof(T1)));
        } else {
            gammaSize = hasGamma ? RoundUp<T1>(shapeC * (sizeof(T2) / sizeof(T1))) : 0;
            betaSize = hasBeta ? RoundUp<T1>(shapeC * (sizeof(T2) / sizeof(T1))) : 0;
            quantScaleSize = RoundUp<T1>(shapeQuantScale * (FLOAT_BYTE_SIZE / sizeof(T1)));
        }
        int32_t meanOutSize = 0;
        int32_t rstdOutSize = 0;
        if constexpr (IsSameType<T2, half>::value || IsSameType<T2, bfloat16_t>::value) {
            meanOutSize = RoundUp<T1>(realNumPerCore);
            rstdOutSize = RoundUp<T1>(realNumPerCore);
        }
        int32_t meanOffset = ubElemNum - meanSize - rstdSize - gammaSize - betaSize - quantScaleSize - meanOutSize -
                             rstdOutSize;
        int32_t rstdOffset = meanOffset + meanSize;

        xPhase1Tensor = ubTensor;
        tMeanTensor = ubTensor[tMeanOffset].template ReinterpretCast<float>();
        tVarTensor = ubTensor[tVarOffset].template ReinterpretCast<float>();
        dichotomyAddTensor = ubTensor[dichotomyAddOffset].template ReinterpretCast<float>();
        meanTensor = ubTensor[meanOffset].template ReinterpretCast<float>();
        rstdTensor = ubTensor[rstdOffset].template ReinterpretCast<float>();

        AssignPhase2Tensors(ubTensor, rstdOffset, rstdSize, gammaSize, betaSize, meanOutSize, rstdOutSize);
    }

    __aicore__ inline void AssignPhase2Tensors(LocalTensor<T1> ubTensor, int32_t rstdOffset, int32_t rstdSize,
                                               int32_t gammaSize, int32_t betaSize, int32_t meanOutSize,
                                               int32_t rstdOutSize)
    {
        int32_t xPhase2Size = processSize * BUFFER_NUM;
        xPhase2Tensor = ubTensor;
        yTensor = ubTensor[xPhase2Size].template ReinterpretCast<int8_t>();
        int32_t curOffset = rstdOffset + rstdSize;

        if constexpr (IsSameType<T2, half>::value || IsSameType<T2, bfloat16_t>::value) {
            meanOutTensor = ubTensor[curOffset];
            curOffset += meanOutSize;
            rstdOutTensor = ubTensor[curOffset];
            curOffset += rstdOutSize;
        }

        if (hasGamma) {
            gammaTensor = ubTensor[curOffset].template ReinterpretCast<T2>();
            curOffset += gammaSize;
        }
        if (hasBeta) {
            betaTensor = ubTensor[curOffset].template ReinterpretCast<T2>();
            curOffset += betaSize;
        }
        quantScaleTensor = ubTensor[curOffset].template ReinterpretCast<float>();
    }

private:
    const GroupNormSiluQuantRegbaseTilingData* tiling;
    TPipe pipe;

    GlobalTensor<T1> xGm;
    GlobalTensor<T2> gammaGm;
    GlobalTensor<T2> betaGm;

    GlobalTensor<int8_t> siluGm;
    GlobalTensor<float> quantScaleGm;
    GlobalTensor<T1> meanGm;
    GlobalTensor<T1> rstdGm;
    GlobalTensor<T2> meanT2Gm;
    GlobalTensor<T2> rstdT2Gm;

    TBuf<> innerBuf;

    int64_t blockIdx;
    int64_t blockNum;
    bool hasGamma{false};
    bool hasBeta{false};
    int64_t numPerCore;
    int64_t numGroups;
    int64_t elemNum;
    int64_t ubSize;
    int64_t parallelN;
    int64_t parallelNAlign;
    int64_t welfordLoopCount;
    int64_t welfordLoopTail;
    bool welfordAlign{false};
    int64_t shapeC;
    int64_t shapeD;
    int64_t shapeQuantScale;
    bool perChannelScale{false};
    int64_t ubElemNum;
    int64_t hwNum;
    int64_t hwNumAlign;
    int64_t loopNum;
    int64_t loopTail;
    int64_t processSize;
    int64_t innerLoopNum;
    int64_t innerLoopTail;
    int64_t innerNumPerCore{MAX_ONCE_NUM_PER_CORE};
    int64_t dichotomyAddPower;
    int64_t dichotomyAddK;
    int64_t dichotomyAddLastNum;
    float eps;
    bool activateSilu{true};

    int64_t count{0};

    LocalTensor<T1> xPhase1Tensor;
    LocalTensor<T1> xPhase2Tensor;
    LocalTensor<T2> gammaTensor;
    LocalTensor<T2> betaTensor;

    LocalTensor<float> tMeanTensor;
    LocalTensor<float> tVarTensor;
    LocalTensor<float> dichotomyAddTensor;
    LocalTensor<float> meanTensor;
    LocalTensor<float> rstdTensor;
    LocalTensor<T1> meanOutTensor;
    LocalTensor<T1> rstdOutTensor;

    LocalTensor<float> quantScaleTensor;
    LocalTensor<int8_t> yTensor;
};
} // namespace GroupNormSiluQuant

#endif
