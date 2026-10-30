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
 * \file mul_addn_align_bf16.h
 * \brief
 */
#ifndef MUL_ADDN_ALIGN_BF16_H
#define MUL_ADDN_ALIGN_BF16_H
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t Align_DIV_COEF = 2;
constexpr int32_t NUM_BFLOAT = 64;
constexpr int32_t NUM_NOT_BFLOAT = 128;

template <typename T1, typename T2>
class KernelMulAddnAlignF16
{
public:
    __aicore__ inline KernelMulAddnAlignF16()
    {}
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, const MulAddnTilingData* tiling_data, TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        dataAlign = blockBytes / sizeof(T1);

        dataAlignCompute = 0;
        if constexpr (std::is_same<T1, bfloat16_t>::value) {
            dataAlignCompute = dataAlign / Align_DIV_COEF;
        } else {
            dataAlignCompute = dataAlign;
        }

        N = tiling_data->N;
        shapeB = tiling_data->shapeB;
        shapeM = tiling_data->shapeM;
        shapeN = tiling_data->shapeN;
        shapeNAlign = tiling_data->shapeNAlign;
        useCoreNum = tiling_data->useCoreNum;
        coreTaskNum = tiling_data->coreTaskNum;
        lastCoreTaskNum = tiling_data->lastCoreTaskNum;

        mLoopNum = tiling_data->mLoopNum;
        mNum = tiling_data->mNum;
        mNumTail = tiling_data->mNumTail;

        startOffset = curBlockIdx * coreTaskNum;
        if (curBlockIdx < useCoreNum - 1) {
            thisCoreLoopNum = coreTaskNum;
        } else if (curBlockIdx == useCoreNum - 1) {
            thisCoreLoopNum = lastCoreTaskNum;
        } else {
            thisCoreLoopNum = 0;
        }

        if constexpr (std::is_same<T1, bfloat16_t>::value) {
            typeNum = NUM_BFLOAT;
            shapeNLoop = (shapeNAlign + typeNum - 1) / typeNum;
            shapeNLast = shapeNAlign - (shapeNLoop - 1) * typeNum;
        } else {
            typeNum = NUM_NOT_BFLOAT;
            shapeNLoop = (shapeNAlign + typeNum - 1) / typeNum;
            shapeNLast = shapeNAlign - (shapeNLoop - 1) * typeNum;
        }

        eventIdMte2ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte3ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T1*>(y), shapeB * shapeM * shapeN);
        x1_ = x1;
        x2_ = x2;
    }

    __aicore__ inline void InitBuffer()
    {
        uint64_t mNumAlign = (mNum + dataAlign - 1) / dataAlign * dataAlign;
        pipe->InitBuffer(inputX1Ub, mNumAlign * sizeof(T1));
        pipe->InitBuffer(inputX1BroadUb, mNumAlign * dataAlign * sizeof(T2));
        pipe->InitBuffer(inputX2Ub, shapeNAlign * sizeof(T1));
        pipe->InitBuffer(outputPingUb, shapeNAlign * mNum * sizeof(T2));
        pipe->InitBuffer(outputPongUb, shapeNAlign * mNum * sizeof(T2));

        if constexpr (std::is_same<T1, bfloat16_t>::value) {
            pipe->InitBuffer(x1CastUb, mNumAlign * sizeof(T2));
            pipe->InitBuffer(x2CastUb, mNumAlign * sizeof(T2));
            pipe->InitBuffer(outCastUb, shapeNAlign * mNum * sizeof(T1));
        }
    }

    __aicore__ inline void GetLocalTensor()
    {
        inputX1Local = inputX1Ub.Get<T1>();
        inputX1BroadLocal = inputX1BroadUb.Get<T2>();
        inputX2Local = inputX2Ub.Get<T1>();
        outputPingLocal = outputPingUb.Get<T2>();
        outputPongLocal = outputPongUb.Get<T2>();

        if constexpr (std::is_same<T1, bfloat16_t>::value) {
            x1CastLocal = x1CastUb.Get<T2>();
            x2CastLocal = x2CastUb.Get<T2>();
            outCastLocal = outCastUb.Get<T1>();
        }
    }

    __aicore__ inline void ReleaseEventID()
    {
        pipe->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
        pipe->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
        pipe->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
    }

    __aicore__ inline void Process()
    {
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        for (uint64_t taskLoop = 0; taskLoop < thisCoreLoopNum; taskLoop++) {
            for (uint64_t mLoop = 0; mLoop < mLoopNum; mLoop++) {
                uint64_t thisLoopNum = mLoop == mLoopNum - 1 ? mNumTail : mNum;
                uint64_t offsetX1 = (startOffset + taskLoop) * shapeM + mLoop * mNum;
                uint64_t offsetX2 = (startOffset + taskLoop) * shapeN;
                uint64_t offsetY = offsetX1 * shapeN;

                const uint32_t dimNum = 2;
                const uint32_t dstShape[dimNum] = {
                    static_cast<uint32_t>(thisLoopNum), static_cast<uint32_t>(dataAlignCompute)};
                const uint32_t srcX1Shape[dimNum] = {static_cast<uint32_t>(thisLoopNum), 1};
                copyInParamsV1 = {1, (uint32_t)(thisLoopNum * sizeof(T1)), 0, 0, 0};
                copyInParamsV2 = {1, (uint32_t)(shapeN * sizeof(T1)), 0, 0, 0};

                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                Duplicate(outputPongLocal, (T2)0, shapeNAlign * thisLoopNum);
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                for (uint64_t n = 0; n < N; n++) {
                    inputX1Gm.SetGlobalBuffer(GetTensorAddr(x1_, n));
                    inputX2Gm.SetGlobalBuffer(GetTensorAddr(x2_, n));
                    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                    DataCopyPad(inputX1Local, inputX1Gm[offsetX1], copyInParamsV1, padParamsFloat);
                    DataCopyPad(inputX2Local, inputX2Gm[offsetX2], copyInParamsV2, padParamsFloat);

                    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                    if constexpr (std::is_same<T1, bfloat16_t>::value) {
                        Cast(x1CastLocal, inputX1Local, RoundMode::CAST_NONE, thisLoopNum);
                        Cast(x2CastLocal, inputX2Local, RoundMode::CAST_NONE, shapeN);
                    } else {
                        x1CastLocal = inputX1Local;
                        x2CastLocal = inputX2Local;
                    }

                    AscendC::BroadCast<T2, dimNum, 1>(inputX1BroadLocal, x1CastLocal, dstShape, srcX1Shape);

                    for (uint64_t loopIdx = 0; loopIdx < shapeNLoop - 1; loopIdx++) {
                        AscendC::Mul(
                            outputPingLocal[typeNum * loopIdx], inputX1BroadLocal, x2CastLocal[typeNum * loopIdx],
                            typeNum, thisLoopNum,
                            {1, 0, 1, static_cast<uint8_t>(shapeNAlign / dataAlignCompute), 1, 0});
                    }
                    AscendC::Mul(
                        outputPingLocal[typeNum * (shapeNLoop - 1)], inputX1BroadLocal,
                        x2CastLocal[typeNum * (shapeNLoop - 1)], shapeNLast, thisLoopNum,
                        {1, 0, 1, static_cast<uint8_t>(shapeNAlign / dataAlignCompute), 1, 0});
                    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                    AscendC::Add(outputPongLocal, outputPongLocal, outputPingLocal, shapeNAlign * thisLoopNum);
                }
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);

                if constexpr (std::is_same<T1, bfloat16_t>::value) {
                    Cast(outCastLocal, outputPongLocal, RoundMode::CAST_RINT, shapeNAlign * thisLoopNum);

                } else {
                    outCastLocal = outputPongLocal;
                }

                SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                if (shapeNAlign == shapeN) {
                    DataCopyPad(
                        outputGm[offsetY], outCastLocal,
                        {(uint16_t)1, (uint32_t)(shapeN * thisLoopNum * sizeof(T1)), 0, 0, 0});
                } else {
                    DataCopyPad(
                        outputGm[offsetY], outCastLocal,
                        {(uint16_t)thisLoopNum, (uint32_t)(shapeN * sizeof(T1)), 0, 0, 0});
                }
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            }
        }
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }

private:
    __aicore__ inline __gm__ T1* GetTensorAddr(GM_ADDR indexListPtr, const int offset)
    {
        __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(indexListPtr);
        uint64_t tensorPtrOffset = *dataAddr;
        __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
        return reinterpret_cast<__gm__ T1*>(*(tensorPtr + offset));
    }

private:
    TPipe* pipe;
    GM_ADDR x1_;
    GM_ADDR x2_;
    GlobalTensor<T1> inputX1Gm, inputX2Gm, outputGm;
    TBuf<TPosition::VECCALC> inputX1Ub, inputX1BroadUb, inputX2Ub, outputPingUb, outputPongUb, x1CastUb, x2CastUb,
        outCastUb, tmpUb;
    LocalTensor<T1> inputX1Local, inputX2Local, outCastLocal, tmpLocal;
    LocalTensor<T2> x1CastLocal, x2CastLocal, inputX1BroadLocal, outputPingLocal, outputPongLocal;

    uint64_t typeNum = 0;
    uint64_t curBlockIdx;
    uint64_t dataAlign, dataAlignCompute, blockBytes = 32;
    uint64_t N, shapeB, shapeM, shapeN, shapeNAlign, shapeNLoop, shapeNLast;
    uint64_t mNum, mLoopNum, mNumTail, startOffset;
    uint64_t thisCoreNumTail, thisCoreLoopNum, coreTaskNum, useCoreNum, lastCoreTaskNum;
    DataCopyExtParams copyInParamsV1, copyInParamsV2, copyInParamsV3, copyOutParams;
    DataCopyPadExtParams<T1> padParamsFloat{false, 0, 0, 0};
    event_t eventIdVToMte2, eventIdVToMte3, eventIdMte2ToV, eventIdMte3ToV;
};

#endif // MUL_ADDN_ALIGN_H