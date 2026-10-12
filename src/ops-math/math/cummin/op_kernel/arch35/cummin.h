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
 * \file Cummin_regbase_common.h
 * \brief Cummin
 */

#ifndef Cummin_REGBASE_COMMON_H
#define Cummin_REGBASE_COMMON_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "cummin_struct.h"
#include "cummin_tiling_key.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/load_store_utils.h"
namespace CumminRegbase {

using namespace AscendC;

constexpr static uint32_t BLOCK_SIZE = 32;
constexpr static uint32_t VREGTENSOR_SIZE = 256;
constexpr static uint32_t BUFFER_NUM = 1;
constexpr static uint32_t THREAD_DIM = 2048;

template <typename X, typename ARGMIN>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) void SimtCompute(__gm__ X* xGm, __gm__ X* yGm, __gm__ ARGMIN* argminGm, int64_t M,
                                                      int64_t R, int64_t N, int64_t count)
{
    const int64_t threadNum = blockDim.x;

    const int64_t RN = R * N;
    int64_t blockId = threadIdx.x;
    int64_t idx = (blockId / N) * RN + (blockId % N);

    while (idx < count) {
        X minVal = xGm[idx];
        ARGMIN minIdx = static_cast<ARGMIN>(0);

        yGm[idx] = minVal;
        argminGm[idx] = minIdx;

        for (int64_t j = 1; j < R; ++j) {
            const int64_t curr_idx = idx + j * N;
            const X curr_val = xGm[curr_idx];

            bool is_not_smaller = (minVal < curr_val);
            minVal = is_not_smaller ? minVal : curr_val;
            minIdx = is_not_smaller ? minIdx : static_cast<ARGMIN>(j);

            yGm[curr_idx] = minVal;
            argminGm[curr_idx] = minIdx;
        }
        blockId += threadNum;
        idx = (blockId / N) * RN + (blockId % N);
    }
}

template <typename X, typename ARGMIN>
class Cummin {
public:
    __aicore__ inline Cummin(TPipe& tpipe, CumminRegbaseTilingData& tilingData) : tpipe(tpipe), tiling(tilingData){};

public:
    __aicore__ inline void InitGMBuffers(GM_ADDR x, GM_ADDR y, GM_ADDR argmin, int64_t offset);
    __aicore__ inline void InitBuffers(int64_t yBufferSize, int64_t rows);
    __aicore__ inline void InitBuffersForN(int64_t yBufferSize);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR argmin);
    __aicore__ inline void InitForSimT(GM_ADDR x, GM_ADDR y, GM_ADDR argmin);
    __aicore__ inline void InsertSync(const HardEvent& event);

    __aicore__ inline void CopyIn(LocalTensor<X>& tensor, uint16_t copyRows, uint32_t dataNum, int64_t xOffset);

    template <typename C>
    __aicore__ inline void CopyOut(GlobalTensor<C>& gm, LocalTensor<C>& tensor, uint16_t copyRows, uint32_t dataNum,
                                   int64_t offset, bool needcast);

    __aicore__ inline void ComputeM();
    __aicore__ inline void ComputeSimT();
    __aicore__ inline void ComputeR(CumminSplitInfo info);
    __aicore__ inline void ComputeN(CumminSplitInfo info);
    __aicore__ inline void ComputeReservedM(GM_ADDR x, GM_ADDR y, GM_ADDR argmin);
    __aicore__ inline void DoCompare(int64_t computeTimes, int64_t offset, int64_t regComputeTimes,
                                     uint32_t reservedRegComputeTimes, int32_t duplicate);

public:
    TPipe& tpipe;
    // tilingData
    CumminRegbaseTilingData& tiling;
    int64_t M = 0;
    int64_t N = 0;
    int64_t R = 0;
    int64_t alignedN = 0;
    bool needCast = sizeof(ARGMIN) == 8;
    int64_t castOffset = 0;
    int64_t splitM = 0;
    int64_t computeM = 0;
    int64_t ReservedM = 0;
    int64_t mRowsPerCore = 0;
    int64_t corenum = 0;
    int64_t perGroupCoreNum = 0;
    int64_t gmOffset = 0;
    int64_t computeLength = 0;
    int64_t allocLength = 0;
    int64_t stride = 0;
    int64_t blockId = 0;
    int64_t blockCount = 0;
    int64_t computeCountForSimt = 0;
    uint32_t dSize = sizeof(X);
    uint32_t regTensorPerComputeLength = VREGTENSOR_SIZE / sizeof(X);
    CumminSplitInfo generalProcessInfo;
    CumminSplitInfo resevedMgeneralProcessInfo;

    GlobalTensor<X> xGm;
    GlobalTensor<X> yGm;
    GlobalTensor<int32_t> argminGm;

    TQue<TPosition::VECIN, BUFFER_NUM> xTmpQue;
    TQue<TPosition::VECIN, BUFFER_NUM> argminTmpQue;
    TQue<TPosition::VECIN, BUFFER_NUM> yQue;
    TQue<TPosition::VECIN, BUFFER_NUM> argminQue;

    LocalTensor<X> tmpXTensor;
    LocalTensor<int32_t> tmpArgminTensor;
    LocalTensor<X> yTensor;
    LocalTensor<int32_t> argminTensor;
};

template <typename X, typename ARGMIN>
__aicore__ inline void Cummin<X, ARGMIN>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR argmin)
{
    blockId = GetBlockIdx();
    blockCount = GetBlockNum();
    corenum = tiling.coreNum;
    perGroupCoreNum = tiling.perGroupCoreNum;
    M = tiling.M;
    N = tiling.N;
    R = tiling.R;

    alignedN = Ops::Base::CeilAlign(N, static_cast<int64_t>(BLOCK_SIZE / dSize));

    splitM = tiling.splitM;
    computeM = tiling.computeM;
    ReservedM = tiling.ReservedM;

    generalProcessInfo = tiling.generalProcessInfo;
    mRowsPerCore = tiling.mRowsPerCore;

    if (blockId < corenum && blockId % perGroupCoreNum < tiling.formerCore) {
        resevedMgeneralProcessInfo = tiling.formerCoreProcessInfo;
        gmOffset = blockCount * mRowsPerCore * R * N + (blockId / perGroupCoreNum) * R * N +
                   (blockId % perGroupCoreNum) * tiling.formerCoreComputeLength;
        computeLength = tiling.formerCoreComputeLength;

    } else {
        resevedMgeneralProcessInfo = tiling.tailCoreProcessInfo;
        gmOffset = blockCount * mRowsPerCore * R * N + (blockId / perGroupCoreNum) * R * N +
                   (blockId % perGroupCoreNum) * tiling.tailCoreComputeLength + tiling.formerCore;
        computeLength = tiling.tailCoreComputeLength;
    }

    auto offset = blockId * mRowsPerCore * N * R;

    InitGMBuffers(x, y, argmin, offset);
}

template <typename X, typename ARGMIN>
__aicore__ inline void Cummin<X, ARGMIN>::InitForSimT(GM_ADDR x, GM_ADDR y, GM_ADDR argmin)
{
    blockId = GetBlockIdx();
    blockCount = GetBlockNum();
    M = tiling.M;
    N = tiling.N;
    R = tiling.R;
    computeM = M / blockCount;
    ReservedM = M % blockCount;
    auto offset = 0;
    if (blockId < ReservedM) {
        computeM = computeM + 1;
        offset = blockId * computeM * N * R;
    } else {
        offset = blockId * computeM * N * R + ReservedM * N * R;
    }
    computeCountForSimt = computeM * N * R;
    InitGMBuffers(x, y, argmin, offset);
}

template <typename X, typename ARGMIN>
__aicore__ inline void Cummin<X, ARGMIN>::ComputeSimT()
{
    asc_vf_call<SimtCompute<X, ARGMIN>>(dim3(THREAD_DIM), (__gm__ X*)(xGm.GetPhyAddr()), (__gm__ X*)(yGm.GetPhyAddr()),
                                        (__gm__ ARGMIN*)(argminGm.GetPhyAddr()), M, R, N, computeCountForSimt);
}

template <typename X, typename ARGMIN>
__aicore__ inline void Cummin<X, ARGMIN>::InitGMBuffers(GM_ADDR x, GM_ADDR y, GM_ADDR argmin, int64_t offset)
{
    xGm.SetGlobalBuffer((__gm__ X*)x + offset);
    yGm.SetGlobalBuffer((__gm__ X*)y + offset);
    argminGm.SetGlobalBuffer((__gm__ int32_t*)argmin + offset * sizeof(ARGMIN) / sizeof(int32_t));
}
template <typename X, typename ARGMIN>
__aicore__ inline void Cummin<X, ARGMIN>::InitBuffers(int64_t yBufferSize, int64_t rows)
{
    auto argminBuffersize = Ops::Base::CeilAlign(yBufferSize, static_cast<int64_t>(VREGTENSOR_SIZE / dSize));
    tpipe.InitBuffer(xTmpQue, BUFFER_NUM, yBufferSize * dSize);
    tpipe.InitBuffer(argminTmpQue, BUFFER_NUM, argminBuffersize * sizeof(int32_t));
    tpipe.InitBuffer(yQue, BUFFER_NUM, yBufferSize * rows * dSize);
    tmpXTensor = xTmpQue.AllocTensor<X>();
    tmpArgminTensor = argminTmpQue.AllocTensor<int32_t>();
    yTensor = yQue.AllocTensor<X>();

    tpipe.InitBuffer(argminQue, BUFFER_NUM, yBufferSize * rows * sizeof(ARGMIN));
    argminTensor = argminQue.AllocTensor<int32_t>();

    if (needCast) {
        castOffset = yBufferSize * rows;
    }
}

template <typename X, typename ARGMIN>
__aicore__ inline void Cummin<X, ARGMIN>::InsertSync(const HardEvent& event)
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    switch (event) {
        case HardEvent::V_MTE3:
            SetFlag<HardEvent::V_MTE3>(eventID);
            WaitFlag<HardEvent::V_MTE3>(eventID);
            break;
        case HardEvent::V_MTE2:
            SetFlag<HardEvent::V_MTE2>(eventID);
            WaitFlag<HardEvent::V_MTE2>(eventID);
            break;
        case HardEvent::MTE3_MTE2:
            SetFlag<HardEvent::MTE3_MTE2>(eventID);
            WaitFlag<HardEvent::MTE3_MTE2>(eventID);
            break;
        case HardEvent::MTE2_V:
            SetFlag<HardEvent::MTE2_V>(eventID);
            WaitFlag<HardEvent::MTE2_V>(eventID);
            break;
        case HardEvent::MTE3_V:
            SetFlag<HardEvent::MTE3_V>(eventID);
            WaitFlag<HardEvent::MTE3_V>(eventID);
            break;
        default:
            break;
    }
}

template <typename X, typename ARGMIN>
__aicore__ inline void Cummin<X, ARGMIN>::CopyIn(LocalTensor<X>& tensor, uint16_t copyRows, uint32_t dataNum,
                                                 int64_t xOffset)
{
    InsertSync(HardEvent::MTE3_MTE2);
    DataCopyPadExtParams<X> dataCopyPadExtParams = {false, 0, 0, 0};
    DataCopyExtParams copyParams{copyRows, dataNum * dSize, stride * dSize, 0, 0};
    DataCopyPad(tensor, this->xGm[xOffset], copyParams, dataCopyPadExtParams);
    InsertSync(HardEvent::MTE2_V);
}

template <typename X, typename ARGMIN>
template <typename C>
__aicore__ inline void Cummin<X, ARGMIN>::CopyOut(GlobalTensor<C>& gm, LocalTensor<C>& tensor, uint16_t copyRows,
                                                  uint32_t dataNum, int64_t offset, bool needCast)
{
    uint32_t size = sizeof(C);
    if (needCast) {
        size = sizeof(ARGMIN);
        offset *= 2;
        auto castTensor = tensor.template ReinterpretCast<int64_t>();
        Cast(castTensor, tensor[castOffset], RoundMode::CAST_NONE, castOffset);
    }
    InsertSync(HardEvent::V_MTE3);
    uint32_t ubStride = (Ops::Base::CeilAlign(dataNum, static_cast<uint32_t>(BLOCK_SIZE / dSize)) -
                         Ops::Base::CeilAlign(dataNum, static_cast<uint32_t>(BLOCK_SIZE / size))) /
                        (BLOCK_SIZE / size);

    DataCopyExtParams copyParams{copyRows, dataNum * size, ubStride, stride * size, 0};
    DataCopyPad(gm[offset], tensor, copyParams);

    InsertSync(HardEvent::MTE3_MTE2);
    InsertSync(HardEvent::MTE3_V);
}

template <typename X, typename ARGMIN>
__aicore__ inline void Cummin<X, ARGMIN>::ComputeM()
{
    InitBuffers(alignedN, computeM * R);
    allocLength = alignedN;
    auto rows = computeM;
    auto regComputeTimes = Ops::Base::CeilDiv(alignedN, static_cast<int64_t>(VREGTENSOR_SIZE / dSize));
    auto reservedRegComputeTimes = alignedN % regTensorPerComputeLength;

    for (auto i = 0; i < splitM; i++) {
        if (i == splitM - 1 && ReservedM > 0) {
            rows = ReservedM;
        }

        auto offset = i * R * N * computeM;
        CopyIn(yTensor, static_cast<uint16_t>(rows * R), N, offset);

        for (auto j = 0; j < rows; j++) {
            DataCopy(tmpXTensor, yTensor[j * R * alignedN], alignedN);

            auto dupLen = Ops::Base::CeilAlign(alignedN, static_cast<int64_t>(VREGTENSOR_SIZE / dSize));
            Duplicate<int32_t>(tmpArgminTensor, 0, dupLen);
            Duplicate<int32_t>(argminTensor[j * R * alignedN + castOffset], 0, dupLen);
            DoCompare(R - 1, j * R * alignedN + alignedN, regComputeTimes, reservedRegComputeTimes, 1);
        }
        CopyOut(yGm, yTensor, R * rows, N, offset, false);
        CopyOut(argminGm, argminTensor, R * rows, N, offset, needCast);
    }
}

template <typename X, typename ARGMIN>
__aicore__ inline void Cummin<X, ARGMIN>::ComputeR(CumminSplitInfo info)
{
    InitBuffers(info.allocLength, info.computeR);
    allocLength = info.allocLength;

    auto regComputeTimes = Ops::Base::CeilDiv(info.allocLength, static_cast<int64_t>(VREGTENSOR_SIZE / dSize));
    auto reservedRegComputeTimes = info.allocLength % regTensorPerComputeLength;

    for (auto i = 0; i < mRowsPerCore; i++) {
        auto rows = info.computeR;
        for (auto j = 0; j < info.splitR; j++) {
            if (j == info.splitR - 1 && info.reservedR > 0) {
                rows = info.reservedR;
            }

            auto offset = i * R * N + j * info.computeR * N;
            CopyIn(yTensor, static_cast<uint16_t>(rows), info.computeLength, offset);

            if (j == 0) {
                auto dupLen = Ops::Base::CeilAlign(info.allocLength, static_cast<int64_t>(VREGTENSOR_SIZE / dSize));
                DataCopy(tmpXTensor, yTensor, info.allocLength);
                Duplicate<int32_t>(argminTensor, 0, dupLen);
                Duplicate<int32_t>(tmpArgminTensor, 0, dupLen);
            }

            DoCompare(rows, 0, regComputeTimes, reservedRegComputeTimes, j * info.computeR);

            CopyOut(yGm, yTensor, rows, info.computeLength, offset, false);
            CopyOut(argminGm, argminTensor, rows, info.computeLength, offset, needCast);
        }
    }
}

template <typename X, typename ARGMIN>
__aicore__ inline void Cummin<X, ARGMIN>::ComputeN(CumminSplitInfo info)
{
    InitBuffers(info.computeN, 1);
    allocLength = info.computeN;

    for (auto i = 0; i < mRowsPerCore; i++) {
        auto length = info.computeN;
        auto regComputeTimes = Ops::Base::CeilDiv(info.computeN, static_cast<int64_t>(VREGTENSOR_SIZE / dSize));
        auto reservedRegComputeTimes = info.computeN % regTensorPerComputeLength;

        for (auto j = 0; j < info.splitN; j++) {
            if (j == info.splitN - 1 && info.reservedN > 0) {
                length = info.reservedN;
                regComputeTimes = Ops::Base::CeilDiv(info.reservedN, static_cast<int64_t>(VREGTENSOR_SIZE / dSize));
                reservedRegComputeTimes = info.reservedN % regTensorPerComputeLength;
            }
            for (auto k = 0; k < R; k++) {
                auto offset = i * R * N + j * info.computeN + k * N;
                if (k == 0) {
                    CopyIn(tmpXTensor, static_cast<uint16_t>(1), length, offset);
                    Duplicate<int32_t>(tmpArgminTensor, 0,
                                       Ops::Base::CeilAlign(length, static_cast<int64_t>(VREGTENSOR_SIZE / dSize)));
                    CopyOut(yGm, tmpXTensor, 1, length, offset, false);
                    if (needCast) {
                        DataCopy(argminTensor[castOffset], tmpArgminTensor,
                                 Ops::Base::CeilAlign(length, static_cast<int64_t>(VREGTENSOR_SIZE / dSize)));
                        CopyOut(argminGm, argminTensor, 1, length, offset, needCast);
                    } else {
                        CopyOut(argminGm, tmpArgminTensor, 1, length, offset, needCast);
                    }

                    continue;
                }
                CopyIn(yTensor, static_cast<uint16_t>(1), length, offset);
                DoCompare(1, 0, regComputeTimes, reservedRegComputeTimes, k);
                CopyOut(yGm, tmpXTensor, 1, length, offset, false);
                CopyOut(argminGm, argminTensor, 1, length, offset, needCast);
            }
        }
    }
}

template <typename X, typename ARGMIN>
__aicore__ inline void Cummin<X, ARGMIN>::ComputeReservedM(GM_ADDR x, GM_ADDR y, GM_ADDR argmin)
{
    InitGMBuffers(x, y, argmin, gmOffset);
    tpipe.Reset();
    stride = N - computeLength;
    mRowsPerCore = 1;
    if (tiling.reservedRows && GetBlockIdx() < corenum) {
        if (resevedMgeneralProcessInfo.isNFullyLoad) {
            ComputeR(resevedMgeneralProcessInfo);
        } else {
            ComputeN(resevedMgeneralProcessInfo);
        }
    }
}

template <typename X, typename ARGMIN>
__aicore__ inline void Cummin<X, ARGMIN>::DoCompare(int64_t computeTimes, int64_t offset, int64_t regComputeTimes,
                                                    uint32_t reservedRegComputeTimes, int32_t duplicate)
{
    __VEC_SCOPE__
    {
        __local_mem__ X* selectXAddr = (__local_mem__ X*)tmpXTensor.GetPhyAddr();
        __local_mem__ int32_t* selectargminAddr = (__local_mem__ int32_t*)tmpArgminTensor.GetPhyAddr();
        __local_mem__ X* yAddr = (__local_mem__ X*)yTensor.GetPhyAddr() + offset;
        __local_mem__ int32_t* argminAddr = (__local_mem__ int32_t*)argminTensor.GetPhyAddr() + offset + castOffset;

        if constexpr (sizeof(X) == 4) {
            Reg::RegTensor<X> xReg;
            Reg::RegTensor<X> yReg;
            Reg::RegTensor<int32_t> argminReg;
            Reg::RegTensor<int32_t> dupReg;
            AscendC::Reg::MaskReg yMaskReg = AscendC::Reg::CreateMask<X, AscendC::Reg::MaskPattern::ALL>();

            AscendC::Reg::MaskReg cmpReg;
            for (int i = 0; i < regComputeTimes; i++) {
                if (i == regComputeTimes - 1 && reservedRegComputeTimes != 0) {
                    yMaskReg = Reg::UpdateMask<X>(reservedRegComputeTimes);
                }
                int64_t tmpOffset = i * VREGTENSOR_SIZE / dSize;
                AscendC::Reg::DataCopy(xReg, selectXAddr + tmpOffset);
                AscendC::Reg::DataCopy(argminReg, selectargminAddr + tmpOffset);
                for (int j = 0; j < computeTimes; j++) {
                    auto yOffset = tmpOffset + j * allocLength;
                    AscendC::Reg::DataCopy(yReg, yAddr + yOffset);
                    AscendC::Reg::Duplicate(dupReg, j + duplicate);
                    AscendC::Reg::Compare<X, CMPMODE::LT>(cmpReg, xReg, yReg, yMaskReg);
                    AscendC::Reg::Select(xReg, xReg, yReg, cmpReg);
                    AscendC::Reg::Select(argminReg, argminReg, dupReg, cmpReg);
                    AscendC::Reg::DataCopy(yAddr + yOffset, xReg, yMaskReg);
                    AscendC::Reg::DataCopy(argminAddr + yOffset, argminReg, yMaskReg);
                }
                AscendC::Reg::DataCopy(selectXAddr + tmpOffset, xReg, yMaskReg);
                AscendC::Reg::DataCopy(selectargminAddr + tmpOffset, argminReg, yMaskReg);
            }
        } else {
            Reg::RegTensor<X> xReg;
            Reg::RegTensor<X> yReg;
            Reg::RegTensor<int32_t> argminReg1;
            Reg::RegTensor<int32_t> argminReg2;
            Reg::RegTensor<int32_t> dupReg;
            AscendC::Reg::MaskReg yMaskReg = AscendC::Reg::CreateMask<X, AscendC::Reg::MaskPattern::ALL>();

            AscendC::Reg::MaskReg argminMaskReg1;
            AscendC::Reg::MaskReg argminMaskReg2;
            AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::LOWEST>(argminMaskReg1, yMaskReg);
            AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::HIGHEST>(argminMaskReg2, yMaskReg);

            AscendC::Reg::MaskReg cmpReg;
            AscendC::Reg::MaskReg cmpReg1;
            AscendC::Reg::MaskReg cmpReg2;
            for (int i = 0; i < regComputeTimes; i++) {
                if (i == regComputeTimes - 1 && reservedRegComputeTimes != 0) {
                    yMaskReg = Reg::UpdateMask<X>(reservedRegComputeTimes);
                    AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::LOWEST>(argminMaskReg1, yMaskReg);
                    AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::HIGHEST>(argminMaskReg2, yMaskReg);
                }
                int64_t tmpOffset = i * VREGTENSOR_SIZE / dSize;
                AscendC::Reg::DataCopy(xReg, selectXAddr + tmpOffset);
                AscendC::Reg::DataCopy(argminReg1, selectargminAddr + tmpOffset);
                AscendC::Reg::DataCopy(argminReg2, selectargminAddr + tmpOffset + 64);
                for (int j = 0; j < computeTimes; j++) {
                    auto yOffset = tmpOffset + j * allocLength;
                    AscendC::Reg::DataCopy(yReg, yAddr + yOffset);
                    AscendC::Reg::Duplicate(dupReg, j + duplicate);

                    AscendC::Reg::Compare<X, CMPMODE::LT>(cmpReg, xReg, yReg, yMaskReg);
                    AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::LOWEST>(cmpReg1, cmpReg);
                    AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::HIGHEST>(cmpReg2, cmpReg);

                    AscendC::Reg::Select(xReg, xReg, yReg, cmpReg);
                    AscendC::Reg::Select(argminReg1, argminReg1, dupReg, cmpReg1);
                    AscendC::Reg::Select(argminReg2, argminReg2, dupReg, cmpReg2);

                    AscendC::Reg::DataCopy(yAddr + yOffset, xReg, yMaskReg);
                    AscendC::Reg::DataCopy(argminAddr + yOffset, argminReg1, argminMaskReg1);
                    AscendC::Reg::DataCopy(argminAddr + yOffset + 64, argminReg2, argminMaskReg2);
                }
                AscendC::Reg::DataCopy(selectXAddr + tmpOffset, xReg, yMaskReg);
                AscendC::Reg::DataCopy(selectargminAddr + tmpOffset, argminReg1, argminMaskReg1);
                AscendC::Reg::DataCopy(selectargminAddr + tmpOffset + 64, argminReg2, argminMaskReg2);
            }
        }
    }
}

} // namespace CumminRegbase

#endif
