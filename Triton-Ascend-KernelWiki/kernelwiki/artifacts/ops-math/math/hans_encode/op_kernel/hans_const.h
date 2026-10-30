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
 * \file hans_const.h
 * \brief
 */
#ifndef HANS_CONST_H
#define HANS_CONST_H
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
namespace HansCommonNs {
using namespace AscendC;

constexpr int64_t BLOCK_SIZE = 64;
constexpr int64_t EACH_LOOOP_REPEAT_TIMES = 64;
constexpr int64_t EACH_LOOOP_PROCESS_NUM = 4096;
constexpr int64_t PDF_LENGTH = 256;
constexpr int64_t EACH_REPEAT_BYTES = 256;
constexpr int64_t META_INFO_NUM = 128;
constexpr int64_t ENCODE_TAIL_INFO_BYTES_PER_CORE = 8448;
constexpr int64_t BYTE_BIT_NUM = 8;
constexpr int64_t CALC_BUF_SIZE = 128 * 1024;
constexpr int64_t MANTISSA_MASK_NUM = 1024;
constexpr int64_t REPEAT_STRIDE = 8;
constexpr int32_t INT32_LOW_16_BIT_MASK = 1431655765;
constexpr int32_t HOST_START_IDX = 64;
constexpr int32_t DEVICE_START_IDX = 8;
constexpr int32_t LOW_BLOCK_MASK = 252645135;

constexpr int32_t UB_SIZE_MAX = 192 * 1024;
constexpr int32_t COM_HEADER_OFFSET_COREUSE = 1;
constexpr int32_t COM_HEADER_OFFSET_LOOPS = 2;
constexpr int32_t COM_HEADER_OFFSET_FIXEDLOOPS = 3;
constexpr int32_t COM_HEADER_OFFSET_VARLOOPS = 4;
constexpr int32_t COM_HEADER_OFFSET_MATEINFO_BY_BYTES = 512;
constexpr int32_t EACH_LOOOP_STATE_READ_NUM = 64;
constexpr int32_t EACH_LOOOP_COMPRESS_READ_NUM = 4096;
constexpr int32_t MASK_128 = 128;
constexpr int32_t MASK_64 = 64;
constexpr int32_t PROCESS_3072 = 3072;
constexpr int32_t PROCESS_4096 = 4096;

constexpr int32_t CONST_2 = 2;
constexpr int32_t CONST_3 = 3;
constexpr int32_t CONST_4 = 4;
constexpr int32_t CONST_8 = 8;
constexpr int32_t CONST_15 = 15;
constexpr int32_t CONST_16 = 16;
constexpr int32_t CONST_24 = 24;
constexpr int32_t CONST_32 = 32;
constexpr int32_t CONST_64 = 64;
constexpr int32_t CONST_128 = 128;
constexpr int32_t CONST_256 = 256;
constexpr int32_t CONST_192 = 192;
constexpr int32_t CONST_512 = 512;
constexpr int32_t CONST_768 = 768;
constexpr int32_t CONST_1024 = 1024;
constexpr int32_t CONST_2048 = 2048;
constexpr int32_t CONST_4096 = 4096;
constexpr int32_t CONST_4160 = 4160;
constexpr int32_t CONST_8192 = 8192;
constexpr int32_t CONST_65535 = 65535;

// for ARRAY_LENGTH
constexpr int32_t ARRAY_LEN4 = 4;

// bit op
constexpr int32_t SHIFT_LEFT_24 = 16777216;
constexpr uint32_t MANTISSABITMASK_SEC0 = 1977048437;
constexpr uint32_t MANTISSABITMASK_SEC1 = 1568003933;
constexpr uint32_t MANTISSABITMASK_SEC2 = 3613226455;

class EventManager
{
public:
    __aicore__ inline EventManager()
    {}

    __aicore__ inline void InitEvent()
    {
        // MTE3
        this->eventMTE3MTE2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        this->eventMTE3MTE2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        this->eventMTE3SPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_S>());
        this->eventMTE3SPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_S>());
        this->eventMTE3VPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        this->eventMTE3VPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        // MTE2
        this->eventMTE2VPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        this->eventMTE2VPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        this->eventMTE2MTE3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
        this->eventMTE2MTE3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
        // V
        this->eventVMTE3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        this->eventVMTE3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        this->eventVSPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_S>());
        this->eventVSPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_S>());
        this->eventVMTE2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        this->eventVMTE2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());

        // S
        this->eventSVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::S_V>());
        this->eventSVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::S_V>());
    }

    __aicore__ inline void ReleaseEvent()
    {
        // MTE3
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(this->eventMTE3MTE2Ping);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(this->eventMTE3MTE2Pong);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_S>(this->eventMTE3SPing);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_S>(this->eventMTE3SPong);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_V>(this->eventMTE3VPing);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_V>(this->eventMTE3VPong);

        // MTE2
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(this->eventMTE2VPing);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(this->eventMTE2VPong);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(this->eventMTE2MTE3Ping);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(this->eventMTE2MTE3Pong);

        // V
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(this->eventVMTE3Ping);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(this->eventVMTE3Pong);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(this->eventVSPing);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(this->eventVSPong);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(this->eventVMTE2Ping);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(this->eventVMTE2Pong);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::S_V>(this->eventSVPing);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::S_V>(this->eventSVPong);
    }

    // CONTROL
    AscendC::TEventID eventMTE3MTE2Ping;
    AscendC::TEventID eventMTE3MTE2Pong;
    AscendC::TEventID eventMTE3SPing;
    AscendC::TEventID eventMTE3SPong;
    AscendC::TEventID eventMTE3VPing;
    AscendC::TEventID eventMTE3VPong;
    AscendC::TEventID eventMTE2VPing;
    AscendC::TEventID eventMTE2VPong;
    AscendC::TEventID eventMTE2MTE3Ping;
    AscendC::TEventID eventMTE2MTE3Pong;
    // V
    AscendC::TEventID eventVMTE3Ping;
    AscendC::TEventID eventVMTE3Pong;
    AscendC::TEventID eventVSPing;
    AscendC::TEventID eventVSPong;
    AscendC::TEventID eventVMTE2Ping;
    AscendC::TEventID eventVMTE2Pong;

    // S
    AscendC::TEventID eventSVPing;
    AscendC::TEventID eventSVPong;
};

struct DecodeLoopConfig {
    int64_t compressPtr;
    int64_t copmpressPtrCurrentCoreStart;
    int64_t decompressUint8Ptr;
    int64_t firstMoveInSize;
    int64_t lastEncodedRemainLoopNum;
    int64_t mantissaPtr;
    int32_t mantissaFirstLoopNum;
    int32_t repeatTimes;
    int32_t stateBufferBitNumber;
    int32_t bitLevel;
    int32_t loopDataIndex;
    uint64_t negativeZeroSum;
    int32_t currentCoreDecompressPtr;
    int32_t bigLoop;
};

__aicore__ inline void Uint162Int32Helper(
    LocalTensor<uint16_t> srcUint16, LocalTensor<uint16_t> transposeUint16, LocalTensor<int32_t> dstInt32,
    LocalTensor<uint16_t> constZero, int32_t processNum = EACH_LOOOP_PROCESS_NUM)
{
    const int processNumUint16 = 256;
    const int dataBlockSize = 16;
    const int dataBlockNum = 16;
    const int byteScaleInt162Int32 = 2;
    uint8_t repeatTimes = processNum / processNumUint16;
    AscendC::TransDataTo5HDParams transDataParams = {
        false, false, repeatTimes, dataBlockSize / dataBlockSize, processNumUint16 / dataBlockSize};
    int transposeStep = 2;
    int transposeOffset = 8;
    AscendC::LocalTensor<uint16_t> dstLocalList[dataBlockSize];
    for (int i = 0; i < dataBlockNum; i++) {
        dstLocalList[i] = transposeUint16[dataBlockSize * repeatTimes * i];
    }
    AscendC::LocalTensor<uint16_t> srcLocalList[dataBlockSize];
    for (int i = 0; i < dataBlockNum; i++) {
        srcLocalList[i] = srcUint16[dataBlockSize * i];
    }
    AscendC::TransDataTo5HD(dstLocalList, srcLocalList, transDataParams);
    AscendC::PipeBarrier<PIPE_V>();
    transDataParams.dstRepStride = processNumUint16 * byteScaleInt162Int32 / dataBlockSize;
    transDataParams.srcRepStride = dataBlockSize / dataBlockSize;
    for (int i = 0; i < dataBlockNum; i++) {
        dstLocalList[i] = dstInt32.template ReinterpretCast<uint16_t>()[dataBlockSize * transposeStep * i];
    }
    for (int i = 0; i < dataBlockNum; i += transposeStep) {
        srcLocalList[i] = transposeUint16[dataBlockSize * repeatTimes * (i / transposeStep)];
        srcLocalList[i + 1] = constZero[0];
    }
    AscendC::TransDataTo5HD(dstLocalList, srcLocalList, transDataParams);
    PipeBarrier<PIPE_V>();
    for (int i = 0; i < dataBlockNum; i++) {
        dstLocalList[i] = dstInt32.template ReinterpretCast<
            uint16_t>()[dataBlockSize * transposeStep * i + transposeOffset * byteScaleInt162Int32];
    }
    for (int i = 0; i < dataBlockNum; i += transposeStep) {
        srcLocalList[i] = transposeUint16[dataBlockSize * repeatTimes * (i / transposeStep + transposeOffset)];
        srcLocalList[i + 1] = constZero[0];
    }
    AscendC::TransDataTo5HD(dstLocalList, srcLocalList, transDataParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void MallocLocalTensor(
    TBuf<TPosition::VECCALC>& calcBuf, LocalTensor<T>& Tensor, int32_t TensorSize, uint64_t& ubOffset,
    int32_t Scale = 1)
{
    Tensor = calcBuf.GetWithOffset<T>(TensorSize, ubOffset);
    ubOffset += TensorSize * sizeof(T) * Scale;
}

__aicore__ inline void SortPdf(
    LocalTensor<int32_t>& pdfUb, LocalTensor<int32_t>& pdfRangeIndexLocal, LocalTensor<float>& pdfSortLocal,
    LocalTensor<float>& pdfMrgsortLocal, int32_t repeatTimes)
{
    Sort32<float>(
        pdfSortLocal, pdfUb.template ReinterpretCast<float>(), pdfRangeIndexLocal.ReinterpretCast<uint32_t>(),
        repeatTimes);
    PipeBarrier<PIPE_V>();
    const int validBitMode[3] = {3, 7, 15};
    int sortInexArr[4] = {0, 64, 128, 192};
    MrgSortSrcList<float> srcList1(
        pdfSortLocal[sortInexArr[0]], pdfSortLocal[sortInexArr[1]], pdfSortLocal[sortInexArr[CONST_2]],
        pdfSortLocal[sortInexArr[CONST_3]]);
    uint16_t elementLengths[ARRAY_LEN4] = {CONST_32, CONST_32, CONST_32, CONST_32};
    MrgSort4Info srcInfo(elementLengths, false, validBitMode[CONST_2], 1);
    MrgSort<float>(pdfMrgsortLocal, srcList1, srcInfo);
    PipeBarrier<PIPE_V>();
    int sortInexArrSec[4] = {256, 320, 384, 448};
    MrgSortSrcList<float> srcList2(
        pdfSortLocal[sortInexArrSec[0]], pdfSortLocal[sortInexArrSec[1]], pdfSortLocal[sortInexArrSec[CONST_2]],
        pdfSortLocal[sortInexArrSec[CONST_3]]);
    MrgSort<float>(pdfMrgsortLocal[sortInexArrSec[0]], srcList2, srcInfo);
    PipeBarrier<PIPE_V>();
    int sortInexArrLast[4] = {0, 256, 0, 256};
    MrgSortSrcList<float> srcList3(
        pdfMrgsortLocal[sortInexArrLast[0]], pdfMrgsortLocal[sortInexArrLast[1]],
        pdfMrgsortLocal[sortInexArrLast[CONST_2]], pdfMrgsortLocal[sortInexArrLast[CONST_3]]);
    uint16_t eleLengths[ARRAY_LEN4] = {CONST_128, CONST_128, 0, 0};
    MrgSort4Info src256Info(eleLengths, false, validBitMode[0], 1);
    MrgSort<float>(pdfSortLocal, srcList3, src256Info);
    PipeBarrier<PIPE_V>();
}

template <bool IF_BF16>
__aicore__ inline void SetIdxValue(
    LocalTensor<int32_t>* pdfSortIndexReverseUb, LocalTensor<float>& pdfSortLocal, int32_t pdfLen)
{
    int32_t pdfValue = 0;
    int32_t valueInterval = 2;
    for (int32_t i_0 = 0; i_0 < pdfLen; ++i_0) {
        pdfValue = pdfSortLocal.ReinterpretCast<int32_t>().GetValue(valueInterval * i_0 + 1);
        if (IF_BF16) {
            pdfSortIndexReverseUb->ReinterpretCast<uint16_t>().SetValue(i_0, (uint16_t)(pdfValue * PDF_LENGTH));
        } else {
            pdfSortIndexReverseUb->SetValue(i_0, pdfValue * SHIFT_LEFT_24);
        }
    }
}

template <bool IF_BF16>
__aicore__ inline void GetPdfSortIndex(
    TBuf<TPosition::VECCALC>& calcBuf, GlobalTensor<int32_t>& pdfGm, EventManager& eventManager, uint64_t& ubOffset,
    LocalTensor<int32_t>* pdfSortIndexReverseUb)
{
    LocalTensor<int32_t> pdfUb;
    LocalTensor<int32_t> pdfRangeIndexLocal;
    LocalTensor<float> pdfSortLocal;
    LocalTensor<float> pdfMrgsortLocal;
    int pdfUbSize = 256;
    int pdfRangeIndexLocalSize = 256;
    int pdfSortLocalSize = 512;
    int pdfMrgsortLocalSize = 512;
    MallocLocalTensor(calcBuf, pdfUb, pdfUbSize, ubOffset);
    MallocLocalTensor(calcBuf, pdfRangeIndexLocal, pdfRangeIndexLocalSize, ubOffset);
    MallocLocalTensor(calcBuf, pdfSortLocal, pdfSortLocalSize, ubOffset);
    MallocLocalTensor(calcBuf, pdfMrgsortLocal, pdfMrgsortLocalSize, ubOffset);
    DataCopyParams copyParams{1, static_cast<uint16_t>(PDF_LENGTH * sizeof(int32_t) / sizeof(uint8_t)), 0, 0};
    DataCopyPadParams padParams{true, 0, 0, 0};
    DataCopyPad(pdfUb, pdfGm, copyParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventManager.eventMTE2VPing);
    CreateVecIndex(pdfRangeIndexLocal, (int32_t)0, PDF_LENGTH);
    PipeBarrier<PIPE_V>();
    int sortInt32NumPerRepeat = 32;
    int repeatTimes = PDF_LENGTH / sortInt32NumPerRepeat;
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventManager.eventMTE2VPing);
    SortPdf(pdfUb, pdfRangeIndexLocal, pdfSortLocal, pdfMrgsortLocal, repeatTimes);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventManager.eventVSPing);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventManager.eventVSPing);
    if (pdfSortIndexReverseUb != nullptr) {
        SetIdxValue<IF_BF16>(pdfSortIndexReverseUb, pdfSortLocal, PDF_LENGTH);
    }
}

}  // namespace HansCommonNs

#endif // HANS_CONST_H
