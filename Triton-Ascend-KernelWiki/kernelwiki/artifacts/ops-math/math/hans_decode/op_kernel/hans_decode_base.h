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
 * \file hans_decode_base.h
 * \brief
 */
#ifndef HANS_DECODE_BASE_H
#define HANS_DECODE_BASE_H
#ifdef __CCE_UT_TEST__
#include "../../hans_encode/op_kernel/hans_const.h"
#else
#include "../hans_encode/hans_const.h"
#endif


namespace HansDecodeNS {
using namespace HansCommonNs;
using namespace AscendC;

struct HansDecodeInitConfig {
    GM_ADDR mantissaGm;
    GM_ADDR fixedGm;
    GM_ADDR varGm;
    GM_ADDR pdfGm;
    GM_ADDR recoverGm;
    const HansDecodeTilingData* tilingData;
};

template <bool IF_BF16>
class HansDecode
{
public:
    __aicore__ inline HansDecode(){};

    __aicore__ inline void Init(TPipe* pipeIn, const HansDecodeInitConfig& config)
    {
        this->pipe = pipeIn;
        this->pipe->InitBuffer(this->calcBuf, UB_SIZE_MAX);
        this->eventManager = HansCommonNs::EventManager();
        this->reshuff = config.tilingData->reshuff;
        this->compressDeviceGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(config.fixedGm));
        this->compressDeviceWoHeaderGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(config.fixedGm) + COM_HEADER_OFFSET_MATEINFO_BY_BYTES);
        this->compressHostGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(config.varGm));
        this->mantissaGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(config.mantissaGm));
        this->pdfGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(config.pdfGm), PDF_LENGTH);
        this->outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(config.recoverGm));
        this->outputGm_uint8.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(config.recoverGm));
        this->mantissaUint8Size = config.tilingData->mantissaByteSize;
        this->recoverExpUint8Size = config.tilingData->recoverExpByteSize;
        this->recoverTensorUint8Size = config.tilingData->recoverByteSize;
        this->compressTensorUint8Size = config.tilingData->fixedByteSize - COM_HEADER_OFFSET_MATEINFO_BY_BYTES;
        InitInfo();
    }

    __aicore__ inline void Process();

    __aicore__ inline void InitInfo()
    {
        this->id = GetBlockIdx();
        this->actualUseCore = this->compressDeviceGm.GetValue(COM_HEADER_OFFSET_COREUSE);
        int32_t dtype_size = this->recoverTensorUint8Size / this->recoverExpUint8Size;

        if (this->id < this->actualUseCore) {
            int32_t processLoopSumCore = this->compressDeviceGm.GetValue(COM_HEADER_OFFSET_LOOPS);
            this->currentCoreAllExpNumel = (processLoopSumCore / this->actualUseCore) * EACH_LOOOP_REPEAT_TIMES;
            if (this->id == this->actualUseCore - 1) {
                this->currentCoreAllExpNumel =
                    this->currentCoreAllExpNumel + (processLoopSumCore % this->actualUseCore) * EACH_LOOOP_REPEAT_TIMES;
            }
            InitCompressInfo(processLoopSumCore, dtype_size);
            InitHostInfo(processLoopSumCore, dtype_size);
        }
    }

    __aicore__ inline void InitCompressInfo(int32_t processLoopSumCore, int32_t dtype_size)
    {
        if (this->reshuff == false) {
            this->compressPtrCurrentCoreStartOffset = (this->compressTensorUint8Size / this->actualUseCore) * id;
            this->currentCoreCompressUint8Size = this->compressDeviceGm.GetValue(DEVICE_START_IDX + id);
            this->currentCoreCompressOffset =
                (this->compressTensorUint8Size / this->actualUseCore) * id + this->currentCoreCompressUint8Size;
        } else {
            this->currentCoreCompressOffset = 0;
            int cur_len = 0;
            for (int32_t k = 0; k < this->id + 1; ++k) {
                this->currentCoreCompressOffset += this->compressDeviceGm.GetValue(DEVICE_START_IDX + k);
                if (k == GetBlockIdx()) {
                    cur_len = this->compressDeviceGm.GetValue(DEVICE_START_IDX + k);
                }
            }
            this->compressPtrCurrentCoreStartOffset = this->currentCoreCompressOffset - cur_len;
        }
    }

    __aicore__ inline void InitHostInfo(int32_t processLoopSumCore, int32_t dtype_size)
    {
        this->currentCoreHostExpSize = this->compressDeviceGm.GetValue(HOST_START_IDX + id);
        this->currentCoreHostExpOffset = 0;
        for (int32_t i = 0; i < this->id; i++) {
            int64_t exp_length = this->compressDeviceGm.GetValue(HOST_START_IDX + i);
            this->currentCoreHostExpOffset = this->currentCoreHostExpOffset + exp_length;
        }
        this->currentCoreHostMantissaSize = this->currentCoreHostExpSize * (dtype_size - 1);
        this->currentCoreRecoverTensorUint8Size = this->currentCoreAllExpNumel - this->currentCoreHostExpSize;
        this->currentCoreRecoverTensorOffset =
            (processLoopSumCore / this->actualUseCore) * EACH_LOOOP_REPEAT_TIMES * this->id +
            this->currentCoreRecoverTensorUint8Size;
        this->currentCoreMantissaDeviceUint8Size = (dtype_size - 1) * this->currentCoreRecoverTensorUint8Size;
        this->currentCoreMantissaDeviceOffset =
            (processLoopSumCore / this->actualUseCore) * EACH_LOOOP_REPEAT_TIMES * this->id * (dtype_size - 1) +
            this->currentCoreMantissaDeviceUint8Size;
    }

private:
    GlobalTensor<int32_t> pdfGm;
    GlobalTensor<int32_t> compressDeviceGm;
    GlobalTensor<uint8_t> compressDeviceWoHeaderGm;
    GlobalTensor<uint8_t> mantissaGm;
    GlobalTensor<uint8_t> compressHostGm;
    GlobalTensor<uint16_t> outputGm;
    GlobalTensor<uint8_t> outputGm_uint8;
    LocalTensor<int32_t> pdfUb;
    LocalTensor<int32_t> pdfRangeIndexLocal;
    LocalTensor<float> pdfSortLocal;
    LocalTensor<float> pdfMrgsortLocal;
    LocalTensor<int16_t> inputUb;
    LocalTensor<int16_t> backupInputUb;
    LocalTensor<int32_t> inputBufferUb;
    LocalTensor<int32_t> vnchTempUb;
    LocalTensor<int32_t> constZeroUb;
    LocalTensor<int32_t> constOneUb;
    LocalTensor<int32_t> const16Ub;
    LocalTensor<int32_t> stateBufferBitCmpUb;
    LocalTensor<int32_t> stateBufferShlScaleUb;
    LocalTensor<int32_t> bitLevelMaskZeroUb;
    LocalTensor<int32_t> bitLevelMaskFirstUb;
    LocalTensor<int32_t> bitLevelMaskSecondUb;
    LocalTensor<int16_t> stateBufferRemainUb;
    LocalTensor<int16_t> stateBufferLowUb;
    LocalTensor<int32_t> stateBufferUb;
    LocalTensor<int32_t> stateBufferBitNumberUb;
    LocalTensor<int32_t> bitLevelMaskZeroAndScaleUb;
    LocalTensor<int32_t> outputPdfIndexInt32Ub;
    LocalTensor<int32_t> outputExpInt32Ub;
    LocalTensor<int32_t> int32LowMask;
    LocalTensor<int32_t> negativeZeroCmpBitMask;
    LocalTensor<int32_t> negativeBitTempSpace;
    LocalTensor<int32_t> bitShlMulScaleUb;
    LocalTensor<int32_t> recoverMantissaInt32Ub;
    LocalTensor<half> recoverMantissaFp16Ub;
    LocalTensor<int32_t> recoverMantissaFp162s32Ub;
    LocalTensor<int32_t> recoverMantissaFp16GatherUb;
    LocalTensor<int32_t> recoverMantissaUint8Ub;
    LocalTensor<uint8_t> hostExpUbs;
    LocalTensor<uint8_t> hostExpUbp;
    LocalTensor<uint8_t> mantissaUbs;
    LocalTensor<uint8_t> mantissaUbp;
    LocalTensor<uint8_t> mantissaFp16Ubs;
    LocalTensor<uint8_t> mantissaFp16Ubp;
    LocalTensor<uint8_t> mantissaInt32Ubs;
    LocalTensor<uint8_t> mantissaInt32Ubp;
    LocalTensor<uint8_t> hostExpUb[2];
    LocalTensor<uint8_t> mantissaUb[2];
    LocalTensor<uint8_t> deviceExpUb[2];
    LocalTensor<uint8_t> recoverUb[2];
    TPipe* pipe;
    TBuf<TPosition::VECCALC> calcBuf;
    HansCommonNs::EventManager eventManager;
    int32_t id;
    int32_t actualUseCore;
    int64_t compressTensorUint8Size;
    int64_t mantissaUint8Size;
    int64_t recoverExpUint8Size;
    int64_t recoverTensorUint8Size;
    uint64_t ubOffset{0};
    bool reshuff;
    uint64_t rsvdCnt;
    int64_t currentCoreCompressUint8Size;
    int64_t currentCoreCompressOffset;
    int64_t compressPtrCurrentCoreStartOffset;
    int64_t currentCoreRecoverTensorUint8Size;
    int64_t currentCoreRecoverTensorOffset;
    int64_t currentCoreMantissaDeviceUint8Size;
    int64_t currentCoreMantissaDeviceOffset;
    int64_t currentCoreHostExpSize;
    int64_t currentCoreHostExpOffset;
    int64_t currentCoreHostMantissaSize;
    int64_t currentCoreAllExpNumel;

private:
    __aicore__ inline void MallocUbforDecompress();
    __aicore__ inline void MallocUbforH2D(int32_t process_num, int32_t dtype_size);
    __aicore__ inline void InitLocalTensor();
    __aicore__ inline void PreProcess(DecodeLoopConfig* config);
    __aicore__ inline void PrepareMantissa(DecodeLoopConfig* config);
    __aicore__ inline void GetNegativeBitNum(DecodeLoopConfig* config);
    __aicore__ inline void DataCopyAndSelCompare(DecodeLoopConfig* config);
    __aicore__ inline void UpdateStateBuffer(DecodeLoopConfig* config);
    __aicore__ inline void ReadDevice(DecodeLoopConfig* config);
    __aicore__ inline void DecompressSymbol(LocalTensor<int32_t> pdfSortIndexReverseUb, int32_t loopIdx);
    __aicore__ inline void MoveOutBf16(DecodeLoopConfig* config, int32_t loopIdx);
    __aicore__ inline void MoveOutFp32(
        LocalTensor<uint32_t> mantissaBitMask, DecodeLoopConfig* config, int32_t loopIdx);
    __aicore__ inline void MoveOut(LocalTensor<uint32_t> mantissaBitMask, DecodeLoopConfig* config, int32_t loopIdx);
    __aicore__ inline void InitLoopConfig(DecodeLoopConfig* config);
    __aicore__ inline void Decompress(
        LocalTensor<int32_t> pdfSortIndexReverseUb, LocalTensor<uint32_t> mantissaBitMask);
    __aicore__ inline void ProcessBf16BeforeH2D(int32_t index, uint32_t size, uint32_t expShlDistance);
    __aicore__ inline void ProcessFp32BeforeH2D(
        LocalTensor<uint32_t> mantissaBitMask, int32_t index, uint32_t size, int32_t process_num,
        uint32_t expShlDistance);
    __aicore__ inline void DataCopy2OutputGm(int32_t outputOffsetNum, int32_t dtype_size, int32_t index, uint32_t size);
    __aicore__ inline void Host2Device(LocalTensor<uint32_t> mantissaBitMask);
};

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::MallocUbforDecompress()
{
    MallocLocalTensor(calcBuf, inputUb, CONST_4160, ubOffset);
    MallocLocalTensor(calcBuf, backupInputUb, CONST_4160, ubOffset);
    MallocLocalTensor(calcBuf, inputBufferUb, CONST_4096, ubOffset);
    MallocLocalTensor(calcBuf, vnchTempUb, CONST_4096, ubOffset);
    MallocLocalTensor(calcBuf, bitLevelMaskSecondUb, CONST_4096, ubOffset);
    MallocLocalTensor(calcBuf, stateBufferRemainUb, CONST_4096, ubOffset);
    MallocLocalTensor(calcBuf, stateBufferLowUb, CONST_4096, ubOffset, 2);
    MallocLocalTensor(calcBuf, stateBufferUb, CONST_4096, ubOffset);
    MallocLocalTensor(calcBuf, outputPdfIndexInt32Ub, CONST_4096, ubOffset);
    MallocLocalTensor(calcBuf, outputExpInt32Ub, CONST_4096, ubOffset);
    MallocLocalTensor(calcBuf, int32LowMask, CONST_4096, ubOffset);
    MallocLocalTensor(calcBuf, bitLevelMaskFirstUb, CONST_512, ubOffset);
    MallocLocalTensor(calcBuf, constZeroUb, CONST_256, ubOffset);
    MallocLocalTensor(calcBuf, constOneUb, CONST_64, ubOffset);
    MallocLocalTensor(calcBuf, const16Ub, CONST_64, ubOffset);
    MallocLocalTensor(calcBuf, stateBufferBitCmpUb, CONST_64, ubOffset);
    MallocLocalTensor(calcBuf, stateBufferShlScaleUb, CONST_64, ubOffset);
    MallocLocalTensor(calcBuf, bitLevelMaskZeroUb, CONST_64, ubOffset);
    MallocLocalTensor(calcBuf, stateBufferBitNumberUb, CONST_64, ubOffset);
    MallocLocalTensor(calcBuf, bitLevelMaskZeroAndScaleUb, CONST_64, ubOffset);
    MallocLocalTensor(calcBuf, negativeBitTempSpace, CONST_64, ubOffset);
    MallocLocalTensor(calcBuf, bitShlMulScaleUb, CONST_64, ubOffset);
    MallocLocalTensor(calcBuf, negativeZeroCmpBitMask, CONST_8, ubOffset);
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::MallocUbforH2D(int32_t process_num, int32_t dtype_size)
{
    MallocLocalTensor(calcBuf, hostExpUbs, process_num, ubOffset);
    MallocLocalTensor(calcBuf, hostExpUbp, process_num, ubOffset);
    MallocLocalTensor(calcBuf, mantissaUbs, process_num * (dtype_size - 1), ubOffset);
    MallocLocalTensor(calcBuf, mantissaUbp, process_num * (dtype_size - 1), ubOffset);
    MallocLocalTensor(calcBuf, mantissaFp16Ubs, process_num * (dtype_size) * 2, ubOffset);
    MallocLocalTensor(calcBuf, mantissaFp16Ubp, process_num * (dtype_size) * 2, ubOffset);
    MallocLocalTensor(calcBuf, mantissaInt32Ubs, process_num * (dtype_size - 1) * 4, ubOffset);
    MallocLocalTensor(calcBuf, mantissaInt32Ubp, process_num * (dtype_size - 1) * 4, ubOffset);
    hostExpUb[0] = hostExpUbs;
    hostExpUb[1] = hostExpUbp;
    mantissaUb[0] = mantissaUbs;
    mantissaUb[1] = mantissaUbp;
    deviceExpUb[0] = mantissaFp16Ubs;
    deviceExpUb[1] = mantissaFp16Ubp;
    recoverUb[0] = mantissaInt32Ubs;
    recoverUb[1] = mantissaInt32Ubp;
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::InitLocalTensor()
{
    for (int32_t k = 0; k < CONST_16; ++k) {
        bitShlMulScaleUb.SetValue(k, (1 << k) - 1);
    }
    SetVectorMask<uint64_t, MaskMode::NORMAL>(0xffffffffffffffff, 0x0);
    Duplicate<int16_t>(inputUb, (int16_t)0, CONST_4160);
    PipeBarrier<PIPE_V>();
    Duplicate<int16_t>(backupInputUb, (int16_t)0, CONST_4160);
    PipeBarrier<PIPE_V>();
    Duplicate<int16_t>(stateBufferRemainUb, (int16_t)1, CONST_4096);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(int32LowMask, (int32_t)1, CONST_4096);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(constZeroUb, (int32_t)0, CONST_256);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(negativeBitTempSpace, (int32_t)0, CONST_64);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(constOneUb, (int32_t)1, CONST_64);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(const16Ub, (int32_t)CONST_16, CONST_64);
    PipeBarrier<PIPE_V>();
    SetVectorMask<uint64_t, MaskMode::NORMAL>(0xffffffffffffffff, 0xffffffffffffffff);
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::PreProcess(DecodeLoopConfig* config)
{
    DataCopy(
        stateBufferBitNumberUb.template ReinterpretCast<uint8_t>(),
        this->compressDeviceWoHeaderGm[config->compressPtr - CONST_256], CONST_256);
    config->compressPtr = config->compressPtr - CONST_256;
    DataCopy(
        stateBufferRemainUb.template ReinterpretCast<uint8_t>(),
        this->compressDeviceWoHeaderGm[config->compressPtr - CONST_4096 * 2], CONST_4096 * 2);
    config->compressPtr = config->compressPtr - CONST_4096 * 2;
    config->firstMoveInSize = config->compressPtr - CONST_8192 >= 0 ? CONST_8192 : config->compressPtr;
    DataCopy(
        inputUb.template ReinterpretCast<uint8_t>(),
        this->compressDeviceWoHeaderGm[config->compressPtr - config->firstMoveInSize], config->firstMoveInSize);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventManager.eventMTE2VPing);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventManager.eventMTE2VPing);
    Uint162Int32Helper(
        stateBufferRemainUb.template ReinterpretCast<uint16_t>(), vnchTempUb.template ReinterpretCast<uint16_t>(),
        stateBufferUb, constZeroUb.template ReinterpretCast<uint16_t>());
    Uint162Int32Helper(
        inputUb.template ReinterpretCast<uint16_t>(), vnchTempUb.template ReinterpretCast<uint16_t>(), inputBufferUb,
        constZeroUb.template ReinterpretCast<uint16_t>());
    Adds(stateBufferBitNumberUb, stateBufferBitNumberUb, CONST_16, EACH_LOOOP_REPEAT_TIMES);
    PipeBarrier<PIPE_V>();
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::PrepareMantissa(DecodeLoopConfig* config)
{
    recoverMantissaInt32Ub = outputExpInt32Ub.template ReinterpretCast<int32_t>();
    recoverMantissaFp16Ub = stateBufferRemainUb.template ReinterpretCast<half>();
    recoverMantissaFp162s32Ub = stateBufferLowUb.template ReinterpretCast<int32_t>();
    recoverMantissaFp16GatherUb = bitLevelMaskSecondUb.template ReinterpretCast<int32_t>();
    recoverMantissaUint8Ub = bitLevelMaskSecondUb[CONST_2048].template ReinterpretCast<int32_t>();
    if (IF_BF16) {
        config->mantissaPtr = config->mantissaPtr - config->mantissaFirstLoopNum * CONST_64;
        DataCopy(
            recoverMantissaInt32Ub.template ReinterpretCast<uint8_t>(), this->mantissaGm[config->mantissaPtr],
            config->mantissaFirstLoopNum * CONST_64);
    } else {
        config->mantissaPtr = config->mantissaPtr - config->mantissaFirstLoopNum * CONST_192;
        DataCopy(
            recoverMantissaInt32Ub.template ReinterpretCast<uint8_t>(), this->mantissaGm[config->mantissaPtr],
            config->mantissaFirstLoopNum * CONST_192);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventManager.eventMTE2VPing);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventManager.eventMTE2VPing);
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::GetNegativeBitNum(DecodeLoopConfig* config)
{
    Adds(
        bitLevelMaskZeroUb, inputBufferUb[config->loopDataIndex - EACH_LOOOP_STATE_READ_NUM], 0,
        EACH_LOOOP_STATE_READ_NUM);
    PipeBarrier<PIPE_V>();
    config->compressPtr = config->compressPtr - EACH_LOOOP_STATE_READ_NUM * 2;
    Sub(stateBufferBitNumberUb, stateBufferBitNumberUb, bitLevelMaskZeroUb, EACH_LOOOP_STATE_READ_NUM);
    PipeBarrier<PIPE_V>();
    Compare(
        negativeZeroCmpBitMask.template ReinterpretCast<uint8_t>(),
        stateBufferBitNumberUb.template ReinterpretCast<float>(), const16Ub.template ReinterpretCast<float>(),
        AscendC::CMPMODE::LE, EACH_LOOOP_STATE_READ_NUM);
    PipeBarrier<PIPE_V>();
    GatherMask(
        negativeBitTempSpace.template ReinterpretCast<uint32_t>(), constOneUb.template ReinterpretCast<uint32_t>(),
        negativeZeroCmpBitMask.template ReinterpretCast<uint32_t>(), true, EACH_REPEAT_BYTES / sizeof(uint32_t),
        {1, 1, REPEAT_STRIDE, 0}, this->rsvdCnt);
    PipeBarrier<PIPE_V>();
    config->negativeZeroSum = AscendC::AscendCUtils::GetRsvdCnt();
    config->loopDataIndex = config->loopDataIndex - EACH_LOOOP_STATE_READ_NUM;
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::DataCopyAndSelCompare(DecodeLoopConfig* config)
{
    DataCopy(
        inputUb.template ReinterpretCast<uint8_t>(),
        this->compressDeviceWoHeaderGm[config->compressPtr - config->repeatTimes * EACH_LOOOP_STATE_READ_NUM * 2],
        config->repeatTimes * EACH_LOOOP_STATE_READ_NUM * 2);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventManager.eventMTE2VPing);
    config->repeatTimes = (config->currentCoreDecompressPtr % EACH_LOOOP_COMPRESS_READ_NUM != 0) ?
                              config->lastEncodedRemainLoopNum :
                              EACH_LOOOP_STATE_READ_NUM;
    Compare(
        stateBufferBitNumberUb.template ReinterpretCast<float>(), const16Ub.template ReinterpretCast<float>(),
        AscendC::CMPMODE::LE, uint64_t(EACH_LOOOP_STATE_READ_NUM),
        {1, 1, 1, REPEAT_STRIDE, REPEAT_STRIDE, REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    AscendC::BinaryRepeatParams repeatParams = {1, 1, 1, REPEAT_STRIDE, REPEAT_STRIDE, REPEAT_STRIDE};
    AscendC::Select<float, AscendC::SELMODE::VSEL_CMPMASK_SPR>(
        stateBufferBitCmpUb.template ReinterpretCast<float>(), constOneUb.template ReinterpretCast<float>(),
        constZeroUb.template ReinterpretCast<float>(), 1, repeatParams);
    PipeBarrier<PIPE_V>();
    Muls(stateBufferShlScaleUb, stateBufferBitCmpUb, CONST_65535, EACH_LOOOP_REPEAT_TIMES);
    PipeBarrier<PIPE_V>();
    Adds(stateBufferShlScaleUb, stateBufferShlScaleUb, 1, EACH_LOOOP_REPEAT_TIMES);
    PipeBarrier<PIPE_V>();
    Brcb(bitLevelMaskFirstUb, stateBufferShlScaleUb, CONST_8, {1, REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    Brcb(bitLevelMaskSecondUb, bitLevelMaskFirstUb, EACH_LOOOP_REPEAT_TIMES, {1, REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::UpdateStateBuffer(DecodeLoopConfig* config)
{
    Mul(stateBufferUb.template ReinterpretCast<int32_t>(), stateBufferUb, bitLevelMaskSecondUb,
        config->repeatTimes * EACH_LOOOP_STATE_READ_NUM);
    PipeBarrier<PIPE_V>();
    for (int32_t index = EACH_LOOOP_STATE_READ_NUM - 1; index >= 0; --index) {
        config->stateBufferBitNumber = stateBufferBitNumberUb.ReinterpretCast<int32_t>().GetValue(index);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(eventManager.eventSVPing);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventManager.eventSVPing);
        if (config->stateBufferBitNumber <= CONST_16) {
            config->loopDataIndex = config->loopDataIndex - EACH_LOOOP_STATE_READ_NUM;
            Add(stateBufferUb[index * BLOCK_SIZE], stateBufferUb[index * BLOCK_SIZE],
                inputBufferUb[config->loopDataIndex], EACH_LOOOP_REPEAT_TIMES);
        }
    }
    PipeBarrier<PIPE_V>();
    Muls(stateBufferShlScaleUb, stateBufferBitCmpUb, CONST_16, EACH_LOOOP_REPEAT_TIMES);
    PipeBarrier<PIPE_V>();
    Add(stateBufferBitNumberUb, stateBufferBitNumberUb, stateBufferShlScaleUb, EACH_LOOOP_REPEAT_TIMES);
    config->loopDataIndex = (config->compressPtr - config->copmpressPtrCurrentCoreStart) >= CONST_8192 ?
                                CONST_4096 :
                                (config->compressPtr - config->copmpressPtrCurrentCoreStart) / CONST_128 * CONST_64;
    PipeBarrier<PIPE_V>();
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventManager.eventMTE2VPing);
    Uint162Int32Helper(
        inputUb.template ReinterpretCast<uint16_t>(), vnchTempUb.template ReinterpretCast<uint16_t>(), inputBufferUb,
        constZeroUb.template ReinterpretCast<uint16_t>());
    PipeBarrier<PIPE_V>();
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::ReadDevice(DecodeLoopConfig* config)
{
    bool read = (config->negativeZeroSum > 0) && ((config->compressPtr - config->copmpressPtrCurrentCoreStart) >=
                                                  config->negativeZeroSum * EACH_LOOOP_STATE_READ_NUM * 2);
    if (read) {
        if (config->negativeZeroSum * BLOCK_SIZE > config->loopDataIndex) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventManager.eventVMTE2Pong);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventManager.eventVMTE2Pong);
            DataCopy(
                backupInputUb.template ReinterpretCast<uint8_t>(),
                this->compressDeviceWoHeaderGm[config->compressPtr - config->negativeZeroSum * BLOCK_SIZE * 2],
                config->negativeZeroSum * BLOCK_SIZE * 2);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventManager.eventMTE2VPong);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventManager.eventMTE2VPong);
            Uint162Int32Helper(
                backupInputUb.template ReinterpretCast<uint16_t>(), vnchTempUb.template ReinterpretCast<uint16_t>(),
                inputBufferUb, constZeroUb.template ReinterpretCast<uint16_t>());
            PipeBarrier<PIPE_V>();
            config->loopDataIndex = config->negativeZeroSum * BLOCK_SIZE;
        }
        config->compressPtr = config->compressPtr - (config->negativeZeroSum * BLOCK_SIZE) * 2;
        config->repeatTimes = (config->compressPtr - config->copmpressPtrCurrentCoreStart) >= CONST_8192 ?
                                  EACH_LOOOP_REPEAT_TIMES :
                                  (config->compressPtr - config->copmpressPtrCurrentCoreStart) / CONST_128;

        DataCopyAndSelCompare(config);
        UpdateStateBuffer(config);
        config->negativeZeroSum = 0;
    }
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::DecompressSymbol(
    LocalTensor<int32_t> pdfSortIndexReverseUb, int32_t loopIdx)
{
    Mul(stateBufferLowUb.template ReinterpretCast<int16_t>(), stateBufferUb.template ReinterpretCast<int16_t>(),
        int32LowMask.template ReinterpretCast<int16_t>(), uint64_t(CONST_128), EACH_LOOOP_REPEAT_TIMES,
        {1, 1, 1, REPEAT_STRIDE, REPEAT_STRIDE, REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    Muls(bitLevelMaskZeroAndScaleUb, bitLevelMaskZeroUb, CONST_4, EACH_LOOOP_REPEAT_TIMES);
    PipeBarrier<PIPE_V>();
    Gather(
        bitLevelMaskZeroAndScaleUb.template ReinterpretCast<uint32_t>(),
        bitShlMulScaleUb.template ReinterpretCast<uint32_t>(),
        bitLevelMaskZeroAndScaleUb.template ReinterpretCast<uint32_t>(), (uint32_t)0, EACH_LOOOP_REPEAT_TIMES);
    PipeBarrier<PIPE_V>();
    Brcb(bitLevelMaskFirstUb, bitLevelMaskZeroAndScaleUb, CONST_8, {1, REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    Brcb(bitLevelMaskSecondUb, bitLevelMaskFirstUb, EACH_LOOOP_REPEAT_TIMES, {1, REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    if (loopIdx > 0) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventManager.eventMTE3VPing);
    }
    And(outputPdfIndexInt32Ub.template ReinterpretCast<uint16_t>(),
        stateBufferLowUb.template ReinterpretCast<uint16_t>(),
        bitLevelMaskSecondUb.template ReinterpretCast<uint16_t>(), uint64_t(MASK_128), EACH_LOOOP_REPEAT_TIMES,
        {1, 1, 1, REPEAT_STRIDE, REPEAT_STRIDE, REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    if (IF_BF16) {
        Muls(bitLevelMaskSecondUb, outputPdfIndexInt32Ub, CONST_2, EACH_LOOOP_REPEAT_TIMES * EACH_LOOOP_REPEAT_TIMES);
        PipeBarrier<PIPE_V>();
        Gather(
            outputPdfIndexInt32Ub.template ReinterpretCast<uint16_t>(),
            pdfSortIndexReverseUb.template ReinterpretCast<uint16_t>(),
            bitLevelMaskSecondUb.template ReinterpretCast<uint32_t>(), (uint32_t)0, CONST_8 * CONST_16 * CONST_32);
    } else {
        Muls(outputPdfIndexInt32Ub, outputPdfIndexInt32Ub, CONST_4, EACH_LOOOP_REPEAT_TIMES * EACH_LOOOP_REPEAT_TIMES);
        PipeBarrier<PIPE_V>();

        Gather(
            outputPdfIndexInt32Ub.template ReinterpretCast<uint32_t>(),
            pdfSortIndexReverseUb.template ReinterpretCast<uint32_t>(),
            outputPdfIndexInt32Ub.template ReinterpretCast<uint32_t>(), (uint32_t)0, CONST_8 * CONST_16 * CONST_32);
    }
    PipeBarrier<PIPE_V>();
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::MoveOutBf16(DecodeLoopConfig* config, int32_t loopIdx)
{
    Cast(
        recoverMantissaFp16Ub, recoverMantissaInt32Ub.template ReinterpretCast<uint8_t>(),
        AscendC::RoundMode::CAST_NONE, EACH_LOOOP_PROCESS_NUM);
    PipeBarrier<PIPE_V>();
    Cast(
        recoverMantissaFp16Ub.template ReinterpretCast<int16_t>(), recoverMantissaFp16Ub,
        AscendC::RoundMode::CAST_ROUND, EACH_LOOOP_PROCESS_NUM);
    PipeBarrier<PIPE_V>();
    Or(outputPdfIndexInt32Ub.template ReinterpretCast<uint16_t>(),
       outputPdfIndexInt32Ub.template ReinterpretCast<uint16_t>(),
       recoverMantissaFp16Ub.template ReinterpretCast<uint16_t>(), EACH_LOOOP_PROCESS_NUM);
    PipeBarrier<PIPE_V>();
    if (loopIdx < config->bigLoop - 1) {
        config->mantissaPtr = config->mantissaPtr - EACH_LOOOP_PROCESS_NUM;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventManager.eventVMTE2Pong);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventManager.eventVMTE2Pong);
        DataCopy(
            recoverMantissaInt32Ub.template ReinterpretCast<uint8_t>(), this->mantissaGm[config->mantissaPtr],
            EACH_LOOOP_PROCESS_NUM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventManager.eventMTE2VPong);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventManager.eventVMTE3Ping);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventManager.eventVMTE3Ping);
    DataCopy(
        this->outputGm[(int32_t)config->decompressUint8Ptr], outputPdfIndexInt32Ub.template ReinterpretCast<uint16_t>(),
        config->repeatTimes * CONST_64);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventManager.eventMTE3VPing);
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::MoveOutFp32(
    LocalTensor<uint32_t> mantissaBitMask, DecodeLoopConfig* config, int32_t loopIdx)
{
    int32_t mantissaRepeatTimes = 16;
    int32_t mantissaLoops = ((config->repeatTimes + mantissaRepeatTimes - 1) / mantissaRepeatTimes);
    for (int32_t loop = 0; loop < mantissaLoops; ++loop) {
        if (loop == mantissaLoops - 1) {
            mantissaRepeatTimes = config->repeatTimes - mantissaRepeatTimes * (mantissaLoops - 1);
        } else {
            mantissaRepeatTimes = mantissaRepeatTimes;
        }
        Cast(
            recoverMantissaFp16Ub, recoverMantissaInt32Ub.template ReinterpretCast<uint8_t>()[loop * PROCESS_3072],
            AscendC::RoundMode::CAST_NONE, PROCESS_3072);
        PipeBarrier<PIPE_V>();
        Cast(recoverMantissaFp162s32Ub, recoverMantissaFp16Ub, AscendC::RoundMode::CAST_CEIL, PROCESS_3072);
        PipeBarrier<PIPE_V>();
        GatherMask(
            recoverMantissaFp16GatherUb.template ReinterpretCast<uint16_t>(),
            recoverMantissaFp162s32Ub.template ReinterpretCast<uint16_t>(),
            mantissaBitMask.template ReinterpretCast<uint16_t>(), true, 3 * mantissaRepeatTimes * EACH_REPEAT_BYTES / sizeof(uint16_t),
            {1, 1, REPEAT_STRIDE, 0}, this->rsvdCnt);
        PipeBarrier<PIPE_V>();
        Cast(
            recoverMantissaFp16GatherUb.template ReinterpretCast<half>(),
            recoverMantissaFp16GatherUb.template ReinterpretCast<int16_t>(), AscendC::RoundMode::CAST_CEIL,
            PROCESS_4096);
        PipeBarrier<PIPE_V>();
        Cast(
            recoverMantissaUint8Ub.template ReinterpretCast<uint8_t>(),
            recoverMantissaFp16GatherUb.template ReinterpretCast<half>(), AscendC::RoundMode::CAST_CEIL, PROCESS_4096);
        PipeBarrier<PIPE_V>();
        Add(outputPdfIndexInt32Ub[CONST_1024 * loop], outputPdfIndexInt32Ub[CONST_1024 * loop], recoverMantissaUint8Ub,
            mantissaRepeatTimes * CONST_64);
        PipeBarrier<PIPE_V>();
    }
    if (loopIdx < config->bigLoop - 1) {
        config->mantissaPtr = config->mantissaPtr - CONST_64 * CONST_192;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventManager.eventVMTE2Pong);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventManager.eventVMTE2Pong);
        DataCopy(
            recoverMantissaInt32Ub.template ReinterpretCast<uint8_t>(), this->mantissaGm[config->mantissaPtr],
            CONST_64 * CONST_192);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventManager.eventMTE2VPong);
    }

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventManager.eventVMTE3Ping);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventManager.eventVMTE3Ping);
    DataCopy(
        this->outputGm[config->decompressUint8Ptr * 2], outputPdfIndexInt32Ub.template ReinterpretCast<uint16_t>(),
        config->repeatTimes * 2 * CONST_64);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventManager.eventMTE3VPing);
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::MoveOut(
    LocalTensor<uint32_t> mantissaBitMask, DecodeLoopConfig* config, int32_t loopIdx)
{
    if (config->currentCoreDecompressPtr % PROCESS_4096 != 0) {
        config->decompressUint8Ptr -= (config->currentCoreDecompressPtr % PROCESS_4096);
        config->currentCoreDecompressPtr -= (config->currentCoreDecompressPtr % PROCESS_4096);
        config->repeatTimes = config->lastEncodedRemainLoopNum;
    } else {
        config->decompressUint8Ptr -= PROCESS_4096;
        config->currentCoreDecompressPtr -= PROCESS_4096;
        config->repeatTimes = EACH_LOOOP_REPEAT_TIMES;
    }
    if (IF_BF16) {
        MoveOutBf16(config, loopIdx);
    } else {
        MoveOutFp32(mantissaBitMask, config, loopIdx);
    }
    PipeBarrier<PIPE_V>();
    Adds(bitLevelMaskFirstUb, bitLevelMaskFirstUb, 1, CONST_8 * CONST_64);
    PipeBarrier<PIPE_V>();
    Cast(
        bitLevelMaskFirstUb.template ReinterpretCast<float>(), bitLevelMaskFirstUb, AscendC::RoundMode::CAST_CEIL,
        CONST_512);
    PipeBarrier<PIPE_V>();
    Brcb(bitLevelMaskSecondUb, bitLevelMaskFirstUb, config->repeatTimes, {1, REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    Cast(
        stateBufferUb.template ReinterpretCast<float>(), stateBufferUb, AscendC::RoundMode::CAST_CEIL,
        config->repeatTimes * CONST_64);
    PipeBarrier<PIPE_V>();
    Div(stateBufferUb.template ReinterpretCast<float>(), stateBufferUb.template ReinterpretCast<float>(),
        bitLevelMaskSecondUb.template ReinterpretCast<float>(), config->repeatTimes * CONST_64);
    PipeBarrier<PIPE_V>();
    Cast(
        stateBufferUb, stateBufferUb.template ReinterpretCast<float>(), AscendC::RoundMode::CAST_FLOOR,
        config->repeatTimes * CONST_64);
    PipeBarrier<PIPE_V>();
    if (loopIdx < config->bigLoop - 1) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventManager.eventMTE2VPong);
    }
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::InitLoopConfig(DecodeLoopConfig* config)
{
    config->lastEncodedRemainLoopNum = (this->currentCoreRecoverTensorUint8Size / CONST_64) % CONST_64;
    config->mantissaPtr = this->currentCoreMantissaDeviceOffset;
    config->mantissaFirstLoopNum = CONST_64;
    if (config->lastEncodedRemainLoopNum > 0) {
        config->mantissaFirstLoopNum = config->lastEncodedRemainLoopNum;
    }
    config->repeatTimes = 0;
    config->stateBufferBitNumber = 0;
    config->bitLevel = 0;
    config->loopDataIndex = config->firstMoveInSize / 2;
    config->negativeZeroSum = 0;
    config->currentCoreDecompressPtr = this->currentCoreRecoverTensorUint8Size;
    config->bigLoop =
        ((this->currentCoreRecoverTensorUint8Size / EACH_LOOOP_REPEAT_TIMES) + EACH_LOOOP_REPEAT_TIMES - 1) /
        EACH_LOOOP_REPEAT_TIMES;
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::Decompress(
    LocalTensor<int32_t> pdfSortIndexReverseUb, LocalTensor<uint32_t> mantissaBitMask)
{
    if (((int32_t)id) < actualUseCore) {
        MallocUbforDecompress();
        PipeBarrier<PIPE_ALL>();
        InitLocalTensor();
        DecodeLoopConfig loopConfig;
        loopConfig.compressPtr = this->currentCoreCompressOffset;
        loopConfig.firstMoveInSize = 0;
        loopConfig.copmpressPtrCurrentCoreStart = this->compressPtrCurrentCoreStartOffset;
        loopConfig.decompressUint8Ptr = this->currentCoreRecoverTensorOffset;
        PreProcess(&loopConfig);
        InitLoopConfig(&loopConfig);
        PrepareMantissa(&loopConfig);
        for (int32_t loop = 0; loop < loopConfig.bigLoop; loop++) {
            GetNegativeBitNum(&loopConfig);
            ReadDevice(&loopConfig);
            DecompressSymbol(pdfSortIndexReverseUb, loop);
            MoveOut(mantissaBitMask, &loopConfig, loop);
        }
        if (loopConfig.bigLoop > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventManager.eventMTE3VPing);
        }
    }
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::ProcessBf16BeforeH2D(int32_t index, uint32_t size, uint32_t expShlDistance)
{
    Cast(deviceExpUb[index].template ReinterpretCast<half>(), hostExpUb[index], AscendC::RoundMode::CAST_NONE, size);
    Cast(recoverUb[index].template ReinterpretCast<half>(), mantissaUb[index], AscendC::RoundMode::CAST_NONE, size);
    PipeBarrier<PIPE_V>();
    Cast(
        recoverUb[index].template ReinterpretCast<int16_t>(), recoverUb[index].template ReinterpretCast<half>(),
        AscendC::RoundMode::CAST_TRUNC, size);
    Cast(
        deviceExpUb[index].template ReinterpretCast<int16_t>(), deviceExpUb[index].template ReinterpretCast<half>(),
        AscendC::RoundMode::CAST_TRUNC, size);
    PipeBarrier<PIPE_V>();
    ShiftLeft(
        deviceExpUb[index].template ReinterpretCast<uint16_t>(),
        deviceExpUb[index].template ReinterpretCast<uint16_t>(), (uint16_t)expShlDistance, size);
    PipeBarrier<PIPE_V>();
    Or(recoverUb[index].template ReinterpretCast<uint16_t>(), deviceExpUb[index].template ReinterpretCast<uint16_t>(),
       recoverUb[index].template ReinterpretCast<uint16_t>(), size);
    PipeBarrier<PIPE_V>();
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::ProcessFp32BeforeH2D(
    LocalTensor<uint32_t> mantissaBitMask, int32_t index, uint32_t size, int32_t process_num, uint32_t expShlDistance)
{
    Cast(
        deviceExpUb[index].template ReinterpretCast<half>(), mantissaUb[index], AscendC::RoundMode::CAST_NONE,
        (size * (sizeof(float) - 1)));
    PipeBarrier<PIPE_V>();
    Cast(
        recoverUb[index].template ReinterpretCast<int32_t>(), deviceExpUb[index].template ReinterpretCast<half>(),
        AscendC::RoundMode::CAST_TRUNC, (size * (sizeof(float) - 1)));
    PipeBarrier<PIPE_V>();
    SetVectorMask<uint64_t, MaskMode::NORMAL>((uint64_t)-1, (uint64_t)-1);
    GatherMask(
        deviceExpUb[index].template ReinterpretCast<uint16_t>(), recoverUb[index].template ReinterpretCast<uint16_t>(),
        mantissaBitMask.template ReinterpretCast<uint16_t>(), true, (size * (sizeof(float) - 1) * 2) / CONST_128 * EACH_REPEAT_BYTES / sizeof(uint16_t),
        {1, 1, REPEAT_STRIDE, 0}, this->rsvdCnt);
    PipeBarrier<PIPE_V>();
    Cast(
        deviceExpUb[index].template ReinterpretCast<half>(), deviceExpUb[index].template ReinterpretCast<int16_t>(),
        AscendC::RoundMode::CAST_NONE, (size * sizeof(float)));
    PipeBarrier<PIPE_V>();
    Cast(
        recoverUb[index].template ReinterpretCast<uint8_t>(), deviceExpUb[index].template ReinterpretCast<half>(),
        AscendC::RoundMode::CAST_TRUNC, (size * sizeof(float)));
    PipeBarrier<PIPE_V>();
    Cast(
        deviceExpUb[index].template ReinterpretCast<half>(), hostExpUb[index].template ReinterpretCast<uint8_t>(),
        AscendC::RoundMode::CAST_NONE, (size));
    PipeBarrier<PIPE_V>();
    Cast(
        recoverUb[index].template ReinterpretCast<int32_t>()[process_num],
        deviceExpUb[index].template ReinterpretCast<half>(), AscendC::RoundMode::CAST_TRUNC, (size));
    PipeBarrier<PIPE_V>();
    ShiftLeft(
        recoverUb[index].template ReinterpretCast<uint32_t>()[process_num],
        recoverUb[index].template ReinterpretCast<uint32_t>()[process_num], (uint32_t)expShlDistance, size);
    PipeBarrier<PIPE_V>();
    Add(recoverUb[index].template ReinterpretCast<int32_t>(), recoverUb[index].template ReinterpretCast<int32_t>(),
        recoverUb[index].template ReinterpretCast<int32_t>()[process_num], size);
    PipeBarrier<PIPE_V>();
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::DataCopy2OutputGm(
    int32_t outputOffsetNum, int32_t dtype_size, int32_t index, uint32_t size)
{
    if (IF_BF16) {
        DataCopy(
            this->outputGm[this->currentCoreRecoverTensorOffset + outputOffsetNum / dtype_size],
            recoverUb[index].template ReinterpretCast<uint16_t>(),
            {(uint16_t)(size / BLOCK_SIZE), (uint16_t)(2 * dtype_size), (uint16_t)0, (uint16_t)0});
    } else {
        DataCopy(
            this->outputGm_uint8[outputOffsetNum + currentCoreRecoverTensorOffset * dtype_size], recoverUb[index],
            {(uint16_t)(size / BLOCK_SIZE), (uint16_t)(2 * dtype_size), (uint16_t)0, (uint16_t)0});
    }
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::Host2Device(LocalTensor<uint32_t> mantissaBitMask)
{
    if (this->currentCoreHostExpSize <= 0) {
        return;
    }
    int32_t dtype_size = IF_BF16 ? sizeof(half) : sizeof(float);
    uint32_t expShlDistance = IF_BF16 ? CONST_8 : CONST_24;
    int process_num = PROCESS_3072;
    MallocUbforH2D(process_num, dtype_size);
    int64_t dataSizeRemain = this->currentCoreHostExpSize;
    int inputOffsetNum = 0;
    int outputOffsetNum = 0;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventManager.eventMTE3VPing);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventManager.eventMTE3VPong);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventManager.eventMTE3MTE2Ping);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventManager.eventMTE3MTE2Pong);
    for (int32_t i = 0; dataSizeRemain > 0; i++) {
        uint32_t size = dataSizeRemain >= process_num ? process_num : dataSizeRemain;
        int32_t index = (i & 1) ? 0 : 1;
        auto mte3_v_id = (i & 1) ? eventManager.eventMTE3VPing : eventManager.eventMTE3VPong;
        auto mte3_mte2_id = (i & 1) ? eventManager.eventMTE3MTE2Ping : eventManager.eventMTE3MTE2Pong;
        auto mte2_v_id = (i & 1) ? eventManager.eventMTE2VPing : eventManager.eventMTE2VPong;
        auto v_mte3_id = (i & 1) ? eventManager.eventVMTE3Ping : eventManager.eventVMTE3Pong;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3_v_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3_mte2_id);
        DataCopyPad(
            hostExpUb[index], this->compressHostGm[this->currentCoreHostExpOffset + inputOffsetNum],
            {1, static_cast<uint16_t>(size), 0, 0}, {true, 0, 0, 0});
        DataCopyPad(
            mantissaUb[index],
            this->mantissaGm[this->currentCoreMantissaDeviceOffset + inputOffsetNum * (dtype_size - 1)],
            {(uint16_t)(dtype_size - 1), static_cast<uint16_t>(size), 0, 0}, {true, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2_v_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2_v_id);
        if (IF_BF16) {
            ProcessBf16BeforeH2D(index, size, expShlDistance);
        } else {
            ProcessFp32BeforeH2D(mantissaBitMask, index, size, process_num, expShlDistance);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(v_mte3_id);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(v_mte3_id);
        DataCopy2OutputGm(outputOffsetNum, dtype_size, index, size);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3_v_id);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3_mte2_id);
        dataSizeRemain -= size;
        inputOffsetNum += size;
        outputOffsetNum += size * dtype_size;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventManager.eventMTE3VPing);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventManager.eventMTE3VPong);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventManager.eventMTE3MTE2Ping);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventManager.eventMTE3MTE2Pong);
}

template <bool IF_BF16>
__aicore__ inline void HansDecode<IF_BF16>::Process()
{
    SetVectorMask<uint64_t, MaskMode::NORMAL>((uint64_t)-1, (uint64_t)-1);
    SetMaskNorm();
    SetAtomicNone();
    uint64_t ubOffset_backup = 0;
    this->eventManager.InitEvent();
    if (this->id < this->actualUseCore) {
        int32_t pdfSortIndexReverseUbSize = PDF_LENGTH * sizeof(int32_t);
        LocalTensor<int32_t> pdfSortIndexReverseUb =
            calcBuf.GetWithOffset<int32_t>(pdfSortIndexReverseUbSize, ubOffset);
        ubOffset += pdfSortIndexReverseUbSize;
        int32_t mantissaBitMaskNum = CONST_768;
        int32_t mantissaBitMaskUbSize = mantissaBitMaskNum * sizeof(uint32_t);
        auto mantissaBitMask = calcBuf.GetWithOffset<uint32_t>(mantissaBitMaskUbSize, ubOffset);
        ubOffset += mantissaBitMaskUbSize;
        ubOffset_backup = ubOffset;
        int maskRepeatStride = 3;
        for (int32_t i = 0; i < PDF_LENGTH; ++i) {
            mantissaBitMask.SetValue(i * maskRepeatStride, MANTISSABITMASK_SEC0);
            mantissaBitMask.SetValue(i * maskRepeatStride + 1, MANTISSABITMASK_SEC1);
            mantissaBitMask.SetValue(i * maskRepeatStride + maskRepeatStride - 1, MANTISSABITMASK_SEC2);
        }
        GetPdfSortIndex<IF_BF16>(calcBuf, pdfGm, eventManager, ubOffset, &pdfSortIndexReverseUb);
        ubOffset = ubOffset_backup;
        AscendC::SetFlag<AscendC::HardEvent::S_V>(eventManager.eventSVPing);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventManager.eventSVPing);
        int32_t deviceEncodedLoops = compressDeviceGm.GetValue(COM_HEADER_OFFSET_FIXEDLOOPS);
        if (deviceEncodedLoops > 0) {
            Decompress(pdfSortIndexReverseUb, mantissaBitMask);
            ubOffset = ubOffset_backup;
        }
        int32_t hostEncodedLoops = compressDeviceGm.GetValue(COM_HEADER_OFFSET_VARLOOPS);
        if (hostEncodedLoops > 0) {
            Host2Device(mantissaBitMask);
            ubOffset = ubOffset_backup;
        }
    }
    this->eventManager.ReleaseEvent();
}

} // namespace HansDecodeNS

#endif
