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
 * \file hans_encode_base.h
 * \brief
 */
#ifndef HANS_ENCODE_BASE_H
#define HANS_ENCODE_BASE_H

namespace HansEncodeNS {
using namespace HansCommonNs;
using namespace AscendC;

template <typename dataType>
class HansEncode {
 public:
  __aicore__ inline HansEncode() {
  }
  __aicore__ inline void Init(TPipe* pipe_, const HansEncodeInitConfig& config) {
    const HansEncodeTilingData* tilingData = config.tilingData;
    if (GetBlockIdx() >= tilingData->processCoreDim) {
      return;
    }
    this->pipe = pipe_;
    this->eventManager = HansCommonNs::EventManager();
    this->reshuff = tilingData->reshuff;
    this->dtypeSize = sizeof(dataType);
    this->processCoreDim = tilingData->processCoreDim;
    this->processLoopPerCore = tilingData->processLoopPerCore;
    this->processLoopCurrentCore =
        GetBlockIdx() < tilingData->processCoreDim - 1 ? this->processLoopPerCore : tilingData->processLoopLastCore;
    int64_t processMantissaPerCore = this->processLoopPerCore * BLOCK_SIZE * (this->dtypeSize - 1);
    int64_t prcoessNumelCurrentCore = this->processLoopCurrentCore * BLOCK_SIZE;
    int64_t processMantissaCurrentCore = prcoessNumelCurrentCore * (this->dtypeSize - 1);
    this->outputDeviceSizeCurrentCore =
        GetBlockIdx() < processCoreDim - 1 ? tilingData->fixedLengthPerCore : tilingData->fixedLengthLastCore;
    this->pdfGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(config.pdfGm), PDF_LENGTH);
    this->inputGm.SetGlobalBuffer(
      reinterpret_cast<__gm__ dataType*>(config.inputGm) + processLoopPerCore * BLOCK_SIZE * GetBlockIdx(), 
      prcoessNumelCurrentCore);
    if (reshuff) {
      this->fixedGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint8_t*>(config.workspace) + tilingData->fixedLengthPerCore * GetBlockIdx() +
        META_INFO_NUM * sizeof(int32_t), this->outputDeviceSizeCurrentCore);
      this->workspaceGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint8_t*>(config.fixedGm),
        tilingData->fixedLengthPerCore * (processCoreDim - 1) + tilingData->fixedLengthLastCore);
    } else {
      this->fixedGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint8_t*>(config.fixedGm) + tilingData->fixedLengthPerCore * GetBlockIdx() +
        META_INFO_NUM * sizeof(int32_t), this->outputDeviceSizeCurrentCore);
      this->workspaceGm.SetGlobalBuffer( reinterpret_cast<__gm__ uint8_t*>(config.fixedGm),
        tilingData->fixedLengthPerCore * (processCoreDim - 1) + tilingData->fixedLengthLastCore);
    }
    this->outputDeviceHeaderGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(config.fixedGm), 128);
    this->outputMantissaGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint8_t*>(config.outputMantissaGm) + processMantissaPerCore * GetBlockIdx(),
        processMantissaCurrentCore);
    this->varGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(config.varGm), tilingData->varLength);
    this->tileDataLength = EACH_LOOOP_PROCESS_NUM;
    this->tileNum = prcoessNumelCurrentCore / this->tileDataLength;
    this->tileRemain = prcoessNumelCurrentCore % this->tileDataLength;
    this->pipe->InitBuffer(this->inputQue, 1, this->dtypeSize * this->tileDataLength);
    this->pipe->InitBuffer(this->pdfQue, 1, sizeof(int32_t) * PDF_LENGTH);
    this->pipe->InitBuffer(this->outputMantissaQue, 1, this->tileDataLength * (this->dtypeSize - 1));
    this->pipe->InitBuffer(this->outputDeviceQue, 1,
                           (this->tileDataLength + EACH_LOOOP_REPEAT_TIMES) * sizeof(uint16_t));
    this->pipe->InitBuffer(this->calcBuf, CALC_BUF_SIZE);
  }

  __aicore__ inline void Process() {
    if (GetBlockIdx() >= this->processCoreDim) {
      return;
    }
    ProcessPdf();
    Prepare4Encode();
    ProcessEncode();
    PostProcess();
  }

 private:
  __aicore__ inline void ProcessPdf() {
    this->eventManager.InitEvent();
    this->inputLocal = this->inputQue.template AllocTensor<dataType>();
    this->pdfLocal = this->pdfQue.template AllocTensor<int32_t>();
    this->outputDeviceLocal = this->outputDeviceQue.template AllocTensor<uint16_t>();
    this->outputMantissaLocal = this->outputMantissaQue.template AllocTensor<uint8_t>();

    this->pdfSortIndexLocal = calcBuf.GetWithOffset<int32_t>(PDF_LENGTH, this->bufOffset);
    this->bufOffset += PDF_LENGTH * sizeof(int32_t);
    PdfSort();
  }

  __aicore__ inline void Prepare4Encode() {
    this->inputHalfLocal = calcBuf.GetWithOffset<half>(this->dtypeSize * EACH_LOOOP_PROCESS_NUM, this->bufOffset);
    this->bufOffset += this->dtypeSize * EACH_LOOOP_PROCESS_NUM * sizeof(half);
    this->ConstOneLocal = calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_REPEAT_TIMES, this->bufOffset);
    this->bufOffset += EACH_LOOOP_REPEAT_TIMES * sizeof(int32_t);
    this->ConstZeroLocal = calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_REPEAT_TIMES, this->bufOffset);
    this->bufOffset += EACH_LOOOP_REPEAT_TIMES * sizeof(int32_t);
    this->fp32MantissaMaskLocal = calcBuf.GetWithOffset<uint16_t>(MANTISSA_MASK_NUM, this->bufOffset);
    this->bufOffset += MANTISSA_MASK_NUM * sizeof(uint16_t);
    this->inputExpHalfLocal = calcBuf.GetWithOffset<half>(EACH_LOOOP_PROCESS_NUM, this->bufOffset);
    this->bufOffset += EACH_LOOOP_PROCESS_NUM * sizeof(half);
    this->inputExpUint32Local = calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_PROCESS_NUM, this->bufOffset);
    this->bufOffset += EACH_LOOOP_PROCESS_NUM * sizeof(int32_t);
    this->statePdfIndexSortLocal = calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_PROCESS_NUM, this->bufOffset);
    this->bufOffset += EACH_LOOOP_PROCESS_NUM * sizeof(int32_t);
    this->stateBufferLocal = calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_PROCESS_NUM, this->bufOffset);
    this->bufOffset += EACH_LOOOP_PROCESS_NUM * sizeof(int32_t);
    this->statePdfMaxUseBitLocal =
        calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_REPEAT_TIMES + EACH_LOOOP_REPEAT_TIMES, this->bufOffset);
    this->bufOffset += (EACH_LOOOP_REPEAT_TIMES + EACH_LOOOP_REPEAT_TIMES) * sizeof(int32_t);
    this->bitLevelMaskZeroLocal = calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_REPEAT_TIMES, this->bufOffset);
    this->bufOffset += EACH_LOOOP_REPEAT_TIMES * sizeof(int32_t);
    this->stateBufferBitNumberLocal = calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_REPEAT_TIMES, this->bufOffset);
    this->bufOffset += EACH_LOOOP_REPEAT_TIMES * sizeof(int32_t);
    this->encodeHeaderLocal = calcBuf.GetWithOffset<int32_t>(META_INFO_NUM, this->bufOffset);
    this->bufOffset += META_INFO_NUM * sizeof(int32_t);
    this->bitShlMulScaleLocal = calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_REPEAT_TIMES, this->bufOffset);
    this->bufOffset += EACH_LOOOP_REPEAT_TIMES * sizeof(int32_t);
    this->reduceLowBlockMaskLocal = calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_REPEAT_TIMES, this->bufOffset);
    this->bufOffset += EACH_LOOOP_REPEAT_TIMES * sizeof(int32_t);
    for (int32_t i = 0; i < sizeof(uint16_t) * BYTE_BIT_NUM; ++i) {
      this->bitShlMulScaleLocal.SetValue(i, (int32_t)(1 << i));
    }

    Duplicate<int32_t>(this->encodeHeaderLocal, 0, META_INFO_NUM);
    if (GetBlockIdx() == 0) {
      PipeBarrier<PIPE_ALL>();
      DataCopyPad(this->outputDeviceHeaderGm, this->encodeHeaderLocal,
                  {1, static_cast<uint16_t>(META_INFO_NUM * sizeof(int32_t)), 0, 0});
      PipeBarrier<PIPE_ALL>();
    }
    Duplicate<uint16_t>(this->fp32MantissaMaskLocal, (uint16_t)30583ULL, MANTISSA_MASK_NUM);
    Duplicate<int32_t>(this->bitLevelMaskZeroLocal, 0, EACH_LOOOP_REPEAT_TIMES);
    Duplicate<int32_t>(this->ConstOneLocal, 1, EACH_LOOOP_REPEAT_TIMES);
    Duplicate<int32_t>(this->ConstZeroLocal, 0, EACH_LOOOP_REPEAT_TIMES);
    Duplicate<int32_t>(this->stateBufferBitNumberLocal, 0, EACH_LOOOP_REPEAT_TIMES);
    Duplicate<int32_t>(this->reduceLowBlockMaskLocal, LOW_BLOCK_MASK, EACH_LOOOP_REPEAT_TIMES);
    Duplicate<int32_t>(this->stateBufferLocal, 0, EACH_LOOOP_PROCESS_NUM);
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void ProcessEncode() {
    if (this->outputDeviceSizeCurrentCore - ENCODE_TAIL_INFO_BYTES_PER_CORE >= 0) {
      CopyIn(0, this->tileDataLength * this->dtypeSize);
      PipeBarrier<PIPE_ALL>();
      SetFlag<HardEvent::MTE3_V>(this->eventManager.eventMTE3VPing);
      int32_t tileRemainNum = this->tileRemain > 0 ? 1 : 0;
      for (int32_t i = 0; i < this->tileNum + tileRemainNum; i++) {
        int32_t processDataLength = i < this->tileNum ? this->tileDataLength : this->tileRemain;
        Uint8Convert2Fp16(processDataLength);
        if (i < this->tileNum) {
          int32_t copySize =
              i < this->tileNum - 1 ? this->tileDataLength * this->dtypeSize : this->tileRemain * this->dtypeSize;
          CopyIn((i + 1) * this->tileDataLength, copySize);
          SetFlag<HardEvent::MTE2_V>(this->eventManager.eventMTE2VPing);
        }
        SeparateAndCopyOut(i * this->tileDataLength * (this->dtypeSize - 1), processDataLength);
        SetFlag<HardEvent::MTE3_V>(this->eventManager.eventMTE3VPong);
        PdfSortMapping(processDataLength);
        if (SpaceVerify(processDataLength)) {
          if (i < this->tileNum) {
            WaitFlag<HardEvent::MTE2_V>(this->eventManager.eventMTE2VPing);
          }
          WaitFlag<HardEvent::MTE3_V>(this->eventManager.eventMTE3VPong);
          break;
        }
        PdfSortAdd2Buffer(processDataLength);
        WaitFlag<HardEvent::MTE3_V>(this->eventManager.eventMTE3VPing);
        OverflowProcess(processDataLength);
        CopyOut(processDataLength);
        SetFlag<HardEvent::MTE3_V>(this->eventManager.eventMTE3VPing);
        if (i < this->tileNum) {
          WaitFlag<HardEvent::MTE2_V>(this->eventManager.eventMTE2VPing);
        }
        WaitFlag<HardEvent::MTE3_V>(this->eventManager.eventMTE3VPong);
      }
      WaitFlag<HardEvent::MTE3_V>(this->eventManager.eventMTE3VPing);
      DeviceTailProcess();
    }
  }

  __aicore__ inline void PostProcess() {
    Sync();
    this->inputQue.template FreeTensor<dataType>(this->inputLocal);
    this->pdfQue.template FreeTensor<int32_t>(this->pdfLocal);
    this->outputDeviceQue.template FreeTensor<uint16_t>(this->outputDeviceLocal);
    this->outputMantissaQue.template FreeTensor<uint8_t>(this->outputMantissaLocal);
    this->pipe->Reset();
    this->eventManager.InitEvent();

    if (this->reshuff) {
      ReshuffDevice();
    } else {
      HostTailProcess();
    }
    this->eventManager.ReleaseEvent();
  }

  __aicore__ inline void PdfSort() {
    int32_t pdfSortAndIndexNum = 512;
    int32_t blockSize = 32;
    int32_t mrg4SortMode = 15;
    int32_t mrg2SortMode = 3;
    int32_t idx0 = 0;
    int32_t idx1 = 64;
    int32_t idx2 = 128;
    int32_t idx3 = 192;
    int32_t idx4 = 256;
    int32_t idx5 = 320;
    int32_t idx6 = 384;
    int32_t idx7 = 448;
    LocalTensor<int32_t> pdfRangeIndexLocal = calcBuf.GetWithOffset<int32_t>(PDF_LENGTH, this->bufOffset);
    this->bufOffset += PDF_LENGTH * sizeof(int32_t);
    LocalTensor<float> pdfSort = calcBuf.GetWithOffset<float>(pdfSortAndIndexNum, this->bufOffset);
    this->bufOffset += pdfSortAndIndexNum * sizeof(float);
    LocalTensor<float> pdfMrgsort = calcBuf.GetWithOffset<float>(pdfSortAndIndexNum, this->bufOffset);
    this->bufOffset += pdfSortAndIndexNum * sizeof(float);
    DataCopyPad(this->pdfLocal, this->pdfGm, {1, static_cast<uint16_t>(PDF_LENGTH * sizeof(int32_t)), 0, 0},
                {true, 0, 0, 0});
    PipeBarrier<PIPE_ALL>();
    CreateVecIndex(pdfRangeIndexLocal, (int32_t)0, PDF_LENGTH);
    PipeBarrier<PIPE_V>();
    Sort32<float>(pdfSort, this->pdfLocal.template ReinterpretCast<float>(),
                  pdfRangeIndexLocal.ReinterpretCast<uint32_t>(), PDF_LENGTH / blockSize);
    PipeBarrier<PIPE_V>();
    MrgSortSrcList<float> srcList1(pdfSort[idx0], pdfSort[idx1], pdfSort[idx2], pdfSort[idx3]);
    uint16_t elementLengths[4] = {32, 32, 32, 32};
    MrgSort4Info srcInfo(elementLengths, false, mrg4SortMode, 1);
    MrgSort<float>(pdfMrgsort, srcList1, srcInfo);
    MrgSortSrcList<float> srcList2(pdfSort[idx4], pdfSort[idx5], pdfSort[idx6], pdfSort[idx7]);
    MrgSort<float>(pdfMrgsort[idx4], srcList2, srcInfo);
    PipeBarrier<PIPE_V>();
    MrgSortSrcList<float> srcList3(pdfMrgsort[idx0], pdfMrgsort[idx4], pdfMrgsort[0], pdfMrgsort[idx4]);
    uint16_t eleLengths[4] = {128, 128, 0, 0};
    MrgSort4Info src256Info(eleLengths, false, mrg2SortMode, 1);
    MrgSort<float>(pdfSort, srcList3, src256Info);
    PipeBarrier<PIPE_V>();
    int32_t uint64BitNum = 64;
    int32_t uint16BitNum = 16;
    int32_t pdfStep = 2;
    for (int32_t i = 0; i < PDF_LENGTH; i++) {
      int32_t index = pdfSort.ReinterpretCast<int32_t>().GetValue(pdfStep * i + 1);
      int32_t metaValue = i == 0 ? (1 << (sizeof(uint16_t) * BYTE_BIT_NUM)) + i
                                 : (int32_t)((uint64BitNum - (int32_t)(ScalarCountLeadingZero((uint64_t)i))) << uint16BitNum) + i;
      this->pdfSortIndexLocal.SetValue(index, metaValue);
    }
    this->bufOffset -= (PDF_LENGTH * sizeof(int32_t) + EACH_LOOOP_REPEAT_TIMES * BYTE_BIT_NUM * sizeof(float) +
                        EACH_LOOOP_REPEAT_TIMES * BYTE_BIT_NUM * sizeof(float));
  }

  __aicore__ inline void CopyIn(int32_t copyOffset, int32_t copySize) {
    if (copySize == 0) {
      return;
    }
    SetFlag<HardEvent::V_MTE2>(this->eventManager.eventVMTE2Ping);
    WaitFlag<HardEvent::V_MTE2>(this->eventManager.eventVMTE2Ping);
    DataCopyParams copyParams{1, static_cast<uint16_t>(copySize), 0, 0};
    DataCopyPadParams padParams{true, 0, 0, 0};
    DataCopyPad(this->inputLocal, this->inputGm[copyOffset], copyParams, padParams);
  }

  __aicore__ inline void Uint8Convert2Fp16(int32_t processDataLength) {
    Cast(this->inputHalfLocal, this->inputLocal.template ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE,
         processDataLength * this->dtypeSize);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void SeparateAndCopyOut(int32_t copyOffset, int32_t processDataLength) {
    uint8_t expPattern = this->dtypeSize == sizeof(float) ? 6 : 2;
    GatherMaskParams expParams;
    expParams.src0BlockStride = 1;
    expParams.repeatTimes =
        static_cast<uint8_t>(processDataLength * this->dtypeSize / (EACH_REPEAT_BYTES / sizeof(uint16_t)));
    expParams.src0RepeatStride = REPEAT_STRIDE;
    expParams.src1RepeatStride = 0;
    GatherMask(this->inputExpHalfLocal, this->inputHalfLocal, expPattern, false, 0, expParams, this->rsvdCnt);
    PipeBarrier<PIPE_V>();
    if (this->dtypeSize == sizeof(float)) {
      GatherMask(
          this->inputHalfLocal.template ReinterpretCast<uint16_t>(),
          this->inputHalfLocal.template ReinterpretCast<uint16_t>(),
          this->fp32MantissaMaskLocal.template ReinterpretCast<uint16_t>(), true,
          expParams.repeatTimes * EACH_REPEAT_BYTES / sizeof(uint16_t), {1, 1, 8, 0}, this->rsvdCnt);
    } else {
      GatherMask(this->inputHalfLocal, this->inputHalfLocal, 1, false, 0, expParams, this->rsvdCnt);
    }
    PipeBarrier<PIPE_V>();
    Cast(this->outputMantissaLocal, this->inputHalfLocal, RoundMode::CAST_TRUNC,
         processDataLength * (this->dtypeSize - 1));
    PipeBarrier<PIPE_V>();
    Cast(this->inputExpUint32Local, this->inputExpHalfLocal, RoundMode::CAST_TRUNC, processDataLength);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(this->eventManager.eventVMTE3Ping);
    WaitFlag<HardEvent::V_MTE3>(this->eventManager.eventVMTE3Ping);
    DataCopyParams copyParams{1, static_cast<uint16_t>((this->dtypeSize - 1) * processDataLength), 0, 0};
    DataCopyPad(this->outputMantissaGm[copyOffset], this->outputMantissaLocal, copyParams);
  }

  __aicore__ inline void PdfSortMapping(int32_t processDataLength) {
    Duplicate<int32_t>(this->statePdfMaxUseBitLocal, 0, EACH_LOOOP_REPEAT_TIMES + EACH_LOOOP_REPEAT_TIMES);
    Muls(this->inputExpUint32Local, this->inputExpUint32Local, (int32_t)sizeof(int32_t), processDataLength);
    PipeBarrier<PIPE_V>();
    Gather(this->inputExpUint32Local, this->pdfSortIndexLocal,
           this->inputExpUint32Local.template ReinterpretCast<uint32_t>(), (uint32_t)0, processDataLength);
    PipeBarrier<PIPE_V>();
    Mul(this->statePdfIndexSortLocal.template ReinterpretCast<int16_t>(),
        this->inputExpUint32Local.template ReinterpretCast<int16_t>(),
        this->ConstOneLocal.template ReinterpretCast<int16_t>(), EACH_LOOOP_REPEAT_TIMES * sizeof(int16_t),
        (uint8_t)(processDataLength / BLOCK_SIZE), {1, 1, 0, 8, 8, 0});
    PipeBarrier<PIPE_V>();
    ShiftRight(this->inputExpUint32Local.template ReinterpretCast<uint32_t>(),
               this->inputExpUint32Local.template ReinterpretCast<uint32_t>(),
               (uint32_t)(sizeof(uint16_t) * BYTE_BIT_NUM), processDataLength);
    PipeBarrier<PIPE_V>();
    WholeReduceMax<float>(this->statePdfMaxUseBitLocal.template ReinterpretCast<float>(),
                          this->inputExpUint32Local.template ReinterpretCast<float>(), EACH_LOOOP_REPEAT_TIMES,
                          (uint8_t)(processDataLength / BLOCK_SIZE), 1, 1, REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    GatherMask(this->bitLevelMaskZeroLocal, this->statePdfMaxUseBitLocal, 1, false, 0, {1, 2, 8, 0}, this->rsvdCnt);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline bool SpaceVerify(int32_t processDataLength) {
    if ((this->currentCoreOutputAcculmulateSize + ENCODE_TAIL_INFO_BYTES_PER_CORE) >
        this->outputDeviceSizeCurrentCore - ENCODE_TAIL_INFO_BYTES_PER_CORE) {
      LocalTensor<int32_t> overflowCheckLocal =
          calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_REPEAT_TIMES, this->bufOffset);
      this->bufOffset += EACH_LOOOP_REPEAT_TIMES * sizeof(int32_t);
      LocalTensor<int32_t> stateBufferBitCmpLocal =
          calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_REPEAT_TIMES, this->bufOffset);
      this->bufOffset += EACH_LOOOP_REPEAT_TIMES * sizeof(int32_t);
      PipeBarrier<PIPE_V>();
      Add(overflowCheckLocal, this->stateBufferBitNumberLocal, this->bitLevelMaskZeroLocal, EACH_LOOOP_REPEAT_TIMES);
      Duplicate<int32_t>(stateBufferBitCmpLocal, 0, EACH_LOOOP_REPEAT_TIMES);
      PipeBarrier<PIPE_V>();
      CompareScalar(stateBufferBitCmpLocal.ReinterpretCast<uint8_t>(), overflowCheckLocal.ReinterpretCast<float>(),
                    this->const16Float, AscendC::CMPMODE::GT, EACH_LOOOP_REPEAT_TIMES);
      PipeBarrier<PIPE_V>();
      GatherMask(
          this->inputHalfLocal.template ReinterpretCast<uint32_t>(),
          overflowCheckLocal.template ReinterpretCast<uint32_t>(),
          stateBufferBitCmpLocal.template ReinterpretCast<uint32_t>(), true, EACH_REPEAT_BYTES / sizeof(uint32_t),
          {1, 1, 8, 0}, this->rsvdCnt);
      uint64_t compressBufferOffset = AscendC::AscendCUtils::GetRsvdCnt();
      this->bufOffset -= (EACH_LOOOP_REPEAT_TIMES * sizeof(int32_t) + EACH_LOOOP_REPEAT_TIMES * sizeof(int32_t));
      if ((currentCoreOutputAcculmulateSize + EACH_LOOOP_REPEAT_TIMES * sizeof(int32_t) +
           compressBufferOffset * EACH_LOOOP_REPEAT_TIMES * sizeof(uint16_t)) >
          this->outputDeviceSizeCurrentCore - ENCODE_TAIL_INFO_BYTES_PER_CORE) {
        return true;
      }
    }
    return false;
  }

  __aicore__ inline void PdfSortAdd2Buffer(int32_t processDataLength) {
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> bitLevelMaskZeroShlScaleLocal =
        calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_REPEAT_TIMES, this->bufOffset);
    this->bufOffset += EACH_LOOOP_REPEAT_TIMES * sizeof(int32_t);
    LocalTensor<int32_t> bitLevelMaskFirstShlScaleLocal =
        calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_REPEAT_TIMES * BYTE_BIT_NUM, this->bufOffset);
    this->bufOffset += EACH_LOOOP_REPEAT_TIMES * BYTE_BIT_NUM * sizeof(int32_t);
    Muls(bitLevelMaskZeroShlScaleLocal, this->bitLevelMaskZeroLocal, (int32_t)sizeof(int32_t), EACH_LOOOP_REPEAT_TIMES);
    PipeBarrier<PIPE_V>();
    Gather(bitLevelMaskZeroShlScaleLocal, this->bitShlMulScaleLocal,
           bitLevelMaskZeroShlScaleLocal.template ReinterpretCast<uint32_t>(), (uint32_t)0, EACH_LOOOP_REPEAT_TIMES);
    PipeBarrier<PIPE_V>();
    Brcb(bitLevelMaskFirstShlScaleLocal, bitLevelMaskZeroShlScaleLocal, BYTE_BIT_NUM, {1, 8});
    PipeBarrier<PIPE_V>();
    Mul(this->stateBufferLocal, this->stateBufferLocal, bitLevelMaskFirstShlScaleLocal, EACH_LOOOP_REPEAT_TIMES,
        (uint8_t)(processDataLength / BLOCK_SIZE), {1, 1, 0, 8, 8, 1});
    PipeBarrier<PIPE_V>();
    Add(this->stateBufferLocal, this->stateBufferLocal, this->statePdfIndexSortLocal, processDataLength);
    PipeBarrier<PIPE_V>();
    Add(this->stateBufferBitNumberLocal, this->stateBufferBitNumberLocal, this->bitLevelMaskZeroLocal,
        EACH_LOOOP_REPEAT_TIMES);
    PipeBarrier<PIPE_V>();
    this->bufOffset -= (BLOCK_SIZE * sizeof(int32_t) + EACH_LOOOP_REPEAT_TIMES * BYTE_BIT_NUM * sizeof(int32_t));
  }

  __aicore__ inline void OverflowProcess(int32_t processDataLength) {
    LocalTensor<int32_t> stateBufferBitCmpLocal = calcBuf.GetWithOffset<int32_t>(BLOCK_SIZE, this->bufOffset);
    this->bufOffset += BLOCK_SIZE * sizeof(int32_t);
    LocalTensor<int32_t> overflowFlagLocal = calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_REPEAT_TIMES, this->bufOffset);
    this->bufOffset += EACH_LOOOP_REPEAT_TIMES * sizeof(int32_t);
    LocalTensor<int32_t> overflowFlagBrcbLocal =
        calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_REPEAT_TIMES * BYTE_BIT_NUM, this->bufOffset);
    this->bufOffset += EACH_LOOOP_REPEAT_TIMES * BYTE_BIT_NUM * sizeof(int32_t);
    LocalTensor<int32_t> overflowFlagBrcbReduceLocal =
        calcBuf.GetWithOffset<int32_t>(EACH_LOOOP_PROCESS_NUM / (sizeof(uint16_t) * BYTE_BIT_NUM), this->bufOffset);
    this->bufOffset += EACH_LOOOP_PROCESS_NUM / (sizeof(uint16_t) * BYTE_BIT_NUM) * sizeof(int32_t);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(stateBufferBitCmpLocal, 0, BLOCK_SIZE);
    CompareScalar(stateBufferBitCmpLocal.ReinterpretCast<uint8_t>(),
                  this->stateBufferBitNumberLocal.template ReinterpretCast<float>(), this->const16Float,
                  AscendC::CMPMODE::GT, EACH_LOOOP_REPEAT_TIMES);
    PipeBarrier<PIPE_V>();
    Select(overflowFlagLocal.ReinterpretCast<float>(), stateBufferBitCmpLocal.ReinterpretCast<uint8_t>(),
           this->ConstOneLocal.template ReinterpretCast<float>(),
           this->ConstZeroLocal.template ReinterpretCast<float>(), AscendC::SELMODE::VSEL_CMPMASK_SPR, BLOCK_SIZE);
    PipeBarrier<PIPE_V>();
    Brcb(overflowFlagBrcbLocal, overflowFlagLocal, BYTE_BIT_NUM, {1, 8});
    PipeBarrier<PIPE_V>();
    GatherMask(
        overflowFlagBrcbReduceLocal.template ReinterpretCast<uint32_t>(),
        overflowFlagBrcbLocal.template ReinterpretCast<uint32_t>(),
        this->reduceLowBlockMaskLocal.template ReinterpretCast<uint32_t>(), true, CONST_8 * EACH_REPEAT_BYTES / sizeof(uint32_t),
        {1, 1, 8, 0}, this->rsvdCnt);
    PipeBarrier<PIPE_V>();
    Muls(overflowFlagBrcbReduceLocal, overflowFlagBrcbReduceLocal, (int32_t)INT32_LOW_16_BIT_MASK,
         EACH_LOOOP_PROCESS_NUM / (sizeof(uint16_t) * BYTE_BIT_NUM));
    PipeBarrier<PIPE_V>();
    GatherMask(
	    this->outputDeviceLocal.template ReinterpretCast<uint16_t>(),
        this->stateBufferLocal.template ReinterpretCast<uint16_t>(),
        overflowFlagBrcbReduceLocal.template ReinterpretCast<uint16_t>(), true,
        processDataLength / BLOCK_SIZE * EACH_REPEAT_BYTES / sizeof(uint16_t), {1, 1, 8, 0}, this->rsvdCnt);
    this->compressBufferOffset = AscendC::AscendCUtils::GetRsvdCnt();
    GatherOutput(overflowFlagBrcbLocal, overflowFlagLocal, processDataLength);
    this->bufOffset -= (BLOCK_SIZE * sizeof(int32_t) + EACH_LOOOP_REPEAT_TIMES * sizeof(int32_t) +
                        EACH_LOOOP_REPEAT_TIMES * BYTE_BIT_NUM * sizeof(int32_t) +
                        EACH_LOOOP_PROCESS_NUM / (sizeof(uint16_t) * BYTE_BIT_NUM) * sizeof(int32_t));
  }

  __aicore__ inline void GatherOutput(LocalTensor<int32_t>& overflowFlagBrcbLocal, 
                                      LocalTensor<int32_t>& overflowFlagLocal, int32_t processDataLength) {
    Adds(overflowFlagBrcbLocal, overflowFlagBrcbLocal, -1, EACH_LOOOP_REPEAT_TIMES * BYTE_BIT_NUM);
    PipeBarrier<PIPE_V>();
    Mul(this->statePdfIndexSortLocal, this->stateBufferLocal, overflowFlagBrcbLocal, BLOCK_SIZE,
        (uint8_t)(processDataLength / BLOCK_SIZE), {1, 1, 0, 8, 8, 1});
    PipeBarrier<PIPE_V>();
    ShiftRight(this->stateBufferLocal.template ReinterpretCast<uint32_t>(),
               this->stateBufferLocal.template ReinterpretCast<uint32_t>(), (uint32_t)(sizeof(uint16_t) * BYTE_BIT_NUM),
               processDataLength);
    PipeBarrier<PIPE_V>();
    Sub(this->stateBufferLocal, this->stateBufferLocal, this->statePdfIndexSortLocal, processDataLength);
    PipeBarrier<PIPE_V>();
    Muls(overflowFlagLocal, overflowFlagLocal, (int32_t)(sizeof(uint16_t) * BYTE_BIT_NUM), BLOCK_SIZE);
    PipeBarrier<PIPE_V>();
    Sub(this->stateBufferBitNumberLocal, this->stateBufferBitNumberLocal, overflowFlagLocal, BLOCK_SIZE);
    PipeBarrier<PIPE_V>();
    GatherMask(this->outputDeviceLocal.template ReinterpretCast<uint16_t>()[this->compressBufferOffset],
               this->bitLevelMaskZeroLocal.template ReinterpretCast<uint16_t>(), 1, false, 0, {1, 1, 8, 0},
               this->rsvdCnt);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void CopyOut(int32_t processDataLength) {
    SetFlag<HardEvent::V_MTE3>(this->eventManager.eventVMTE3Ping);
    WaitFlag<HardEvent::V_MTE3>(this->eventManager.eventVMTE3Ping);
    DataCopyParams copyParams{1, static_cast<uint16_t>((BLOCK_SIZE + this->compressBufferOffset) * 2), 0, 0};
    DataCopyPad(this->fixedGm[this->currentCoreOutputAcculmulateSize],
                this->outputDeviceLocal.template ReinterpretCast<uint8_t>(), copyParams);
    this->currentCoreOutputAcculmulateSize += (BLOCK_SIZE + this->compressBufferOffset) * sizeof(uint16_t);
    this->currentCorePrcoessAcculmulateSize += processDataLength;
    this->currentCoreMantissaAcculmulateSize += processDataLength * (this->dtypeSize - 1);
  }

  __aicore__ inline void DeviceTailProcess() {
    GatherMask(this->outputDeviceLocal.template ReinterpretCast<uint16_t>(),
               this->stateBufferLocal.template ReinterpretCast<uint16_t>(), 1, false, 0, {1, BLOCK_SIZE, 8, 0},
               this->rsvdCnt);
    SetFlag<HardEvent::V_MTE3>(this->eventManager.eventVMTE3Ping);
    WaitFlag<HardEvent::V_MTE3>(this->eventManager.eventVMTE3Ping);
    PipeBarrier<PIPE_MTE3>();
    DataCopyPad(this->fixedGm[this->currentCoreOutputAcculmulateSize],
                this->outputDeviceLocal.template ReinterpretCast<uint8_t>(), {1, static_cast<uint16_t>(8192), 0, 0});
    this->currentCoreOutputAcculmulateSize += EACH_LOOOP_PROCESS_NUM * sizeof(uint16_t);
    SetFlag<HardEvent::MTE3_V>(this->eventManager.eventMTE3VPing);
    WaitFlag<HardEvent::MTE3_V>(this->eventManager.eventMTE3VPing);
    Adds(this->outputDeviceLocal.template ReinterpretCast<int32_t>(),
         this->stateBufferBitNumberLocal.template ReinterpretCast<int32_t>(), 0, BLOCK_SIZE);
    SetFlag<HardEvent::V_MTE3>(this->eventManager.eventVMTE3Ping);
    WaitFlag<HardEvent::V_MTE3>(this->eventManager.eventVMTE3Ping);
    PipeBarrier<PIPE_MTE3>();
    DataCopyPad(this->fixedGm[this->currentCoreOutputAcculmulateSize],
                this->outputDeviceLocal.template ReinterpretCast<uint8_t>(),
                {1, static_cast<uint16_t>(EACH_LOOOP_REPEAT_TIMES * sizeof(int32_t)), 0, 0});
    this->currentCoreOutputAcculmulateSize += BLOCK_SIZE * sizeof(int32_t);
    PipeBarrier<PIPE_MTE3>();
  }

  __aicore__ inline void Sync() {
    this->remainInputSize = this->processLoopCurrentCore * BLOCK_SIZE - this->currentCorePrcoessAcculmulateSize;
    SetFlag<HardEvent::V_S>(this->eventManager.eventVSPing);
    WaitFlag<HardEvent::V_S>(this->eventManager.eventVSPing);
    SetFlag<HardEvent::MTE3_S>(this->eventManager.eventMTE3SPing);
    WaitFlag<HardEvent::MTE3_S>(this->eventManager.eventMTE3SPing);

    int32_t magicNumber = 12138;
    int32_t sumUseCoreIndex = 1;
    int32_t sumLoopsIndex = 2;
    int32_t sumFixedLoopsIndex = 3;
    int32_t sumVarLoopsIndex = 4;

    this->encodeHeaderLocal.SetValue(0, magicNumber);
    this->encodeHeaderLocal.SetValue(sumUseCoreIndex, 1);
    this->encodeHeaderLocal.SetValue(sumLoopsIndex, (int32_t)this->processLoopCurrentCore);
    this->encodeHeaderLocal.SetValue(sumFixedLoopsIndex, this->currentCorePrcoessAcculmulateSize / BLOCK_SIZE);
    this->encodeHeaderLocal.SetValue(sumVarLoopsIndex, this->remainInputSize / BLOCK_SIZE);
    this->encodeHeaderLocal.SetValue(DEVICE_START_IDX + GetBlockIdx(), this->currentCoreOutputAcculmulateSize);
    this->encodeHeaderLocal.SetValue(HOST_START_IDX + GetBlockIdx(), this->remainInputSize);
    SyncAll();
    SetAtomicAdd<int32_t>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(512), 0, 0};
    DataCopyPad(this->outputDeviceHeaderGm, this->encodeHeaderLocal, copyParams);
    SetAtomicNone();
    SyncAll();
  }

  __aicore__ inline void ReshuffDevice() {
    SyncAll();
    int32_t processByteEachLoop = 16384;
    static constexpr int32_t doubleBufferNum = 2;
    this->pipe->InitBuffer(this->inputQue, doubleBufferNum, processByteEachLoop);
    LocalTensor<uint8_t> inputLocalList[doubleBufferNum] = {this->inputQue.template AllocTensor<uint8_t>(),
                                                            this->inputQue.template AllocTensor<uint8_t>()};

    DataCopyPad(this->encodeHeaderLocal, this->outputDeviceHeaderGm, {1, static_cast<uint16_t>(512), 0, 0},
                {true, 0, 0, 0});
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(this->workspaceGm, this->encodeHeaderLocal.template ReinterpretCast<uint8_t>(),
                {1, static_cast<uint16_t>(512), 0, 0});
    PipeBarrier<PIPE_ALL>();
    int64_t outputOffsetNum = 512;
    for (int32_t k = 0; k < GetBlockIdx(); ++k) {
      outputOffsetNum = outputOffsetNum + this->encodeHeaderLocal.GetValue(DEVICE_START_IDX + k);
    }

    int64_t dataSizeRemain = this->encodeHeaderLocal.GetValue(8 + GetBlockIdx());
    int64_t inputOffsetNum = 0;

    SetFlag<HardEvent::MTE3_MTE2>(this->eventManager.eventMTE3MTE2Ping);
    SetFlag<HardEvent::MTE3_MTE2>(this->eventManager.eventMTE3MTE2Pong);
    for (int32_t i = 0; dataSizeRemain > 0; i++) {
      uint32_t size = dataSizeRemain > processByteEachLoop ? processByteEachLoop : dataSizeRemain;
      int32_t index = (i & 1) ? 0 : 1;
      TEventID MTE2MTE3 = (i & 1) ? this->eventManager.eventMTE2MTE3Ping : this->eventManager.eventMTE2MTE3Pong;
      TEventID MTE3MTE2 = (i & 1) ? this->eventManager.eventMTE3MTE2Ping : this->eventManager.eventMTE3MTE2Pong;
      WaitFlag<HardEvent::MTE3_MTE2>(MTE3MTE2);
      DataCopyPad(inputLocalList[index], this->fixedGm[inputOffsetNum], {1, static_cast<uint16_t>(size), 0, 0},
                  {true, 0, 0, 0});
      SetFlag<HardEvent::MTE2_MTE3>(MTE2MTE3);
      WaitFlag<HardEvent::MTE2_MTE3>(MTE2MTE3);
      DataCopyPad(this->workspaceGm[outputOffsetNum], inputLocalList[index], {1, static_cast<uint16_t>(size), 0, 0});
      dataSizeRemain -= processByteEachLoop;
      inputOffsetNum += processByteEachLoop;
      outputOffsetNum += processByteEachLoop;
      SetFlag<HardEvent::MTE3_MTE2>(MTE3MTE2);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(this->eventManager.eventMTE3MTE2Ping);
    WaitFlag<HardEvent::MTE3_MTE2>(this->eventManager.eventMTE3MTE2Pong);

    this->inputQue.template FreeTensor<uint8_t>(inputLocalList[0]);
    this->inputQue.template FreeTensor<uint8_t>(inputLocalList[1]);
  }

  __aicore__ inline void InitBufferInHostTail(int32_t processByteEachLoop){
    this->pipe->InitBuffer(this->inputQue, doubleBufferNum, processByteEachLoop);
    this->pipe->InitBuffer(this->outputMantissaQue, doubleBufferNum,
                           processByteEachLoop / this->dtypeSize * (this->dtypeSize - 1));
    this->pipe->InitBuffer(this->outputHostQue, doubleBufferNum, processByteEachLoop / this->dtypeSize);
    this->pipe->InitBuffer(this->calcBuf, CALC_BUF_SIZE);
    for(int32_t i = 0; i < doubleBufferNum; i++){
      inputLocalList[i] = this->inputQue.template AllocTensor<dataType>();
      outputHostLocalList[i] = this->outputHostQue.template AllocTensor<uint8_t>();
      outputMantissaLocalList[i] = this->outputMantissaQue.template AllocTensor<uint8_t>();
    }
  }

  __aicore__ inline void AllocTensorInHostTail(int32_t processByteEachLoop){
      this->bufOffset = 0;
      for(int i = 0; i <  doubleBufferNum; i++){
        inputReaminFp16LocalList[i] = calcBuf.GetWithOffset<half>(processByteEachLoop, 
                                      bufOffset + processByteEachLoop * doubleBufferNum * i);
      }
      this->bufOffset += processByteEachLoop * doubleBufferNum * sizeof(half);

      for(int i = 0; i <  doubleBufferNum; i++){
        inputReaminExpFp16LocalList[i] = calcBuf.GetWithOffset<half>((processByteEachLoop / this->dtypeSize), 
                          bufOffset +  (processByteEachLoop / this->dtypeSize) * doubleBufferNum * i);
      }
      this->bufOffset += (processByteEachLoop / this->dtypeSize) * doubleBufferNum * sizeof(half);
      for(int i = 0; i <  doubleBufferNum; i++){
        inputReaminMantissaFp16LocalList[i] = 
              calcBuf.GetWithOffset<half>((processByteEachLoop / this->dtypeSize) * (this->dtypeSize - 1), 
              bufOffset + (processByteEachLoop / this->dtypeSize) * (this->dtypeSize - 1) * doubleBufferNum * i);
      }
      this->bufOffset += (processByteEachLoop / this->dtypeSize) * 
                          (this->dtypeSize - 1) * doubleBufferNum * sizeof(half);
      this->encodeHeaderLocal = calcBuf.GetWithOffset<int32_t>(META_INFO_NUM, this->bufOffset);
      this->bufOffset += META_INFO_NUM * sizeof(int32_t);
      this->fp32MantissaMaskLocal = calcBuf.GetWithOffset<uint16_t>(MANTISSA_MASK_NUM, this->bufOffset);
      this->bufOffset += MANTISSA_MASK_NUM * sizeof(uint16_t);
  }

  __aicore__ inline void ProcessInHostTailDb(int32_t loopId,uint32_t size , int64_t& inputOffsetNum,
                        int64_t& ouputHostExpOffsetNum, int64_t& outputMantissaOffsetNum){
      int32_t index = (loopId & 1) ? 0 : 1;
      TEventID MTE3MTE2 = (loopId & 1) ? this->eventManager.eventMTE3MTE2Ping : this->eventManager.eventMTE3MTE2Pong;
      TEventID MTE2V = (loopId & 1) ? this->eventManager.eventMTE2VPing : this->eventManager.eventMTE2VPong;
      TEventID VMTE3 = (loopId & 1) ? this->eventManager.eventVMTE3Ping : this->eventManager.eventVMTE3Pong;
      WaitFlag<HardEvent::MTE3_MTE2>(MTE3MTE2);
      DataCopyParams copyParams{1, static_cast<uint16_t>(size), 0, 0};
      DataCopyPadParams padParams{true, 0, 0, 0};
      DataCopyPad(inputLocalList[index], this->inputGm[inputOffsetNum], copyParams, padParams);
      SetFlag<HardEvent::MTE2_V>(MTE2V);
      WaitFlag<HardEvent::MTE2_V>(MTE2V);
      Cast(inputReaminFp16LocalList[index], inputLocalList[index].template ReinterpretCast<uint8_t>(),
           RoundMode::CAST_NONE, size);
      PipeBarrier<PIPE_V>();
      uint8_t expPattern = this->dtypeSize == sizeof(int32_t) ? 6 : 2;
      GatherMaskParams expParams;
      expParams.src0BlockStride = 1;
      expParams.repeatTimes = static_cast<uint8_t>(size / (EACH_REPEAT_BYTES / sizeof(uint16_t)));
      expParams.src0RepeatStride = REPEAT_STRIDE;
      expParams.src1RepeatStride = 0;
      GatherMask(inputReaminExpFp16LocalList[index], inputReaminFp16LocalList[index], expPattern, false, 0, expParams,
                 this->rsvdCnt);
      PipeBarrier<PIPE_V>();
      if (this->dtypeSize == sizeof(int32_t)) {
        GatherMask(
            inputReaminMantissaFp16LocalList[index].template ReinterpretCast<uint16_t>(),
            inputReaminFp16LocalList[index].template ReinterpretCast<uint16_t>(),
            this->fp32MantissaMaskLocal.template ReinterpretCast<uint16_t>(), true, expParams.repeatTimes * EACH_REPEAT_BYTES / sizeof(uint16_t),
            {1, 1, 8, 0}, this->rsvdCnt);
      } else {
        GatherMask(inputReaminMantissaFp16LocalList[index], inputReaminFp16LocalList[index], 1, false, 0, expParams,
                   this->rsvdCnt);
      }
      PipeBarrier<PIPE_V>();
      Cast(outputHostLocalList[index], inputReaminExpFp16LocalList[index], RoundMode::CAST_TRUNC,
           size / this->dtypeSize);
      PipeBarrier<PIPE_V>();
      Cast(outputMantissaLocalList[index], inputReaminMantissaFp16LocalList[index], RoundMode::CAST_TRUNC,
           size / this->dtypeSize * (this->dtypeSize - 1));
      PipeBarrier<PIPE_V>();
      SetFlag<HardEvent::V_MTE3>(VMTE3);
      WaitFlag<HardEvent::V_MTE3>(VMTE3);
      DataCopyPad(this->varGm[ouputHostExpOffsetNum], outputHostLocalList[index],
                  {1, static_cast<uint16_t>(size / this->dtypeSize), 0, 0});
      DataCopyPad(this->outputMantissaGm[outputMantissaOffsetNum], outputMantissaLocalList[index],
                  {1, static_cast<uint16_t>(size - size / this->dtypeSize), 0, 0});
      SetFlag<HardEvent::MTE3_MTE2>(MTE3MTE2);
      inputOffsetNum += size / this->dtypeSize;
      ouputHostExpOffsetNum += size / this->dtypeSize;
      outputMantissaOffsetNum += size / this->dtypeSize * (this->dtypeSize - 1);
  }

  __aicore__ inline void HostTailProcess() {
    SyncAll();
    if (this->remainInputSize == 0) {
      return;
    }

    int32_t processByteEachLoop = 12288;
    InitBufferInHostTail(processByteEachLoop);
    AllocTensorInHostTail(processByteEachLoop);
    Duplicate<uint16_t>(this->fp32MantissaMaskLocal, (uint16_t)30583ULL, MANTISSA_MASK_NUM);
    PipeBarrier<PIPE_ALL>();

    DataCopyPad(this->encodeHeaderLocal, this->outputDeviceHeaderGm, {1, static_cast<uint16_t>(512), 0, 0},
                {true, 0, 0, 0});
    int64_t hostGmOffset = 0;
    for (int32_t k = 0; k < GetBlockIdx(); ++k) {
      hostGmOffset = hostGmOffset + this->encodeHeaderLocal.GetValue(HOST_START_IDX + k);
    }

    int64_t inputOffsetNum = currentCorePrcoessAcculmulateSize;
    int64_t outputMantissaOffsetNum = currentCoreMantissaAcculmulateSize;
    int64_t ouputHostExpOffsetNum = hostGmOffset;
    int64_t dataSizeRemain = this->remainInputSize * this->dtypeSize;
    SetFlag<HardEvent::MTE3_MTE2>(this->eventManager.eventMTE3MTE2Ping);
    SetFlag<HardEvent::MTE3_MTE2>(this->eventManager.eventMTE3MTE2Pong);
    for (int32_t i = 0; dataSizeRemain > 0; i++) {
      uint32_t size = dataSizeRemain > processByteEachLoop ? processByteEachLoop : dataSizeRemain;
      ProcessInHostTailDb(i, size, inputOffsetNum, ouputHostExpOffsetNum, outputMantissaOffsetNum);
      dataSizeRemain -= size;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(this->eventManager.eventMTE3MTE2Ping);
    WaitFlag<HardEvent::MTE3_MTE2>(this->eventManager.eventMTE3MTE2Pong);
    PipeBarrier<PIPE_ALL>();
    this->inputQue.template FreeTensor<dataType>(inputLocalList[0]);
    this->inputQue.template FreeTensor<dataType>(inputLocalList[1]);
    this->outputHostQue.template FreeTensor<uint8_t>(outputHostLocalList[0]);
    this->outputHostQue.template FreeTensor<uint8_t>(outputHostLocalList[1]);
    this->outputMantissaQue.template FreeTensor<uint8_t>(outputMantissaLocalList[0]);
    this->outputMantissaQue.template FreeTensor<uint8_t>(outputMantissaLocalList[1]);
  }

 private:
  // tiling param
  bool reshuff;
  int32_t dtypeSize;
  int32_t processCoreDim;
  int64_t processLoopPerCore;
  int64_t processLoopCurrentCore;
  int32_t tileDataLength;
  int32_t tileNum;
  int32_t tileRemain;
  // calculate param
  int32_t bufOffset = 0;
  int64_t outputDeviceSizeCurrentCore;
  int64_t outputHostSizeCurrentCore;
  int64_t currentCoreOutputAcculmulateSize = 0;
  int64_t currentCorePrcoessAcculmulateSize = 0;
  int64_t currentCoreMantissaAcculmulateSize = 0;
  int64_t remainInputSize = 0;
  float const16Float = 2.2420775429197073e-44;
  uint64_t rsvdCnt;
  uint64_t compressBufferOffset;
  int32_t debugOffset = 0;

  GlobalTensor<dataType> inputGm;
  GlobalTensor<int32_t> pdfGm;
  GlobalTensor<uint8_t> fixedGm;
  GlobalTensor<uint8_t> workspaceGm;
  GlobalTensor<uint8_t> outputMantissaGm;
  GlobalTensor<uint8_t> varGm;
  GlobalTensor<int32_t> outputDeviceHeaderGm;
  // input
  LocalTensor<dataType> inputLocal;
  LocalTensor<int32_t> pdfLocal;
  // output
  LocalTensor<uint16_t> outputDeviceLocal;
  LocalTensor<uint8_t> outputHostLocal;
  LocalTensor<uint8_t> outputMantissaLocal;
  // calculate
  LocalTensor<half> inputHalfLocal;
  LocalTensor<int32_t> pdfSortIndexLocal;
  LocalTensor<int32_t> ConstOneLocal;
  LocalTensor<int32_t> ConstZeroLocal;
  LocalTensor<uint16_t> fp32MantissaMaskLocal;
  LocalTensor<half> inputExpHalfLocal;
  LocalTensor<int32_t> inputExpUint32Local;
  LocalTensor<int32_t> statePdfIndexSortLocal;
  LocalTensor<int32_t> statePdfMaxUseBitLocal;
  LocalTensor<int32_t> bitLevelMaskZeroLocal;
  LocalTensor<int32_t> stateBufferLocal;
  LocalTensor<int32_t> stateBufferBitNumberLocal;
  LocalTensor<int32_t> encodeHeaderLocal;
  LocalTensor<int32_t> bitShlMulScaleLocal;
  LocalTensor<int32_t> reduceLowBlockMaskLocal;

  TPipe* pipe = nullptr;
  TQue<TPosition::VECIN, 1> inputQue;
  TQue<TPosition::VECIN, 1> pdfQue;
  TBuf<TPosition::VECCALC> calcBuf;
  TQue<TPosition::VECOUT, 1> outputMantissaQue;
  TQue<TPosition::VECOUT, 1> outputDeviceQue;
  TQue<TPosition::VECOUT, 1> outputHostQue;
  HansCommonNs::EventManager eventManager;

  static constexpr int32_t doubleBufferNum = 2;
  LocalTensor<dataType> inputLocalList[doubleBufferNum];
  LocalTensor<uint8_t> outputHostLocalList[doubleBufferNum];
  LocalTensor<uint8_t> outputMantissaLocalList[doubleBufferNum];
  LocalTensor<half> inputReaminFp16LocalList[doubleBufferNum];
  LocalTensor<half> inputReaminExpFp16LocalList[doubleBufferNum];
  LocalTensor<half> inputReaminMantissaFp16LocalList[doubleBufferNum];
};
}  // namespace HansEncodeNS
#endif  // HANS_ENCODE_BASE_H
