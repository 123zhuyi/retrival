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
 * \file hans_pdf_statistic_base.h
 * \brief
 */
#ifndef HANS_PDF_STATISTIC_BASE_H
#define HANS_PDF_STATISTIC_BASE_H

namespace HansEncodeNS {
using namespace HansCommonNs;
using namespace AscendC;

struct HansEncodeInitConfig {
    GM_ADDR inputGm;
    GM_ADDR pdfGm;
    GM_ADDR outputMantissaGm;
    GM_ADDR fixedGm;
    GM_ADDR varGm;
    GM_ADDR workspace;
    const HansEncodeTilingData* tilingData;
};

template <typename dataType>
class AnsPdfStatistic
{
public:
    __aicore__ inline AnsPdfStatistic()
    {}
    __aicore__ inline void Init(TPipe* pipe_, const HansEncodeInitConfig& config)
    {
        const HansEncodeTilingData* tilingData = config.tilingData;
        if (GetBlockIdx() >= tilingData->processCoreDim) {
            return;
        }
        this->pipe = pipe_;
        this->dtypeSize = sizeof(dataType);
        this->processCoreDim = tilingData->processCoreDim;
        int64_t prcoessNumelCurrentCore = GetBlockIdx() < tilingData->processCoreDim - 1 ?
                                              tilingData->processLoopPerCore * BLOCK_SIZE :
                                              tilingData->processLoopLastCore * BLOCK_SIZE;
        this->tileDataLength = EACH_LOOOP_PROCESS_NUM;
        this->tileNum = prcoessNumelCurrentCore / this->tileDataLength;
        this->tileRemain = prcoessNumelCurrentCore % this->tileDataLength;

        this->inputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ dataType*>(config.inputGm) +
                tilingData->processLoopPerCore * BLOCK_SIZE * GetBlockIdx(),
            prcoessNumelCurrentCore);
        this->pdfGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(config.pdfGm), PDF_LENGTH);

        this->pipe->InitBuffer(this->inputQue, 1, this->dtypeSize * this->tileDataLength);
        this->pipe->InitBuffer(this->pdfQue, 1, sizeof(int32_t) * PDF_LENGTH);
        this->pipe->InitBuffer(this->compareQue, 1, this->tileDataLength / BYTE_BIT_NUM);
        this->pipe->InitBuffer(this->expQue, 1, this->dtypeSize * this->tileDataLength);
    }

    __aicore__ inline void Process()
    {
        // 1. init
        if (GetBlockIdx() >= this->processCoreDim) {
            return;
        }
        this->inputLocal = this->inputQue.template AllocTensor<dataType>();
        this->pdfLocal = this->pdfQue.template AllocTensor<int32_t>();
        this->expLocal = this->expQue.template AllocTensor<dataType>();
        this->compareLocal = this->compareQue.template AllocTensor<uint8_t>();
        for (int32_t i = 0; i < PDF_LENGTH; i++) {
            outStaticArray[i] = 0;
        }
        Duplicate<int32_t>(this->pdfLocal, 0, PDF_LENGTH);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(this->pdfGm, this->pdfLocal, {1, static_cast<uint16_t>(PDF_LENGTH * sizeof(int32_t)), 0, 0});
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        // 2. compute
        for (int32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i * this->tileDataLength, this->tileDataLength * this->dtypeSize);
            ComputePdf(this->tileDataLength);
        }
        PipeBarrier<PIPE_ALL>();
        // 3. remain
        if (this->tileRemain > 0) {
            CopyIn(this->tileNum * this->tileDataLength, this->tileRemain * this->dtypeSize);
            ComputePdf(this->tileRemain);
        }
        PipeBarrier<PIPE_ALL>();
        // 4. copy out
        CopyOut();
        // 5. free
        this->expQue.template FreeTensor<dataType>(this->expLocal);
        this->pdfQue.template FreeTensor<int32_t>(this->pdfLocal);
        this->inputQue.template FreeTensor<dataType>(this->inputLocal);
        SyncAll();
        this->pipe->Reset();
    }

protected:
    __aicore__ inline void CopyIn(int32_t copyOffset, int32_t copySize)
    {
        DataCopyParams copyParams{1, static_cast<uint16_t>(copySize), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyPad(this->inputLocal, this->inputGm[copyOffset], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputePdf(int32_t computeLength)
    {
        using intType = typename std::conditional<sizeof(dataType) == sizeof(float), uint32_t, uint16_t>::type;
        LocalTensor<intType> dataLocalInt = this->inputLocal.template ReinterpretCast<intType>();
        LocalTensor<intType> compareLocalInt = this->compareLocal.template ReinterpretCast<intType>();
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));

        intType shrDistance((this->dtypeSize - 1) * BYTE_BIT_NUM);
        ShiftRight(dataLocalInt, dataLocalInt, shrDistance, computeLength);
        PipeBarrier<PIPE_V>();
        uint64_t rsvdCnt = 0;
        uint8_t repeatTimes = computeLength / (EACH_REPEAT_BYTES / this->dtypeSize);
        GatherMaskParams params;
        params.src0BlockStride = 1;
        params.repeatTimes = static_cast<uint8_t>(computeLength / (EACH_REPEAT_BYTES / this->dtypeSize));
        params.src0RepeatStride = REPEAT_STRIDE;
        params.src1RepeatStride = 0;
        if (computeLength % (EACH_REPEAT_BYTES / this->dtypeSize) == 0) {
            int32_t accNum = 0;
            for (int32_t i = 0; i < PDF_LENGTH; i++) {
                if (accNum == computeLength) {
                    break;
                }
                intType intValue(i);
                dataType scalarValue = *reinterpret_cast<dataType*>(&intValue);
                CompareScalar(
                    compareLocalInt.template ReinterpretCast<uint8_t>(), this->inputLocal, scalarValue, CMPMODE::EQ,
                    computeLength);
                PipeBarrier<PIPE_V>();
                GatherMask(
                    this->expLocal.template ReinterpretCast<intType>(),
                    this->inputLocal.template ReinterpretCast<intType>(),
                    compareLocalInt.template ReinterpretCast<intType>(), true, repeatTimes * EACH_REPEAT_BYTES / sizeof(intType),
                    {1, 1, 8, 0}, this->rsvdCnt);
                rsvdCnt = AscendC::AscendCUtils::GetRsvdCnt();
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(eventVS);
                WaitFlag<HardEvent::V_S>(eventVS);
                this->outStaticArray[i] += rsvdCnt;
                accNum += rsvdCnt;
            }
        } else {
            for (int32_t i = 0; i < computeLength; i++) {
                int32_t index = static_cast<int32_t>(dataLocalInt.GetValue(i));
                this->outStaticArray[index]++;
            }
        }
    }

    __aicore__ inline void CopyOut()
    {
        for (int32_t i = 0; i < PDF_LENGTH; i++) {
            this->pdfLocal.SetValue(i, this->outStaticArray[i]);
        }
        PipeBarrier<PIPE_ALL>();
        SetAtomicAdd<int32_t>();
        DataCopyPad(this->pdfGm, this->pdfLocal, {1, static_cast<uint16_t>(1024), 0, 0});
        SetAtomicNone();
    }

private:
    // tiling param
    int32_t outStaticArray[PDF_LENGTH];
    int32_t dtypeSize;
    int32_t processCoreDim;
    int32_t tileDataLength;
    int32_t tileNum;
    int32_t tileRemain;
    uint64_t rsvdCnt;

    // tensor
    GlobalTensor<dataType> inputGm;
    GlobalTensor<int32_t> pdfGm;

    LocalTensor<dataType> inputLocal;
    LocalTensor<dataType> expLocal;
    LocalTensor<uint8_t> compareLocal;
    LocalTensor<int32_t> pdfLocal;

    TPipe* pipe = nullptr;
    TQue<TPosition::VECIN, 1> inputQue;
    TQue<TPosition::VECIN, 1> expQue;
    TQue<TPosition::VECIN, 1> compareQue;
    TQue<TPosition::VECOUT, 1> pdfQue;
};
} // namespace HansEncodeNS
#endif // HANS_PDF_STATISTIC_BASE_H
