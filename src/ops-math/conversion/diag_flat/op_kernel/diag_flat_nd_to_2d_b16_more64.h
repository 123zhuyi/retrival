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
 * \file diag_flat_nd_to_2d_b16_more64.h
 * \brief
 */
#ifndef DIAG_FLAT_ND_TO_2D_B16_MORE_64
#define DIAG_FLAT_ND_TO_2D_B16_MORE_64

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "diag_flat_common.h"

using namespace AscendC;

namespace DiagFlat {

template <typename T>
class DiagFlatND2To2DB16More64 {
public:
    __aicore__ inline DiagFlatND2To2DB16More64(AscendC::TPipe* p) : pipe(p){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                const DiagV2TilingData* __restrict__ tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint16_t iter);
    __aicore__ inline void Compute(uint16_t iter);
    __aicore__ inline void CopyOut(uint16_t iter);
    __aicore__ inline void CreateAuxMatrix();
    __aicore__ inline void ParseTilingData(const DiagV2TilingData* __restrict__ tilingData);
    __aicore__ inline static constexpr bool IsDataCopyPadSupport()
    {
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310 || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        return true;
#else
        return false;
#endif
    };

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<QuePosition::VECCALC> assistBuf_;

    GlobalTensor<T> gmInput_;
    GlobalTensor<T> gmOutput_;
    GlobalTensor<T> gmAssist_;
    GlobalTensor<T> gmWorkspace_;
    GlobalTensor<int32_t> syncGlobal_;

    TQue<QuePosition::VECIN, 1> workQueue_;

    int64_t offset_{0};
    int64_t inputNum_{0};
    int64_t inputIdx_{0};
    int64_t usedCoreNum_{1};
    int64_t normalCoreHandleNum_{64};
    int64_t lastCoreHandleNum_{64};
    int64_t curHandleNum_{64};
    int64_t totalCoreNum_{1};
};

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16More64<T>::ParseTilingData(const DiagV2TilingData* __restrict__ tilingData)
{
    inputNum_ = tilingData->inputNum;
    usedCoreNum_ = tilingData->usedCoreNum;
    totalCoreNum_ = tilingData->totalCoreNum;
    normalCoreHandleNum_ = tilingData->normalCoreHandleNum;
    lastCoreHandleNum_ = tilingData->lastCoreHandleNum;
    offset_ = tilingData->diagonal;

    inputIdx_ = GetBlockIdx() * normalCoreHandleNum_;
    // [尾核处理] && [非尾核处理]
    if (GetBlockIdx() == usedCoreNum_ - 1) {
        if (lastCoreHandleNum_ % 64 != 0) {
            inputIdx_ -= (64 - (lastCoreHandleNum_ % 64));
        }
        curHandleNum_ = (lastCoreHandleNum_ + 64 - 1) / 64 * 64;
    } else {
        curHandleNum_ = normalCoreHandleNum_;
    }
}

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16More64<T>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                         const DiagV2TilingData* __restrict__ tilingData)
{
    // 解析tiling数据
    ParseTilingData(tilingData);

    // 创建辅助矩阵
    CreateAuxMatrix();

    DiagFlatInitB16Common<T, IsDataCopyPadSupport()>(pipe, syncGlobal_, workQueue_, gmOutput_, gmInput_, inputQueue_,
                                                     outputQueue_, input, output, workspace, totalCoreNum_, inputNum_,
                                                     offset_, inputIdx_);
}

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16More64<T>::CopyIn(uint16_t iter)
{
    LocalTensor<T> ubInput = inputQueue_.AllocTensor<T>();
    DataCopy(ubInput, gmInput_[iter * 64 * 2], 64 * 2);
    inputQueue_.EnQue<T>(ubInput);
}

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16More64<T>::Compute(uint16_t iter)
{
    LocalTensor<T> ubInput = inputQueue_.DeQue<T>();
    LocalTensor<T> ubOutput = outputQueue_.AllocTensor<T>();
    LocalTensor<T> ubAssist = assistBuf_.Get<T>();

    // 参数设置
    uint64_t mask = 128;
    uint64_t repeatTime = 64;
    uint8_t dstBlkStride = 1;
    uint8_t src0BlkStride = 1;
    uint8_t src1BlkStride = 1;
    uint8_t dstRepStride = 32;
    uint8_t src0RepStride = 0;
    uint8_t src1RepStride = 32;

    // 类型转换
    LocalTensor<int16_t> CastUbInput = ubInput.template ReinterpretCast<int16_t>();
    LocalTensor<int16_t> CastUbOutput = ubOutput.template ReinterpretCast<int16_t>();
    LocalTensor<int16_t> CastUbAssist = ubAssist.template ReinterpretCast<int16_t>();

    BinaryRepeatParams params = {dstBlkStride, src0BlkStride, src1BlkStride,
                                 dstRepStride, src0RepStride, src1RepStride};
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    And(CastUbOutput[0], CastUbInput[0], CastUbAssist[0], mask, repeatTime, params);
    And(CastUbOutput[128], CastUbInput[128], CastUbAssist[128], mask, repeatTime, params);
    And(CastUbOutput[128 * 2], CastUbInput[128 * 2], CastUbAssist[128 * 2], mask, repeatTime, params);
    And(CastUbOutput[128 * 3], CastUbInput[128 * 3], CastUbAssist[128 * 3], mask, repeatTime, params);

    outputQueue_.EnQue<T>(ubOutput);
    inputQueue_.FreeTensor(ubInput);
}

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16More64<T>::CopyOut(uint16_t iter)
{
    LocalTensor<T> ubOutput = outputQueue_.DeQue<T>();
    LocalTensor<T> ubAssist = assistBuf_.Get<T>();

    // 区分为output的 [对齐场景,且小于65535个数] && [非对齐场景]进行拷贝策略
    /*
    [CopyOut 逻辑伪代码]:

     if {
         if (对齐场景 && gap < 65535){
             ...
         } else {
             ...
         }
     }
     */

    int64_t width = (inputNum_ + abs(offset_));
    int64_t BLOCK_SIZE = 32;
    uint16_t blockCount = 64;
    uint16_t blockLen = 64 * 2 * sizeof(int64_t) / BLOCK_SIZE;
    uint16_t srcStride = 0;
    uint16_t dstStride = width * 2 * sizeof(int64_t) / BLOCK_SIZE - blockLen;

    if (width % 64 == 0 && dstStride < 65535) {
        const DataCopyParams intriParams = {blockCount, blockLen, srcStride, dstStride};
        auto rowIdx1 = iter * 64;
        int64_t gmOffset1 = (inputNum_ + abs(offset_)) * rowIdx1 + (offset_ > 0 ? offset_ : 0) +
                            (iter * 64 + inputIdx_);
        DataCopy(gmOutput_[gmOffset1 * 2], ubOutput[0], intriParams);
    } else {
        for (int32_t i = 0; i < 64; i++) {
            auto rowIdx2 = iter * 64 + i;
            int64_t gmOffset2 = (inputNum_ + abs(offset_)) * rowIdx2 + (offset_ > 0 ? offset_ : 0) +
                                (iter * 64 + inputIdx_);
            DataCopy(gmOutput_[gmOffset2 * 2], ubOutput[64 * i * 2], 64 * 2);
        }
    }
    outputQueue_.FreeTensor(ubOutput);
}

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16More64<T>::Process()
{
    if (GetBlockIdx() >= usedCoreNum_) {
        return;
    }

    int32_t loops = curHandleNum_ / 64;
    for (int32_t i = 0; i < loops; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

/*
 * 辅助矩阵的创建函数
 */
template <typename T>
__aicore__ inline void DiagFlatND2To2DB16More64<T>::CreateAuxMatrix()
{
    pipe->InitBuffer(assistBuf_, 64 * 128 * sizeof(int64_t));
    LocalTensor<int16_t> ubAssist = assistBuf_.Get<int16_t>();

    constexpr int64_t loops = 64 * 4;
    constexpr int16_t scalarValue = 0;
    constexpr uint8_t calCount = 64 * 2;
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    for (int32_t i = 0; i < loops; i++) {
        Duplicate(ubAssist[calCount * i], scalarValue, calCount);
    }
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    int64_t fillVal = -1;
    constexpr int32_t repeatTime = 64;
    constexpr int32_t step = 128;
    LocalTensor<int64_t> tmpBuf = assistBuf_.Get<int64_t>();
    for (int32_t i = 0; i < repeatTime; i++) {
        tmpBuf.SetValue(step * i + i * 2, fillVal);
        tmpBuf.SetValue(step * i + (i * 2 + 1), fillVal);
    }
}
} // namespace DiagFlat

#endif // DIAG_FLAT_ND_TO_2D_B16_MORE_64
