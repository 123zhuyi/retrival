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
 * \file diag_flat_nd_to_2d_b16_less.h
 * \brief
 */
#ifndef DIAG_FLAT_ND_TO_2D_B16_LESS_64
#define DIAG_FLAT_ND_TO_2D_B16_LESS_64

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "diag_flat_common.h"

using namespace AscendC;

namespace DiagFlat {

template <typename T>
class DiagFlatND2To2DB16Less64 {
public:
    __aicore__ inline DiagFlatND2To2DB16Less64(AscendC::TPipe* p) : pipe(p){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                const DiagV2TilingData* __restrict__ tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint16_t iter);
    __aicore__ inline void Compute(uint16_t iter);
    __aicore__ inline void CopyOut(uint16_t iter);
    __aicore__ inline void ConstructZeroMatrix();
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
    TQue<QuePosition::VECIN, 1> workQueue_;
    TBuf<QuePosition::VECCALC> assistBuf_;

    GlobalTensor<T> gmInput_;
    GlobalTensor<T> gmOutput_;
    GlobalTensor<T> gmAssist_;
    GlobalTensor<T> gmWorkspace_;
    GlobalTensor<int32_t> syncGlobal_;

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
__aicore__ inline void DiagFlatND2To2DB16Less64<T>::ParseTilingData(const DiagV2TilingData* __restrict__ tilingData)
{
    inputNum_ = tilingData->inputNum;
    usedCoreNum_ = tilingData->usedCoreNum;
    totalCoreNum_ = tilingData->totalCoreNum;
    normalCoreHandleNum_ = tilingData->normalCoreHandleNum;
    lastCoreHandleNum_ = tilingData->lastCoreHandleNum;
    offset_ = tilingData->diagonal;
    inputIdx_ = GetBlockIdx() * normalCoreHandleNum_;
}

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16Less64<T>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                         const DiagV2TilingData* __restrict__ tilingData)
{
    // 解析tiling数据
    ParseTilingData(tilingData);

    DiagFlatInitB16Common<T, IsDataCopyPadSupport()>(pipe, syncGlobal_, workQueue_, gmOutput_, gmInput_, inputQueue_,
                                                     outputQueue_, input, output, workspace, totalCoreNum_, inputNum_,
                                                     offset_, inputIdx_);

    // 矩阵清零
    ConstructZeroMatrix();
}

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16Less64<T>::CopyIn(uint16_t iter)
{
    LocalTensor<T> ubInput = inputQueue_.AllocTensor<T>();
    DataCopy(ubInput, gmInput_[iter * 64 * 2], 64 * 2);
    inputQueue_.EnQue<T>(ubInput);
}

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16Less64<T>::Compute(uint16_t iter)
{
    LocalTensor<T> ubInput = inputQueue_.DeQue<T>();
    LocalTensor<T> ubOutput = outputQueue_.AllocTensor<T>();
    LocalTensor<T> ubAssist = assistBuf_.Get<T>();

    // 元素个数小于64的场景
    for (int32_t i = 0; i < inputNum_; i++) {
        ubAssist.SetValue(64 * 2 * i + i * 2, ubInput.GetValue(i * 2));
        ubAssist.SetValue(64 * 2 * i + (i * 2 + 1), ubInput.GetValue(i * 2 + 1));
    }

    outputQueue_.EnQue<T>(ubOutput);
    inputQueue_.FreeTensor(ubInput);
}

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16Less64<T>::CopyOut(uint16_t iter)
{
    LocalTensor<T> ubOutput = outputQueue_.DeQue<T>();
    LocalTensor<T> ubAssist = assistBuf_.Get<T>();

    int64_t gmOffset = 0;
    int64_t ONCE_COPY_NUM = ONE_BLK_SIZE / sizeof(T);
    for (int32_t i = 0; i < inputNum_; i++) {
        gmOffset = (inputNum_ + abs(offset_)) * i + (offset_ > 0 ? offset_ : 0);
        DataCopy(gmOutput_[gmOffset * 2], ubAssist[64 * i * 2],
                 (inputNum_ * 2 + ONCE_COPY_NUM - 1) / ONCE_COPY_NUM * ONCE_COPY_NUM);
    }
    outputQueue_.FreeTensor(ubOutput);
}

template <typename T>
__aicore__ inline void DiagFlatND2To2DB16Less64<T>::Process()
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
 * 清零矩阵的创建函数
 */
template <typename T>
__aicore__ inline void DiagFlatND2To2DB16Less64<T>::ConstructZeroMatrix()
{
    pipe->InitBuffer(assistBuf_, 64 * 64 * 2 * sizeof(T));
    LocalTensor<int16_t> ubAssist = assistBuf_.Get<int16_t>();
    constexpr uint8_t BLOCK_LENGTH = 32;
    int16_t scalarValue = 0;
    uint64_t calCount = 64 * 64 * 2 * sizeof(T) / sizeof(int16_t);
    Duplicate(ubAssist, scalarValue, calCount);
}
} // namespace DiagFlat

#endif // DIAG_FLAT_ND_TO_2D_B16_LESS_64
