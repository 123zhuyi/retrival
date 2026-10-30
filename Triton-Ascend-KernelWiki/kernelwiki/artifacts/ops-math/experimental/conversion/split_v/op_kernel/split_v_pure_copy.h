/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef OP_KERNEL_SPLIT_V_PURE_COPY_H_
#define OP_KERNEL_SPLIT_V_PURE_COPY_H_
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "split_v_tiling_data.h"

using namespace AscendC;

template <typename T>
class SplitVPureCopy {
public:
    __aicore__ inline SplitVPureCopy() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, SplitVTilingDataPureCopy* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t BLOCK_SIZE = 32;

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t length, uint32_t alignedLength);
    __aicore__ inline void CopyOut(uint64_t offset, uint32_t length, uint32_t alignedLength);

private:
    ListTensorDesc outList;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> que;
    uint32_t tileLength = 0;
    uint32_t lastTileLength = 0;
    uint32_t tileNum = 0;
    uint32_t blockIdx = 0;
    uint32_t coreNum = 0;
    uint32_t alignedNum = 0;
    uint64_t blockLength = 0;
    uint64_t offset = 0;
    uint64_t outputNum = 0;
};

template <typename T>
__aicore__ inline void SplitVPureCopy<T>::Init(GM_ADDR input, GM_ADDR output, SplitVTilingDataPureCopy* tiling,
                                               TPipe* pipe)
{
    blockIdx = GetBlockIdx();
    coreNum = GetBlockNum();
    if (blockIdx >= coreNum) {
        return;
    }

    outList.Init((__gm__ void*)output);
    outputNum = outList.GetSize();
    if (outputNum == 0) {
        return;
    }

    tileLength = tiling->tileLength;
    if (tileLength == 0) {
        return;
    }

    if (blockIdx < coreNum - 1) {
        lastTileLength = tiling->formerLastTileLength;
        tileNum = tiling->formerTileNum;
        offset = blockIdx * tiling->formerLength;
        blockLength = tiling->formerLength;
    } else {
        lastTileLength = tiling->tailLastTileLength;
        tileNum = tiling->tailTileNum;
        offset = tiling->formerLength * (coreNum - 1);
        blockLength = tiling->tailLength;
    }

    if (tileNum == 0 || blockLength == 0) {
        return;
    }

    inputGm.SetGlobalBuffer((__gm__ T*)input + offset, blockLength);
    outputGm.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(0) + offset, blockLength);

    pipe->InitBuffer(que, BUFFER_NUM, tileLength * sizeof(T));
    alignedNum = BLOCK_SIZE / sizeof(T);
}

template <typename T>
__aicore__ inline void SplitVPureCopy<T>::Process()
{
    if (tileNum == 0 || blockLength == 0 || outputNum == 0) {
        return;
    }

    for (uint32_t i = 0; i < tileNum; ++i) {
        uint32_t length = (i == tileNum - 1) ? lastTileLength : tileLength;
        uint32_t alignedLength = (length + alignedNum - 1) / alignedNum * alignedNum;
        uint64_t tileOffset = static_cast<uint64_t>(i) * tileLength;
        CopyIn(tileOffset, length, alignedLength);
        CopyOut(tileOffset, length, alignedLength);
    }
}

template <typename T>
__aicore__ inline void SplitVPureCopy<T>::CopyIn(uint64_t offset, uint32_t length, uint32_t alignedLength)
{
    LocalTensor<T> inLocal = que.AllocTensor<T>();
    if (length == alignedLength) {
        DataCopy(inLocal, inputGm[offset], alignedLength);
    } else {
        DataCopyParams dataCopyParams = {1, 0, 0, 0};
        DataCopyPadParams dataCopyPadParams = {true, 0, 0, 0};
        dataCopyParams.blockLen = length * sizeof(T);
        dataCopyPadParams.rightPadding = alignedLength - length;
        DataCopyPad(inLocal, inputGm[offset], dataCopyParams, dataCopyPadParams);
    }
    que.EnQue<T>(inLocal);
}

template <typename T>
__aicore__ inline void SplitVPureCopy<T>::CopyOut(uint64_t offset, uint32_t length, uint32_t alignedLength)
{
    LocalTensor<T> outLocal = que.DeQue<T>();
    if (length == alignedLength) {
        DataCopy(outputGm[offset], outLocal, length);
    } else {
        DataCopyParams dataCopyParams = {1, 0, 0, 0};
        dataCopyParams.blockLen = length * sizeof(T);
        DataCopyPad(outputGm[offset], outLocal, dataCopyParams);
    }
    que.FreeTensor(outLocal);
}

#endif // OP_KERNEL_SPLIT_V_PURE_COPY_H_
