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
 * \file mem_set.h
 * \brief mem_set struct
 */
#ifndef MEM_SET_ARCH35_H_
#define MEM_SET_ARCH35_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/platform_util.h"
#include "mem_set_struct.h"

namespace MemSetSpc {
using namespace AscendC;

constexpr uint32_t OUT_BUFFER_NUM = 2;
constexpr uint64_t BLOCK_SIZE = 512;

template <uint16_t inputCount>
class MemSet {
public:
    __aicore__ inline MemSet(const MemSetTilingData<inputCount>& tilingData, TPipe& pipe, GM_ADDR* xAddr)
        : tilingData_(tilingData), pipe_(pipe), xAddr_(xAddr){};
    __aicore__ inline void Init();
    __aicore__ inline void Process();

private:
    template <typename T>
    __aicore__ inline void DupProcess(int i, bool isFloat);
    template <typename T>
    __aicore__ inline void DupCounter(int i, bool isFloat, int dupSize);
    template <typename T>
    __aicore__ inline void CopyOut(GlobalTensor<T>& xGm, int64_t addrShift, int dupSize);

private:
    const MemSetTilingData<inputCount>& tilingData_;
    TPipe& pipe_;
    GM_ADDR* xAddr_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    uint32_t blockIdx_;
    int64_t perCoreSizes_[inputCount];
};

template <uint16_t inputCount>
__aicore__ inline void MemSet<inputCount>::Init()
{
    pipe_.InitBuffer(outQueue_, OUT_BUFFER_NUM, tilingData_.halfUbSize);
    blockIdx_ = GetBlockIdx();
    return;
}

template <uint16_t inputCount>
__aicore__ inline void MemSet<inputCount>::Process()
{
    for (uint16_t i = 0; i < tilingData_.inputCount; i++) {
        if (blockIdx_ >= tilingData_.useCore[i]) {
            continue;
        }
        if (blockIdx_ == tilingData_.useCore[i] - 1) {
            perCoreSizes_[i] = tilingData_.lastCoreSizes[i];
        } else {
            perCoreSizes_[i] = tilingData_.perCoreSizes[i];
        }
        if (perCoreSizes_[i] == 0){
            continue;
        }
        switch (tilingData_.listType[i]) {
            case DT_FLOAT:
                DupProcess<float>(i, true);
                break;
            case DT_FLOAT16:
                DupProcess<half>(i, true);
                break;
            case DT_INT8:
                DupProcess<int8_t>(i, false);
                break;
            case DT_INT32:
                DupProcess<int32_t>(i, false);
                break;
            case DT_UINT8:
                DupProcess<uint8_t>(i, false);
                break;
            case DT_INT16:
                DupProcess<int16_t>(i, false);
                break;
            case DT_UINT16:
                DupProcess<uint16_t>(i, false);
                break;
            case DT_UINT32:
                DupProcess<uint32_t>(i, false);
                break;
            case DT_INT64:
                DupProcess<int64_t>(i, false);
                break;
            case DT_UINT64:
                DupProcess<uint64_t>(i, false);
                break;
            default:
                DupProcess<uint32_t>(i, false);
                break;
        }
    }
}

template <uint16_t inputCount>
template <typename T>
__aicore__ inline void MemSet<inputCount>::DupProcess(int i, bool isFloat)
{
    int halfUbSizeForCount = tilingData_.halfUbSize / sizeof(T);
    int64_t loopNum = (perCoreSizes_[i] + halfUbSizeForCount - 1) / halfUbSizeForCount;
    int perLoopSize = (perCoreSizes_[i] + loopNum - 1) / loopNum;
    int lastLoopSize = perCoreSizes_[i] - perLoopSize * (loopNum - 1);
    int dupSize;
    GlobalTensor<T> xGm;
    xGm.SetGlobalBuffer((__gm__ T*)xAddr_[i] + blockIdx_ * tilingData_.perCoreSizes[i]);
    for (int j = 0; j < loopNum; j++) {
        if (j == loopNum - 1) {
            dupSize = lastLoopSize;
        } else {
            dupSize = perLoopSize;
        }
        DupCounter<T>(i, isFloat, dupSize);
        CopyOut<T>(xGm, j * perLoopSize, dupSize);
    }
}

template <uint16_t inputCount>
template <typename T>
__aicore__ inline void MemSet<inputCount>::DupCounter(int i, bool isFloat, int dupSize)
{
    LocalTensor<T> localBuffer = outQueue_.AllocTensor<T>();
    if (isFloat) {
        Duplicate<T>(localBuffer, static_cast<T>(tilingData_.floatValue[i]), dupSize);
    } else {
        Duplicate<T>(localBuffer, static_cast<T>(tilingData_.intValue[i]), dupSize);
    }
    outQueue_.EnQue<T>(localBuffer);
}

template <uint16_t inputCount>
template <typename T>
__aicore__ inline void MemSet<inputCount>::CopyOut(GlobalTensor<T>& xGm, int64_t addrShift, int dupSize)
{
    LocalTensor<T> localBuffer = outQueue_.DeQue<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = dupSize * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(xGm[addrShift], localBuffer, dataCopyParams);
    outQueue_.FreeTensor<T>(localBuffer);
}

} // namespace MemSetSpc
#endif // MEM_SET_STRUCT_H_