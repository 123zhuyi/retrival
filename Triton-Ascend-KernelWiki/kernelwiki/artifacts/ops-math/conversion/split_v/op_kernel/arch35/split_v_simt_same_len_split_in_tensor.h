/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_V_SIMT_SAME_LEN_SPLIT_IN_TENSOR_H
#define SPLIT_V_SIMT_SAME_LEN_SPLIT_IN_TENSOR_H
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "simt_api/asc_simt.h"

#ifdef __DAV_FPGA__
constexpr int32_t THREAD_DIM_SLSIT = 512;
#else
constexpr int32_t THREAD_DIM_SLSIT = 2048;
#endif

namespace SplitV
{
using namespace AscendC;

template <typename T>  // T原始数据类型  U做sizeSplits的数据类型
class SplitVSIMTSameLenSplitTensor
{  // 每个核处理的数据小于128K，才会有这个模版，所以数据类型都可以降级
public:
    __aicore__ inline SplitVSIMTSameLenSplitTensor(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SplitSIMTTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T* GetTensorAddr(uint32_t index);

private:
    const SplitSIMTTilingData* tilingData_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    ListTensorDesc outputList_;
    int32_t blockIdx_ = 0;
    int32_t blockNums_ = 0;

    int32_t mLen_ = 0;
    int32_t nLen_ = 0;
    uint32_t splitSize_ = 0;
    int32_t splitNum_ = 0;
    int32_t mainTensorSplitSize_ = 0;
    int32_t tailTensorSplitSize_ = 0;
    int32_t tensorSplitNum_ = 0;
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_SLSIT) __aicore__
    void ProcessSplitSameLenTensor(__gm__ T* inputGM, __gm__ volatile T* outputGM, int32_t outputSize, int32_t nCurLen,
                                   int32_t nLen, int32_t preSumN, int32_t partLen, int32_t partOffset, uint32_t shift,
                                   uint32_t m)
{
    for (uint32_t idx = threadIdx.x; idx < outputSize; idx += blockDim.x) {
        int32_t row = Simt::UintDiv(idx, m, shift);
        int32_t col = idx - row * nCurLen;
        int32_t inputIndex = row * nLen + col + preSumN + partOffset;
        int32_t outputIndex = row * partLen + col + partOffset;
        outputGM[outputIndex] = inputGM[inputIndex];
    }
}

template <typename T>
__aicore__ inline void SplitVSIMTSameLenSplitTensor<T>::Init(GM_ADDR x, GM_ADDR y,
                                                             const SplitSIMTTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    blockNums_ = GetBlockNum();
    tilingData_ = tilingData;
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    outputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(y));
    mLen_ = tilingData_->mSize;
    nLen_ = tilingData_->nSize;
    splitSize_ = tilingData_->splitSize;
    splitNum_ = tilingData_->splitNum;
    mainTensorSplitSize_ = tilingData_->mainTensorSplitSize;
    tailTensorSplitSize_ = tilingData_->tailTensorSplitSize;
    tensorSplitNum_ = tilingData_->tensorSplitNum;
}

template <typename T>
__aicore__ inline void SplitVSIMTSameLenSplitTensor<T>::Process()
{
    if (blockIdx_ >= tilingData_->realCoreNum) {
        return;
    }

    uint32_t nCurLen = 0;
    uint32_t tensorIdx = blockIdx_ / tensorSplitNum_;
    if ((blockIdx_ + 1) % tensorSplitNum_ == 0) {
        nCurLen = tailTensorSplitSize_;
    } else {
        nCurLen = mainTensorSplitSize_;
    }

    uint32_t shift = 0;
    uint32_t m = 0;
    GetUintDivMagicAndShift(m, shift, nCurLen);

    int32_t partLen = splitSize_;
    // tilingData_->colOffset[blockIdx_] - 自己块的长度 - 前1整块的offset
    int32_t partIdx = blockIdx_ % tensorSplitNum_;
    int32_t partOffset = partIdx * mainTensorSplitSize_;
    int32_t colOffset = tensorIdx * splitSize_;

    uint32_t outputSize = mLen_ * nCurLen;
    yGm_.SetGlobalBuffer(GetTensorAddr(tensorIdx));
    asc_vf_call<ProcessSplitSameLenTensor<T>>(dim3(THREAD_DIM_SLSIT), (__gm__ T*)(xGm_.GetPhyAddr()),
                                                (__gm__ volatile T*)(yGm_.GetPhyAddr()), outputSize, nCurLen, nLen_,
                                                colOffset, partLen, partOffset, shift, m);
}

template <typename T>
__aicore__ inline __gm__ T* SplitVSIMTSameLenSplitTensor<T>::GetTensorAddr(uint32_t index)
{
    return outputList_.GetDataPtr<T>(index);
}

}  // namespace SplitV
#endif  // namespace SplitV
