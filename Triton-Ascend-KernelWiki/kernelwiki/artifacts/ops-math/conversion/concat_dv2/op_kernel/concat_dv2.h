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
 * \file concat_dv2.h
 * \brief
 */
#ifndef CONCAT_DV2_H
#define CONCAT_DV2_H
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;
constexpr int64_t BUFFER_NUM = 1;
constexpr int64_t UB_BLOCK_SIZE = 32;

template <typename T>
class ConcatDV2
{
public:
    __aicore__ inline ConcatDV2(AscendC::TPipe &pipe) : pipe_(pipe) {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dst, const ConcatDV2TilingData& tilingData) {
        blockIdx_ = AscendC::GetBlockIdx();
        usedCoreNum = AscendC::GetBlockNum();
        elePerLoop_ = tilingData.elePerLoop;
        elePercore_ = tilingData.elePercore;
        ubLoop_ = tilingData.ubLoop;
        sameDimSize_ = tilingData.sameDimSize;
        
        if (blockIdx_ != 0) {
            startTensorIdx_ = tilingData.endTensorIdx[blockIdx_ - 1];
            startTensorOffset_ = tilingData.endTensorOffset[blockIdx_ - 1];
        }
        endTensorOffset_ = tilingData.endTensorOffset[blockIdx_];
        endTensorIdx_ = tilingData.endTensorIdx[blockIdx_];

        inputList_ = AscendC::ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        desc_.SetShapeAddr(buf); // 用于获取shape信息
        inputNum_ = inputList_.GetSize();
        
        blockOffset_ = blockIdx_ * elePercore_;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + blockOffset_);
        pipe_.InitBuffer(inOutQueue_, BUFFER_NUM, elePerLoop_ * sizeof(T));
        if (blockIdx_ == AscendC::GetBlockNum() - 1) {
            elePercore_ = tilingData.eleTailCore;
            ubLoop_ = tilingData.ubLoopTail;
        }
    }

    __aicore__ inline void Process() {
        int64_t curTensorIdx = 0;
        int64_t srcGmOffset = 0;
        int64_t dstGmOffset = 0;
        for (int64_t i = 0; i < ubLoop_; i++) {
            AscendC::LocalTensor<T> srcLocal = inOutQueue_.AllocTensor<T>();
            int64_t ubOffset = 0;
            int64_t totalInputSize = 0;
            for (int64_t j = startTensorIdx_; j < endTensorIdx_ + 1; j++) {
                inputList_.GetDesc(desc_, j);
                int64_t inputSize = desc_.GetShape(0) * sameDimSize_;
                int64_t startOffset = 0;
                if (j == startTensorIdx_) {
                    inputSize = inputSize - startTensorOffset_;
                    startOffset = startTensorOffset_;
                } 
                inputSize = (j == endTensorIdx_) ?  endTensorOffset_ : inputSize;
                inputSize = (startTensorIdx_ == endTensorIdx_)? (endTensorOffset_ - startTensorOffset_) : inputSize;
                if (inputSize == 0) {
                    continue;
                }
                int64_t preInputSize = totalInputSize;
                totalInputSize += inputSize;
                // 当前循坏需要处理的tensor
                if (preInputSize < (i + 1) * elePerLoop_ && totalInputSize > i * elePerLoop_) {
                    if (curTensorIdx != j) {
                        srcGmOffset = 0;
                    }
                    curTensorIdx = j;
                    srcGlobal_.SetGlobalBuffer(inputList_.GetDataPtr<T>(j) + startOffset + srcGmOffset);
                    int64_t copyNum = (totalInputSize >= (i + 1) * elePerLoop_) ?
                                        elePerLoop_ :  ((preInputSize > i * elePerLoop_) ?
                                        inputSize : (totalInputSize - i * elePerLoop_));
                    copyNum = ubOffset + copyNum <= elePerLoop_ ? copyNum : elePerLoop_ - ubOffset;
                    uint16_t blockLen = static_cast<uint16_t>(copyNum * sizeof(T) / UB_BLOCK_SIZE);
                    AscendC::DataCopyParams copyParamsIn{1, blockLen, 0, 0};
                    AscendC::DataCopy(srcLocal[ubOffset], srcGlobal_, copyParamsIn);
                    ubOffset += copyNum;
                    srcGmOffset += copyNum;
                }
            }
            inOutQueue_.EnQue(srcLocal);
            AscendC::LocalTensor<T> dstLocal = inOutQueue_.DeQue<T>();
            int64_t copyNum = i == ubLoop_ - 1 ? elePercore_ - i * elePerLoop_ : elePerLoop_;
            uint16_t blockLen = static_cast<uint16_t>(copyNum * sizeof(T) / UB_BLOCK_SIZE);
            AscendC::DataCopyParams copyParamsOut{1, blockLen, 0, 0};
            AscendC::DataCopy(dstGlobal_[dstGmOffset], dstLocal, copyParamsOut);
            dstGmOffset += copyNum;
            inOutQueue_.FreeTensor(dstLocal);
        }
    }

private:
    TPipe &pipe_;
    uint32_t inputNum_{0};
    uint32_t elePerLoop_{0};
    int64_t blockIdx_{0};
    int64_t usedCoreNum{0};
    int64_t elePercore_{0};
    int64_t ubLoop_{0};
    int64_t sameDimSize_{0};
    int64_t startTensorIdx_{0};
    int64_t startTensorOffset_{0};
    int64_t endTensorIdx_{0};
    int64_t endTensorOffset_{0};
    int64_t blockOffset_{0};
    uint64_t buf[10];
    
    AscendC::TensorDesc<T> desc_;
    AscendC::ListTensorDesc inputList_;

    GlobalTensor<T> dstGlobal_;
    GlobalTensor<T> srcGlobal_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inOutQueue_;
};
#endif