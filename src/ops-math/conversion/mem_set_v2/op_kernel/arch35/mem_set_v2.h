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
 * \file mem_set_v2.h
 * \brief mem_set_v2 struct
 */
#ifndef MEM_SET_V2_STRUCT_H_
#define MEM_SET_V2_STRUCT_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/platform_util.h"

namespace MemSetV2 {
using namespace AscendC;

constexpr uint32_t DEFAULT_BUFFER_NUM = 1;
constexpr uint64_t BLOCK_SIZE = 65536;
constexpr uint32_t DTYPE_INT = 0;
constexpr uint32_t DTYPE_FLOAT = 1;
constexpr uint32_t MAX_GM_NUM = 256;
constexpr uint32_t AlignSize = 32;

template <typename TILINGDATA>
class MemSetV2 {
public:
    __aicore__ inline MemSetV2(const TILINGDATA& tilingData, TPipe& pipe) : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitBufferSizeInfoByTilingData();
    __aicore__ inline void InitBufferListByGMList();
    __aicore__ inline void SetBufferSizeZero(const uint32_t& gmIndex);
    __aicore__ inline void CalculateBufferSize(const uint32_t& gmIndex, const uint32_t& dealNum);
    __aicore__ inline void DistinguishDTypeInf(const uint32_t& gmIndex);
    __aicore__ inline uint32_t GetDtypeByGmIndex(const uint32_t& gmIndex);
    template <typename T>
    __aicore__ inline void DoDupAndOut(const uint32_t& gmIndex, const T& value);
    template <typename T>
    __aicore__ inline T GetIntValueByGmIndex(const uint32_t& gmIndex);
    template <typename T>
    __aicore__ inline T GetFloatValueByGmIndex(const uint32_t& gmIndex);
    template <typename T>
    __aicore__ inline void Compute(const uint64_t& size, const T& value);
    template <typename T>
    __aicore__ inline void CopyOut(
        const uint32_t& gmIndex, const uint64_t& startOffset, LocalTensor<T> localBuffer, const uint64_t& size);
    __aicore__ inline bfloat16_t GetBfloat16ByFloat(const float& floatValue);

private:
    const TILINGDATA& tilingData_;
    TPipe& pipe_;
    TQue<QuePosition::VECOUT, DEFAULT_BUFFER_NUM> outQueue_;
    TBuf<QuePosition::VECCALC> floatBuffer;
    TBuf<QuePosition::VECCALC> bfloat16Buffer;
    uint32_t blockIdx_;
    uint32_t validNum_; // 共多少个有效GM
    uint64_t coreNum_;
    uint64_t buf_[10] = {0};
    ListTensorDesc inputList_;
    ListTensorDesc outputList_;

    // 以下为描述DB Buffer的数据，每个核一样，通过InitBufferSizeInfoByTilingData()初始化
    uint64_t bufferSize_;    // DB的总buffer大小
    uint64_t perBufferSize_; // 每个分区buffer大小。perBufferSize_ = bufferSize_ / DEFAULT_BUFFER_NUM

    // 以下为描述当前核在每个GM的数据，每个核不一样，通过InitBufferListByGMList()初始化
    uint64_t selfBufferOffsetList_[MAX_GM_NUM]; // 当前核处理每个GM的第一个block的偏移量，单位是Byte
    uint64_t selfFullBufferNumList_
        [MAX_GM_NUM]; // 当前核针对每个GM里需要的block融合成的整块buffer数，一次DB操作处理一个整块buffer
    uint64_t
        selfTailBufferSizeList_[MAX_GM_NUM]; // 除了上述整块buffer外，剩余的buffer大小，也占用一次DB操作，单位为Byte
};

template <typename TILINGDATA>
__aicore__ inline void MemSetV2<TILINGDATA>::Init(GM_ADDR x, GM_ADDR y)
{
    if (GetBlockIdx() > tilingData_.realCoreNum - 1) {
        return;
    }
    InitBufferSizeInfoByTilingData();
    pipe_.InitBuffer(outQueue_, DEFAULT_BUFFER_NUM, perBufferSize_);
    pipe_.InitBuffer(floatBuffer, AlignSize);
    pipe_.InitBuffer(bfloat16Buffer, AlignSize);
    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
    outputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(y));
    blockIdx_ = GetBlockIdx();
    validNum_ = tilingData_.processGMNum;
    coreNum_ = tilingData_.realCoreNum;
    InitBufferListByGMList();
    return;
}

template <typename TILINGDATA>
__aicore__ inline void MemSetV2<TILINGDATA>::InitBufferSizeInfoByTilingData()
{
    bufferSize_ = tilingData_.ubSize - 2 * DEFAULT_BUFFER_NUM * AlignSize;
    perBufferSize_ = bufferSize_ / DEFAULT_BUFFER_NUM;
    return;
}

template <typename TILINGDATA>
__aicore__ inline void MemSetV2<TILINGDATA>::SetBufferSizeZero(const uint32_t& gmIndex)
{
    selfBufferOffsetList_[gmIndex] = 0;
    selfFullBufferNumList_[gmIndex] = 0;
    selfTailBufferSizeList_[gmIndex] = 0;
    return;
}

template <typename TILINGDATA>
__aicore__ inline void MemSetV2<TILINGDATA>::CalculateBufferSize(const uint32_t& gmIndex, const uint32_t& dealNum)
{
    uint64_t totalSize;
    if (tilingData_.tailSizeList[gmIndex] == 0 ||
        blockIdx_ != ((tilingData_.startIndexList[gmIndex] + tilingData_.blockNumList[gmIndex] - 1) < (coreNum_ - 1) ?
                          (tilingData_.startIndexList[gmIndex] + tilingData_.blockNumList[gmIndex] - 1) :
                          (coreNum_ - 1))) {
        // 尾块的处理交给编号最大的block
        totalSize = dealNum * BLOCK_SIZE;
    } else {
        totalSize = dealNum * BLOCK_SIZE - BLOCK_SIZE + tilingData_.tailSizeList[gmIndex];
    }

    selfFullBufferNumList_[gmIndex] = totalSize / perBufferSize_;
    selfTailBufferSizeList_[gmIndex] = totalSize % perBufferSize_;
    return;
}

template <typename TILINGDATA>
__aicore__ inline void MemSetV2<TILINGDATA>::InitBufferListByGMList()
{
    for (uint32_t i = 0; i < validNum_; i++) {
        uint32_t dealNum = 0;     // 表示当前核共需要处理GM[i]里的多少个block
        uint64_t startOffset = 0; // 表示相对当前GM[i]的起始地址偏移了多少个block，copyout时使用

        /*
        当前blockIdx_的dealNum计算公式：
            dealNum[blockIdx_] = (blockNumList_[i] + startIndexList_[i] - blockIdx_ + coreNum_ - 1) / coreNum_
            （实际dealNum[blockIdx_]在起始核之前的需要减1，先按照计算不减1的累加，后面再通过smaller减掉）
            设 x = blockNumList_[i] + startIndexList_[i] + coreNum_ - 1 = k * coreNum_ + tail，（即k = x /
        coreNum_，tail = x % coreNum_） 则 dealNum[blockIdx_] = (x - blockIdx_) / coreNum_
        当前blockIdx_的startOffset计算公式：
            startOffset[blockIdx_] = dealNum[0] + ... + dealNum[blockIdx_ - 1]
            = (x - 0) / coreNum_ + (x - 1) / coreNum_ + ... + (x - (blockIdx_ - 1)) / coreNum_
            = (k * coreNum_ + tail - 0) / coreNum_ + (k * coreNum_ + tail - 1) / coreNum_ + ... + (k * coreNum_ + tail -
        (blockIdx_ - 1)) / coreNum_ = ((k * coreNum_) / coreNum_) * ((blockIdx_ -1) + 1) + (tail - 0) / coreNum_ + (tail
        - 1) / coreNum_ + ... + (tail - (blockIdx_ - 1)) / coreNum_ 先将k * coreNum提出来。如果blockIdx_ >
        tail则存在需要减1，可以根据blockIdx_和tail的大小关系算出减多少个1 = k * ((blockIdx_ -1) + 1) + (tail - 0) /
        coreNum_ + (tail - 1) / coreNum_ + ... + (tail - (blockIdx_ -1)) / coreNum_ = k * (blockIdx_) + biggerOrZero
            再减去dealNum计算方式中起始核之前多加的1：smaller = blockIdx_ < startIndexList_[i] ? blockIdx_ :
        startIndexList_[i]; 最终结果为： startOffset[blockIdx_] = k * (blockIdx_) + biggerOrZero - smaller
        */
        uint32_t blockAddIndex = tilingData_.blockNumList[i] + tilingData_.startIndexList[i] + coreNum_ - 1;
        dealNum = (blockAddIndex - blockIdx_) / coreNum_;
        if (blockIdx_ < tilingData_.startIndexList[i]) {
            dealNum -= 1;
        }

        // 当前block不用处理该GM[i]
        if (dealNum == 0) {
            SetBufferSizeZero(i);
            continue;
        }

        int32_t temp = blockIdx_ - 1 - blockAddIndex % coreNum_;
        uint32_t biggerOrZero = temp > 0 ? temp : 0;
        uint32_t smaller = blockIdx_ < tilingData_.startIndexList[i] ? blockIdx_ : tilingData_.startIndexList[i];
        startOffset = blockIdx_ * (blockAddIndex / coreNum_) - biggerOrZero - smaller;

        selfBufferOffsetList_[i] = (startOffset * BLOCK_SIZE);
        CalculateBufferSize(i, dealNum);
    }
    return;
}

template <typename TILINGDATA>
__aicore__ inline void MemSetV2<TILINGDATA>::Process()
{
    if (GetBlockIdx() > tilingData_.realCoreNum - 1) {
        return;
    }
    for (uint32_t gmIndex = 0; gmIndex < validNum_; gmIndex++) {
        DistinguishDTypeInf(gmIndex);
    }
    return;
}

template <typename TILINGDATA>
__aicore__ inline void MemSetV2<TILINGDATA>::DistinguishDTypeInf(const uint32_t& gmIndex)
{
    uint32_t dtype = GetDtypeByGmIndex(gmIndex);
    if (dtype == 0) {
        float value = GetFloatValueByGmIndex<float>(gmIndex);
        DoDupAndOut<float>(gmIndex, value);
    } else if (dtype == 1) {
        half value = GetFloatValueByGmIndex<half>(gmIndex);
        DoDupAndOut<half>(gmIndex, value);
    } else if (dtype == 27) {
        float floatValue = GetFloatValueByGmIndex<float>(gmIndex);
        bfloat16_t value = GetBfloat16ByFloat(floatValue);
        DoDupAndOut<bfloat16_t>(gmIndex, value);
    } else if (dtype == 2) {
        int8_t value = GetIntValueByGmIndex<int8_t>(gmIndex);
        DoDupAndOut<int8_t>(gmIndex, value);
    } else if (dtype == 3) {
        int32_t value = GetIntValueByGmIndex<int32_t>(gmIndex);
        DoDupAndOut<int32_t>(gmIndex, value);
    } else if (dtype == 4) {
        uint8_t value = GetIntValueByGmIndex<uint8_t>(gmIndex);
        DoDupAndOut<uint8_t>(gmIndex, value);
    } else if (dtype == 12) {
        int8_t value = GetIntValueByGmIndex<int8_t>(gmIndex);
        DoDupAndOut<int8_t>(gmIndex, value);
    } else if (dtype == 10) {
        uint64_t value = GetIntValueByGmIndex<uint64_t>(gmIndex);
        DoDupAndOut<uint64_t>(gmIndex, value);
    } else if (dtype == 9) {
        int64_t value = GetIntValueByGmIndex<int64_t>(gmIndex);
        DoDupAndOut<int64_t>(gmIndex, value);
    } else if (dtype == 8) {
        uint32_t value = GetIntValueByGmIndex<uint32_t>(gmIndex);
        DoDupAndOut<uint32_t>(gmIndex, value);
    } else if (dtype == 6) {
        int16_t value = GetIntValueByGmIndex<int16_t>(gmIndex);
        DoDupAndOut<int16_t>(gmIndex, value);
    } else if (dtype == 7) {
        uint16_t value = GetIntValueByGmIndex<uint16_t>(gmIndex);
        DoDupAndOut<uint16_t>(gmIndex, value);
    }
    return;
}

template <typename TILINGDATA>
template <typename T>
__aicore__ inline void MemSetV2<TILINGDATA>::DoDupAndOut(const uint32_t& gmIndex, const T& value)
{
    uint64_t offset = selfBufferOffsetList_[gmIndex];
    LocalTensor<T> localBuffer = outQueue_.AllocTensor<T>();
    uint32_t fullLoopNum = selfFullBufferNumList_[gmIndex];
    LocalTensor<T> dstLocal;
    if (fullLoopNum > 0) {
        Duplicate<T>(localBuffer, value, perBufferSize_ / sizeof(T));
        outQueue_.EnQue<T>(localBuffer);
        dstLocal = outQueue_.DeQue<T>();
    }
    for (uint32_t loop = 0; loop < fullLoopNum; loop++) {
        CopyOut<T>(gmIndex, offset, dstLocal, perBufferSize_);
        offset += perBufferSize_;
    }

    if (selfTailBufferSizeList_[gmIndex] > 0) {
        if (fullLoopNum <= 0) {
            Duplicate<T>(localBuffer, value, selfTailBufferSizeList_[gmIndex] / sizeof(T));
            outQueue_.EnQue<T>(localBuffer);
            dstLocal = outQueue_.DeQue<T>();
        }
        CopyOut<T>(gmIndex, offset, dstLocal, selfTailBufferSizeList_[gmIndex]);
    }
    outQueue_.FreeTensor(localBuffer);

    return;
}

template <typename TILINGDATA>
__aicore__ inline uint32_t MemSetV2<TILINGDATA>::GetDtypeByGmIndex(const uint32_t& gmIndex)
{
    return tilingData_.dtypesList[gmIndex];
}

template <typename TILINGDATA>
template <typename T>
__aicore__ inline T MemSetV2<TILINGDATA>::GetIntValueByGmIndex(const uint32_t& gmIndex)
{
    return static_cast<T>(tilingData_.valuesIntList[gmIndex]);
}

template <typename TILINGDATA>
template <typename T>
__aicore__ inline T MemSetV2<TILINGDATA>::GetFloatValueByGmIndex(const uint32_t& gmIndex)
{
    return static_cast<T>(tilingData_.valuesFloatList[gmIndex]);
}

template <typename TILINGDATA>
__aicore__ inline bfloat16_t MemSetV2<TILINGDATA>::GetBfloat16ByFloat(const float& floatValue)
{
    LocalTensor<float> floatUb = floatBuffer.Get<float>();
    LocalTensor<bfloat16_t> bfloat16Ub = bfloat16Buffer.Get<bfloat16_t>();
    Duplicate<float>(floatUb, floatValue, 1);
    Cast(bfloat16Ub, floatUb, RoundMode::CAST_RINT, 1);
    PipeBarrier<PIPE_ALL>();
    return bfloat16Ub.GetValue(0);
}

template <typename TILINGDATA>
template <typename T>
__aicore__ inline void MemSetV2<TILINGDATA>::Compute(const uint64_t& size, const T& value)
{
    return;
}

template <typename TILINGDATA>
template <typename T>
__aicore__ inline void MemSetV2<TILINGDATA>::CopyOut(
    const uint32_t& gmIndex, const uint64_t& startOffset, LocalTensor<T> localBuffer, const uint64_t& size)
{
    GlobalTensor<T> dstGlobal;
    dstGlobal.SetGlobalBuffer(
        (__gm__ T*)outputList_.GetDataPtr<T>(gmIndex) + startOffset / sizeof(T), size / sizeof(T));
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(size), 0, 0, 0};
    DataCopyPad<T>(dstGlobal, localBuffer, copyParams);
    return;
}
} // namespace MemSetV2
#endif // MEM_SET_V2_STRUCT_H_