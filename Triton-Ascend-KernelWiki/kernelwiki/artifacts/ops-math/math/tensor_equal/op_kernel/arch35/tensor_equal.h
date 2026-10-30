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
 * \file tensor_equal.h
 * \brief
 */

#ifndef TENSOR_EQUAL_H_
#define TENSOR_EQUAL_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace TensorEqual {
using namespace AscendC;

constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t EMPTY_SHAPE_TILINGKEY = 101;
constexpr int64_t DIFF_SHAPE_TILINGKEY = 111;
constexpr int64_t OUTPUT_SIZE = 256;
constexpr uint8_t NORMAL_OUTPUT = 1;
constexpr uint8_t DIFF_SHAPE_OUTPUT = 0;

template <typename T>
class TensorEqualKernel {
    using InputType = std::conditional_t<std::is_integral_v<T>, uint8_t, T>;

public:
    __aicore__ inline TensorEqualKernel(const TensorEqualTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR input_y, GM_ADDR output_z, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t dataLen);
    __aicore__ inline void Compute(int64_t dataLen);

private:
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inputXQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inputYQueue_;
    TBuf<QuePosition::VECCALC> saveBuf_;
    TBuf<QuePosition::VECCALC> resultBuf_;

    GlobalTensor<InputType> inputXGm_;
    GlobalTensor<InputType> inputYGm_;
    GlobalTensor<uint8_t> outputZGm_;

    int64_t blockIdx_ = 0;
    int64_t blockOffset_ = 0;

    int64_t totalCoreNum_;
    int64_t ubFactor_;
    int64_t bufferSize_;
    TPipe& pipe_;
    const TensorEqualTilingData& tilingData_;
};

template <typename T>
__aicore__ inline void TensorEqualKernel<T>::Init(GM_ADDR input_x, GM_ADDR input_y, GM_ADDR output_z, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }

    blockOffset_ = blockIdx_ * (sizeof(T) / sizeof(InputType)) *
                   ((tilingData_.perCoreLoopTimes - 1) * tilingData_.ubFactor + tilingData_.perCoreTailFactor);
    bufferSize_ = tilingData_.ubFactor * sizeof(T);

    inputXGm_.SetGlobalBuffer((__gm__ InputType*)(input_x) + blockOffset_);
    inputYGm_.SetGlobalBuffer((__gm__ InputType*)(input_y) + blockOffset_);
    outputZGm_.SetGlobalBuffer((__gm__ uint8_t*)(output_z));

    if (blockIdx_ == 0) {
        uint32_t initOutput = 1;
        uint8_t globalInitValue = tilingData_.tilingKey == DIFF_SHAPE_TILINGKEY ? DIFF_SHAPE_OUTPUT : NORMAL_OUTPUT;
        InitGlobalMemory(outputZGm_, initOutput, globalInitValue);
        auto mteWaitMTE3EventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(mteWaitMTE3EventID);
        WaitFlag<HardEvent::MTE3_MTE2>(mteWaitMTE3EventID);
    }

    pipe_.InitBuffer(inputXQueue_, DOUBLE_BUFFER, bufferSize_);
    pipe_.InitBuffer(inputYQueue_, DOUBLE_BUFFER, bufferSize_);
    pipe_.InitBuffer(resultBuf_, OUTPUT_SIZE);
    pipe_.InitBuffer(saveBuf_, OUTPUT_SIZE);
    SyncAll();
}

template <typename T>
__aicore__ inline void TensorEqualKernel<T>::CopyIn(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams inParams = {1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<InputType> padParams = {false, 0, 0, 0};

    LocalTensor<InputType> xLocal = inputXQueue_.AllocTensor<InputType>();
    LocalTensor<InputType> yLocal = inputYQueue_.AllocTensor<InputType>();

    int64_t gmOffset = offset * (sizeof(T) / sizeof(InputType));
    DataCopyPad(xLocal, inputXGm_[gmOffset], inParams, padParams);
    DataCopyPad(yLocal, inputYGm_[gmOffset], inParams, padParams);

    inputXQueue_.EnQue(xLocal);
    inputYQueue_.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void TensorEqualKernel<T>::Compute(int64_t dataLen)
{
    LocalTensor<InputType> xLocal = inputXQueue_.DeQue<InputType>();
    LocalTensor<InputType> yLocal = inputYQueue_.DeQue<InputType>();
    __ubuf__ InputType* inputXAddr = (__ubuf__ InputType*)xLocal.GetPhyAddr();
    __ubuf__ InputType* inputYAddr = (__ubuf__ InputType*)yLocal.GetPhyAddr();
    uint16_t strideVReg = Ops::Base::GetVRegSize();
    uint32_t dataLenVf = dataLen * sizeof(T);
    uint16_t repeatTimes = (dataLenVf + strideVReg - 1) / strideVReg;
    LocalTensor<uint32_t> resultLocal = resultBuf_.Get<uint32_t>();
    __ubuf__ uint32_t* resultAddr = (__ubuf__ uint32_t*)resultLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<InputType> xReg, yReg;
        AscendC::Reg::RegTensor<uint16_t> tmpU16Reg, tmpU16RegAdd;
        AscendC::Reg::RegTensor<uint32_t> tmpU32Reg;
        AscendC::Reg::AddrReg offSetReg;
        AscendC::Reg::MaskReg bakMaskRegHigh, bakMaskRegLow, maskReg, cmpMaskReg;
        AscendC::Reg::MaskReg allMaskReg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::MaskReg bakMaskReg = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALLF>();
        for (uint16_t i = 0; i < repeatTimes; i++) {
            offSetReg = AscendC::Reg::CreateAddrReg<uint8_t>(i, strideVReg);
            AscendC::Reg::DataCopy(xReg, inputXAddr, offSetReg);
            AscendC::Reg::DataCopy(yReg, inputYAddr, offSetReg);
            maskReg = AscendC::Reg::UpdateMask<uint8_t>(dataLenVf);
            AscendC::Reg::Compare<InputType, CMPMODE::NE>(cmpMaskReg, xReg, yReg, maskReg);
            AscendC::Reg::MaskOr(bakMaskReg, bakMaskReg, cmpMaskReg, allMaskReg);
        }
        if constexpr (std::is_same<InputType, uint8_t>::value) {
            AscendC::Reg::Duplicate(tmpU16Reg, 1);
            AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::HIGHEST>(bakMaskRegHigh, bakMaskReg);
            AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::LOWEST>(bakMaskRegLow, bakMaskReg);
            AscendC::Reg::ReduceMax(tmpU16RegAdd, tmpU16Reg, bakMaskRegHigh);
            AscendC::Reg::ReduceMax(tmpU16Reg, tmpU16Reg, bakMaskRegLow);
            AscendC::Reg::Add(tmpU16Reg, tmpU16Reg, tmpU16RegAdd, allMaskReg);
            AscendC::Reg::UnPack<uint32_t, uint16_t, AscendC::Reg::HighLowPart::LOWEST>(tmpU32Reg, tmpU16Reg);
            AscendC::Reg::DataCopy(resultAddr, tmpU32Reg, allMaskReg);
        } else if constexpr (std::is_same<InputType, half>::value) {
            AscendC::Reg::Duplicate(tmpU16Reg, 1);
            AscendC::Reg::ReduceMax(tmpU16Reg, tmpU16Reg, bakMaskReg);
            AscendC::Reg::UnPack<uint32_t, uint16_t, AscendC::Reg::HighLowPart::LOWEST>(tmpU32Reg, tmpU16Reg);
            AscendC::Reg::DataCopy(resultAddr, tmpU32Reg, allMaskReg);
        } else {
            AscendC::Reg::Duplicate(tmpU32Reg, 1);
            AscendC::Reg::ReduceMax(tmpU32Reg, tmpU32Reg, bakMaskReg);
            AscendC::Reg::DataCopy(resultAddr, tmpU32Reg, allMaskReg);
        }
    }
    inputXQueue_.FreeTensor(xLocal);
    inputYQueue_.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void TensorEqualKernel<T>::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum || tilingData_.tilingKey == DIFF_SHAPE_TILINGKEY ||
        tilingData_.tilingKey == EMPTY_SHAPE_TILINGKEY) {
        return;
    }
    LocalTensor<uint32_t> saveLocal = saveBuf_.Get<uint32_t>();
    AscendC::Duplicate<uint32_t>(saveLocal, 0, 1);

    int64_t loopSize_ = blockIdx_ == tilingData_.usedCoreNum - 1 ? tilingData_.tailCoreLoopTimes :
                                                                   tilingData_.perCoreLoopTimes;
    int64_t tailFactor_ = blockIdx_ == tilingData_.usedCoreNum - 1 ? tilingData_.tailCoreTailFactor :
                                                                     tilingData_.perCoreTailFactor;

    int64_t offset = 0;

    for (int64_t idx = 0; idx < loopSize_ - 1; idx++) {
        offset = idx * tilingData_.ubFactor;
        CopyIn(offset, tilingData_.ubFactor);
        Compute(tilingData_.ubFactor);
        LocalTensor<uint32_t> resultLocal = resultBuf_.Get<uint32_t>();
        saveLocal = saveLocal + resultLocal;
    }
    offset = (loopSize_ - 1) * tilingData_.ubFactor;
    CopyIn(offset, tailFactor_);
    Compute(tailFactor_);

    LocalTensor<uint32_t> resultLocal = resultBuf_.Get<uint32_t>();
    saveLocal = saveLocal + resultLocal;

    auto sWaitVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(sWaitVEventID);
    WaitFlag<HardEvent::V_S>(sWaitVEventID);

    if (saveLocal(0) != 0 && outputZGm_.GetValue(0) == NORMAL_OUTPUT) {
        outputZGm_.SetValue(0, 0);
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_ALL>(outputZGm_);
        return;
    }
}
} // namespace TensorEqual

#endif // TENSOR_EQUAL_H_
