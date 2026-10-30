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
 * \file add_n_regbase.h
 * \brief
 */

#ifndef OP_KERNEL_V35_ADD_N_REGBASE_H
#define OP_KERNEL_V35_ADD_N_REGBASE_H
#include "kernel_operator_list_tensor_intf.h"
constexpr int64_t DB_BUFFER = 2;

namespace AddNRegbase {
using namespace AscendC;

template <typename T>
class AddNRegbase {
public:
    __aicore__ inline AddNRegbase(const AddNTilingData& tilingData, TPipe* pipe)
        : tilingData_(tilingData), pPipe(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T* GetTensorAddr(int64_t index, int64_t offset);
    __aicore__ inline void InitParams();
    __aicore__ inline void InitAndSetBuffer(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void LoopProcess(int64_t index);
    __aicore__ inline void CopyInAndCompute(LocalTensor<T> yLocal, int64_t base, int64_t loop, int64_t offset);
    __aicore__ inline void CopyInX2(int64_t base, int64_t index, int64_t offset);
    __aicore__ inline void CopyInX1(int64_t base, int64_t offset);
    __aicore__ inline void CopyOut(int64_t offset);
    __aicore__ inline void AddVF(LocalTensor<T> x1Local, LocalTensor<T> x2Local);

    constexpr static int64_t GROUP_SIZE = 8;
    const AddNTilingData& tilingData_;
    TPipe* pPipe;
    GlobalTensor<T> x1Gm, x2Gm;
    GlobalTensor<T> yGm;

    TQue<QuePosition::VECIN, DB_BUFFER> inQueue1, inQueue2;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue;

    ListTensorDesc inputList_;
    uint32_t blockIdx_{0};
    int64_t blockNum_{0};
    int64_t blockTail_{0};
    int64_t gmOffset{0};
    int64_t ubLoopNum{0};
    int64_t ubTailNum{0};
    int64_t ubBaseNum{0};
    int64_t calcNum{0};
    int64_t rightPadding{0};
};

template <typename T>
__aicore__ inline __gm__ T* AddNRegbase<T>::GetTensorAddr(int64_t index, int64_t offset)
{
    return inputList_.GetDataPtr<T>(index) + offset;
}

template <typename T>
__aicore__ inline void AddNRegbase<T>::Init(GM_ADDR x, GM_ADDR y)
{
    InitParams();
    InitAndSetBuffer(x, y);
}

template <typename T>
__aicore__ inline void AddNRegbase<T>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    blockNum_ = (tilingData_.dim0 + tilingData_.blockFormer - 1) / tilingData_.blockFormer;
    gmOffset = blockIdx_ * tilingData_.blockFormer;
    ubLoopNum = (blockIdx_ == blockNum_ - 1) ? tilingData_.ubLoopOfTailBlock : tilingData_.ubLoopOfFormerBlock;
    ubTailNum = (blockIdx_ == blockNum_ - 1) ? tilingData_.ubTailOfTailBlock : tilingData_.ubTailOfFormerBlock;
    ubBaseNum = tilingData_.ubFormer > tilingData_.blockFormer ? tilingData_.blockFormer : tilingData_.ubFormer;
    calcNum = ubBaseNum;
    rightPadding = (calcNum * sizeof(T) + 31) / 32 * 32 / sizeof(T) - calcNum;
}

template <typename T>
__aicore__ inline void AddNRegbase<T>::InitAndSetBuffer(GM_ADDR x, GM_ADDR y)
{
    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
    x1Gm.SetGlobalBuffer((__gm__ T*)inputList_.GetDataPtr<T>(0));
    x2Gm.SetGlobalBuffer((__gm__ T*)inputList_.GetDataPtr<T>(0));
    yGm.SetGlobalBuffer((__gm__ T*)y);

    pPipe->InitBuffer(inQueue1, 2, tilingData_.ubFormer * sizeof(T));
    pPipe->InitBuffer(inQueue2, 2, tilingData_.ubFormer * sizeof(T));
    pPipe->InitBuffer(outQueue, 2, tilingData_.ubFormer * sizeof(T));
}

template <typename T>
__aicore__ inline void AddNRegbase<T>::Process()
{
    if (blockIdx_ >= blockNum_) {
        return;
    }

    for (int64_t i = 0; i < ubLoopNum; i++) {
        if (i == ubLoopNum - 1) {
            calcNum = ubTailNum;
            rightPadding = (calcNum * sizeof(T) + 31) / 32 * 32 / sizeof(T) - calcNum;
        }
        LoopProcess(i);
    }

    return;
}

template <typename T>
__aicore__ inline void AddNRegbase<T>::LoopProcess(int64_t index)
{
    int64_t offset = index * ubBaseNum + gmOffset;

    // calc first N
    LocalTensor<T> yLocal = outQueue.template AllocTensor<T>();
    Duplicate(yLocal, static_cast<T>(0.0), calcNum);
    CopyInAndCompute(yLocal, 0, tilingData_.firstN, offset);

    // calc loop N
    for (int64_t j = 0; j < tilingData_.loopN; j++) {
        CopyInAndCompute(yLocal, tilingData_.firstN + j * GROUP_SIZE, GROUP_SIZE, offset);
    }

    outQueue.template EnQue<T>(yLocal);
    CopyOut(offset);
}

template <typename T>
__aicore__ inline void AddNRegbase<T>::CopyInAndCompute(LocalTensor<T> yLocal, int64_t base, int64_t loop,
                                                        int64_t offset)
{
    CopyInX1(base, offset);
    LocalTensor<T> x1Local = inQueue1.template DeQue<T>();

    for (int64_t i = 1; i < loop; i++) {
        CopyInX2(base, i, offset);
        LocalTensor<T> x2Local = inQueue2.template DeQue<T>();
        AddVF(x1Local, x2Local);
        inQueue2.FreeTensor(x2Local);
    }
    AddVF(yLocal, x1Local);
    inQueue1.FreeTensor(x1Local);
}

template <typename T>
__aicore__ inline void AddNRegbase<T>::CopyInX2(int64_t base, int64_t index, int64_t offset)
{
    x2Gm.SetGlobalBuffer(GetTensorAddr(base + index, offset));
    LocalTensor<T> inLocal = inQueue2.template AllocTensor<T>();

    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(calcNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(rightPadding), 0};
    DataCopyPad(inLocal, x2Gm, copyParams, padParams);

    inQueue2.template EnQue(inLocal);
}

template <typename T>
__aicore__ inline void AddNRegbase<T>::CopyInX1(int64_t base, int64_t offset)
{
    x1Gm.SetGlobalBuffer(GetTensorAddr(base, offset));
    LocalTensor<T> x1Local = inQueue1.template AllocTensor<T>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(calcNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(rightPadding), 0};
    DataCopyPad(x1Local, x1Gm, copyParams, padParams);
    inQueue1.template EnQue(x1Local);
}

template <typename T>
__aicore__ inline void AddNRegbase<T>::CopyOut(int64_t offset)
{
    LocalTensor<T> yLocal = outQueue.template DeQue<T>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(calcNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm[offset], yLocal, copyParams);
    outQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void AddNRegbase<T>::AddVF(LocalTensor<T> x1Local, LocalTensor<T> x2Local)
{
    __local_mem__ T* x1Addr = (__local_mem__ T*)x1Local.GetPhyAddr();
    __local_mem__ T* x2Addr = (__local_mem__ T*)x2Local.GetPhyAddr();
    uint32_t dtypeSize = sizeof(T);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoop = (calcNum + VL - 1) / VL;
    uint32_t sreg = static_cast<uint32_t>(calcNum);
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> vreg1;
        AscendC::Reg::RegTensor<T> vreg2;
        AscendC::Reg::RegTensor<T> vreg3;
        AscendC::Reg::MaskReg mask;
        for (uint16_t i = 0; i < vfLoop; i++) {
            mask = AscendC::Reg::UpdateMask<T>(sreg);
            AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_NORM>(vreg1, x1Addr + i * VL);
            AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_NORM>(vreg2, x2Addr + i * VL);
            AscendC::Reg::Add(vreg3, vreg1, vreg2, mask);
            AscendC::Reg::DataCopy<T, AscendC::Reg::StoreDist::DIST_NORM>(x1Addr + i * VL, vreg3, mask);
        }
    }
}

} // namespace AddNRegbase

#endif
