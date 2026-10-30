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
 * \file add_n_regbase_fullload.h
 * \brief
 */

#ifndef OP_KERNEL_V35_ADD_N_REGBASE_FULLLOAD_H
#define OP_KERNEL_V35_ADD_N_REGBASE_FULLLOAD_H
#include "kernel_operator_list_tensor_intf.h"
namespace AddNRegbaseFullLoad {
using namespace AscendC;

template <typename T>
class AddNRegbaseFullLoad {
public:
    __aicore__ inline AddNRegbaseFullLoad(const AddNTilingData& tilingData, TPipe* pipe)
        : tilingData_(tilingData), pPipe(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T* GetTensorAddr(int64_t index);
    __aicore__ inline void InitParams();
    __aicore__ inline void InitAndSetBuffer(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void LoopProcess(int64_t index);
    __aicore__ inline void CopyInAndCompute(LocalTensor<T> yLocal);
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyOut();
    __aicore__ inline void AddVF(LocalTensor<T> x1Local, LocalTensor<T> x2Local);

    constexpr static int64_t GROUP_SIZE = 8;
    constexpr static int64_t PADDING_NUM = 32; // 32内存对齐
    const AddNTilingData& tilingData_;
    TPipe* pPipe;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;

    ListTensorDesc inputList_;
    uint32_t blockIdx_{0};
    int64_t blockNum_{0};
    int64_t blockTail_{0};
    int64_t gmOffset{0};
    int64_t ubLoopNum{0};
    int64_t ubTailNum{0};
    int64_t ubBaseNum{0};
    int64_t calcNum{0};
    int64_t dim0Aline{0};
    int64_t dimPadding{0};
};

template <typename T>
__aicore__ inline void AddNRegbaseFullLoad<T>::Init(GM_ADDR x, GM_ADDR y)
{
    InitParams();
    InitAndSetBuffer(x, y);
}

template <typename T>
__aicore__ inline void AddNRegbaseFullLoad<T>::InitParams()
{
    dim0Aline = (tilingData_.dim0 * sizeof(T) + (PADDING_NUM - 1)) / PADDING_NUM * PADDING_NUM / sizeof(T);
    dimPadding = (tilingData_.dim0 * sizeof(T) + (PADDING_NUM - 1)) / PADDING_NUM * PADDING_NUM / sizeof(T) -
                 tilingData_.dim0;
}

template <typename T>
__aicore__ inline void AddNRegbaseFullLoad<T>::InitAndSetBuffer(GM_ADDR x, GM_ADDR y)
{
    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
    xGm.SetGlobalBuffer((__gm__ T*)inputList_.GetDataPtr<T>(0));
    yGm.SetGlobalBuffer((__gm__ T*)y);

    pPipe->InitBuffer(inQueue, 1, dim0Aline * tilingData_.sizeN * sizeof(T));
    pPipe->InitBuffer(outQueue, 1, dim0Aline * sizeof(T));
}

template <typename T>
__aicore__ inline void AddNRegbaseFullLoad<T>::Process()
{
    LocalTensor<T> yLocal = outQueue.template AllocTensor<T>();
    Duplicate(yLocal, static_cast<T>(0.0), tilingData_.dim0);
    CopyInAndCompute(yLocal);
    outQueue.template EnQue<T>(yLocal);
    CopyOut();
    return;
}

template <typename T>
__aicore__ inline void AddNRegbaseFullLoad<T>::CopyInAndCompute(LocalTensor<T> yLocal)
{
    CopyIn();
    LocalTensor<T> xLocal = inQueue.template DeQue<T>();
    for (int64_t i = 1; i < tilingData_.firstN; i++) {
        AddVF(xLocal, xLocal[i * dim0Aline]);
    }
    for (int64_t i = 0; i < tilingData_.loopN; i++) {
        for (int64_t j = 1; j < GROUP_SIZE; j++) {
            AddVF(xLocal[i * GROUP_SIZE * dim0Aline + tilingData_.firstN * dim0Aline],
                  xLocal[i * GROUP_SIZE * dim0Aline + tilingData_.firstN * dim0Aline + j * dim0Aline]);
        }
        AddVF(xLocal, xLocal[i * GROUP_SIZE * dim0Aline + tilingData_.firstN * dim0Aline]);
    }
    AddVF(yLocal, xLocal);
    inQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void AddNRegbaseFullLoad<T>::CopyIn()
{
    LocalTensor<T> xLocal = inQueue.template AllocTensor<T>();
    for (int64_t i = 0; i < tilingData_.sizeN; i++) {
        xGm.SetGlobalBuffer((__gm__ T*)inputList_.GetDataPtr<T>(i));
        DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(tilingData_.dim0 * sizeof(T)),
                                        0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(dimPadding), 0};
        DataCopyPad(xLocal[i * dim0Aline], xGm, copyParams, padParams);
    }
    inQueue.template EnQue(xLocal);
}

template <typename T>
__aicore__ inline void AddNRegbaseFullLoad<T>::CopyOut()
{
    LocalTensor<T> yLocal = outQueue.template DeQue<T>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(tilingData_.dim0 * sizeof(T)), 0, 0,
                                    0};
    DataCopyPad(yGm, yLocal, copyParams);
    outQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void AddNRegbaseFullLoad<T>::AddVF(LocalTensor<T> x1Local, LocalTensor<T> x2Local)
{
    __local_mem__ T* x1Addr = (__local_mem__ T*)x1Local.GetPhyAddr();
    __local_mem__ T* x2Addr = (__local_mem__ T*)x2Local.GetPhyAddr();
    uint32_t dtypeSize = sizeof(T);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoop = (dim0Aline + VL - 1) / VL;
    uint32_t sreg = static_cast<uint32_t>(dim0Aline);
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
} // namespace AddNRegbaseFullLoad

#endif
