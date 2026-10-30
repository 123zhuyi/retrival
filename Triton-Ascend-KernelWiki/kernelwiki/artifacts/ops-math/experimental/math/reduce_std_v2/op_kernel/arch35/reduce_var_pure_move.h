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
 * \file reduce_var_pure_move.h
 * \brief reduce_var_pure_move
 */
#ifndef REDUCE_VAR_PURE_MOVE_H
#define REDUCE_VAR_PURE_MOVE_H

#include "reduce_var_vf_common.h"

namespace ReduceOpTmpl {
using namespace AscendC;

template<typename T>
class ReduceVarPureMove {
public:
    __aicore__ inline ReduceVarPureMove() {};
    __aicore__ inline void InitTiling(const ReduceVarTilingData* tilingData) {
        tiling_ = tilingData;
    }
    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR x, GM_ADDR var, GM_ADDR mean);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerCore();
    __aicore__ inline void VFPureMove(__local_mem__ T *meanLocalAddr, __local_mem__ T *varLocalAddr,
        uint32_t calCount, float varScale);

private:
    int64_t blockIdx_ = 0;
    TPipe *pipe_ = nullptr;

    GlobalTensor<T> inputGM_;
    GlobalTensor<T> varGM_;
    GlobalTensor<T> meanGM_;

    TBuf<> meanBuf_;
    TBuf<> varBuf_;

    const ReduceVarTilingData* tiling_ = nullptr;
    int64_t tailLoopLen_ = 0;
    int64_t addrOffset_ = 0;

    uint64_t loopStartIdx_ = 0;
    uint64_t loopEndIdx_ = 0;

    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};
    DataCopyExtParams copyOutParams_{1, 0, 0, 0, 0};
};

template <typename T>
__aicore__ inline void ReduceVarPureMove<T>::Init(TPipe* pipeIn, GM_ADDR x, GM_ADDR var, GM_ADDR mean)
{
    blockIdx_ = GetBlockIdx();
    pipe_ = pipeIn;

    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    varGM_.SetGlobalBuffer((__gm__ T*)var);
    if (mean != nullptr) {
        meanGM_.SetGlobalBuffer((__gm__ T*)mean);
    }
    pipe_->InitBuffer(meanBuf_, tiling_->reduceOpTiling.basicBlock);
    pipe_->InitBuffer(varBuf_, tiling_->reduceOpTiling.basicBlock);
}

template <typename T>
__aicore__ inline void ReduceVarPureMove<T>::Process()
{
    loopStartIdx_ = blockIdx_ * tiling_->reduceOpTiling.factorACntPerCore;
    loopEndIdx_ = loopStartIdx_ + tiling_->reduceOpTiling.factorACntPerCore;              
    if (unlikely(loopEndIdx_ > tiling_->reduceOpTiling.factorATotalCnt)) {
        loopEndIdx_ = tiling_->reduceOpTiling.factorATotalCnt;
    }

    tailLoopLen_ = tiling_->reduceOpTiling.outSize % tiling_->reduceOpTiling.ubFactorA;
    ProcessPerCore();
}

template <typename T>
__aicore__ inline void ReduceVarPureMove<T>::VFPureMove(__local_mem__ T *meanLocalAddr,
    __local_mem__ T *varLocalAddr,
    uint32_t calCount,
    float varScale)
{
    uint16_t loopCount = Ops::Base::CeilDiv(calCount, ReduceOpTmpl::VL_FP32);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> meanReg;
        AscendC::MicroAPI::RegTensor<float> varReg;
        
        AscendC::MicroAPI::MaskReg pregLoop;
        uint32_t sreg0 = calCount;
        for (uint16_t index = 0; index < loopCount; index++) {
            pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
            // ub -> regTensor
            ReduceOpTmpl::LoadOneTensorForDtypeT<T>(meanLocalAddr, meanReg, pregLoop, index * VL_FP32);
            // sub
            AscendC::MicroAPI::Sub(varReg, meanReg, meanReg, pregLoop);
            // mul
            AscendC::MicroAPI::Mul(varReg, varReg, varReg, pregLoop);
            // muls
            AscendC::MicroAPI::Muls(varReg, varReg, varScale, pregLoop);
            // regTensor -> ub
            ReduceOpTmpl::StoreOneTensorForDtypeT<T>(varLocalAddr, varReg, pregLoop, index * VL_FP32);
        }
    }
}

template <typename T>
__aicore__ inline void ReduceVarPureMove<T>::ProcessPerCore()
{
    int64_t copyElementNum = tiling_->reduceOpTiling.ubFactorA;
    for (int64_t loopIdx = loopStartIdx_; loopIdx < loopEndIdx_; loopIdx++) {
        if (loopIdx == tiling_->reduceOpTiling.factorATotalCnt - 1 && tailLoopLen_ != 0) {
            copyElementNum = tailLoopLen_;
        }

        copyOutParams_.blockLen = copyElementNum * sizeof(T);

        LocalTensor<T> meanLocalIn = meanBuf_.Get<T>();
        LocalTensor<T> varLocalIn = varBuf_.Get<T>();
        __local_mem__ T *meanLocalAddr = (__local_mem__ T *)meanLocalIn.GetPhyAddr();
        __local_mem__ T *varLocalAddr = (__local_mem__ T *)varLocalIn.GetPhyAddr();

        int32_t calCount = static_cast<int32_t>(copyElementNum);
        float varScale = static_cast<float>(tiling_->varFactor);
        if (tiling_->correctionInvalid == 1) {
            varScale = static_cast<float>(NAN);
        }

        DataCopyPad(meanLocalIn, inputGM_[loopIdx * tiling_->reduceOpTiling.ubFactorA], copyOutParams_, padParams_);
        
        Ops::Base::ReduceOpTmpl::SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        VFPureMove(meanLocalAddr, varLocalAddr, static_cast<uint32_t>(calCount), varScale);
        Ops::Base::ReduceOpTmpl::SetEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        
        if (tiling_->isMeanOut != 0) {
            DataCopyPad(meanGM_[loopIdx * tiling_->reduceOpTiling.ubFactorA], meanLocalIn, copyOutParams_);
        }

        DataCopyPad(varGM_[loopIdx * tiling_->reduceOpTiling.ubFactorA], varLocalIn, copyOutParams_);
        Ops::Base::ReduceOpTmpl::SetEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
}
}

#endif  // REDUCE_VAR_PURE_MOVE_H