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
 * \file fused_mul_add_n_align_base.h
 * \brief A2 (DAV_2201) shared CRTP base for FusedMulAddN kernel templates.
 *        Hosts the共享 Init/Process/CopyIn/CopyOut + members. The only per-variant
 *        difference (the elementwise math y = x1*x3[0] + x2) is delegated to the
 *        derived class via compile-time dispatch (CRTP, no virtual functions —
 *        virtual dispatch is不可用 on the AscendC device side).
 *
 *        Derived classes must provide:
 *          - InitScalarAndExtraBuffers(x3, pipe): read x3[0] scalar and init any
 *            extra TBuf (e.g. fp32 cast buffers); called at the end of base Init.
 *          - ComputeImpl(x1Local, x2Local, yLocal, curNum): write yLocal from
 *            x1Local/x2Local (DeQue/Alloc/EnQue/Free are handled by the base).
 */
#ifndef FUSED_MUL_ADD_N_ALIGN_BASE_H
#define FUSED_MUL_ADD_N_ALIGN_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace FusedMulAddNNs {

using namespace AscendC;

constexpr int32_t FMAN_BUFFER_NUM = 2; // Double Buffer

template <typename Derived, typename T>
class FusedMulAddNAlignBase {
public:
    __aicore__ inline FusedMulAddNAlignBase() {}

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, const FusedMulAddNTilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        int64_t blockNum = tilingData->blockNum;
        blockFormer_ = tilingData->blockFormer;
        ubFormer_ = tilingData->ubFormer;

        // 本核处理的元素数 + ub 循环参数（former 核 / 尾核区分）
        if (blockIdx < blockNum - 1) {
            blockLength_ = blockFormer_;
            ubLoop_ = tilingData->ubLoopOfFormerBlock;
            ubTail_ = tilingData->ubTailOfFormerBlock;
        } else if (blockIdx == blockNum - 1) {
            blockLength_ = tilingData->blockTail;
            ubLoop_ = tilingData->ubLoopOfTailBlock;
            ubTail_ = tilingData->ubTailOfTailBlock;
        } else {
            blockLength_ = 0;
            ubLoop_ = 0;
            ubTail_ = 0;
        }

        int64_t gmOffset = blockFormer_ * blockIdx;
        int64_t gmLen = (blockLength_ > 0) ? blockLength_ : 1;
        inputGmX1_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1) + gmOffset, gmLen);
        inputGmX2_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2) + gmOffset, gmLen);
        outputGmY_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + gmOffset, gmLen);
        inputGmX3_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x3), 1);

        pipe_->InitBuffer(inQueueX1_, FMAN_BUFFER_NUM, ubFormer_ * sizeof(T));
        pipe_->InitBuffer(inQueueX2_, FMAN_BUFFER_NUM, ubFormer_ * sizeof(T));
        pipe_->InitBuffer(outQueueY_, FMAN_BUFFER_NUM, ubFormer_ * sizeof(T));

        // 唯一的 per-variant 初始化（x3 标量取值 + 可选 cast 中间 buffer）下沉到派生类
        static_cast<Derived*>(this)->InitScalarAndExtraBuffers(inputGmX3_, ubFormer_);
    }

    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < ubLoop_; i++) {
            int64_t curNum = (i == (ubLoop_ - 1)) ? ubTail_ : ubFormer_;
            if (curNum <= 0) {
                continue;
            }
            CopyIn(i, curNum);
            Compute(curNum);
            CopyOut(i, curNum);
        }
    }

protected:
    __aicore__ inline void CopyIn(int64_t progress, int64_t curNum)
    {
        LocalTensor<T> x1Local = inQueueX1_.template AllocTensor<T>();
        LocalTensor<T> x2Local = inQueueX2_.template AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x1Local, inputGmX1_[progress * ubFormer_], copyParams, padParams);
        DataCopyPad(x2Local, inputGmX2_[progress * ubFormer_], copyParams, padParams);
        inQueueX1_.EnQue(x1Local);
        inQueueX2_.EnQue(x2Local);
    }

    // 共享的 Compute 外壳：DeQue 输入 / Alloc 输出 / EnQue / Free，
    // 中间逐元素计算 y = x1 * x3[0] + x2 由派生类 ComputeImpl 提供（编译期分发）。
    __aicore__ inline void Compute(int64_t curNum)
    {
        LocalTensor<T> x1Local = inQueueX1_.template DeQue<T>();
        LocalTensor<T> x2Local = inQueueX2_.template DeQue<T>();
        LocalTensor<T> yLocal = outQueueY_.template AllocTensor<T>();

        static_cast<Derived*>(this)->ComputeImpl(x1Local, x2Local, yLocal, curNum);

        outQueueY_.template EnQue<T>(yLocal);
        inQueueX1_.FreeTensor(x1Local);
        inQueueX2_.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t curNum)
    {
        LocalTensor<T> yLocal = outQueueY_.template DeQue<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGmY_[progress * ubFormer_], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

protected:
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, FMAN_BUFFER_NUM> inQueueX1_;
    TQue<QuePosition::VECIN, FMAN_BUFFER_NUM> inQueueX2_;
    TQue<QuePosition::VECOUT, FMAN_BUFFER_NUM> outQueueY_;

    GlobalTensor<T> inputGmX1_;
    GlobalTensor<T> inputGmX2_;
    GlobalTensor<T> inputGmX3_;
    GlobalTensor<T> outputGmY_;

    int64_t blockFormer_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubFormer_ = 0;
    int64_t ubLoop_ = 0;
    int64_t ubTail_ = 0;
};

} // namespace FusedMulAddNNs

#endif // FUSED_MUL_ADD_N_ALIGN_BASE_H
