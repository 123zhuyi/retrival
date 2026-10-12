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
 * \file data_compare_empty.h
 * \brief DataCompare 空 tensor 模板 kernel 类
 *
 * EMPTY_A：usedCoreNum=0，所有核早退，输出空 tensor
 * EMPTY_R：usedCoreNum>0，输出 0.0f（Reducer::empty_r_output_value）
 * 不带 isTailR 模板参数（All Reduce 简化）
 */
#ifndef OPS_DATA_COMPARE_EMPTY_H_
#define OPS_DATA_COMPARE_EMPTY_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "data_compare_tiling_data.h"
#include "data_compare_tiling_key.h"

namespace NsDataCompare {

using namespace AscendC;

constexpr uint32_t kEmptyVlBytes = 256;
constexpr uint32_t kEmptyRepF32 = kEmptyVlBytes / sizeof(float);

template <typename DType>
class DataCompareEmpty {
public:
    using D_T = DType;

    __aicore__ inline DataCompareEmpty() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const DataCompareTilingData* td);
    __aicore__ inline void Process();

private:
    int32_t usedCoreNum_ = 0;
    int64_t aTotal_ = 0;
    int64_t aUbFactor_ = 0;
    int32_t aBigCoreCnt_ = 0;
    int64_t aBigCoreLoopCnt_ = 0;
    int64_t aSmallCoreLoopCnt_ = 0;

    GlobalTensor<D_T> x1Gm_;
    GlobalTensor<D_T> x2Gm_;
    GlobalTensor<float> yGm_;
    TPipe pipe_;
    TQue<QuePosition::VECOUT, 1> outQue_;
};

template <typename DType>
__aicore__ inline void DataCompareEmpty<DType>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const DataCompareTilingData* td)
{
    x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(x1));
    x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(x2));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    usedCoreNum_ = td->usedCoreNum;
    aTotal_ = td->axisShape[0];
    aUbFactor_ = td->aUbFactor;
    aBigCoreCnt_ = td->aBigCoreCnt;
    aBigCoreLoopCnt_ = td->aBigCoreLoopCnt;
    aSmallCoreLoopCnt_ = td->aSmallCoreLoopCnt;

    // outBuf：fp32，大小 = CeilAlign(aUbFactor * sizeof(float), blockSize)
    int64_t outBufSize = td->postReduceUbSize;
    if (outBufSize <= 0) {
        outBufSize = 256; // 最小分配
    }
    pipe_.InitBuffer(outQue_, 1, outBufSize);
}

template <typename DType>
__aicore__ inline void DataCompareEmpty<DType>::Process()
{
    int64_t blockIdx = GetBlockIdx();
    if (blockIdx >= usedCoreNum_)
        return; // EMPTY_A: usedCoreNum=0 → 所有核早退

    // EMPTY_R：输出 0.0f
    int64_t aStart, aEnd;
    if (blockIdx < aBigCoreCnt_) {
        aStart = blockIdx * aBigCoreLoopCnt_ * aUbFactor_;
        aEnd = aStart + aBigCoreLoopCnt_ * aUbFactor_;
    } else {
        aStart =
            aBigCoreCnt_ * aBigCoreLoopCnt_ * aUbFactor_ + (blockIdx - aBigCoreCnt_) * aSmallCoreLoopCnt_ * aUbFactor_;
        aEnd = aStart + aSmallCoreLoopCnt_ * aUbFactor_;
    }
    if (aEnd > aTotal_)
        aEnd = aTotal_;

    for (int64_t aOff = aStart; aOff < aEnd; aOff += aUbFactor_) {
        int64_t aLen = aUbFactor_;
        if (aOff + aLen > aTotal_)
            aLen = aTotal_ - aOff;

        // Duplicate 0.0f → outBuf
        auto outLocal = outQue_.AllocTensor<float>();
        __ubuf__ float* outBase = reinterpret_cast<__ubuf__ float*>(outLocal.GetPhyAddr());
        uint32_t count = static_cast<uint32_t>(aLen);
        uint16_t repU16 = static_cast<uint16_t>((count + kEmptyRepF32 - 1) / kEmptyRepF32);

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> idReg;
            AscendC::Reg::Duplicate(idReg, 0.0f);
            AscendC::Reg::MaskReg mask;
            uint32_t remaining = count;
            for (uint16_t i = 0; i < repU16; ++i) {
                int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kEmptyRepF32);
                mask = AscendC::Reg::UpdateMask<float>(remaining);
                AscendC::Reg::StoreAlign(outBase + off, idReg, mask);
            }
        }
        outQue_.EnQue(outLocal);

        // CopyOut
        auto outDeq = outQue_.DeQue<float>();
        DataCopyExtParams cpExt;
        cpExt.blockLen = static_cast<uint32_t>(aLen * sizeof(float));
        cpExt.blockCount = 1;
        cpExt.srcStride = 0;
        cpExt.dstStride = 0;
        cpExt.rsv = 0;
        DataCopyPad(yGm_[aOff], outDeq, cpExt);
        outQue_.FreeTensor(outDeq);
    }
}

} // namespace NsDataCompare

#endif // OPS_DATA_COMPARE_EMPTY_H_
