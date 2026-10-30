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
 * \file concat_base.h
 * \brief concat base
 */
#ifndef CONCAT_BASE_H
#define CONCAT_BASE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/platform_util.h"

namespace Concat {
using namespace AscendC;
using namespace Ops::Base;
constexpr int64_t BUFFER_NUM_ZERO_AXIS = 2;
constexpr int64_t MAX_DIM_SIZE = 8;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t INDEX_SIZE = 512;
constexpr int64_t BYTES_PER_BLOCK = GetUbBlockSize();
constexpr int64_t ARRAY_SIZE = 10;
constexpr int64_t PRELOAD_DIM1_SIZE = 2;
constexpr int32_t DIGIT_TWO = 2;
constexpr int32_t DIGIT_THREE = 3;
constexpr int32_t DIGIT_FOUR = 4;
using SplitInfo = struct {
    int64_t startIdx;
    int64_t startOffset;
    int64_t endIdx;
    int64_t endOffset;
};

using SplitLoopParam = struct {
    int64_t startTensorIdx;
    int64_t handleTensorNum;
    int64_t perLoopDealTensorNum;
    int64_t tailLoopDealTensorNum;
    int64_t loopSize;
    int64_t tensorStride;
};

template <typename T>
struct VciTypeGet;
template <>
struct VciTypeGet<uint32_t> {
    using T = int32_t;
};
template <>
struct VciTypeGet<uint16_t> {
    using T = int16_t;
};

template <typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using trueType = IntegralConstant<bool, true>;
using falseType = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public falseType {};
template <typename Tp>
struct IsSame<Tp, Tp> : public trueType {};

template <typename T>
__aicore__ inline void Copy(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint32_t rows,
                            uint32_t cols, uint32_t rowStride)
{
    constexpr uint32_t vfLen = GetVRegSize() / sizeof(T);
    uint16_t size0 = rows;
    uint16_t size1 = cols / vfLen;
    uint32_t tail = cols - size1 * vfLen;
    uint32_t main = vfLen;

    auto dstAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
    auto srcAddr = (__ubuf__ T*)srcLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::UnalignReg u0;
        AscendC::Reg::UnalignReg uReg;
        AscendC::Reg::RegTensor<T> vd0;
        AscendC::Reg::DataCopyUnAlignPre(u0, srcAddr);
        for (uint16_t i = 0; i < size0; i++) {
            auto curDstAddr = dstAddr + i * rowStride;
            for (uint16_t j = 0; j < size1; j++) {
                AscendC::Reg::DataCopyUnAlign(vd0, u0, srcAddr, main);
                AscendC::Reg::DataCopyUnAlign(curDstAddr, vd0, uReg, main);
            }
            AscendC::Reg::DataCopyUnAlign(vd0, u0, srcAddr, tail);
            AscendC::Reg::DataCopyUnAlign(curDstAddr, vd0, uReg, tail);
            AscendC::Reg::DataCopyUnAlignPost(curDstAddr, uReg, 0);
        }
    }
}

template <typename T, typename U, int32_t ScatterNum>
__aicore__ inline void ScatterConcat(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint32_t rows,
                                     uint32_t cols, uint32_t rowStride)
{
    constexpr uint32_t vfLen = GetVRegSize() / sizeof(U);
    uint16_t size0 = rows;
    uint32_t tail = cols - cols / vfLen * vfLen;
    if (tail == 0) {
        tail = vfLen;
    }
    uint32_t main = vfLen;

    auto dstAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
    auto srcAddr = (__ubuf__ T*)srcLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::UnalignReg u0;
        AscendC::Reg::RegTensor<T> gather0;
        AscendC::Reg::RegTensor<T> gather1;
        AscendC::Reg::RegTensor<T> gather2;
        AscendC::Reg::RegTensor<T> gather3;
        AscendC::Reg::RegTensor<T> dst;
        AscendC::Reg::RegTensor<U> index;
        using regType = typename VciTypeGet<U>::T;
        AscendC::Reg::RegTensor<regType> tmp;
        AscendC::Reg::Arange(tmp, 0);
        index = (AscendC::Reg::RegTensor<U>&)tmp;
        auto allVf = main;
        auto tailVf = tail;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(allVf);
        AscendC::Reg::MaskReg p1 = AscendC::Reg::UpdateMask<U>(tailVf);
        AscendC::Reg::DataCopyUnAlignPre(u0, srcAddr);
        for (uint16_t i = 0; i < size0; i++) {
            auto curDstAddr = dstAddr + i * rowStride;
            if constexpr (ScatterNum > DIGIT_THREE) {
                AscendC::Reg::DataCopyUnAlign(gather3, u0, srcAddr, main);
                if constexpr (sizeof(T) == 1) {
                    AscendC::Reg::UnPack((AscendC::Reg::RegTensor<uint16_t>&)dst, gather3);
                    AscendC::Reg::DataCopyScatter(curDstAddr, dst, index, p0);
                } else {
                    AscendC::Reg::DataCopyScatter(curDstAddr, gather3, index, p0);
                }
                curDstAddr = curDstAddr + main;
            }
            if constexpr (ScatterNum > DIGIT_TWO) {
                AscendC::Reg::DataCopyUnAlign(gather2, u0, srcAddr, main);
                if constexpr (sizeof(T) == 1) {
                    AscendC::Reg::UnPack((AscendC::Reg::RegTensor<uint16_t>&)dst, gather2);
                    AscendC::Reg::DataCopyScatter(curDstAddr, dst, index, p0);
                } else {
                    AscendC::Reg::DataCopyScatter(curDstAddr, gather2, index, p0);
                }
                curDstAddr = curDstAddr + main;
            }
            if constexpr (ScatterNum > 1) {
                AscendC::Reg::DataCopyUnAlign(gather1, u0, srcAddr, main);
                if constexpr (sizeof(T) == 1) {
                    AscendC::Reg::UnPack((AscendC::Reg::RegTensor<uint16_t>&)dst, gather1);
                    AscendC::Reg::DataCopyScatter(curDstAddr, dst, index, p0);
                } else {
                    AscendC::Reg::DataCopyScatter(curDstAddr, gather1, index, p0);
                }
                curDstAddr = curDstAddr + main;
            }
            AscendC::Reg::DataCopyUnAlign(gather0, u0, srcAddr, tail);
            if constexpr (sizeof(T) == 1) {
                AscendC::Reg::UnPack((AscendC::Reg::RegTensor<uint16_t>&)dst, gather0);
                AscendC::Reg::DataCopyScatter(curDstAddr, dst, index, p1);
            } else {
                AscendC::Reg::DataCopyScatter(curDstAddr, gather0, index, p1);
            }
        }
    }
}

template <typename TILINGDATA>
__aicore__ inline int64_t GetTensorDim0Stride(const TILINGDATA& tilingData_, int64_t idx, int64_t dim1)
{
    int64_t dim0stride = dim1;
    if (tilingData_.isNonContiguous) {
        dim0stride = static_cast<int64_t>(tilingData_.arrays.strideList[idx]);
    }
    return dim0stride;
}

template <typename TILINGDATA, typename T>
__aicore__ inline int64_t GetNonConDimSize(const TILINGDATA& tilingData_, int64_t idx, ListTensorDesc inputList_,
                                           TensorDesc<T> desc_)
{
    int64_t concatDimSize_ = 0;
    if (tilingData_.isNonContiguous) {
        concatDimSize_ = static_cast<int64_t>(tilingData_.arrays.concatDimList[idx]);
    } else {
        inputList_.GetDesc(desc_, idx);
        concatDimSize_ = desc_.GetShape(tilingData_.dim);
    }
    return concatDimSize_;
}
} // namespace Concat
#endif
