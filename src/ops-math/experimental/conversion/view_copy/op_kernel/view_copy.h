/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file view_copy.h
 * \brief ViewCopy kernel implementation.
 */

#ifndef VIEWCOPY_H_
#define VIEWCOPY_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "view_copy_tiling_data.h"

namespace NsViewCopy {

using namespace AscendC;

constexpr int64_t VIEWCOPY_COPY_BUFFER_BYTES = 65504;
constexpr int64_t VIEWCOPY_DATA_BLOCK_BYTES = 64;
constexpr int64_t VIEWCOPY_INT64_MAX = 9223372036854775807LL;

template <typename T>
class ViewCopy {
public:
    __aicore__ inline ViewCopy() {}
    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR dstSize, GM_ADDR dstStride, GM_ADDR dstStorageOffset, GM_ADDR src,
                                GM_ADDR srcSize, GM_ADDR srcStride, GM_ADDR srcStorageOffset, GM_ADDR y,
                                const ViewCopyTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const ViewCopyTilingData* tilingData);
    __aicore__ inline int64_t ReadMeta(GM_ADDR addr, int64_t index, uint64_t bufferSize) const;
    __aicore__ inline bool LoadRuntimeMetadata();
    __aicore__ inline void BuildSuffixDstMax();
    __aicore__ inline int64_t CalcDstSpan() const;
    __aicore__ inline int64_t CalcViewNum() const;
    __aicore__ inline bool IsDstMaybeOverlapped() const;
    __aicore__ inline int64_t CompareTensorIteratorDims(int64_t dim0, int64_t dim1) const;
    __aicore__ inline void BuildTensorIteratorOrder(int64_t* order) const;
    __aicore__ inline int64_t GcdPositive(int64_t lhs, int64_t rhs) const;
    __aicore__ inline int64_t ModPositive(int64_t value, int64_t divisor) const;
    __aicore__ inline bool ModInversePositive(int64_t value, int64_t divisor, int64_t& inverse) const;
    __aicore__ inline void BuildOrderSuffixDstInfo(const int64_t* order, int64_t* suffixDstMax,
                                                   int64_t* suffixDstGcd) const;
    __aicore__ inline bool SolveOneDimForDst(int64_t depth, int64_t target, int64_t srcBase, const int64_t* order,
                                             int64_t& srcOffset) const;
    __aicore__ inline bool SolveTwoDimsForDst(int64_t depth, int64_t target, int64_t srcBase, const int64_t* order,
                                              int64_t& srcOffset) const;
    __aicore__ inline bool FindLastSrcOffsetForDst(int64_t dstRel, const int64_t* order,
                                                   const int64_t* orderSuffixDstMax, const int64_t* orderSuffixDstGcd,
                                                   int64_t& srcOffset) const;
    __aicore__ inline int64_t FloorDivPositive(int64_t value, int64_t divisor) const;
    __aicore__ inline int64_t CalcContiguousTailElems() const;
    __aicore__ inline int64_t CalcThreeDimChunkElems(int64_t blockNum) const;
    __aicore__ inline bool IsThreeDimTranspose2ContiguousSrc() const;
    __aicore__ inline bool GetContiguousSliceInfo(int64_t& sliceDim, int64_t& sliceNum, int64_t& sliceElems) const;
    __aicore__ inline int64_t CalcContiguousSliceChunkElems(int64_t sliceNum, int64_t sliceElems,
                                                            int64_t blockNum) const;
    __aicore__ inline void LinearToOffset(int64_t linear, int64_t& srcOffset, int64_t& dstOffset) const;
    __aicore__ inline int64_t CeilDivPositive(int64_t value, int64_t divisor) const;
    __aicore__ inline bool PrepareDimRange(int64_t dim, int64_t srcBase, int64_t dstBase, int64_t& begin,
                                           int64_t& end) const;
    __aicore__ inline void CopyElement(GlobalTensor<T>& src, GlobalTensor<T>& dst, int64_t srcOffset,
                                       int64_t dstOffset);
    __aicore__ inline void CopySpan(GlobalTensor<T>& src, GlobalTensor<T>& dst, int64_t srcOffset, int64_t dstOffset,
                                    int64_t elemCount);
    __aicore__ inline void CopyStridedRow(GlobalTensor<T>& src, GlobalTensor<T>& dst, int64_t srcOffset,
                                          int64_t dstOffset, int64_t elemCount);
    __aicore__ inline void CopyThreeDimTranspose2Row(int64_t row);
    __aicore__ inline void CopyThreeDimTranspose2Chunk(int64_t row, int64_t col, int64_t elemCount);
    __aicore__ inline void ProcessThreeDimTranspose2Rows(int64_t blockIdx, int64_t blockNum);
    __aicore__ inline void ProcessThreeDimTranspose2Linear(int64_t blockIdx, int64_t blockNum);
    __aicore__ inline void ProcessContiguousSlices(int64_t blockIdx, int64_t blockNum);
    __aicore__ inline void ProcessLinearViewWrites(int64_t linearStart, int64_t linearEnd);
    __aicore__ inline void ProcessOverlapViewWrites();
    __aicore__ inline void ProcessOverlapDstWrites(int64_t blockIdx, int64_t blockNum);
    __aicore__ inline void ProcessLastDim(int64_t srcBase, int64_t dstBase);
    __aicore__ inline void ProcessViewWrites();

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> dataBuf_;
    GlobalTensor<T> dstGm_;
    GlobalTensor<T> srcGm_;
    GlobalTensor<T> yGm_;

    GM_ADDR dstSizeGm_ = nullptr;
    GM_ADDR dstStrideGm_ = nullptr;
    GM_ADDR dstStorageOffsetGm_ = nullptr;
    GM_ADDR srcStrideGm_ = nullptr;
    GM_ADDR srcStorageOffsetGm_ = nullptr;

    int64_t storageNum_ = 0;
    int64_t srcStorageNum_ = 0;
    int64_t ndim_ = 0;
    int64_t metaTypeBytes_ = 0;
    int64_t viewNum_ = 0;
    int64_t metadataReady_ = 0;
    int64_t dstOverlap_ = 0;
    int64_t dstSpan_ = 0;
    int64_t srcOffset_ = 0;
    int64_t dstOffset_ = 0;
    int64_t copyStart_ = 0;
    int64_t copyEnd_ = 0;
    int64_t sizes_[VIEWCOPY_MAX_DIMS] = {0};
    int64_t srcStrides_[VIEWCOPY_MAX_DIMS] = {0};
    int64_t dstStrides_[VIEWCOPY_MAX_DIMS] = {0};
    int64_t suffixDstMax_[VIEWCOPY_MAX_DIMS + 1] = {0};
    int64_t rangeBegin_[VIEWCOPY_MAX_DIMS] = {0};
    int64_t rangeEnd_[VIEWCOPY_MAX_DIMS] = {0};
    int64_t coord_[VIEWCOPY_MAX_DIMS] = {0};
    int64_t srcStack_[VIEWCOPY_MAX_DIMS] = {0};
    int64_t dstStack_[VIEWCOPY_MAX_DIMS] = {0};
    bool scalarWritten_ = false;
};

template <typename T>
__aicore__ inline void ViewCopy<T>::Init(GM_ADDR dst, GM_ADDR dstSize, GM_ADDR dstStride, GM_ADDR dstStorageOffset,
                                         GM_ADDR src, GM_ADDR srcSize, GM_ADDR srcStride, GM_ADDR srcStorageOffset,
                                         GM_ADDR y, const ViewCopyTilingData* tilingData)
{
    dstSizeGm_ = dstSize;
    (void)srcSize;
    dstStrideGm_ = dstStride;
    dstStorageOffsetGm_ = dstStorageOffset;
    srcStrideGm_ = srcStride;
    srcStorageOffsetGm_ = srcStorageOffset;

    ParseTilingData(tilingData);
    dstGm_.SetGlobalBuffer((__gm__ T*)dst);
    srcGm_.SetGlobalBuffer((__gm__ T*)src);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    pipe_.InitBuffer(dataBuf_, static_cast<uint32_t>(VIEWCOPY_COPY_BUFFER_BYTES));
}

template <typename T>
__aicore__ inline void ViewCopy<T>::ParseTilingData(const ViewCopyTilingData* tilingData)
{
    storageNum_ = tilingData->storageNum;
    srcStorageNum_ = tilingData->srcStorageNum;
    ndim_ = tilingData->ndim;
    metaTypeBytes_ = tilingData->metaTypeBytes;
    viewNum_ = tilingData->viewNum;
    metadataReady_ = tilingData->metadataReady;
    dstOverlap_ = tilingData->dstOverlap;
    dstSpan_ = tilingData->dstSpan;
    if (metadataReady_ != 0 && ndim_ > 0 && ndim_ <= VIEWCOPY_MAX_DIMS) {
        srcOffset_ = tilingData->srcOffset;
        dstOffset_ = tilingData->dstOffset;
        for (int64_t dim = 0; dim < ndim_; ++dim) {
            sizes_[dim] = tilingData->sizes[dim];
            srcStrides_[dim] = tilingData->srcStrides[dim];
            dstStrides_[dim] = tilingData->dstStrides[dim];
        }
        BuildSuffixDstMax();
        if (dstSpan_ <= 0) {
            dstSpan_ = CalcDstSpan();
        }
    }
}

template <typename T>
__aicore__ inline int64_t ViewCopy<T>::ReadMeta(GM_ADDR addr, int64_t index, uint64_t bufferSize) const
{
    if (metaTypeBytes_ == 4) {
        GlobalTensor<int32_t> metaGm;
        metaGm.SetGlobalBuffer((__gm__ int32_t*)addr, static_cast<uint32_t>(bufferSize));
        return static_cast<int64_t>(metaGm.GetValue(static_cast<uint64_t>(index)));
    }
    GlobalTensor<int64_t> metaGm;
    metaGm.SetGlobalBuffer((__gm__ int64_t*)addr, static_cast<uint32_t>(bufferSize));
    return metaGm.GetValue(static_cast<uint64_t>(index));
}

template <typename T>
__aicore__ inline bool ViewCopy<T>::LoadRuntimeMetadata()
{
    if ((metaTypeBytes_ != 4 && metaTypeBytes_ != 8) || ndim_ <= 0 || ndim_ > VIEWCOPY_MAX_DIMS) {
        return false;
    }

    srcOffset_ = ReadMeta(srcStorageOffsetGm_, 0, 1);
    dstOffset_ = ReadMeta(dstStorageOffsetGm_, 0, 1);
    for (int64_t dim = 0; dim < ndim_; ++dim) {
        sizes_[dim] = ReadMeta(dstSizeGm_, dim, static_cast<uint64_t>(ndim_));
        srcStrides_[dim] = ReadMeta(srcStrideGm_, dim, static_cast<uint64_t>(ndim_));
        dstStrides_[dim] = ReadMeta(dstStrideGm_, dim, static_cast<uint64_t>(ndim_));
        if (sizes_[dim] <= 0 || srcStrides_[dim] <= 0 || dstStrides_[dim] <= 0) {
            return false;
        }
    }
    BuildSuffixDstMax();
    dstSpan_ = CalcDstSpan();
    viewNum_ = CalcViewNum();
    metadataReady_ = 1;
    if (dstOverlap_ < 0) {
        dstOverlap_ = IsDstMaybeOverlapped() ? 1 : 0;
    }
    return true;
}

template <typename T>
__aicore__ inline void ViewCopy<T>::BuildSuffixDstMax()
{
    suffixDstMax_[ndim_] = 0;
    for (int64_t dim = ndim_ - 1; dim >= 0; --dim) {
        suffixDstMax_[dim] = suffixDstMax_[dim + 1] + (sizes_[dim] - 1) * dstStrides_[dim];
    }
}

template <typename T>
__aicore__ inline int64_t ViewCopy<T>::CalcDstSpan() const
{
    if (ndim_ <= 0 || ndim_ > VIEWCOPY_MAX_DIMS) {
        return 0;
    }
    return suffixDstMax_[0] + 1;
}

template <typename T>
__aicore__ inline int64_t ViewCopy<T>::CalcViewNum() const
{
    int64_t viewNum = 1;
    for (int64_t dim = 0; dim < ndim_; ++dim) {
        viewNum *= sizes_[dim];
    }
    return viewNum;
}

template <typename T>
__aicore__ inline bool ViewCopy<T>::IsDstMaybeOverlapped() const
{
    if (ndim_ <= 0 || ndim_ > VIEWCOPY_MAX_DIMS) {
        return true;
    }

    int64_t sortedSizes[VIEWCOPY_MAX_DIMS] = {0};
    int64_t sortedStrides[VIEWCOPY_MAX_DIMS] = {0};
    int64_t dimCount = 0;
    for (int64_t dim = 0; dim < ndim_; ++dim) {
        const int64_t size = sizes_[dim];
        const int64_t stride = dstStrides_[dim];
        if (size <= 0) {
            return false;
        }
        if (size == 1) {
            continue;
        }
        if (stride <= 0) {
            return true;
        }

        int64_t pos = dimCount;
        while (pos > 0 && sortedStrides[pos - 1] > stride) {
            sortedSizes[pos] = sortedSizes[pos - 1];
            sortedStrides[pos] = sortedStrides[pos - 1];
            --pos;
        }
        sortedSizes[pos] = size;
        sortedStrides[pos] = stride;
        ++dimCount;
    }

    int64_t requiredSpan = 1;
    for (int64_t idx = 0; idx < dimCount; ++idx) {
        if (sortedStrides[idx] < requiredSpan) {
            return true;
        }
        const int64_t extent = sortedSizes[idx] - 1;
        if (extent <= 0) {
            continue;
        }
        if (sortedStrides[idx] > (VIEWCOPY_INT64_MAX - requiredSpan) / extent) {
            return true;
        }
        requiredSpan += extent * sortedStrides[idx];
    }
    return false;
}

template <typename T>
__aicore__ inline int64_t ViewCopy<T>::CompareTensorIteratorDims(int64_t dim0, int64_t dim1) const
{
    const int64_t dstStride0 = dstStrides_[dim0];
    const int64_t dstStride1 = dstStrides_[dim1];
    if (dstStride0 != 0 && dstStride1 != 0) {
        if (dstStride0 < dstStride1) {
            return -1;
        }
        if (dstStride0 > dstStride1) {
            return 1;
        }
        if (sizes_[dim0] > sizes_[dim1]) {
            return 1;
        }
    }

    const int64_t srcStride0 = srcStrides_[dim0];
    const int64_t srcStride1 = srcStrides_[dim1];
    if (srcStride0 != 0 && srcStride1 != 0) {
        if (srcStride0 < srcStride1) {
            return -1;
        }
        if (srcStride0 > srcStride1) {
            return 1;
        }
        if (sizes_[dim0] > sizes_[dim1]) {
            return 1;
        }
    }
    return 0;
}

template <typename T>
__aicore__ inline void ViewCopy<T>::BuildTensorIteratorOrder(int64_t* order) const
{
    int64_t perm[VIEWCOPY_MAX_DIMS] = {0};
    for (int64_t dim = 0; dim < ndim_; ++dim) {
        perm[dim] = ndim_ - 1 - dim;
    }

    for (int64_t i = 1; i < ndim_; ++i) {
        int64_t dim1 = i;
        int64_t dim0 = i - 1;
        while (dim0 >= 0) {
            const int64_t comparison = CompareTensorIteratorDims(perm[dim0], perm[dim1]);
            if (comparison > 0) {
                const int64_t temp = perm[dim0];
                perm[dim0] = perm[dim1];
                perm[dim1] = temp;
                dim1 = dim0;
            } else if (comparison < 0) {
                break;
            }
            --dim0;
        }
    }

    for (int64_t dim = 0; dim < ndim_; ++dim) {
        order[dim] = perm[ndim_ - 1 - dim];
    }
}

template <typename T>
__aicore__ inline int64_t ViewCopy<T>::GcdPositive(int64_t lhs, int64_t rhs) const
{
    if (lhs < 0) {
        lhs = -lhs;
    }
    if (rhs < 0) {
        rhs = -rhs;
    }
    while (rhs != 0) {
        const int64_t tmp = lhs % rhs;
        lhs = rhs;
        rhs = tmp;
    }
    return lhs;
}

template <typename T>
__aicore__ inline int64_t ViewCopy<T>::ModPositive(int64_t value, int64_t divisor) const
{
    int64_t result = value % divisor;
    return result < 0 ? result + divisor : result;
}

template <typename T>
__aicore__ inline bool ViewCopy<T>::ModInversePositive(int64_t value, int64_t divisor, int64_t& inverse) const
{
    int64_t t = 0;
    int64_t newT = 1;
    int64_t r = divisor;
    int64_t newR = ModPositive(value, divisor);
    while (newR != 0) {
        const int64_t quotient = r / newR;
        const int64_t nextT = t - quotient * newT;
        t = newT;
        newT = nextT;
        const int64_t nextR = r - quotient * newR;
        r = newR;
        newR = nextR;
    }
    if (r != 1) {
        return false;
    }
    inverse = ModPositive(t, divisor);
    return true;
}

template <typename T>
__aicore__ inline void ViewCopy<T>::BuildOrderSuffixDstInfo(const int64_t* order, int64_t* suffixDstMax,
                                                            int64_t* suffixDstGcd) const
{
    suffixDstMax[ndim_] = 0;
    suffixDstGcd[ndim_] = 0;
    for (int64_t idx = ndim_ - 1; idx >= 0; --idx) {
        const int64_t dim = order[idx];
        suffixDstMax[idx] = suffixDstMax[idx + 1] + (sizes_[dim] - 1) * dstStrides_[dim];
        suffixDstGcd[idx] = GcdPositive(dstStrides_[dim], suffixDstGcd[idx + 1]);
    }
}

template <typename T>
__aicore__ inline bool ViewCopy<T>::SolveOneDimForDst(int64_t depth, int64_t target, int64_t srcBase,
                                                      const int64_t* order, int64_t& srcOffset) const
{
    const int64_t dim = order[depth];
    const int64_t stride = dstStrides_[dim];
    if (stride <= 0 || target < 0 || target % stride != 0) {
        return false;
    }
    const int64_t coord = target / stride;
    if (coord < 0 || coord >= sizes_[dim]) {
        return false;
    }
    srcOffset = srcBase + coord * srcStrides_[dim];
    return true;
}

template <typename T>
__aicore__ inline bool ViewCopy<T>::SolveTwoDimsForDst(int64_t depth, int64_t target, int64_t srcBase,
                                                       const int64_t* order, int64_t& srcOffset) const
{
    const int64_t dim0 = order[depth];
    const int64_t dim1 = order[depth + 1];
    const int64_t stride0 = dstStrides_[dim0];
    const int64_t stride1 = dstStrides_[dim1];
    if (stride0 <= 0 || stride1 <= 0 || target < 0) {
        return false;
    }

    const int64_t gcd = GcdPositive(stride0, stride1);
    if (gcd <= 0 || target % gcd != 0) {
        return false;
    }

    const int64_t reducedStride0 = stride0 / gcd;
    const int64_t reducedStride1 = stride1 / gcd;
    const int64_t reducedTarget = target / gcd;
    int64_t inverse = 0;
    if (!ModInversePositive(reducedStride0, reducedStride1, inverse)) {
        return false;
    }

    const int64_t size0 = sizes_[dim0];
    const int64_t size1 = sizes_[dim1];
    int64_t coord0 = ModPositive((reducedTarget % reducedStride1) * inverse, reducedStride1);
    const int64_t step = reducedStride1;
    if (step <= 0) {
        return false;
    }
    if (coord0 < size0) {
        coord0 += ((size0 - 1 - coord0) / step) * step;
    } else {
        const int64_t delta = CeilDivPositive(coord0 - (size0 - 1), step);
        coord0 -= delta * step;
    }

    while (coord0 >= 0) {
        const int64_t remain = target - coord0 * stride0;
        if (remain >= 0 && remain % stride1 == 0) {
            const int64_t coord1 = remain / stride1;
            if (coord1 >= 0 && coord1 < size1) {
                srcOffset = srcBase + coord0 * srcStrides_[dim0] + coord1 * srcStrides_[dim1];
                return true;
            }
        }
        coord0 -= reducedStride1;
    }
    return false;
}

template <typename T>
__aicore__ inline bool ViewCopy<T>::FindLastSrcOffsetForDst(int64_t dstRel, const int64_t* order,
                                                            const int64_t* orderSuffixDstMax,
                                                            const int64_t* orderSuffixDstGcd, int64_t& srcOffset) const
{
    if (dstRel < 0 || ndim_ <= 0 || ndim_ > VIEWCOPY_MAX_DIMS || dstRel > orderSuffixDstMax[0]) {
        return false;
    }
    if (orderSuffixDstGcd[0] > 1 && dstRel % orderSuffixDstGcd[0] != 0) {
        return false;
    }

    bool initialized[VIEWCOPY_MAX_DIMS] = {false};
    int64_t lowCoord[VIEWCOPY_MAX_DIMS] = {0};
    int64_t nextCoord[VIEWCOPY_MAX_DIMS] = {0};
    int64_t targetStack[VIEWCOPY_MAX_DIMS + 1] = {0};
    int64_t srcStack[VIEWCOPY_MAX_DIMS + 1] = {0};

    int64_t depth = 0;
    targetStack[0] = dstRel;
    srcStack[0] = srcOffset_;
    while (depth >= 0) {
        const int64_t remainingDims = ndim_ - depth;
        if (remainingDims == 0) {
            if (targetStack[depth] == 0) {
                srcOffset = srcStack[depth];
                return true;
            }
            --depth;
            continue;
        }
        if (remainingDims == 1) {
            if (SolveOneDimForDst(depth, targetStack[depth], srcStack[depth], order, srcOffset)) {
                return true;
            }
            --depth;
            continue;
        }
        if (remainingDims == 2) {
            if (SolveTwoDimsForDst(depth, targetStack[depth], srcStack[depth], order, srcOffset)) {
                return true;
            }
            --depth;
            continue;
        }

        if (!initialized[depth]) {
            initialized[depth] = true;
            const int64_t target = targetStack[depth];
            if (target < 0 || target > orderSuffixDstMax[depth]) {
                initialized[depth] = false;
                --depth;
                continue;
            }
            const int64_t suffixGcd = orderSuffixDstGcd[depth];
            if (suffixGcd > 1 && target % suffixGcd != 0) {
                initialized[depth] = false;
                --depth;
                continue;
            }

            const int64_t dim = order[depth];
            const int64_t stride = dstStrides_[dim];
            const int64_t size = sizes_[dim];
            if (stride <= 0 || size <= 0) {
                return false;
            }

            int64_t low = 0;
            const int64_t need = target - orderSuffixDstMax[depth + 1];
            if (need > 0) {
                low = CeilDivPositive(need, stride);
            }
            int64_t high = target / stride;
            if (high > size - 1) {
                high = size - 1;
            }
            if (low < 0) {
                low = 0;
            }
            if (low > high) {
                initialized[depth] = false;
                --depth;
                continue;
            }

            const int64_t nextGcd = orderSuffixDstGcd[depth + 1];
            if (nextGcd > 1) {
                int64_t found = high + 1;
                for (int64_t coord = high; coord >= low; --coord) {
                    if ((target - coord * stride) % nextGcd == 0) {
                        found = coord;
                        break;
                    }
                }
                if (found > high) {
                    initialized[depth] = false;
                    --depth;
                    continue;
                }
                high = found;
            }
            lowCoord[depth] = low;
            nextCoord[depth] = high;
        }

        if (nextCoord[depth] < lowCoord[depth]) {
            initialized[depth] = false;
            --depth;
            continue;
        }

        const int64_t dim = order[depth];
        const int64_t stride = dstStrides_[dim];
        const int64_t nextGcd = orderSuffixDstGcd[depth + 1];
        int64_t coord = nextCoord[depth];
        if (nextGcd > 1) {
            while (coord >= lowCoord[depth] && (targetStack[depth] - coord * stride) % nextGcd != 0) {
                --coord;
            }
            if (coord < lowCoord[depth]) {
                initialized[depth] = false;
                --depth;
                continue;
            }
        }
        nextCoord[depth] = coord - 1;
        targetStack[depth + 1] = targetStack[depth] - coord * stride;
        srcStack[depth + 1] = srcStack[depth] + coord * srcStrides_[dim];
        ++depth;
        if (depth < ndim_) {
            initialized[depth] = false;
        }
    }
    return false;
}

template <typename T>
__aicore__ inline int64_t ViewCopy<T>::FloorDivPositive(int64_t value, int64_t divisor) const
{
    return value / divisor;
}

template <typename T>
__aicore__ inline int64_t ViewCopy<T>::CalcContiguousTailElems() const
{
    if (ndim_ <= 0 || srcStrides_[ndim_ - 1] != 1 || dstStrides_[ndim_ - 1] != 1) {
        return 1;
    }

    int64_t tailElems = sizes_[ndim_ - 1];
    for (int64_t dim = ndim_ - 2; dim >= 0; --dim) {
        const int64_t expectedSrcStride = sizes_[dim + 1] * srcStrides_[dim + 1];
        const int64_t expectedDstStride = sizes_[dim + 1] * dstStrides_[dim + 1];
        if (srcStrides_[dim] != expectedSrcStride || dstStrides_[dim] != expectedDstStride) {
            break;
        }
        tailElems *= sizes_[dim];
    }
    return tailElems > 0 ? tailElems : 1;
}

template <typename T>
__aicore__ inline int64_t ViewCopy<T>::CalcThreeDimChunkElems(int64_t blockNum) const
{
    (void)blockNum;
    constexpr int64_t BASE_BLOCK_ELEMS = VIEWCOPY_DATA_BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    int64_t chunkElems = BASE_BLOCK_ELEMS;
    if constexpr (sizeof(T) == 4) {
        chunkElems = BASE_BLOCK_ELEMS * 2;
        if (sizes_[1] >= 128) {
            chunkElems = BASE_BLOCK_ELEMS * 4;
        }
    }
    return chunkElems;
}

template <typename T>
__aicore__ inline bool ViewCopy<T>::IsThreeDimTranspose2ContiguousSrc() const
{
    if (ndim_ != 3 || sizes_[2] != 2 || srcStrides_[1] != 2 || srcStrides_[2] != 1 || dstStrides_[1] != 1 ||
        dstStrides_[2] != sizes_[1]) {
        return false;
    }
    const int64_t rowElems = sizes_[1] * sizes_[2];
    return rowElems > 0 && rowElems * static_cast<int64_t>(sizeof(T)) * 2 <= VIEWCOPY_COPY_BUFFER_BYTES;
}

template <typename T>
__aicore__ inline bool ViewCopy<T>::GetContiguousSliceInfo(int64_t& sliceDim, int64_t& sliceNum,
                                                           int64_t& sliceElems) const
{
    if (ndim_ < 4 || sizes_[0] <= 0 || srcStrides_[ndim_ - 1] != 1 || dstStrides_[ndim_ - 1] != 1) {
        return false;
    }

    sliceElems = sizes_[ndim_ - 1];
    int64_t dim = ndim_ - 2;
    for (; dim >= 0; --dim) {
        const int64_t expectedSrcStride = sizes_[dim + 1] * srcStrides_[dim + 1];
        const int64_t expectedDstStride = sizes_[dim + 1] * dstStrides_[dim + 1];
        if (srcStrides_[dim] != expectedSrcStride || dstStrides_[dim] != expectedDstStride) {
            break;
        }
        sliceElems *= sizes_[dim];
    }
    sliceDim = dim + 1;
    if (sliceDim <= 0 || sliceDim >= ndim_ || sliceElems <= 0) {
        return false;
    }

    sliceNum = 1;
    for (int64_t outerDim = 0; outerDim < sliceDim; ++outerDim) {
        sliceNum *= sizes_[outerDim];
    }
    return sliceNum > 0 && sliceNum * sliceElems == viewNum_;
}

template <typename T>
__aicore__ inline int64_t ViewCopy<T>::CalcContiguousSliceChunkElems(int64_t sliceNum, int64_t sliceElems,
                                                                     int64_t blockNum) const
{
    constexpr int64_t BLOCK_ELEMS = VIEWCOPY_DATA_BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    if (sliceNum >= blockNum || sliceElems <= BLOCK_ELEMS) {
        return sliceElems;
    }

    if constexpr (sizeof(T) != 4) {
        return sliceElems;
    }

    int64_t chunksPerSlice = CeilDivPositive(blockNum, sliceNum);
    if (chunksPerSlice <= 1) {
        return sliceElems;
    }

    int64_t chunkElems = CeilDivPositive(sliceElems, chunksPerSlice);
    chunkElems = CeilDivPositive(chunkElems, BLOCK_ELEMS) * BLOCK_ELEMS;
    if (chunkElems >= sliceElems) {
        return sliceElems;
    }
    return chunkElems;
}

template <typename T>
__aicore__ inline void ViewCopy<T>::LinearToOffset(int64_t linear, int64_t& srcOffset, int64_t& dstOffset) const
{
    srcOffset = srcOffset_;
    dstOffset = dstOffset_;
    for (int64_t dim = ndim_ - 1; dim >= 0; --dim) {
        const int64_t coord = linear % sizes_[dim];
        linear /= sizes_[dim];
        srcOffset += coord * srcStrides_[dim];
        dstOffset += coord * dstStrides_[dim];
    }
}

template <typename T>
__aicore__ inline int64_t ViewCopy<T>::CeilDivPositive(int64_t value, int64_t divisor) const
{
    return (value + divisor - 1) / divisor;
}

template <typename T>
__aicore__ inline bool ViewCopy<T>::PrepareDimRange(int64_t dim, int64_t srcBase, int64_t dstBase, int64_t& begin,
                                                    int64_t& end) const
{
    if (dim < 0 || dim >= ndim_ || srcStorageNum_ <= 0 || storageNum_ <= 0 || copyStart_ >= copyEnd_) {
        return false;
    }

    const int64_t size = sizes_[dim];
    const int64_t srcStride = srcStrides_[dim];
    const int64_t dstStride = dstStrides_[dim];
    if (size <= 0 || srcStride <= 0 || dstStride <= 0) {
        return false;
    }

    begin = 0;
    end = size;

    const int64_t srcLowerNeed = -srcBase;
    if (srcLowerNeed > 0) {
        const int64_t srcBegin = CeilDivPositive(srcLowerNeed, srcStride);
        begin = begin > srcBegin ? begin : srcBegin;
    }

    const int64_t srcUpperRemain = srcStorageNum_ - 1 - srcBase;
    if (srcUpperRemain < 0) {
        return false;
    }
    const int64_t srcEnd = srcUpperRemain / srcStride + 1;
    end = end < srcEnd ? end : srcEnd;

    const int64_t dstUpperRemain = copyEnd_ - 1 - dstBase;
    if (dstUpperRemain < 0) {
        return false;
    }
    const int64_t dstEnd = dstUpperRemain / dstStride + 1;
    end = end < dstEnd ? end : dstEnd;

    const int64_t suffixMax = suffixDstMax_[dim + 1];
    const int64_t dstLowerNeed = copyStart_ - suffixMax - dstBase;
    if (dstLowerNeed > 0) {
        const int64_t dstBegin = CeilDivPositive(dstLowerNeed, dstStride);
        begin = begin > dstBegin ? begin : dstBegin;
    }

    if (begin < 0) {
        begin = 0;
    }
    if (end > size) {
        end = size;
    }
    return begin < end;
}

template <typename T>
__aicore__ inline void ViewCopy<T>::CopyElement(GlobalTensor<T>& src, GlobalTensor<T>& dst, int64_t srcOffset,
                                                int64_t dstOffset)
{
    LocalTensor<T> local = dataBuf_.Get<T>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    DataCopyPad(local, src[srcOffset], copyParams, padParams);
    pipe_barrier(PIPE_ALL);
    DataCopyPad(dst[dstOffset], local, copyParams);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
__aicore__ inline void ViewCopy<T>::CopySpan(GlobalTensor<T>& src, GlobalTensor<T>& dst, int64_t srcOffset,
                                             int64_t dstOffset, int64_t elemCount)
{
    if (elemCount <= 0) {
        return;
    }

    constexpr int64_t MAX_COPY_ELEMS = VIEWCOPY_COPY_BUFFER_BYTES / static_cast<int64_t>(sizeof(T));
    constexpr int64_t BLOCK_ELEMS = VIEWCOPY_DATA_BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    LocalTensor<T> local = dataBuf_.Get<T>();
    int64_t copied = 0;

    while (copied < elemCount) {
        int64_t remaining = elemCount - copied;
        int64_t copyElems = remaining;
        if (copyElems > MAX_COPY_ELEMS) {
            copyElems = MAX_COPY_ELEMS;
        }
        copyElems = (copyElems / BLOCK_ELEMS) * BLOCK_ELEMS;

        if (copyElems <= 0) {
            CopyElement(src, dst, srcOffset + copied, dstOffset + copied);
            ++copied;
            continue;
        }

        const int64_t srcCurrent = srcOffset + copied;
        const int64_t dstCurrent = dstOffset + copied;
        if (srcCurrent % BLOCK_ELEMS == 0 && dstCurrent % BLOCK_ELEMS == 0) {
            DataCopy(local, src[srcCurrent], static_cast<uint32_t>(copyElems));
            pipe_barrier(PIPE_ALL);
            DataCopy(dst[dstCurrent], local, static_cast<uint32_t>(copyElems));
        } else {
            DataCopyExtParams copyParams = {
                static_cast<uint16_t>(1), static_cast<uint32_t>(copyElems * static_cast<int64_t>(sizeof(T))),
                static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
            DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                 static_cast<T>(0)};
            DataCopyPad(local, src[srcCurrent], copyParams, padParams);
            pipe_barrier(PIPE_ALL);
            DataCopyPad(dst[dstCurrent], local, copyParams);
        }
        pipe_barrier(PIPE_ALL);
        copied += copyElems;
    }
}

template <typename T>
__aicore__ inline void ViewCopy<T>::CopyStridedRow(GlobalTensor<T>& src, GlobalTensor<T>& dst, int64_t srcOffset,
                                                   int64_t dstOffset, int64_t elemCount)
{
    if (elemCount <= 0) {
        return;
    }

    const int64_t lastDim = ndim_ - 1;
    int64_t copied = 0;
    while (copied < elemCount) {
        CopyElement(src, dst, srcOffset + copied * srcStrides_[lastDim], dstOffset + copied * dstStrides_[lastDim]);
        ++copied;
    }
}

template <typename T>
__aicore__ inline void ViewCopy<T>::CopyThreeDimTranspose2Chunk(int64_t row, int64_t col, int64_t elemCount)
{
    if (elemCount <= 0) {
        return;
    }

    const int64_t inputElems = elemCount * sizes_[2];
    const int64_t srcBase = srcOffset_ + row * srcStrides_[0] + col * srcStrides_[1];
    const int64_t dstBase0 = dstOffset_ + row * dstStrides_[0] + col * dstStrides_[1];
    const int64_t dstBase1 = dstBase0 + dstStrides_[2];
    LocalTensor<T> local = dataBuf_.Get<T>();

    DataCopyExtParams inputParams = {static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(inputElems * static_cast<int64_t>(sizeof(T))),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    DataCopyPad(local, srcGm_[srcBase], inputParams, padParams);
    pipe_barrier(PIPE_ALL);

    const int64_t outBase = inputElems;
    for (int64_t idx = 0; idx < elemCount; ++idx) {
        local.SetValue(static_cast<uint64_t>(outBase + idx), local.GetValue(static_cast<uint64_t>(idx * sizes_[2])));
        local.SetValue(static_cast<uint64_t>(outBase + elemCount + idx),
                       local.GetValue(static_cast<uint64_t>(idx * sizes_[2] + 1)));
    }
    pipe_barrier(PIPE_ALL);

    DataCopyExtParams outputParams = {static_cast<uint16_t>(1),
                                      static_cast<uint32_t>(elemCount * static_cast<int64_t>(sizeof(T))),
                                      static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(yGm_[dstBase0], local[outBase], outputParams);
    pipe_barrier(PIPE_ALL);
    DataCopyPad(yGm_[dstBase1], local[outBase + elemCount], outputParams);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
__aicore__ inline void ViewCopy<T>::CopyThreeDimTranspose2Row(int64_t row)
{
    const int64_t midSize = sizes_[1];
    const int64_t rowElems = midSize * sizes_[2];
    const int64_t srcBase = srcOffset_ + row * srcStrides_[0];
    const int64_t dstBase = dstOffset_ + row * dstStrides_[0];
    LocalTensor<T> local = dataBuf_.Get<T>();

    DataCopyExtParams copyParams = {static_cast<uint16_t>(1),
                                    static_cast<uint32_t>(rowElems * static_cast<int64_t>(sizeof(T))),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    DataCopyPad(local, srcGm_[srcBase], copyParams, padParams);
    pipe_barrier(PIPE_ALL);

    const int64_t outBase = rowElems;
    for (int64_t col = 0; col < midSize; ++col) {
        local.SetValue(static_cast<uint64_t>(outBase + col), local.GetValue(static_cast<uint64_t>(col * 2)));
        local.SetValue(static_cast<uint64_t>(outBase + midSize + col),
                       local.GetValue(static_cast<uint64_t>(col * 2 + 1)));
    }
    pipe_barrier(PIPE_ALL);
    DataCopyPad(yGm_[dstBase], local[outBase], copyParams);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
__aicore__ inline void ViewCopy<T>::ProcessThreeDimTranspose2Rows(int64_t blockIdx, int64_t blockNum)
{
    const int64_t rowNum = sizes_[0];
    const int64_t blockBase = rowNum / blockNum;
    const int64_t blockTail = rowNum % blockNum;
    const int64_t rowStart = blockIdx * blockBase + (blockIdx < blockTail ? blockIdx : blockTail);
    const int64_t rowCount = blockBase + (blockIdx < blockTail ? 1 : 0);
    for (int64_t row = rowStart; row < rowStart + rowCount; ++row) {
        CopyThreeDimTranspose2Row(row);
    }
}

template <typename T>
__aicore__ inline void ViewCopy<T>::ProcessThreeDimTranspose2Linear(int64_t blockIdx, int64_t blockNum)
{
    const int64_t midSize = sizes_[1];
    const int64_t chunkElems = CalcThreeDimChunkElems(blockNum);
    const int64_t chunksPerPlane = CeilDivPositive(midSize, chunkElems);
    const int64_t chunksPerRow = chunksPerPlane;
    const int64_t totalChunks = sizes_[0] * chunksPerRow;
    const int64_t blockBase = totalChunks / blockNum;
    const int64_t blockTail = totalChunks % blockNum;
    int64_t chunk = blockIdx * blockBase + (blockIdx < blockTail ? blockIdx : blockTail);
    const int64_t chunkEnd = chunk + blockBase + (blockIdx < blockTail ? 1 : 0);

    while (chunk < chunkEnd) {
        const int64_t row = chunk / chunksPerRow;
        const int64_t inRowChunk = chunk % chunksPerRow;
        const int64_t col = inRowChunk * chunkElems;
        int64_t count = midSize - col;
        if (count > chunkElems) {
            count = chunkElems;
        }

        if (count == chunkElems) {
            CopyThreeDimTranspose2Chunk(row, col, count);
        } else {
            for (int64_t idx = 0; idx < count; ++idx) {
                for (int64_t plane = 0; plane < sizes_[2]; ++plane) {
                    const int64_t srcOffset = srcOffset_ + row * srcStrides_[0] + (col + idx) * srcStrides_[1] +
                                              plane * srcStrides_[2];
                    const int64_t dstOffset = dstOffset_ + row * dstStrides_[0] + (col + idx) * dstStrides_[1] +
                                              plane * dstStrides_[2];
                    CopyElement(srcGm_, yGm_, srcOffset, dstOffset);
                }
            }
        }
        ++chunk;
    }
}

template <typename T>
__aicore__ inline void ViewCopy<T>::ProcessContiguousSlices(int64_t blockIdx, int64_t blockNum)
{
    int64_t sliceDim = 0;
    int64_t sliceNum = 0;
    int64_t sliceElems = 0;
    if (!GetContiguousSliceInfo(sliceDim, sliceNum, sliceElems)) {
        return;
    }

    const int64_t chunkElems = CalcContiguousSliceChunkElems(sliceNum, sliceElems, blockNum);
    const int64_t chunksPerSlice = CeilDivPositive(sliceElems, chunkElems);
    const int64_t totalChunks = sliceNum * chunksPerSlice;
    const int64_t blockBase = totalChunks / blockNum;
    const int64_t blockTail = totalChunks % blockNum;
    int64_t chunk = blockIdx * blockBase + (blockIdx < blockTail ? blockIdx : blockTail);
    const int64_t chunkEnd = chunk + blockBase + (blockIdx < blockTail ? 1 : 0);

    while (chunk < chunkEnd) {
        const int64_t slice = chunk / chunksPerSlice;
        const int64_t chunkInSlice = chunk % chunksPerSlice;
        const int64_t sliceOffset = chunkInSlice * chunkElems;
        int64_t copyElems = sliceElems - sliceOffset;
        if (copyElems > chunkElems) {
            copyElems = chunkElems;
        }

        int64_t linear = slice;
        int64_t srcOffset = srcOffset_;
        int64_t dstOffset = dstOffset_;
        for (int64_t dim = sliceDim - 1; dim >= 0; --dim) {
            const int64_t coord = linear % sizes_[dim];
            linear /= sizes_[dim];
            srcOffset += coord * srcStrides_[dim];
            dstOffset += coord * dstStrides_[dim];
        }
        CopySpan(srcGm_, yGm_, srcOffset + sliceOffset, dstOffset + sliceOffset, copyElems);
        ++chunk;
    }
}

template <typename T>
__aicore__ inline void ViewCopy<T>::ProcessLinearViewWrites(int64_t linearStart, int64_t linearEnd)
{
    if (linearStart >= linearEnd || ndim_ <= 0) {
        return;
    }

    const int64_t contiguousTailElems = CalcContiguousTailElems();
    int64_t linear = linearStart;
    while (linear < linearEnd) {
        int64_t srcOffset = 0;
        int64_t dstOffset = 0;
        LinearToOffset(linear, srcOffset, dstOffset);

        int64_t chunkElems = 1;
        if (contiguousTailElems > 1) {
            chunkElems = contiguousTailElems - (linear % contiguousTailElems);
            if (chunkElems > linearEnd - linear) {
                chunkElems = linearEnd - linear;
            }
            CopySpan(srcGm_, yGm_, srcOffset, dstOffset, chunkElems);
        } else {
            CopyElement(srcGm_, yGm_, srcOffset, dstOffset);
        }
        linear += chunkElems;
    }
}

template <typename T>
__aicore__ inline void ViewCopy<T>::ProcessOverlapViewWrites()
{
    if (viewNum_ <= 0 || ndim_ <= 0) {
        return;
    }

    int64_t order[VIEWCOPY_MAX_DIMS] = {0};
    BuildTensorIteratorOrder(order);
    for (int64_t linear = 0; linear < viewNum_; ++linear) {
        int64_t remaining = linear;
        int64_t srcOffset = srcOffset_;
        int64_t dstOffset = dstOffset_;
        for (int64_t idx = ndim_ - 1; idx >= 0; --idx) {
            const int64_t dim = order[idx];
            const int64_t coord = remaining % sizes_[dim];
            remaining /= sizes_[dim];
            srcOffset += coord * srcStrides_[dim];
            dstOffset += coord * dstStrides_[dim];
        }
        yGm_.SetValue(static_cast<uint64_t>(dstOffset), srcGm_.GetValue(static_cast<uint64_t>(srcOffset)));
        scalarWritten_ = true;
    }
}

template <typename T>
__aicore__ inline void ViewCopy<T>::ProcessOverlapDstWrites(int64_t blockIdx, int64_t blockNum)
{
    if (viewNum_ <= 0 || ndim_ <= 0) {
        return;
    }
    if (dstSpan_ <= 0) {
        dstSpan_ = CalcDstSpan();
    }
    if (dstSpan_ <= 0) {
        return;
    }

    constexpr int64_t BLOCK_ELEMS = VIEWCOPY_DATA_BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    const int64_t firstAbsBlock = FloorDivPositive(dstOffset_, BLOCK_ELEMS);
    const int64_t lastAbsBlock = FloorDivPositive(dstOffset_ + dstSpan_ - 1, BLOCK_ELEMS);
    const int64_t blockWork = lastAbsBlock - firstAbsBlock + 1;
    if (blockWork <= 0) {
        return;
    }

    const int64_t blockBase = blockWork / blockNum;
    const int64_t blockTail = blockWork % blockNum;
    int64_t block = blockIdx * blockBase + (blockIdx < blockTail ? blockIdx : blockTail);
    const int64_t blockEnd = block + blockBase + (blockIdx < blockTail ? 1 : 0);

    int64_t order[VIEWCOPY_MAX_DIMS] = {0};
    int64_t orderSuffixDstMax[VIEWCOPY_MAX_DIMS + 1] = {0};
    int64_t orderSuffixDstGcd[VIEWCOPY_MAX_DIMS + 1] = {0};
    BuildTensorIteratorOrder(order);
    BuildOrderSuffixDstInfo(order, orderSuffixDstMax, orderSuffixDstGcd);
    while (block < blockEnd) {
        const int64_t absBlock = firstAbsBlock + block;
        int64_t dstRel = absBlock * BLOCK_ELEMS - dstOffset_;
        int64_t dstEnd = dstRel + BLOCK_ELEMS;
        if (dstRel < 0) {
            dstRel = 0;
        }
        if (dstEnd > dstSpan_) {
            dstEnd = dstSpan_;
        }
        while (dstRel < dstEnd) {
            int64_t srcOffset = 0;
            if (FindLastSrcOffsetForDst(dstRel, order, orderSuffixDstMax, orderSuffixDstGcd, srcOffset)) {
                yGm_.SetValue(static_cast<uint64_t>(dstOffset_ + dstRel),
                              srcGm_.GetValue(static_cast<uint64_t>(srcOffset)));
                scalarWritten_ = true;
            }
            ++dstRel;
        }
        ++block;
    }
}

template <typename T>
__aicore__ inline void ViewCopy<T>::ProcessLastDim(int64_t srcBase, int64_t dstBase)
{
    const int64_t lastDim = ndim_ - 1;
    int64_t begin = 0;
    int64_t end = 0;
    if (!PrepareDimRange(lastDim, srcBase, dstBase, begin, end)) {
        return;
    }

    const int64_t count = end - begin;
    const int64_t srcStart = srcBase + begin * srcStrides_[lastDim];
    const int64_t dstStart = dstBase + begin * dstStrides_[lastDim];
    if (srcStrides_[lastDim] == 1 && dstStrides_[lastDim] == 1) {
        CopySpan(srcGm_, yGm_, srcStart, dstStart, count);
    } else {
        CopyStridedRow(srcGm_, yGm_, srcStart, dstStart, count);
    }
}

template <typename T>
__aicore__ inline void ViewCopy<T>::ProcessViewWrites()
{
    if (ndim_ == 1) {
        ProcessLastDim(srcOffset_, dstOffset_);
        return;
    }

    const int64_t lastParentDim = ndim_ - 2;
    int64_t depth = 0;
    srcStack_[0] = srcOffset_;
    dstStack_[0] = dstOffset_;
    if (!PrepareDimRange(0, srcStack_[0], dstStack_[0], rangeBegin_[0], rangeEnd_[0])) {
        return;
    }
    coord_[0] = rangeBegin_[0];

    while (depth >= 0) {
        if (coord_[depth] >= rangeEnd_[depth]) {
            --depth;
            if (depth >= 0) {
                ++coord_[depth];
            }
            continue;
        }

        const int64_t childSrc = srcStack_[depth] + coord_[depth] * srcStrides_[depth];
        const int64_t childDst = dstStack_[depth] + coord_[depth] * dstStrides_[depth];
        if (depth == lastParentDim) {
            ProcessLastDim(childSrc, childDst);
            ++coord_[depth];
            continue;
        }

        const int64_t nextDepth = depth + 1;
        if (PrepareDimRange(nextDepth, childSrc, childDst, rangeBegin_[nextDepth], rangeEnd_[nextDepth])) {
            srcStack_[nextDepth] = childSrc;
            dstStack_[nextDepth] = childDst;
            coord_[nextDepth] = rangeBegin_[nextDepth];
            depth = nextDepth;
        } else {
            ++coord_[depth];
        }
    }
}

template <typename T>
__aicore__ inline void ViewCopy<T>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    if (blockIdx >= blockNum) {
        return;
    }

    if (metadataReady_ == 0 && !LoadRuntimeMetadata()) {
        return;
    }
    if (viewNum_ <= 0) {
        viewNum_ = CalcViewNum();
    }
    if (viewNum_ <= 0) {
        return;
    }

    if (dstOverlap_ < 0) {
        dstOverlap_ = IsDstMaybeOverlapped() ? 1 : 0;
    }
    if (dstOverlap_ != 0) {
        ProcessOverlapDstWrites(blockIdx, blockNum);
        if (scalarWritten_) {
            DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(yGm_);
        }
        return;
    }

    if (IsThreeDimTranspose2ContiguousSrc()) {
        if (sizes_[0] * 4 < blockNum * 3) {
            ProcessThreeDimTranspose2Linear(blockIdx, blockNum);
        } else {
            ProcessThreeDimTranspose2Rows(blockIdx, blockNum);
        }
        if (scalarWritten_) {
            DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(yGm_);
        }
        return;
    }

    int64_t sliceDim = 0;
    int64_t sliceNum = 0;
    int64_t sliceElems = 0;
    if (GetContiguousSliceInfo(sliceDim, sliceNum, sliceElems)) {
        ProcessContiguousSlices(blockIdx, blockNum);
        if (scalarWritten_) {
            DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(yGm_);
        }
        return;
    }

    const int64_t blockBase = viewNum_ / blockNum;
    const int64_t blockTail = viewNum_ % blockNum;
    const int64_t linearStart = blockIdx * blockBase + (blockIdx < blockTail ? blockIdx : blockTail);
    const int64_t linearCount = blockBase + (blockIdx < blockTail ? 1 : 0);
    if (linearCount <= 0) {
        return;
    }
    ProcessLinearViewWrites(linearStart, linearStart + linearCount);
    if (scalarWritten_) {
        DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(yGm_);
    }
}

} // namespace NsViewCopy

#endif // VIEWCOPY_H_
