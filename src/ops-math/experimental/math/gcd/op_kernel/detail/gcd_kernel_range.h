/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DETAIL_GCD_KERNEL_RANGE_H_
#define DETAIL_GCD_KERNEL_RANGE_H_

struct LinearRange {
    int64_t begin;
    int64_t end;
};

struct PackedRange {
    int64_t wordBegin;
    int64_t wordEnd;
    int64_t elementBegin;
    int64_t elementEnd;
};

struct OffsetCursor {
    int64_t coord[GCD_MAX_DIMS];
    int64_t x1Offset;
    int64_t x2Offset;
};

__aicore__ inline void CalcBlockRange(int64_t totalWords, int64_t& begin, int64_t& end)
{
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockNum <= 1) {
        begin = 0;
        end = totalWords;
        return;
    }

    // Keep cross-core boundaries on aligned 32-bit storage-word chunks.
    int64_t wordsPerBlock = AlignUpInt64(CeilDivInt64(totalWords, blockNum), GCD_OUTPUT_WORD_ALIGNMENT);
    begin = blockIdx * wordsPerBlock;
    if (begin > totalWords) {
        begin = totalWords;
    }
    end = begin + wordsPerBlock;
    if (end > totalWords) {
        end = totalWords;
    }
}

__aicore__ inline bool InitLinearRange(int64_t total, LinearRange& range)
{
    if (total <= 0) {
        return false;
    }
    CalcBlockRange(total, range.begin, range.end);
    return range.begin < range.end;
}

__aicore__ inline bool InitPackedRange(int64_t total, int64_t lanesPerWord, PackedRange& range)
{
    if (total <= 0) {
        return false;
    }
    int64_t wordTotal = CeilDivInt64(total, lanesPerWord);
    CalcBlockRange(wordTotal, range.wordBegin, range.wordEnd);
    range.elementBegin = range.wordBegin * lanesPerWord;
    range.elementEnd = range.wordEnd * lanesPerWord;
    if (range.elementEnd > total) {
        range.elementEnd = total;
    }
    return range.elementBegin < range.elementEnd;
}

__aicore__ inline void InitOffsetCursor(const GcdTilingData& tiling, int64_t linear, OffsetCursor& cursor)
{
    cursor.x1Offset = 0;
    cursor.x2Offset = 0;
    for (int64_t dim = 0; dim < GCD_MAX_DIMS; ++dim) {
        cursor.coord[dim] = 0;
    }

    int64_t remaining = linear;
    for (int64_t dim = tiling.rank - 1; dim >= 0; --dim) {
        int64_t extent = tiling.outputDims[dim];
        int64_t coord = remaining % extent;
        remaining = remaining / extent;
        cursor.coord[dim] = coord;
        cursor.x1Offset += coord * tiling.x1Strides[dim];
        cursor.x2Offset += coord * tiling.x2Strides[dim];
    }
}

__aicore__ inline void AdvanceOffsetCursor(const GcdTilingData& tiling, OffsetCursor& cursor)
{
    for (int64_t dim = tiling.rank - 1; dim >= 0; --dim) {
        int64_t nextCoord = cursor.coord[dim] + 1;
        if (nextCoord < tiling.outputDims[dim]) {
            cursor.coord[dim] = nextCoord;
            cursor.x1Offset += tiling.x1Strides[dim];
            cursor.x2Offset += tiling.x2Strides[dim];
            return;
        }

        cursor.x1Offset -= cursor.coord[dim] * tiling.x1Strides[dim];
        cursor.x2Offset -= cursor.coord[dim] * tiling.x2Strides[dim];
        cursor.coord[dim] = 0;
    }
}

__aicore__ inline bool CursorCanSpanInnerDim(const GcdTilingData& tiling, const OffsetCursor& cursor, int64_t span)
{
    int64_t innerDim = tiling.rank - 1;
    return span > 0 && cursor.coord[innerDim] + span <= tiling.outputDims[innerDim];
}

__aicore__ inline bool IsContiguousElementwise(const GcdTilingData& tiling)
{
    int64_t expectedStride = 1;
    for (int64_t dim = tiling.rank - 1; dim >= 0; --dim) {
        if (tiling.x1Strides[dim] != expectedStride || tiling.x2Strides[dim] != expectedStride) {
            return false;
        }
        expectedStride *= tiling.outputDims[dim];
    }
    return true;
}

__aicore__ inline void AdvanceOffsetCursorInnerSpan(const GcdTilingData& tiling, OffsetCursor& cursor, int64_t span)
{
    int64_t innerDim = tiling.rank - 1;
    int64_t nextCoord = cursor.coord[innerDim] + span;
    if (nextCoord < tiling.outputDims[innerDim]) {
        cursor.coord[innerDim] = nextCoord;
        cursor.x1Offset += span * tiling.x1Strides[innerDim];
        cursor.x2Offset += span * tiling.x2Strides[innerDim];
        return;
    }

    cursor.x1Offset -= cursor.coord[innerDim] * tiling.x1Strides[innerDim];
    cursor.x2Offset -= cursor.coord[innerDim] * tiling.x2Strides[innerDim];
    cursor.coord[innerDim] = 0;
    for (int64_t dim = innerDim - 1; dim >= 0; --dim) {
        int64_t nextOuterCoord = cursor.coord[dim] + 1;
        if (nextOuterCoord < tiling.outputDims[dim]) {
            cursor.coord[dim] = nextOuterCoord;
            cursor.x1Offset += tiling.x1Strides[dim];
            cursor.x2Offset += tiling.x2Strides[dim];
            return;
        }

        cursor.x1Offset -= cursor.coord[dim] * tiling.x1Strides[dim];
        cursor.x2Offset -= cursor.coord[dim] * tiling.x2Strides[dim];
        cursor.coord[dim] = 0;
    }
}

template <typename T>
__aicore__ inline void RunLinearScalarContiguous(int64_t begin, int64_t end, GlobalTensor<T>& x1Gm,
                                                 GlobalTensor<T>& x2Gm, GlobalTensor<T>& yGm)
{
    for (int64_t linear = begin; linear < end; ++linear) {
        yGm.SetValue(static_cast<uint64_t>(linear), GcdOp::GcdScalar<T>(x1Gm.GetValue(static_cast<uint64_t>(linear)),
                                                                        x2Gm.GetValue(static_cast<uint64_t>(linear))));
    }
}

template <typename T>
__aicore__ inline void RunLinearScalarStrided(const GcdTilingData& tiling, const LinearRange& range,
                                              GlobalTensor<T>& x1Gm, GlobalTensor<T>& x2Gm, GlobalTensor<T>& yGm)
{
    OffsetCursor cursor;
    InitOffsetCursor(tiling, range.begin, cursor);
    for (int64_t linear = range.begin; linear < range.end; ++linear) {
        yGm.SetValue(static_cast<uint64_t>(linear),
                     GcdOp::GcdScalar<T>(x1Gm.GetValue(static_cast<uint64_t>(cursor.x1Offset)),
                                         x2Gm.GetValue(static_cast<uint64_t>(cursor.x2Offset))));
        if (linear + 1 < range.end) {
            AdvanceOffsetCursor(tiling, cursor);
        }
    }
}

template <typename Kernel, typename T>
__aicore__ inline int64_t RunLinearVectorContiguous(Kernel& kernel, const LinearRange& range, bool vectorEnabled,
                                                    int64_t alignElements, int64_t tileElements, GlobalTensor<T>& x1Gm,
                                                    GlobalTensor<T>& x2Gm, GlobalTensor<T>& yGm)
{
    if (!vectorEnabled) {
        return range.begin;
    }
    int64_t vectorCount = AlignDownInt64(range.end - range.begin, alignElements);
    int64_t vectorEnd = range.begin + vectorCount;
    for (int64_t linear = range.begin; linear < vectorEnd;) {
        int64_t count = MinInt64(vectorEnd - linear, tileElements);
        bool syncTail = NeedScalarTailSync(linear, count, vectorEnd, range.end);
        if (!kernel.ComputeVectorTile(linear, linear, linear, static_cast<int32_t>(count), syncTail)) {
            RunLinearScalarContiguous(linear, linear + count, x1Gm, x2Gm, yGm);
        }
        linear += count;
    }
    return vectorEnd;
}

template <typename Kernel, typename T>
__aicore__ inline void ProcessLinearKernel(Kernel& kernel, const GcdTilingData& tiling, bool contiguousElementwise,
                                           bool vectorEnabled, int64_t alignElements, int64_t tileElements,
                                           GlobalTensor<T>& x1Gm, GlobalTensor<T>& x2Gm, GlobalTensor<T>& yGm)
{
    LinearRange range{0, 0};
    if (!InitLinearRange(tiling.totalNum, range)) {
        return;
    }
    if (!contiguousElementwise) {
        RunLinearScalarStrided(tiling, range, x1Gm, x2Gm, yGm);
        return;
    }
    int64_t vectorEnd = RunLinearVectorContiguous(kernel, range, vectorEnabled, alignElements, tileElements, x1Gm, x2Gm,
                                                  yGm);
    RunLinearScalarContiguous(vectorEnd, range.end, x1Gm, x2Gm, yGm);
}

template <typename Kernel, int64_t LANES_PER_WORD, int64_t BITS_PER_LANE>
__aicore__ inline uint32_t BuildContiguousPackedWord(Kernel& kernel, int64_t word, int64_t total)
{
    uint32_t packed = 0;
    for (int64_t lane = 0; lane < LANES_PER_WORD; ++lane) {
        int64_t linear = word * LANES_PER_WORD + lane;
        uint32_t bits = 0;
        if (linear < total) {
            bits = static_cast<uint32_t>(kernel.ComputeContiguousLaneBits(word, lane));
        }
        packed |= bits << (lane * BITS_PER_LANE);
    }
    return packed;
}

template <typename Kernel, int64_t LANES_PER_WORD, int64_t BITS_PER_LANE>
__aicore__ inline void RunPackedScalarContiguous(Kernel& kernel, int64_t beginWord, int64_t endWord, int64_t total,
                                                 GlobalTensor<uint32_t>& yGm)
{
    for (int64_t word = beginWord; word < endWord; ++word) {
        uint32_t packed = BuildContiguousPackedWord<Kernel, LANES_PER_WORD, BITS_PER_LANE>(kernel, word, total);
        yGm.SetValue(static_cast<uint64_t>(word), packed);
    }
}

template <typename Kernel, int64_t LANES_PER_WORD, int64_t BITS_PER_LANE>
__aicore__ inline uint32_t BuildStridedPackedWord(Kernel& kernel, const GcdTilingData& tiling, int64_t word,
                                                  int64_t elementEnd, OffsetCursor& cursor)
{
    int64_t wordElementBegin = word * LANES_PER_WORD;
    int64_t wordElementEnd = MinInt64(wordElementBegin + LANES_PER_WORD, elementEnd);
    int64_t validLanes = wordElementEnd - wordElementBegin;
    uint32_t packed = 0;
    if (CursorCanSpanInnerDim(tiling, cursor, validLanes)) {
        int64_t innerDim = tiling.rank - 1;
        int64_t x1Base = cursor.x1Offset;
        int64_t x2Base = cursor.x2Offset;
        int64_t x1Step = tiling.x1Strides[innerDim];
        int64_t x2Step = tiling.x2Strides[innerDim];
        for (int64_t lane = 0; lane < LANES_PER_WORD; ++lane) {
            uint32_t bits = 0;
            if (lane < validLanes) {
                bits = static_cast<uint32_t>(
                    kernel.ComputeStridedLaneBits(x1Base + lane * x1Step, x2Base + lane * x2Step));
            }
            packed |= bits << (lane * BITS_PER_LANE);
        }
        if (wordElementEnd < elementEnd) {
            AdvanceOffsetCursorInnerSpan(tiling, cursor, validLanes);
        }
        return packed;
    }

    for (int64_t lane = 0; lane < LANES_PER_WORD; ++lane) {
        int64_t linear = word * LANES_PER_WORD + lane;
        uint32_t bits = 0;
        if (linear < elementEnd) {
            bits = static_cast<uint32_t>(kernel.ComputeStridedLaneBits(cursor.x1Offset, cursor.x2Offset));
            if (linear + 1 < elementEnd) {
                AdvanceOffsetCursor(tiling, cursor);
            }
        }
        packed |= bits << (lane * BITS_PER_LANE);
    }
    return packed;
}

template <typename Kernel, int64_t LANES_PER_WORD, int64_t BITS_PER_LANE>
__aicore__ inline void RunPackedScalarStrided(Kernel& kernel, const GcdTilingData& tiling, const PackedRange& range,
                                              GlobalTensor<uint32_t>& yGm)
{
    OffsetCursor cursor;
    InitOffsetCursor(tiling, range.elementBegin, cursor);
    for (int64_t word = range.wordBegin; word < range.wordEnd; ++word) {
        uint32_t packed = BuildStridedPackedWord<Kernel, LANES_PER_WORD, BITS_PER_LANE>(kernel, tiling, word,
                                                                                        range.elementEnd, cursor);
        yGm.SetValue(static_cast<uint64_t>(word), packed);
    }
}

template <typename Kernel, int64_t LANES_PER_WORD, int64_t BITS_PER_LANE>
__aicore__ inline int64_t RunPackedVectorContiguous(Kernel& kernel, const PackedRange& range, bool vectorEnabled,
                                                    int64_t alignElements, int64_t tileElements,
                                                    GlobalTensor<uint32_t>& yGm)
{
    if (!vectorEnabled) {
        return range.elementBegin;
    }
    int64_t vectorCount = AlignDownInt64(range.elementEnd - range.elementBegin, alignElements);
    int64_t vectorEnd = range.elementBegin + vectorCount;
    for (int64_t linear = range.elementBegin; linear < vectorEnd;) {
        int64_t count = MinInt64(vectorEnd - linear, tileElements);
        bool syncTail = NeedScalarTailSync(linear, count, vectorEnd, range.elementEnd);
        if (!kernel.ComputePackedVectorTile(linear, static_cast<int32_t>(count), syncTail)) {
            RunPackedScalarContiguous<Kernel, LANES_PER_WORD, BITS_PER_LANE>(
                kernel, linear / LANES_PER_WORD, (linear + count) / LANES_PER_WORD, range.elementEnd, yGm);
        }
        linear += count;
    }
    return vectorEnd;
}

template <typename Kernel, int64_t LANES_PER_WORD, int64_t BITS_PER_LANE>
__aicore__ inline void ProcessPackedKernel(Kernel& kernel, const GcdTilingData& tiling, bool contiguousElementwise,
                                           bool vectorEnabled, int64_t alignElements, int64_t tileElements,
                                           GlobalTensor<uint32_t>& yGm)
{
    PackedRange range{0, 0, 0, 0};
    if (!InitPackedRange(tiling.totalNum, LANES_PER_WORD, range)) {
        return;
    }
    if (!contiguousElementwise) {
        RunPackedScalarStrided<Kernel, LANES_PER_WORD, BITS_PER_LANE>(kernel, tiling, range, yGm);
        return;
    }
    int64_t vectorEnd = RunPackedVectorContiguous<Kernel, LANES_PER_WORD, BITS_PER_LANE>(
        kernel, range, vectorEnabled, alignElements, tileElements, yGm);
    RunPackedScalarContiguous<Kernel, LANES_PER_WORD, BITS_PER_LANE>(kernel, vectorEnd / LANES_PER_WORD, range.wordEnd,
                                                                     tiling.totalNum, yGm);
}

template <typename Kernel, int64_t LANES_PER_WORD, int64_t BITS_PER_LANE>
__aicore__ inline void ProcessMixedPackedKernel(Kernel& kernel, const GcdTilingData& tiling,
                                                GlobalTensor<uint32_t>& yGm)
{
    PackedRange range{0, 0, 0, 0};
    if (!InitPackedRange(tiling.totalNum, LANES_PER_WORD, range)) {
        return;
    }
    OffsetCursor cursor;
    InitOffsetCursor(tiling, range.elementBegin, cursor);
    for (int64_t word = range.wordBegin; word < range.wordEnd; ++word) {
        uint32_t packed = 0;
        for (int64_t lane = 0; lane < LANES_PER_WORD; ++lane) {
            int64_t linear = word * LANES_PER_WORD + lane;
            uint32_t bits = 0;
            if (linear < range.elementEnd) {
                bits = static_cast<uint32_t>(kernel.ComputeMixedLaneBits(cursor.x1Offset, cursor.x2Offset));
                if (linear + 1 < range.elementEnd) {
                    AdvanceOffsetCursor(tiling, cursor);
                }
            }
            packed |= bits << (lane * BITS_PER_LANE);
        }
        yGm.SetValue(static_cast<uint64_t>(word), packed);
    }
}

#endif // DETAIL_GCD_KERNEL_RANGE_H_
