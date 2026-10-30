/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file as_strided.h
 * \brief
 */
#ifndef AS_STRIDED_H_
#define AS_STRIDED_H_

#include <cstdint>
#include "as_strided_tiling_data.h"
#include "as_strided_tiling_key.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace NsAsStrided {
using namespace AscendC;

template <typename T, typename IndexT>
class AsStridedKernel {
public:
    __aicore__ inline AsStridedKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR size, GM_ADDR stride, GM_ADDR storageOffset, GM_ADDR y,
                                const AsStridedTilingData* td, TPipe* pipe)
    {
        (void)size;
        (void)stride;
        (void)storageOffset;
        pipe_ = pipe;
        tilingKey_ = td->tilingKey;
        totalOutputElements_ = td->totalOutputElements;
        inputElementCount_ = td->inputElementCount;
        perCoreElements_ = td->perCoreElements;
        lastCoreElements_ = td->lastCoreElements;
        ubElements_ = td->ubElements;
        storageOffset_ = td->storageOffset;
        outputDimNum_ = td->outputDimNum;
        lastDimSize_ = td->lastDimSize;
        lastDimStride_ = td->lastDimStride;
        axis0Elements_ = td->axis0Elements;
        usedCoreNum_ = td->usedCoreNum;
        blockElements_ = td->blockElements;
        inputSpanElements_ = td->inputSpanElements;
        suffixStartDim_ = td->suffixStartDim;
        suffixElements_ = td->suffixElements;
        suffixOuterElements_ = td->suffixOuterElements;

        for (int64_t i = 0; i < AS_STRIDED_MAX_DIMS; ++i) {
            outSize_[i] = td->outSize[i];
            outStride_[i] = td->outStride[i];
            outSizeStride_[i] = td->outSizeStride[i];
        }

        int64_t splitElements = (tilingKey_ == AS_STRIDED_TILING_KEY_CONTIGUOUS ||
                                 tilingKey_ == AS_STRIDED_TILING_KEY_CONTIGUOUS_SMALL_ALIGNED ||
                                 tilingKey_ == AS_STRIDED_TILING_KEY_COMPACT_SPAN ||
                                 tilingKey_ == AS_STRIDED_TILING_KEY_RANK1_STRIDE ||
                                 tilingKey_ == AS_STRIDED_TILING_KEY_RANK1_STRIDE_SPAN) ?
                                    totalOutputElements_ :
                                    (tilingKey_ == AS_STRIDED_TILING_KEY_COMPACT_SUFFIX ? suffixOuterElements_ :
                                                                                          axis0Elements_);
        coreBaseIndex_ = static_cast<int64_t>(GetBlockIdx()) * perCoreElements_;
        if (GetBlockIdx() == GetBlockNum() - 1) {
            currentCoreElements_ = lastCoreElements_;
        } else {
            currentCoreElements_ = perCoreElements_;
        }
        if (coreBaseIndex_ >= splitElements) {
            currentCoreElements_ = 0;
        } else if (coreBaseIndex_ + currentCoreElements_ > splitElements) {
            currentCoreElements_ = splitElements - coreBaseIndex_;
        }

        inputGm_.SetGlobalBuffer((__gm__ T*)x, inputElementCount_);
        outputGm_.SetGlobalBuffer((__gm__ T*)y, totalOutputElements_);

        // Allocate UB buffer(s) if this core has work
        if (currentCoreElements_ > 0 && ubElements_ > 0) {
            if (tilingKey_ == AS_STRIDED_TILING_KEY_COMPACT_SPAN ||
                tilingKey_ == AS_STRIDED_TILING_KEY_COMPACT_SUFFIX) {
                int64_t rawElements = ((inputSpanElements_ + blockElements_ - 1) / blockElements_) * blockElements_;
                if constexpr (sizeof(T) == 1) {
                    const int64_t rawWideStartBytes = AlignUpToBlockBytes(rawElements *
                                                                          static_cast<int64_t>(sizeof(T)));
                    const int64_t rawBufferBytes = rawWideStartBytes + rawElements * static_cast<int64_t>(sizeof(half));
                    const int64_t packedWideStartBytes = AlignUpToBlockBytes(ubElements_ *
                                                                             static_cast<int64_t>(sizeof(T)));
                    const int64_t packedBufferBytes = packedWideStartBytes +
                                                      ubElements_ * static_cast<int64_t>(sizeof(half));
                    pipe_->InitBuffer(copyBuf_, static_cast<uint32_t>(rawBufferBytes));
                    pipe_->InitBuffer(copyBuf2_, static_cast<uint32_t>(packedBufferBytes));
                } else {
                    pipe_->InitBuffer(copyBuf_, static_cast<uint32_t>(rawElements * sizeof(T)));
                    pipe_->InitBuffer(copyBuf2_, static_cast<uint32_t>(ubElements_ * sizeof(T)));
                }
                pipe_->InitBuffer(indexBuf_, static_cast<uint32_t>(ubElements_ * sizeof(uint32_t)));
                return;
            }

            int64_t copyBufferElements = ubElements_;
            if (tilingKey_ == AS_STRIDED_TILING_KEY_GENERAL_STRIDE ||
                tilingKey_ == AS_STRIDED_TILING_KEY_RANK1_STRIDE) {
                copyBufferElements *= blockElements_;
            } else if (tilingKey_ == AS_STRIDED_TILING_KEY_STRIDE1_ROW_BATCH) {
                copyBufferElements *= blockElements_;
            } else if (tilingKey_ == AS_STRIDED_TILING_KEY_RANK1_STRIDE_SPAN) {
                copyBufferElements = (ubElements_ - 1) * lastDimStride_ + 1;
            }
            uint32_t ubBytes = static_cast<uint32_t>(copyBufferElements * sizeof(T));
            pipe_->InitBuffer(copyBuf_, ubBytes);
            if (tilingKey_ == AS_STRIDED_TILING_KEY_CONTIGUOUS ||
                tilingKey_ == AS_STRIDED_TILING_KEY_STRIDE1_ROW_BATCH ||
                tilingKey_ == AS_STRIDED_TILING_KEY_GENERAL_SMALL_SPAN ||
                tilingKey_ == AS_STRIDED_TILING_KEY_RANK1_STRIDE_SPAN) {
                pipe_->InitBuffer(copyBuf2_, ubBytes);
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (currentCoreElements_ <= 0 || ubElements_ <= 0) {
            return;
        }
        switch (tilingKey_) {
            case AS_STRIDED_TILING_KEY_CONTIGUOUS:
                ProcessContiguous();
                break;
            case AS_STRIDED_TILING_KEY_CONTIGUOUS_SMALL_ALIGNED:
                ProcessContiguousSmallAligned();
                break;
            case AS_STRIDED_TILING_KEY_COMPACT_SPAN:
                ProcessCompactSpan();
                break;
            case AS_STRIDED_TILING_KEY_COMPACT_SUFFIX:
                ProcessCompactSuffix();
                break;
            case AS_STRIDED_TILING_KEY_STRIDE_1:
                ProcessStride1();
                break;
            case AS_STRIDED_TILING_KEY_STRIDE1_ROW_BATCH:
                ProcessStride1RowBatch();
                break;
            case AS_STRIDED_TILING_KEY_GENERAL_STRIDE:
                ProcessGeneralStride();
                break;
            case AS_STRIDED_TILING_KEY_GENERAL_SMALL_SPAN:
                ProcessGeneralSmallSpan();
                break;
            case AS_STRIDED_TILING_KEY_RANK1_STRIDE:
                ProcessRank1Stride();
                break;
            case AS_STRIDED_TILING_KEY_RANK1_STRIDE_SPAN:
                ProcessRank1StrideSpan();
                break;
            case AS_STRIDED_TILING_KEY_BROADCAST:
                ProcessBroadcast();
                break;
            default:
                ProcessScalar();
                break;
        }
    }

private:
    // =========================================================================
    // ComputeInputOffset: map flat output index → flat input offset
    // =========================================================================
    __aicore__ inline int64_t ComputeInputOffset(int64_t outputLinearIndex) const
    {
        int64_t remain = outputLinearIndex;
        int64_t inputOffset = storageOffset_;
        for (int64_t dim = 0; dim < outputDimNum_; ++dim) {
            int64_t coord = 0;
            if (dim == outputDimNum_ - 1) {
                coord = remain;
            } else {
                coord = remain / outSizeStride_[dim];
                remain = remain - coord * outSizeStride_[dim];
            }
            inputOffset += coord * outStride_[dim];
        }
        return inputOffset;
    }

    __aicore__ inline int64_t ComputeInputOffsetAndCoords(int64_t outputLinearIndex,
                                                          int64_t coords[AS_STRIDED_MAX_DIMS]) const
    {
        int64_t remain = outputLinearIndex;
        int64_t inputOffset = storageOffset_;
        for (int64_t dim = 0; dim < outputDimNum_; ++dim) {
            int64_t coord = 0;
            if (dim == outputDimNum_ - 1) {
                coord = remain;
            } else {
                coord = remain / outSizeStride_[dim];
                remain = remain - coord * outSizeStride_[dim];
            }
            coords[dim] = coord;
            inputOffset += coord * outStride_[dim];
        }
        return inputOffset;
    }

    __aicore__ inline void AdvanceInputOffset(int64_t coords[AS_STRIDED_MAX_DIMS], int64_t& inputOffset) const
    {
        for (int64_t dim = outputDimNum_ - 1; dim >= 0; --dim) {
            ++coords[dim];
            inputOffset += outStride_[dim];
            if (coords[dim] < outSize_[dim]) {
                return;
            }
            inputOffset -= coords[dim] * outStride_[dim];
            coords[dim] = 0;
        }
    }

    __aicore__ inline void GenerateSuffixMaskBytes(LocalTensor<uint32_t>& maskLocal, int64_t elemSize)
    {
        LocalTensor<int32_t> maskIntLocal = maskLocal.ReinterpretCast<int32_t>();

        if (suffixStartDim_ == outputDimNum_ - 1) {
            ArithProgression<int32_t>(maskIntLocal, static_cast<int32_t>(0),
                                      static_cast<int32_t>(outStride_[suffixStartDim_] * elemSize),
                                      static_cast<int32_t>(suffixElements_));
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }

        if (suffixStartDim_ == outputDimNum_ - 2) {
            const int64_t colCount = outSize_[suffixStartDim_ + 1];
            const int64_t rowCount = suffixElements_ / colCount;
            if ((colCount * static_cast<int64_t>(sizeof(int32_t)) % 32) == 0) {
                const int32_t colStep = static_cast<int32_t>(outStride_[suffixStartDim_ + 1] * elemSize);
                for (int64_t row = 0; row < rowCount; ++row) {
                    const int32_t rowStart = static_cast<int32_t>(row * outStride_[suffixStartDim_] * elemSize);
                    ArithProgression<int32_t>(maskIntLocal[row * colCount], rowStart, colStep,
                                              static_cast<int32_t>(colCount));
                }
                AscendC::PipeBarrier<PIPE_V>();
                return;
            }
        }

        int64_t coords[AS_STRIDED_MAX_DIMS] = {0};
        int64_t inputOffset = 0;
        for (int64_t elem = 0; elem < suffixElements_; ++elem) {
            maskLocal.SetValue(elem, static_cast<uint32_t>(inputOffset * elemSize));
            for (int64_t dim = outputDimNum_ - 1; dim >= suffixStartDim_; --dim) {
                ++coords[dim];
                inputOffset += outStride_[dim];
                if (coords[dim] < outSize_[dim]) {
                    break;
                }
                inputOffset -= coords[dim] * outStride_[dim];
                coords[dim] = 0;
            }
        }
        SyncSToV();
    }

    __aicore__ inline void GenerateSuffixMask(LocalTensor<uint32_t>& maskLocal)
    {
        GenerateSuffixMaskBytes(maskLocal, static_cast<int64_t>(sizeof(T)));
    }

    __aicore__ inline void PackByMaskScalar(LocalTensor<T>& rawLocal, LocalTensor<T>& packedLocal,
                                            LocalTensor<uint32_t>& maskLocal, int64_t inputBaseDelta,
                                            int64_t elements) const
    {
        const int64_t elemSize = static_cast<int64_t>(sizeof(T));
        for (int64_t elem = 0; elem < elements; ++elem) {
            const int64_t rawIndex = inputBaseDelta + static_cast<int64_t>(maskLocal.GetValue(elem)) / elemSize;
            packedLocal.SetValue(elem, rawLocal.GetValue(rawIndex));
        }
    }

    template <typename WideT>
    __aicore__ inline void PackB8FromWide(LocalTensor<WideT>& rawWideLocal, LocalTensor<WideT>& packedWideLocal,
                                          LocalTensor<T>& packedLocal, LocalTensor<uint32_t>& maskLocal,
                                          int64_t inputBaseDelta, int64_t elements) const
    {
        Gather(packedWideLocal, rawWideLocal[inputBaseDelta], maskLocal, 0, static_cast<uint32_t>(elements));
        AscendC::PipeBarrier<PIPE_V>();
        Cast(packedLocal, packedWideLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(elements));
    }

    __aicore__ inline int64_t AlignUpToBlockBytes(int64_t bytes) const
    {
        constexpr int64_t blockBytes = 32;
        return ((bytes + blockBytes - 1) / blockBytes) * blockBytes;
    }

    // =========================================================================
    // ComputeRowInputOffset: get input offset for row 0 of a given axis_0 index
    // =========================================================================
    __aicore__ inline int64_t ComputeRowInputOffset(int64_t rowIndex) const
    {
        return ComputeInputOffset(rowIndex * lastDimSize_);
    }

    __aicore__ inline int64_t ComputeNextRowInputOffset(int64_t rowIndex, int64_t currentOffset) const
    {
        if (outputDimNum_ <= 1) {
            return currentOffset + lastDimStride_;
        }

        int64_t nextOffset = currentOffset;
        for (int64_t dim = outputDimNum_ - 2; dim >= 0; --dim) {
            int64_t rowsPerDim = outSizeStride_[dim] / lastDimSize_;
            if (rowsPerDim <= 0) {
                rowsPerDim = 1;
            }
            int64_t outer = rowIndex / rowsPerDim;
            int64_t coord = outer - (outer / outSize_[dim]) * outSize_[dim];
            if (coord + 1 < outSize_[dim]) {
                return nextOffset + outStride_[dim];
            }
            nextOffset -= (outSize_[dim] - 1) * outStride_[dim];
        }
        return nextOffset;
    }

    // =========================================================================
    // PATH 0: Fully contiguous block-copy with double buffering
    // =========================================================================
    __aicore__ inline void ProcessContiguous()
    {
        LocalTensor<T> copyLocal0 = copyBuf_.Get<T>();
        LocalTensor<T> copyLocal1 = copyBuf2_.Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        int64_t inputBase = storageOffset_ + coreBaseIndex_;
        int64_t outputBase = coreBaseIndex_;
        int64_t totalElements = currentCoreElements_;
        bool useBuf1 = false;

        for (int64_t copied = 0; copied < totalElements; copied += ubElements_) {
            int64_t copyElements = ubElements_;
            if (copyElements > totalElements - copied) {
                copyElements = totalElements - copied;
            }
            DataCopyExtParams loadParams{1, static_cast<uint32_t>(copyElements * sizeof(T)), 0, 0, 0};

            // Double-buffer: ping-pong between copyBuf_ and copyBuf2_
            LocalTensor<T> loadBuf = useBuf1 ? copyLocal1 : copyLocal0;
            LocalTensor<T> storeBuf = useBuf1 ? copyLocal0 : copyLocal1;

            // Load tile N into loadBuf
            DataCopyPad(loadBuf, inputGm_[inputBase + copied], loadParams, padParams);
            SyncMte2ToMte3();

            if (copied > 0) {
                // Store previous tile (tile N-1) from storeBuf — always ubElements_
                DataCopyExtParams storeParams{1, static_cast<uint32_t>(ubElements_ * sizeof(T)), 0, 0, 0};
                DataCopyPad(outputGm_[outputBase + copied - ubElements_], storeBuf, storeParams);
                SyncMte3ToMte2();
            }

            useBuf1 = !useBuf1;
        }

        // Store the final tile
        if (totalElements > 0) {
            int64_t lastCopied = ((totalElements - 1) / ubElements_) * ubElements_;
            int64_t lastElements = totalElements - lastCopied;
            DataCopyExtParams lastParams{1, static_cast<uint32_t>(lastElements * sizeof(T)), 0, 0, 0};
            LocalTensor<T> lastStoreBuf = (((totalElements - 1) / ubElements_) % 2 == 0) ? copyLocal0 : copyLocal1;
            DataCopyPad(outputGm_[outputBase + lastCopied], lastStoreBuf, lastParams);
            SyncMte3ToMte2();
        }
    }

    // =========================================================================
    // PATH 9: Small aligned contiguous copy, single UB buffer
    // =========================================================================
    __aicore__ inline void ProcessContiguousSmallAligned()
    {
        LocalTensor<T> copyLocal = copyBuf_.Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        int64_t inputBase = storageOffset_ + coreBaseIndex_;
        int64_t outputBase = coreBaseIndex_;
        int64_t totalElements = currentCoreElements_;

        for (int64_t copied = 0; copied < totalElements; copied += ubElements_) {
            int64_t copyElements = ubElements_;
            if (copyElements > totalElements - copied) {
                copyElements = totalElements - copied;
            }

            int64_t copyBytes = copyElements * static_cast<int64_t>(sizeof(T));
            if ((copyBytes % 32) == 0) {
                uint16_t blockUnits = static_cast<uint16_t>(copyBytes / 32);
                DataCopyParams copyParams{1, blockUnits, 0, 0};
                if (sizeof(T) == 2 || sizeof(T) == 8) {
                    copyParams.blockCount = blockUnits;
                    copyParams.blockLen = 1;
                }
                DataCopy(copyLocal, inputGm_[inputBase + copied], copyParams);
                SyncMte2ToMte3();
                DataCopy(outputGm_[outputBase + copied], copyLocal, copyParams);
                if (copied + copyElements < totalElements || (sizeof(T) != 8 && sizeof(T) != 2)) {
                    SyncMte3ToMte2();
                }
            } else {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyBytes), 0, 0, 0};
                DataCopyPad(copyLocal, inputGm_[inputBase + copied], copyParams, padParams);
                SyncMte2ToMte3();
                DataCopyPad(outputGm_[outputBase + copied], copyLocal, copyParams);
                if (copied + copyElements < totalElements || (sizeof(T) != 8 && sizeof(T) != 2)) {
                    SyncMte3ToMte2();
                }
            }
        }
    }

    // =========================================================================
    // PATH 10: Whole input span fits in UB. Load it once per core, then gather
    // from UB into large contiguous output tiles.
    // =========================================================================
    __aicore__ inline void ProcessCompactSpan()
    {
        LocalTensor<T> rawLocal = copyBuf_.Get<T>();
        LocalTensor<T> packedLocal = copyBuf2_.Get<T>();
        LocalTensor<uint32_t> maskLocal = indexBuf_.Get<uint32_t>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        DataCopyExtParams loadParams{1, static_cast<uint32_t>(inputSpanElements_ * static_cast<int64_t>(sizeof(T))), 0,
                                     0, 0};
        DataCopyPad(rawLocal, inputGm_[storageOffset_], loadParams, padParams);
        SyncMte2ToV();

        if constexpr (sizeof(T) == 1) {
            const int64_t rawElements = ((inputSpanElements_ + blockElements_ - 1) / blockElements_) * blockElements_;
            const int64_t rawWideStartBytes = AlignUpToBlockBytes(rawElements * static_cast<int64_t>(sizeof(T)));
            const int64_t packedWideStartBytes = AlignUpToBlockBytes(ubElements_ * static_cast<int64_t>(sizeof(T)));
            LocalTensor<half> rawWideLocal = copyBuf_
                                                 .Get<half>()[rawWideStartBytes / static_cast<int64_t>(sizeof(half))];
            LocalTensor<half>
                packedWideLocal = copyBuf2_.Get<half>()[packedWideStartBytes / static_cast<int64_t>(sizeof(half))];
            Cast(rawWideLocal, rawLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(inputSpanElements_));
            AscendC::PipeBarrier<PIPE_V>();

            int64_t coords[AS_STRIDED_MAX_DIMS] = {0};
            for (int64_t done = 0; done < currentCoreElements_; done += ubElements_) {
                int64_t tileElements = ubElements_;
                if (tileElements > currentCoreElements_ - done) {
                    tileElements = currentCoreElements_ - done;
                }

                const int64_t outputOffset = coreBaseIndex_ + done;
                int64_t inputOffset = ComputeInputOffsetAndCoords(outputOffset, coords);
                for (int64_t elem = 0; elem < tileElements; ++elem) {
                    maskLocal.SetValue(elem, static_cast<uint32_t>((inputOffset - storageOffset_) *
                                                                   static_cast<int64_t>(sizeof(half))));
                    AdvanceInputOffset(coords, inputOffset);
                }
                SyncSToV();
                PackB8FromWide(rawWideLocal, packedWideLocal, packedLocal, maskLocal, 0, tileElements);
                SyncVToMte3();

                DataCopyExtParams outParams{1, static_cast<uint32_t>(tileElements * static_cast<int64_t>(sizeof(T))), 0,
                                            0, 0};
                DataCopyPad(outputGm_[outputOffset], packedLocal, outParams);
                SyncMte3ToV();
            }
            return;
        }

        int64_t coords[AS_STRIDED_MAX_DIMS] = {0};
        for (int64_t done = 0; done < currentCoreElements_; done += ubElements_) {
            int64_t tileElements = ubElements_;
            if (tileElements > currentCoreElements_ - done) {
                tileElements = currentCoreElements_ - done;
            }

            const int64_t outputOffset = coreBaseIndex_ + done;
            int64_t inputOffset = ComputeInputOffsetAndCoords(outputOffset, coords);
            if constexpr (sizeof(T) == 8) {
                for (int64_t elem = 0; elem < tileElements; ++elem) {
                    maskLocal.SetValue(
                        elem, static_cast<uint32_t>((inputOffset - storageOffset_) * static_cast<int64_t>(sizeof(T))));
                    AdvanceInputOffset(coords, inputOffset);
                }
                PackByMaskScalar(rawLocal, packedLocal, maskLocal, 0, tileElements);
                SyncSToMte3();
            } else {
                for (int64_t elem = 0; elem < tileElements; ++elem) {
                    maskLocal.SetValue(
                        elem, static_cast<uint32_t>((inputOffset - storageOffset_) * static_cast<int64_t>(sizeof(T))));
                    AdvanceInputOffset(coords, inputOffset);
                }
                SyncSToV();
                Gather(packedLocal, rawLocal, maskLocal, 0, static_cast<uint32_t>(tileElements));
                SyncVToMte3();
            }

            DataCopyExtParams outParams{1, static_cast<uint32_t>(tileElements * static_cast<int64_t>(sizeof(T))), 0, 0,
                                        0};
            DataCopyPad(outputGm_[outputOffset], packedLocal, outParams);
            SyncMte3ToV();
        }
    }

    // =========================================================================
    // PATH 11: Compact span with a reusable suffix gather mask. Each work item
    // is one contiguous output suffix block.
    // =========================================================================
    __aicore__ inline void ProcessCompactSuffix()
    {
        LocalTensor<T> rawLocal = copyBuf_.Get<T>();
        LocalTensor<T> packedLocal = copyBuf2_.Get<T>();
        LocalTensor<uint32_t> maskLocal = indexBuf_.Get<uint32_t>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        if constexpr (sizeof(T) == 1) {
            GenerateSuffixMaskBytes(maskLocal, static_cast<int64_t>(sizeof(half)));
        } else {
            GenerateSuffixMask(maskLocal);
        }
        if constexpr (sizeof(T) == 8) {
            SyncVToS();
        }

        DataCopyExtParams loadParams{1, static_cast<uint32_t>(inputSpanElements_ * static_cast<int64_t>(sizeof(T))), 0,
                                     0, 0};
        DataCopyPad(rawLocal, inputGm_[storageOffset_], loadParams, padParams);
        SyncMte2ToV();

        if constexpr (sizeof(T) == 1) {
            const int64_t rawElements = ((inputSpanElements_ + blockElements_ - 1) / blockElements_) * blockElements_;
            const int64_t rawWideStartBytes = AlignUpToBlockBytes(rawElements * static_cast<int64_t>(sizeof(T)));
            const int64_t packedWideStartBytes = AlignUpToBlockBytes(ubElements_ * static_cast<int64_t>(sizeof(T)));
            LocalTensor<half> rawWideLocal = copyBuf_
                                                 .Get<half>()[rawWideStartBytes / static_cast<int64_t>(sizeof(half))];
            LocalTensor<half>
                packedWideLocal = copyBuf2_.Get<half>()[packedWideStartBytes / static_cast<int64_t>(sizeof(half))];
            Cast(rawWideLocal, rawLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(inputSpanElements_));
            AscendC::PipeBarrier<PIPE_V>();

            if (outputDimNum_ == 2 && suffixStartDim_ == 1) {
                int64_t inputBaseDelta = coreBaseIndex_ * outStride_[0];
                int64_t outputOffset = coreBaseIndex_ * lastDimSize_;
                for (int64_t block = 0; block < currentCoreElements_; ++block) {
                    PackB8FromWide(rawWideLocal, packedWideLocal, packedLocal, maskLocal, inputBaseDelta, lastDimSize_);
                    SyncVToMte3();

                    DataCopyExtParams outParams{
                        1, static_cast<uint32_t>(lastDimSize_ * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
                    DataCopyPad(outputGm_[outputOffset], packedLocal, outParams);
                    SyncMte3ToV();

                    inputBaseDelta += outStride_[0];
                    outputOffset += lastDimSize_;
                }
                return;
            }

            for (int64_t block = 0; block < currentCoreElements_; ++block) {
                const int64_t suffixBlockIndex = coreBaseIndex_ + block;
                const int64_t outputOffset = suffixBlockIndex * suffixElements_;
                const int64_t inputBaseDelta = ComputeInputOffset(outputOffset) - storageOffset_;
                int64_t currentSuffixElements = suffixElements_;
                if (outputOffset + currentSuffixElements > totalOutputElements_) {
                    currentSuffixElements = totalOutputElements_ - outputOffset;
                }
                PackB8FromWide(rawWideLocal, packedWideLocal, packedLocal, maskLocal, inputBaseDelta,
                               currentSuffixElements);
                SyncVToMte3();

                DataCopyExtParams outParams{
                    1, static_cast<uint32_t>(currentSuffixElements * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
                DataCopyPad(outputGm_[outputOffset], packedLocal, outParams);
                SyncMte3ToV();
            }
            return;
        }

        if (outputDimNum_ == 2 && suffixStartDim_ == 1) {
            int64_t inputBaseDelta = coreBaseIndex_ * outStride_[0];
            int64_t outputOffset = coreBaseIndex_ * lastDimSize_;
            for (int64_t block = 0; block < currentCoreElements_; ++block) {
                if constexpr (sizeof(T) == 8) {
                    PackByMaskScalar(rawLocal, packedLocal, maskLocal, inputBaseDelta, lastDimSize_);
                    SyncSToMte3();
                } else {
                    Gather(packedLocal, rawLocal[inputBaseDelta], maskLocal, 0, static_cast<uint32_t>(lastDimSize_));
                    SyncVToMte3();
                }

                DataCopyExtParams outParams{1, static_cast<uint32_t>(lastDimSize_ * static_cast<int64_t>(sizeof(T))), 0,
                                            0, 0};
                DataCopyPad(outputGm_[outputOffset], packedLocal, outParams);
                SyncMte3ToV();

                inputBaseDelta += outStride_[0];
                outputOffset += lastDimSize_;
            }
            return;
        }

        for (int64_t block = 0; block < currentCoreElements_; ++block) {
            const int64_t suffixBlockIndex = coreBaseIndex_ + block;
            const int64_t outputOffset = suffixBlockIndex * suffixElements_;
            const int64_t inputBaseDelta = ComputeInputOffset(outputOffset) - storageOffset_;
            int64_t currentSuffixElements = suffixElements_;
            if (outputOffset + currentSuffixElements > totalOutputElements_) {
                currentSuffixElements = totalOutputElements_ - outputOffset;
            }
            if constexpr (sizeof(T) == 8) {
                PackByMaskScalar(rawLocal, packedLocal, maskLocal, inputBaseDelta, currentSuffixElements);
                SyncSToMte3();
            } else {
                Gather(packedLocal, rawLocal[inputBaseDelta], maskLocal, 0,
                       static_cast<uint32_t>(currentSuffixElements));
                SyncVToMte3();
            }

            DataCopyExtParams outParams{
                1, static_cast<uint32_t>(currentSuffixElements * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
            DataCopyPad(outputGm_[outputOffset], packedLocal, outParams);
            SyncMte3ToV();
        }
    }

    // =========================================================================
    // PATH 1: Last dim stride == 1 — contiguous rows, scattered row starts
    // =========================================================================
    __aicore__ inline void ProcessStride1()
    {
        LocalTensor<T> copyLocal = copyBuf_.Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        int64_t inputOffset = ComputeRowInputOffset(coreBaseIndex_);
        for (int64_t row = 0; row < currentCoreElements_; ++row) {
            int64_t axis0Index = coreBaseIndex_ + row;
            int64_t outputOffset = axis0Index * lastDimSize_;

            // Process row in chunks if it exceeds UB capacity
            for (int64_t elem = 0; elem < lastDimSize_; elem += ubElements_) {
                int64_t chunk = ubElements_;
                if (chunk > lastDimSize_ - elem) {
                    chunk = lastDimSize_ - elem;
                }
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(chunk * sizeof(T)), 0, 0, 0};
                DataCopyPad(copyLocal, inputGm_[inputOffset + elem], copyParams, padParams);
                SyncMte2ToMte3();
                DataCopyPad(outputGm_[outputOffset + elem], copyLocal, copyParams);
                SyncMte3ToMte2();
            }

            if (row + 1 < currentCoreElements_) {
                inputOffset = ComputeNextRowInputOffset(axis0Index, inputOffset);
            }
        }
    }

    __aicore__ inline void ProcessStride1RowBatch()
    {
        LocalTensor<T> rawLocal = copyBuf_.Get<T>();
        LocalTensor<T> packedLocal = copyBuf2_.Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        int64_t elemSize = static_cast<int64_t>(sizeof(T));
        int64_t rowStride = outStride_[outputDimNum_ - 2];
        int64_t srcStrideBytes = (rowStride - lastDimSize_) * elemSize;

        int64_t row = 0;
        while (row < currentCoreElements_) {
            int64_t axis0Index = coreBaseIndex_ + row;
            int64_t runRows = currentCoreElements_ - row;
            int64_t dimCoord = axis0Index - (axis0Index / outSize_[outputDimNum_ - 2]) * outSize_[outputDimNum_ - 2];
            int64_t rowsUntilDimEnd = outSize_[outputDimNum_ - 2] - dimCoord;
            if (runRows > rowsUntilDimEnd) {
                runRows = rowsUntilDimEnd;
            }
            if (runRows > ubElements_) {
                runRows = ubElements_;
            }

            int64_t inputOffset = ComputeRowInputOffset(axis0Index);
            int64_t outputOffset = axis0Index * lastDimSize_;

            DataCopyExtParams loadParams{static_cast<uint16_t>(runRows), static_cast<uint32_t>(lastDimSize_ * elemSize),
                                         static_cast<uint32_t>(srcStrideBytes), 0, 0};
            DataCopyPad(rawLocal, inputGm_[inputOffset], loadParams, padParams);
            SyncMte2ToMte3();

            for (int64_t r = 0; r < runRows; ++r) {
                for (int64_t elem = 0; elem < lastDimSize_; ++elem) {
                    packedLocal.SetValue(r * lastDimSize_ + elem, rawLocal.GetValue(r * blockElements_ + elem));
                }
            }

            DataCopyExtParams outParams{1, static_cast<uint32_t>(runRows * lastDimSize_ * elemSize), 0, 0, 0};
            DataCopyPad(outputGm_[outputOffset], packedLocal, outParams);
            SyncMte3ToMte2();

            row += runRows;
        }
    }

    // =========================================================================
    // PATH 2: General strided last dimension — "搬运即重排"
    // Uses DataCopy srcStride to gather strided elements into contiguous UB.
    // =========================================================================
    __aicore__ inline void ProcessGeneralStride()
    {
        LocalTensor<T> copyLocal = copyBuf_.Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        // DataCopy blockCount is limited to 4095. If lastDimSize exceeds this,
        // we chunk within each row.
        int64_t maxBlockCount = ubElements_; // already capped to ≤4095 by tiling
        int64_t elemSize = static_cast<int64_t>(sizeof(T));
        int64_t srcStrideBytes = (lastDimStride_ - 1) * elemSize;

        int64_t inputOffset = ComputeRowInputOffset(coreBaseIndex_);
        for (int64_t row = 0; row < currentCoreElements_; ++row) {
            int64_t axis0Index = coreBaseIndex_ + row;
            int64_t outputOffset = axis0Index * lastDimSize_;

            for (int64_t done = 0; done < lastDimSize_; done += maxBlockCount) {
                int64_t chunk = maxBlockCount;
                if (chunk > lastDimSize_ - done) {
                    chunk = lastDimSize_ - done;
                }

                // Key optimization: use DataCopy srcStride to skip gaps in input.
                // MTE reads chunk elements from input at stride lastDimStride_
                // and packs them contiguously into UB.
                DataCopyExtParams copyParams{
                    static_cast<uint16_t>(chunk),          // blockCount
                    static_cast<uint32_t>(elemSize),       // blockLen = 1 element
                    static_cast<uint32_t>(srcStrideBytes), // srcStride in bytes
                    0,                                     // dstStride = 0
                    0                                      // reserved
                };
                DataCopyPad(copyLocal, inputGm_[inputOffset + done * lastDimStride_], copyParams, padParams);
                SyncMte2ToMte3();

                for (int64_t elem = 0; elem < chunk; ++elem) {
                    copyLocal.SetValue(elem, copyLocal.GetValue(elem * blockElements_));
                }

                // Write chunk elements contiguously to output
                DataCopyExtParams outParams{1, static_cast<uint32_t>(chunk * elemSize), 0, 0, 0};
                DataCopyPad(outputGm_[outputOffset + done], copyLocal, outParams);
                SyncMte3ToMte2();
            }

            if (row + 1 < currentCoreElements_) {
                inputOffset = ComputeNextRowInputOffset(axis0Index, inputOffset);
            }
        }
    }

    __aicore__ inline void ProcessGeneralSmallSpan()
    {
        LocalTensor<T> rawLocal = copyBuf_.Get<T>();
        LocalTensor<T> packedLocal = copyBuf2_.Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        int64_t elemSize = static_cast<int64_t>(sizeof(T));
        int64_t rowStride = outStride_[outputDimNum_ - 2];
        int64_t spanElements = (lastDimSize_ - 1) * lastDimStride_ + 1;
        if (spanElements > ubElements_) {
            ProcessScalar();
            return;
        }

        int64_t row = 0;
        while (row < currentCoreElements_) {
            int64_t axis0Index = coreBaseIndex_ + row;
            int64_t runRows = currentCoreElements_ - row;
            int64_t dimCoord = axis0Index - (axis0Index / outSize_[outputDimNum_ - 2]) * outSize_[outputDimNum_ - 2];
            int64_t rowsUntilDimEnd = outSize_[outputDimNum_ - 2] - dimCoord;
            if (runRows > rowsUntilDimEnd) {
                runRows = rowsUntilDimEnd;
            }

            while (runRows > 1) {
                int64_t copyElements = (runRows - 1) * rowStride + spanElements;
                int64_t packedElements = runRows * lastDimSize_;
                if (copyElements <= ubElements_ && packedElements <= ubElements_) {
                    break;
                }
                --runRows;
            }

            int64_t copyElements = (runRows - 1) * rowStride + spanElements;
            int64_t packedElements = runRows * lastDimSize_;
            int64_t inputOffset = ComputeRowInputOffset(axis0Index);
            int64_t outputOffset = axis0Index * lastDimSize_;

            DataCopyExtParams loadParams{1, static_cast<uint32_t>(copyElements * elemSize), 0, 0, 0};
            DataCopyPad(rawLocal, inputGm_[inputOffset], loadParams, padParams);
            SyncMte2ToMte3();

            for (int64_t r = 0; r < runRows; ++r) {
                for (int64_t elem = 0; elem < lastDimSize_; ++elem) {
                    packedLocal.SetValue(r * lastDimSize_ + elem,
                                         rawLocal.GetValue(r * rowStride + elem * lastDimStride_));
                }
            }

            DataCopyExtParams outParams{1, static_cast<uint32_t>(packedElements * elemSize), 0, 0, 0};
            DataCopyPad(outputGm_[outputOffset], packedLocal, outParams);
            SyncMte3ToMte2();

            row += runRows;
        }
    }

    __aicore__ inline void ProcessRank1Stride()
    {
        LocalTensor<T> copyLocal = copyBuf_.Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        int64_t maxBlockCount = ubElements_;
        int64_t elemSize = static_cast<int64_t>(sizeof(T));
        int64_t srcStrideBytes = (lastDimStride_ - 1) * elemSize;

        for (int64_t done = 0; done < currentCoreElements_; done += maxBlockCount) {
            int64_t chunk = maxBlockCount;
            if (chunk > currentCoreElements_ - done) {
                chunk = currentCoreElements_ - done;
            }

            DataCopyExtParams copyParams{static_cast<uint16_t>(chunk), static_cast<uint32_t>(elemSize),
                                         static_cast<uint32_t>(srcStrideBytes), 0, 0};
            int64_t outputOffset = coreBaseIndex_ + done;
            int64_t inputOffset = storageOffset_ + outputOffset * lastDimStride_;
            DataCopyPad(copyLocal, inputGm_[inputOffset], copyParams, padParams);
            SyncMte2ToMte3();

            for (int64_t elem = 0; elem < chunk; ++elem) {
                copyLocal.SetValue(elem, copyLocal.GetValue(elem * blockElements_));
            }

            DataCopyExtParams outParams{1, static_cast<uint32_t>(chunk * elemSize), 0, 0, 0};
            DataCopyPad(outputGm_[outputOffset], copyLocal, outParams);
            SyncMte3ToMte2();
        }
    }

    __aicore__ inline void ProcessRank1StrideSpan()
    {
        LocalTensor<T> rawLocal = copyBuf_.Get<T>();
        LocalTensor<T> packedLocal = copyBuf2_.Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        int64_t elemSize = static_cast<int64_t>(sizeof(T));
        int64_t maxChunk = ubElements_;

        for (int64_t done = 0; done < currentCoreElements_; done += maxChunk) {
            int64_t chunk = maxChunk;
            if (chunk > currentCoreElements_ - done) {
                chunk = currentCoreElements_ - done;
            }

            int64_t outputOffset = coreBaseIndex_ + done;
            int64_t inputOffset = storageOffset_ + outputOffset * lastDimStride_;
            int64_t copyElements = (chunk - 1) * lastDimStride_ + 1;

            DataCopyExtParams loadParams{1, static_cast<uint32_t>(copyElements * elemSize), 0, 0, 0};
            DataCopyPad(rawLocal, inputGm_[inputOffset], loadParams, padParams);
            SyncMte2ToMte3();

            for (int64_t elem = 0; elem < chunk; ++elem) {
                packedLocal.SetValue(elem, rawLocal.GetValue(elem * lastDimStride_));
            }

            DataCopyExtParams outParams{1, static_cast<uint32_t>(chunk * elemSize), 0, 0, 0};
            DataCopyPad(outputGm_[outputOffset], packedLocal, outParams);
            SyncMte3ToMte2();
        }
    }

    // =========================================================================
    // PATH 3: Last dim stride == 0 — broadcast single element across the row
    // =========================================================================
    __aicore__ inline void ProcessBroadcast()
    {
        LocalTensor<T> copyLocal = copyBuf_.Get<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        int64_t elemSize = static_cast<int64_t>(sizeof(T));

        int64_t inputOffset = ComputeRowInputOffset(coreBaseIndex_);
        for (int64_t row = 0; row < currentCoreElements_; ++row) {
            int64_t axis0Index = coreBaseIndex_ + row;
            int64_t outputOffset = axis0Index * lastDimSize_;

            // Read 1 element from input into UB[0]
            DataCopyExtParams readOne{1, static_cast<uint32_t>(elemSize), 0, 0, 0};
            DataCopyPad(copyLocal, inputGm_[inputOffset], readOne, padParams);
            SyncMte2ToMte3();

            // Replicate UB[0] across the buffer by chunked DataCopy writes.
            // Each output chunk writes the same source value from UB[0].
            for (int64_t done = 0; done < lastDimSize_; done += ubElements_) {
                int64_t chunk = ubElements_;
                if (chunk > lastDimSize_ - done) {
                    chunk = lastDimSize_ - done;
                }

                // Fill chunk elements of UB with the value in UB[0]
                if (chunk > 1) {
                    for (int64_t i = 1; i < chunk; ++i) {
                        copyLocal.SetValue(i, copyLocal.GetValue(0));
                    }
                }

                DataCopyExtParams outParams{1, static_cast<uint32_t>(chunk * elemSize), 0, 0, 0};
                DataCopyPad(outputGm_[outputOffset + done], copyLocal, outParams);
                SyncMte3ToMte2();
            }

            if (row + 1 < currentCoreElements_) {
                inputOffset = ComputeNextRowInputOffset(axis0Index, inputOffset);
            }
        }
    }

    // =========================================================================
    // PATH 4: Scalar fallback — per-element GetValue/SetValue
    // =========================================================================
    __aicore__ inline void ProcessScalar()
    {
        int64_t start = coreBaseIndex_ * lastDimSize_;
        int64_t end = start + currentCoreElements_ * lastDimSize_;
        for (int64_t outIdx = start; outIdx < end; ++outIdx) {
            int64_t inIdx = ComputeInputOffset(outIdx);
            outputGm_.SetValue(outIdx, inputGm_.GetValue(inIdx));
        }
    }

    // =========================================================================
    // MTE event synchronization
    // =========================================================================
    __aicore__ inline void SyncMte2ToMte3()
    {
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    }

    __aicore__ inline void SyncMte2ToV()
    {
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void SyncSToV()
    {
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);
    }

    __aicore__ inline void SyncVToMte3()
    {
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void SyncVToS()
    {
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
    }

    __aicore__ inline void SyncSToMte3()
    {
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventId);
        WaitFlag<HardEvent::S_MTE3>(eventId);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

private:
    TPipe* pipe_ = nullptr;
    TBuf<TPosition::VECCALC> copyBuf_;
    TBuf<TPosition::VECCALC> copyBuf2_; // double-buffer for contiguous path
    TBuf<TPosition::VECCALC> indexBuf_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    GlobalTensor<IndexT> sizeGm_;
    GlobalTensor<IndexT> strideGm_;
    GlobalTensor<IndexT> offsetGm_;

    int64_t tilingKey_ = 0;
    int64_t totalOutputElements_ = 0;
    int64_t inputElementCount_ = 0;
    int64_t perCoreElements_ = 0;
    int64_t lastCoreElements_ = 0;
    int64_t ubElements_ = 0;
    int64_t storageOffset_ = 0;
    int64_t outputDimNum_ = 0;
    int64_t lastDimSize_ = 0;
    int64_t lastDimStride_ = 0;
    int64_t axis0Elements_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t blockElements_ = 0;
    int64_t inputSpanElements_ = 0;
    int64_t suffixStartDim_ = 0;
    int64_t suffixElements_ = 0;
    int64_t suffixOuterElements_ = 0;
    int64_t coreBaseIndex_ = 0;
    int64_t currentCoreElements_ = 0;
    int64_t outSize_[AS_STRIDED_MAX_DIMS] = {0};
    int64_t outStride_[AS_STRIDED_MAX_DIMS] = {0};
    int64_t outSizeStride_[AS_STRIDED_MAX_DIMS] = {1, 1, 1, 1, 1, 1, 1, 1};
};

} // namespace NsAsStrided

#endif
