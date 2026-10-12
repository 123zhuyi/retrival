/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef EXPERIMENTAL_IM2COL_CHANNEL_TRANSPOSE_H_
#define EXPERIMENTAL_IM2COL_CHANNEL_TRANSPOSE_H_

#include <cstdint>
#include "kernel_operator.h"
#include "im2col_tiling_data.h"

namespace NsIm2col {
using namespace AscendC;

// DAV_2201 C16 transpose path.  T is the bit-preserving 16-bit storage type
// used by both fp16 and bf16 instantiations.
template <typename T>
class Im2colChannelTransposeKernel {
public:
    __aicore__ inline Im2colChannelTransposeKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const Im2colTilingHeader* td, TPipe* pipe)
    {
        pipe_ = pipe;
        c_ = td->c;
        h_ = td->h;
        w_ = td->w;
        kernelH_ = td->kernelH;
        kernelW_ = td->kernelW;
        padTop_ = td->padTop;
        padLeft_ = td->padLeft;
        outH_ = td->outH;
        outW_ = td->outW;
        totalChannelTiles_ = td->totalChannels;
        inputSpatialAligned_ = td->rawRowStrideElements;
        groupTile_ = td->groupBatch;
        outputChannelElements_ = td->outputChannelElements;
        outputSpatial_ = td->outputGroupStrideElements;

        const int64_t block = static_cast<int64_t>(GetBlockIdx());
        const int64_t beforeExtra = block < td->extraChannels ? block : td->extraChannels;
        channelTileBegin_ = block * td->baseChannelsPerCore + beforeExtra;
        channelTileEnd_ = channelTileBegin_ + td->baseChannelsPerCore +
                          (block < td->extraChannels ? EXTRA_WORK_ITEM_COUNT : 0);
        if (channelTileEnd_ > totalChannelTiles_) {
            channelTileEnd_ = totalChannelTiles_;
        }

        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), td->totalInputElements);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), td->totalOutputElements);
        if (channelTileBegin_ >= channelTileEnd_) {
            return;
        }
        pipe_->InitBuffer(outputChwBuf_, td->outBufferBytes);
        pipe_->InitBuffer(inputChwBuf_, td->rawBufferBytes);
        pipe_->InitBuffer(inputHwcBuf_, td->indexBufferBytes);
        pipe_->InitBuffer(patchHwcBuf_, td->outWideBufferBytes);
    }

    __aicore__ inline void Process()
    {
        for (int64_t tile = channelTileBegin_; tile < channelTileEnd_; ++tile) {
            ProcessChannelTile(tile);
        }
    }

private:
    static constexpr int64_t CHANNEL_TILE = 16;
    static constexpr int64_t DATA_BLOCK_BYTES = 32;
    // DAV_2201 B32 vnchwconv consumes eight float columns at a time.  A C16
    // tile is therefore processed as two channel halves and emits two addresses
    // per column in the address table.
    static constexpr int64_t FP32_COLUMNS_PER_TRANSPOSE = 8;
    static constexpr int64_t FP32_CHANNEL_HALF_COUNT = CHANNEL_TILE / FP32_COLUMNS_PER_TRANSPOSE;
    static constexpr int64_t FP32_SECOND_HALF_INDEX = 1;
    static constexpr uint8_t SINGLE_TRANSPOSE_REPEAT = 1;
    static constexpr uint16_t FP32_DST_REPEAT_STRIDE = 2;
    static constexpr uint16_t FP32_SRC_REPEAT_STRIDE = 32;
    static constexpr int64_t EXTRA_WORK_ITEM_COUNT = 1;
    static constexpr uint16_t UNUSED_REPEAT_STRIDE = 0;

    __aicore__ inline int64_t Max(int64_t a, int64_t b) const { return a > b ? a : b; }

    __aicore__ inline int64_t Min(int64_t a, int64_t b) const { return a < b ? a : b; }

    __aicore__ inline int64_t AlignSpatial(int64_t value) const
    {
        return (value + CHANNEL_TILE - 1) / CHANNEL_TILE * CHANNEL_TILE;
    }

    __aicore__ inline void SyncMte2ToV()
    {
        event_t id = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(id);
        WaitFlag<HardEvent::MTE2_V>(id);
    }

    __aicore__ inline void SyncVToMte3()
    {
        event_t id = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(id);
        WaitFlag<HardEvent::V_MTE3>(id);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        event_t id = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(id);
        WaitFlag<HardEvent::MTE3_V>(id);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        event_t id = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(id);
        WaitFlag<HardEvent::MTE3_MTE2>(id);
    }

    __aicore__ inline void TransposeChwToHwc(LocalTensor<T>& inputChw, LocalTensor<T>& inputHwc)
    {
        if constexpr (sizeof(T) == sizeof(uint16_t)) {
            LocalTensor<half> src = inputChw.template ReinterpretCast<half>();
            LocalTensor<half> dst = inputHwc.template ReinterpretCast<half>();
            uint64_t srcList[CHANNEL_TILE];
            uint64_t dstList[CHANNEL_TILE];
            for (int64_t i = 0; i < CHANNEL_TILE; ++i) {
                srcList[i] = reinterpret_cast<uint64_t>(src[i * inputSpatialAligned_].GetPhyAddr());
                dstList[i] = reinterpret_cast<uint64_t>(dst[i * CHANNEL_TILE].GetPhyAddr());
            }
            const uint8_t repeats = static_cast<uint8_t>(inputSpatialAligned_ / CHANNEL_TILE);
            TransDataTo5HDParams params{
                false, false, repeats,
                static_cast<uint16_t>(repeats == SINGLE_TRANSPOSE_REPEAT ? UNUSED_REPEAT_STRIDE : CHANNEL_TILE),
                static_cast<uint16_t>(repeats == SINGLE_TRANSPOSE_REPEAT ? UNUSED_REPEAT_STRIDE :
                                                                           FP32_SECOND_HALF_INDEX)};
            TransDataTo5HD<half>(dstList, srcList, params);
        } else {
            LocalTensor<float> src = inputChw.template ReinterpretCast<float>();
            LocalTensor<float> dst = inputHwc.template ReinterpretCast<float>();
            for (int64_t column = 0; column < inputSpatialAligned_; column += FP32_COLUMNS_PER_TRANSPOSE) {
                uint64_t srcList[CHANNEL_TILE];
                uint64_t dstList[CHANNEL_TILE];
                for (int64_t row = 0; row < CHANNEL_TILE; ++row) {
                    srcList[row] = reinterpret_cast<uint64_t>(src[row * inputSpatialAligned_ + column].GetPhyAddr());
                }
                for (int64_t i = 0; i < FP32_COLUMNS_PER_TRANSPOSE; ++i) {
                    const int64_t dstOffset = (column + i) * CHANNEL_TILE;
                    dstList[FP32_CHANNEL_HALF_COUNT * i] = reinterpret_cast<uint64_t>(dst[dstOffset].GetPhyAddr());
                    dstList[FP32_CHANNEL_HALF_COUNT * i + FP32_SECOND_HALF_INDEX] = reinterpret_cast<uint64_t>(
                        dst[dstOffset + FP32_COLUMNS_PER_TRANSPOSE].GetPhyAddr());
                }
                TransDataTo5HDParams params{false, false, SINGLE_TRANSPOSE_REPEAT, UNUSED_REPEAT_STRIDE,
                                            UNUSED_REPEAT_STRIDE};
                TransDataTo5HD<float>(dstList, srcList, params);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TransposeHwcToChw(LocalTensor<T>& patchHwc, LocalTensor<T>& outputChw,
                                             int64_t planeRowsAligned)
    {
        if constexpr (sizeof(T) == sizeof(uint16_t)) {
            LocalTensor<half> src = patchHwc.template ReinterpretCast<half>();
            LocalTensor<half> dst = outputChw.template ReinterpretCast<half>();
            uint64_t srcList[CHANNEL_TILE];
            uint64_t dstList[CHANNEL_TILE];
            for (int64_t i = 0; i < CHANNEL_TILE; ++i) {
                srcList[i] = reinterpret_cast<uint64_t>(src[i * CHANNEL_TILE].GetPhyAddr());
                dstList[i] = reinterpret_cast<uint64_t>(dst[i * planeRowsAligned].GetPhyAddr());
            }
            const uint8_t repeats = static_cast<uint8_t>(planeRowsAligned / CHANNEL_TILE);
            TransDataTo5HDParams params{
                false, false, repeats,
                static_cast<uint16_t>(repeats == SINGLE_TRANSPOSE_REPEAT ? UNUSED_REPEAT_STRIDE :
                                                                           FP32_SECOND_HALF_INDEX),
                static_cast<uint16_t>(repeats == SINGLE_TRANSPOSE_REPEAT ? UNUSED_REPEAT_STRIDE : CHANNEL_TILE)};
            TransDataTo5HD<half>(dstList, srcList, params);
        } else {
            LocalTensor<float> src = patchHwc.template ReinterpretCast<float>();
            LocalTensor<float> dst = outputChw.template ReinterpretCast<float>();
            const uint8_t repeats = static_cast<uint8_t>(planeRowsAligned / CHANNEL_TILE);
            for (int64_t channelHalf = 0; channelHalf < FP32_CHANNEL_HALF_COUNT; ++channelHalf) {
                uint64_t srcList[CHANNEL_TILE];
                uint64_t dstList[CHANNEL_TILE];
                for (int64_t row = 0; row < CHANNEL_TILE; ++row) {
                    srcList[row] = reinterpret_cast<uint64_t>(
                        src[row * CHANNEL_TILE + channelHalf * FP32_COLUMNS_PER_TRANSPOSE].GetPhyAddr());
                }
                for (int64_t i = 0; i < FP32_COLUMNS_PER_TRANSPOSE; ++i) {
                    const int64_t dstOffset = (channelHalf * FP32_COLUMNS_PER_TRANSPOSE + i) * planeRowsAligned;
                    dstList[FP32_CHANNEL_HALF_COUNT * i] = reinterpret_cast<uint64_t>(dst[dstOffset].GetPhyAddr());
                    dstList[FP32_CHANNEL_HALF_COUNT * i + FP32_SECOND_HALF_INDEX] = reinterpret_cast<uint64_t>(
                        dst[dstOffset + FP32_COLUMNS_PER_TRANSPOSE].GetPhyAddr());
                }
                TransDataTo5HDParams params{
                    false, false, repeats,
                    static_cast<uint16_t>(repeats == SINGLE_TRANSPOSE_REPEAT ? UNUSED_REPEAT_STRIDE :
                                                                               FP32_DST_REPEAT_STRIDE),
                    static_cast<uint16_t>(repeats == SINGLE_TRANSPOSE_REPEAT ? UNUSED_REPEAT_STRIDE :
                                                                               FP32_SRC_REPEAT_STRIDE)};
                TransDataTo5HD<float>(dstList, srcList, params);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FillKernelPlane(LocalTensor<T>& patchHwc, LocalTensor<T>& inputHwc, int64_t localGroup,
                                           int64_t kernelGroup)
    {
        const int64_t kh = kernelGroup / kernelW_;
        const int64_t kw = kernelGroup - kh * kernelW_;
        const int64_t ohBegin = Max(0, padTop_ - kh);
        const int64_t ohEnd = Min(outH_, h_ + padTop_ - kh);
        const int64_t owBegin = Max(0, padLeft_ - kw);
        const int64_t owEnd = Min(outW_, w_ + padLeft_ - kw);
        if (ohBegin >= ohEnd || owBegin >= owEnd) {
            return;
        }

        const int64_t inputH = ohBegin + kh - padTop_;
        const int64_t inputW = owBegin + kw - padLeft_;
        const int64_t validRows = ohEnd - ohBegin;
        const int64_t validCols = owEnd - owBegin;
        const int64_t srcOffset = (inputH * w_ + inputW) * CHANNEL_TILE;
        const int64_t dstOffset = (localGroup * outputSpatial_ + ohBegin * outW_ + owBegin) * CHANNEL_TILE;
        constexpr int64_t blocksPerPosition = CHANNEL_TILE * sizeof(T) / DATA_BLOCK_BYTES;
        DataCopyParams params{static_cast<uint16_t>(validRows), static_cast<uint16_t>(validCols * blocksPerPosition),
                              static_cast<uint16_t>((w_ - validCols) * blocksPerPosition),
                              static_cast<uint16_t>((outW_ - validCols) * blocksPerPosition)};
        DataCopy(patchHwc[dstOffset], inputHwc[srcOffset], params);
    }

    __aicore__ inline void StorePlanes(LocalTensor<T>& outputChw, int64_t channelStart, int64_t groupStart,
                                       int64_t planeRows, int64_t planeRowsAligned)
    {
        const int64_t outputStart = channelStart * outputChannelElements_ + groupStart * outputSpatial_;
        DataCopyExtParams store{
            static_cast<uint16_t>(CHANNEL_TILE), static_cast<uint32_t>(planeRows * static_cast<int64_t>(sizeof(T))), 0,
            static_cast<uint32_t>((outputChannelElements_ - planeRows) * static_cast<int64_t>(sizeof(T))), 0};
        (void)planeRowsAligned;
        DataCopyPad(outputGm_[outputStart], outputChw, store);
    }

    __aicore__ inline void ProcessChannelTile(int64_t channelTile)
    {
        const int64_t channelStart = channelTile * CHANNEL_TILE;
        const int64_t inputSpatial = h_ * w_;
        LocalTensor<T> inputChw = inputChwBuf_.Get<T>();
        LocalTensor<T> inputHwc = inputHwcBuf_.Get<T>();
        LocalTensor<T> patchHwc = patchHwcBuf_.Get<T>();
        LocalTensor<T> outputChw = outputChwBuf_.Get<T>();

        DataCopyExtParams load{static_cast<uint16_t>(CHANNEL_TILE),
                               static_cast<uint32_t>(inputSpatial * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPadExtParams<T> noPad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(inputChw, inputGm_[channelStart * inputSpatial], load, noPad);
        SyncMte2ToV();
        TransposeChwToHwc(inputChw, inputHwc);

        const int64_t kernelArea = kernelH_ * kernelW_;
        for (int64_t groupStart = 0; groupStart < kernelArea; groupStart += groupTile_) {
            const int64_t groups = Min(groupTile_, kernelArea - groupStart);
            const int64_t planeRows = groups * outputSpatial_;
            const int64_t planeRowsAligned = AlignSpatial(planeRows);
            Duplicate<uint16_t>(patchHwc.template ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0),
                                static_cast<uint32_t>(planeRowsAligned * CHANNEL_TILE * sizeof(T) / sizeof(uint16_t)));
            PipeBarrier<PIPE_V>();
            for (int64_t localGroup = 0; localGroup < groups; ++localGroup) {
                FillKernelPlane(patchHwc, inputHwc, localGroup, groupStart + localGroup);
            }
            PipeBarrier<PIPE_V>();
            TransposeHwcToChw(patchHwc, outputChw, planeRowsAligned);
            SyncVToMte3();
            StorePlanes(outputChw, channelStart, groupStart, planeRows, planeRowsAligned);
            // outputChw is a single-buffer transpose destination.
            SyncMte3ToV();
        }
        if (channelTile + 1 < channelTileEnd_) {
            // Some cores own two C16 tiles.  The last MTE3 from the first tile
            // must be ordered before the next tile's MTE2 load even though the
            // vector destination has already been released.
            SyncMte3ToMte2();
        }
    }

private:
    TPipe* pipe_ = nullptr;
    TBuf<TPosition::VECCALC> outputChwBuf_;
    TBuf<TPosition::VECCALC> inputChwBuf_;
    TBuf<TPosition::VECCALC> inputHwcBuf_;
    TBuf<TPosition::VECCALC> patchHwcBuf_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;

    int64_t c_ = 0;
    int64_t h_ = 0;
    int64_t w_ = 0;
    int64_t kernelH_ = 0;
    int64_t kernelW_ = 0;
    int64_t padTop_ = 0;
    int64_t padLeft_ = 0;
    int64_t outH_ = 0;
    int64_t outW_ = 0;
    int64_t totalChannelTiles_ = 0;
    int64_t channelTileBegin_ = 0;
    int64_t channelTileEnd_ = 0;
    int64_t inputSpatialAligned_ = 0;
    int64_t groupTile_ = 0;
    int64_t outputChannelElements_ = 0;
    int64_t outputSpatial_ = 0;
};

} // namespace NsIm2col

#endif // EXPERIMENTAL_IM2COL_CHANNEL_TRANSPOSE_H_
