// ----------------------------------------------------------------------------
// Copyright (c) Huawei Device Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------

#ifndef MASKED_SCALE_KERNEL_ARCH35_H
#define MASKED_SCALE_KERNEL_ARCH35_H

#include "kernel_operator.h"
#include "masked_scale_tiling_data.h"
#include "masked_scale_tiling_key.h"

namespace NsMaskedScale {
using namespace AscendC;

constexpr uint32_t ALIGN_BYTES = 32U;

template <typename T>
__aicore__ inline uint32_t AlignElemCount(uint32_t count)
{
    constexpr uint32_t perBlock = ALIGN_BYTES / sizeof(T);
    return (count + perBlock - 1U) / perBlock * perBlock;
}

template <typename SelfT, typename MaskT>
class MaskedScale {
public:
    __aicore__ inline MaskedScale() {}

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR mask, GM_ADDR y, const MaskedScaleTilingData* tilingData)
    {
        ParseTilingData(tilingData);
        selfGm.SetGlobalBuffer(reinterpret_cast<__gm__ SelfT*>(self), dim0);
        maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ MaskT*>(mask), dim0);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ SelfT*>(y), dim0);

        const uint32_t selfBytes = AlignElemCount<SelfT>(ubFormer) * sizeof(SelfT);
        const uint32_t maskBytes = AlignElemCount<MaskT>(ubFormer) * sizeof(MaskT);
        const uint32_t halfBytes = AlignElemCount<half>(ubFormer) * sizeof(half);
        const uint32_t floatBytes = AlignElemCount<float>(ubFormer) * sizeof(float);

        pipe.InitBuffer(selfBuf, selfBytes);
        pipe.InitBuffer(maskBuf, maskBytes);
        pipe.InitBuffer(outBuf, selfBytes);
        pipe.InitBuffer(tmpHalfBuf, halfBytes);
        pipe.InitBuffer(tmpFloat0Buf, floatBytes);
        pipe.InitBuffer(tmpFloat1Buf, floatBytes);
        pipe.InitBuffer(tmpFloat2Buf, floatBytes);
    }

    __aicore__ inline void Process()
    {
        if (dim0 == 0U || GetBlockIdx() >= blockNum) {
            return;
        }

        const uint64_t blockOffset = static_cast<uint64_t>(GetBlockIdx()) * blockFormer;
        const uint64_t blockLen = (GetBlockIdx() == blockNum - 1U) ? (dim0 - blockOffset) : blockFormer;
        uint64_t processed = 0U;
        while (processed < blockLen) {
            const uint32_t current = (blockLen - processed > ubFormer) ? ubFormer : (blockLen - processed);
            const uint64_t gmOffset = blockOffset + processed;
            CopyIn(gmOffset, current);
            PipeBarrier<PIPE_MTE2>();
            Compute(current);
            PipeBarrier<PIPE_V>();
            CopyOut(gmOffset, current);
            processed += current;
        }
    }

private:
    __aicore__ inline void ParseTilingData(const MaskedScaleTilingData* tilingData)
    {
        dim0 = tilingData->dim0;
        blockFormer = tilingData->blockFormer;
        blockNum = tilingData->blockNum;
        ubFormer = tilingData->ubFormer;
        scaleFloat = tilingData->scaleFloat;
    }

    __aicore__ inline void CopyIn(uint64_t gmOffset, uint32_t count)
    {
        LocalTensor<SelfT> selfLocal = selfBuf.Get<SelfT>();
        LocalTensor<MaskT> maskLocal = maskBuf.Get<MaskT>();
        DataCopyExtParams selfParams{1U, static_cast<uint32_t>(count * sizeof(SelfT)), 0U, 0U, 0U};
        DataCopyExtParams maskParams{1U, static_cast<uint32_t>(count * sizeof(MaskT)), 0U, 0U, 0U};
        DataCopyPad(selfLocal, selfGm[gmOffset], selfParams, {false, 0U, 0U, 0U});
        DataCopyPad(maskLocal, maskGm[gmOffset], maskParams, {false, 0U, 0U, 0U});
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        if constexpr (std::is_same_v<SelfT, half>) {
            ComputeFp16(count);
        } else if constexpr (std::is_same_v<SelfT, float>) {
            ComputeFp32(count);
        } else {
            ComputeBf16(count);
        }
    }

    __aicore__ inline void CopyOut(uint64_t gmOffset, uint32_t count)
    {
        LocalTensor<SelfT> outLocal = outBuf.Get<SelfT>();
        DataCopyExtParams outParams{1U, static_cast<uint32_t>(count * sizeof(SelfT)), 0U, 0U, 0U};
        DataCopyPad(yGm[gmOffset], outLocal, outParams);
    }

    __aicore__ inline LocalTensor<half> MaskAsHalf(uint32_t count)
    {
        LocalTensor<MaskT> maskLocal = maskBuf.Get<MaskT>();
        if constexpr (std::is_same_v<MaskT, half>) {
            return maskLocal.template ReinterpretCast<half>();
        } else {
            LocalTensor<half> maskHalf = tmpHalfBuf.Get<half>();
            if constexpr (std::is_same_v<MaskT, float>) {
                Cast(maskHalf, maskLocal, RoundMode::CAST_ROUND, AlignElemCount<half>(count));
            } else {
                Cast(maskHalf, maskLocal, RoundMode::CAST_NONE, AlignElemCount<half>(count));
            }
            return maskHalf;
        }
    }

    __aicore__ inline LocalTensor<float> MaskAsFloat(uint32_t count)
    {
        LocalTensor<MaskT> maskLocal = maskBuf.Get<MaskT>();
        if constexpr (std::is_same_v<MaskT, float>) {
            return maskLocal.template ReinterpretCast<float>();
        } else {
            LocalTensor<float> maskFloat = tmpFloat1Buf.Get<float>();
            if constexpr (std::is_same_v<MaskT, half>) {
                Cast(maskFloat, maskLocal, RoundMode::CAST_NONE, AlignElemCount<float>(count));
            } else {
                LocalTensor<half> maskHalf = tmpHalfBuf.Get<half>();
                Cast(maskHalf, maskLocal, RoundMode::CAST_ROUND, AlignElemCount<half>(count));
                Cast(maskFloat, maskHalf, RoundMode::CAST_NONE, AlignElemCount<float>(count));
            }
            return maskFloat;
        }
    }

    __aicore__ inline void ComputeFp16(uint32_t count)
    {
        LocalTensor<half> selfLocal = selfBuf.Get<half>();
        LocalTensor<half> outLocal = outBuf.Get<half>();
        LocalTensor<half> maskHalf = MaskAsHalf(count);
        const uint32_t alignedCount = AlignElemCount<half>(count);
        Mul(outLocal, selfLocal, maskHalf, alignedCount);
        Muls(outLocal, outLocal, static_cast<half>(scaleFloat), alignedCount);
    }

    __aicore__ inline void ComputeFp32(uint32_t count)
    {
        LocalTensor<float> selfLocal = selfBuf.Get<float>();
        LocalTensor<float> outLocal = outBuf.Get<float>();
        LocalTensor<float> maskFloat = MaskAsFloat(count);
        const uint32_t alignedCount = AlignElemCount<float>(count);
        Mul(outLocal, selfLocal, maskFloat, alignedCount);
        Muls(outLocal, outLocal, scaleFloat, alignedCount);
    }

    __aicore__ inline void ComputeBf16(uint32_t count)
    {
        LocalTensor<SelfT> selfLocal = selfBuf.Get<SelfT>();
        LocalTensor<SelfT> outLocal = outBuf.Get<SelfT>();
        LocalTensor<float> selfFloat = tmpFloat0Buf.Get<float>();
        LocalTensor<float> maskFloat = MaskAsFloat(count);
        LocalTensor<float> outFloat = tmpFloat2Buf.Get<float>();
        const uint32_t alignedFloatCount = AlignElemCount<float>(count);
        Cast(selfFloat, selfLocal, RoundMode::CAST_NONE, alignedFloatCount);
        Mul(outFloat, selfFloat, maskFloat, alignedFloatCount);
        Muls(outFloat, outFloat, scaleFloat, alignedFloatCount);
        Cast(outLocal, outFloat, RoundMode::CAST_ROUND, AlignElemCount<SelfT>(count));
    }

private:
    TPipe pipe;
    GlobalTensor<SelfT> selfGm;
    GlobalTensor<MaskT> maskGm;
    GlobalTensor<SelfT> yGm;
    TBuf<TPosition::VECCALC> selfBuf;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> outBuf;
    TBuf<TPosition::VECCALC> tmpHalfBuf;
    TBuf<TPosition::VECCALC> tmpFloat0Buf;
    TBuf<TPosition::VECCALC> tmpFloat1Buf;
    TBuf<TPosition::VECCALC> tmpFloat2Buf;

    uint64_t dim0{0U};
    uint32_t blockFormer{0U};
    uint32_t blockNum{0U};
    uint32_t ubFormer{0U};
    float scaleFloat{0.0f};
};
} // namespace NsMaskedScale

#endif // MASKED_SCALE_KERNEL_ARCH35_H
