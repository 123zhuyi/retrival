/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#ifndef __ERF_INV_H__
#define __ERF_INV_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "erf_inv_tiling_data.h"
#include "erf_inv_tiling_key.h"
#include <type_traits>

namespace NsErfInv {

// using namespace inside NsErfInv{} — scope is limited to this namespace block,
// consistent with other ops-math operators.
using namespace AscendC;

template <typename T>
class KernelErfInv {
public:
    __aicore__ inline KernelErfInv() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ErfInvTilingData* tilingData, TPipe* pipePtr)
    {
        this->pipe = pipePtr;
        uint32_t baseElems = tilingData->baseElems;
        uint32_t pivot = tilingData->pivot;
        this->tileSize = tilingData->tileSize;

        // Pivot distribution
        uint32_t blockId = GetBlockIdx();
        this->myElems = baseElems + (blockId < pivot ? 1 : 0);
        this->innerLoops = (this->myElems + this->tileSize - 1) / this->tileSize;
        uint32_t myStart = blockId * baseElems + (blockId < pivot ? blockId : pivot);

        xGm.SetGlobalBuffer((__gm__ T*)x + myStart, this->myElems);
        yGm.SetGlobalBuffer((__gm__ T*)y + myStart, this->myElems);

        // I/O queues in T (for GM transfer)
        uint32_t ioBytes = tileSize * sizeof(T);
        pipe->InitBuffer(inQueue, 2, ioBytes);
        pipe->InitBuffer(outQueue, 2, ioBytes);

        // Float32 compute buffers: Cast block alignment for safety
        uint32_t castAligned = ((tileSize + 15u) / 16u) * 16u * sizeof(float);
        uint32_t padAligned = ((tileSize * sizeof(float) + 31u) / 32u) * 32u;
        uint32_t f32Bytes = (castAligned > padAligned) ? castAligned : padAligned;

        pipe->InitBuffer(xFloatBuf, f32Bytes);
        pipe->InitBuffer(aBuf, f32Bytes);
        pipe->InitBuffer(wBuf, f32Bytes);
        pipe->InitBuffer(r1Buf, f32Bytes);
        pipe->InitBuffer(r2Buf, f32Bytes);
        pipe->InitBuffer(tmpBuf, f32Bytes);
        pipe->InitBuffer(outFloatBuf, f32Bytes);

        // Mask buffer: 1 bit per element, aligned to 32 bytes
        uint32_t maskBytes = ((tileSize / 8 + 31) / 32) * 32;
        pipe->InitBuffer(maskBuf, maskBytes);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->innerLoops; i++) {
            uint32_t curTile = this->tileSize;
            if (i == this->innerLoops - 1) {
                curTile = this->myElems - i * this->tileSize;
            }
            CopyIn(i, curTile);
            Compute(i, curTile);
            CopyOut(i, curTile);
        }
    }

private:
    // Cast input T → fp32 (widening). Mode selection rationale:
    //   - T == float: Adds(·, 0.0f) as a reliable UB-to-UB copy (pure DataCopy or
    //     Cast with same src/dst type may silently no-op between buffer positions
    //     on Ascend hardware).
    //   - T == half:  CAST_NONE. fp16 mantissa + exponent fit exactly in fp32, so
    //     no rounding is needed. CAST_RINT was tested on ascend910_93 and produced
    //     garbage for all fp16 shapes, so we explicitly choose CAST_NONE.
    template <typename U>
    __aicore__ inline void CastInput(LocalTensor<float>& dst, LocalTensor<U>& src, uint32_t count)
    {
        if constexpr (std::is_same_v<U, float>) {
            Adds(dst, src, 0.0f, count);
        } else {
            Cast(dst, src, RoundMode::CAST_NONE, count);
        }
    }

    // Cast fp32 → T (narrowing). Mode selection rationale:
    //   - T == float: Adds(·, 0.0f) as a reliable UB-to-UB copy (see CastInput).
    //   - T == half:  CAST_ROUND (IEEE round-to-nearest-even on narrowing).
    template <typename U>
    __aicore__ inline void CastOutput(LocalTensor<U>& dst, LocalTensor<float>& src, uint32_t count)
    {
        if constexpr (std::is_same_v<U, float>) {
            Adds(dst, src, 0.0f, count);
        } else {
            Cast(dst, src, RoundMode::CAST_ROUND, count);
        }
    }

    // Horner polynomial evaluation: result = c[0]·xⁿ + c[1]·xⁿ⁻¹ + ... + c[n].
    // Rewritten as ((((c[0]·x + c[1])·x + c[2])·x + ...)·x + c[N-1]) for minimal
    // muls/adds. Caller provides the *shifted* input in x; result must be a
    // separate tensor from x (not in-place).
    template <size_t N>
    __aicore__ inline void HornerEval(
        const LocalTensor<float>& result, const LocalTensor<float>& x, const float (&coeffs)[N], uint32_t count)
    {
        Duplicate(result, coeffs[0], count);
        for (size_t i = 1; i < N; i++) {
            Mul(result, x, result, count);
            Adds(result, result, coeffs[i], count);
        }
    }

    __aicore__ inline void CopyIn(uint32_t idx, uint32_t curTile)
    {
        LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
        uint32_t offset = idx * this->tileSize;
        uint32_t blockLen = curTile * sizeof(T);
        DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t idx, uint32_t curTile)
    {
        LocalTensor<T> xLocal = inQueue.DeQue<T>();
        LocalTensor<T> yLocalOut = outQueue.AllocTensor<T>();

        // Process full tile in vector ops: hardware semantics for variable count
        // are fragile when count is small or unaligned. CopyOut only writes curTile
        // elements, so extra garbage past curTile is harmless.
        uint32_t count = this->tileSize;

        // Get float32 compute tensors
        LocalTensor<float> xF = xFloatBuf.Get<float>();
        LocalTensor<float> yF = outFloatBuf.Get<float>();

        // Cast input to float32
        CastInput(xF, xLocal, count);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> aLocal = aBuf.Get<float>();
        LocalTensor<float> wLocal = wBuf.Get<float>();
        LocalTensor<float> r1Local = r1Buf.Get<float>();
        LocalTensor<float> r2Local = r2Buf.Get<float>();
        LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();

        // Step 1: a = |x|, clamped away from 1 to avoid log(0) downstream.
        Abs(aLocal, xF, count);
        Mins(aLocal, aLocal, 0.999999940f, count);

        // Step 2: w = -ln(1 - a²), computed as -ln((1-a)(1+a)) to avoid catastrophic
        // cancellation when |a|→1. Clamp (1-a)(1+a) to 1e-30 before log to keep finite.
        Muls(tmpLocal, aLocal, -1.0f, count);
        Adds(tmpLocal, tmpLocal, 1.0f, count);
        Adds(wLocal, aLocal, 1.0f, count);
        Mul(wLocal, tmpLocal, wLocal, count);
        Maxs(wLocal, wLocal, 1.0e-30f, count);
        Log(wLocal, wLocal, count);
        Muls(wLocal, wLocal, -1.0f, count);

        // Region 1 (w ≤ 5): r1 = P1(w - 2.5), degree-8 Horner polynomial.
        // Coefficients from Mike Giles (2012), "Approximating the erfinv function".
        constexpr float REGION1_COEFFS[] = {2.81022636e-08f,  3.43273939e-07f, -3.52338770e-06f,
                                            -4.39150654e-06f, 2.18580870e-04f, -1.25372503e-03f,
                                            -4.17768164e-03f, 2.46640727e-01f, 1.50140941e+00f};
        Adds(tmpLocal, wLocal, -2.5f, count); // shifted input
        HornerEval(r1Local, tmpLocal, REGION1_COEFFS, count);

        // Region 2 (w > 5): r2 = P2(sqrt(w) - 3.0), degree-8 Horner polynomial.
        constexpr float REGION2_COEFFS[] = {-2.00214257e-04f, 1.00950558e-04f, 1.34934322e-03f,
                                            -3.67342844e-03f, 5.73950773e-03f, -7.62246130e-03f,
                                            9.43887047e-03f,  1.00167406e+00f, 2.83297682e+00f};
        Sqrt(r2Local, wLocal, count);
        Adds(tmpLocal, r2Local, -3.0f, count); // shifted input
        HornerEval(r2Local, tmpLocal, REGION2_COEFFS, count);

        // ========== Blend regions: out = (w > 5.0) ? r2 : r1 ==========
        CompareScalar(maskLocal, wLocal, 5.0f, CMPMODE::GT, count);
        Select(yF, maskLocal, r2Local, r1Local, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);

        // ========== Apply sign: result = polynomial * x ==========
        Mul(yF, yF, xF, count);

        // Cast output back to T
        CastOutput(yLocalOut, yF, count);

        outQueue.EnQue(yLocalOut);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx, uint32_t curTile)
    {
        LocalTensor<T> yLocal = outQueue.DeQue<T>();
        uint32_t offset = idx * this->tileSize;
        uint32_t blockLen = curTile * sizeof(T);
        DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
        DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    TBuf<TPosition::VECCALC> xFloatBuf;
    TBuf<TPosition::VECCALC> aBuf;
    TBuf<TPosition::VECCALC> wBuf;
    TBuf<TPosition::VECCALC> r1Buf;
    TBuf<TPosition::VECCALC> r2Buf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> outFloatBuf;
    TBuf<TPosition::VECCALC> maskBuf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    uint32_t myElems;
    uint32_t tileSize;
    uint32_t innerLoops;
};

} // namespace NsErfInv
#endif // __ERF_INV_H__
