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
 * \file addr_v2.cpp
 * \brief addr_v2 kernel (arch22 / Ascend910B)
 *
 * Design ref: DESIGN.md §6 (architecture), §9 (kernel design)
 *   - 3 分支 tilingKey 分派（§6.2）: 0=WithoutAlpha, 1=WithoutBeta, 2=WithBetaAlpha
 *   - 4 dtype 类别模板分派（§4.2）: Common(float)/Bf16Fp16/Int8/Uint8
 *   - 行级 Muls 外积（§6.3）: outer[i][tile] = vec1[i] * vec2[tile]
 *   - INT8/UINT8/BOOL float32 域 mod256 包裹（§6.7）
 *   - 多核切分 N 维 former/tail（§6.4），UB 切分 M 维 tileM
 *   - 列 tile 外层、行内层循环（§6.6）: vec2 每个 tile 加载一次跨行复用
 *
 * Algorithm ported from verified .asc direct-invoke implementation (_asc_reference/addr_v2_kernel.asc).
 */

#include "kernel_operator.h"
#include "arch22/addr_v2_struct.h"

using namespace AscendC;

// ============================================================================
// dtype 类型判定（编译期，用于 if constexpr 分派）
// BOOL 底层为 int8_t，与 INT8 共用计算路径（与 A5 一致）
// ============================================================================
template <typename T>
struct AddrV2TypeTraits;
template <>
struct AddrV2TypeTraits<float> {
    static constexpr bool IsFloat = true;
    static constexpr bool IsFp16Bf16 = false;
    static constexpr bool IsIntU8 = false;
    static constexpr uint32_t SizeOf = 4;
};
template <>
struct AddrV2TypeTraits<half> {
    static constexpr bool IsFloat = false;
    static constexpr bool IsFp16Bf16 = true;
    static constexpr bool IsIntU8 = false;
    static constexpr uint32_t SizeOf = 2;
};
template <>
struct AddrV2TypeTraits<bfloat16_t> {
    static constexpr bool IsFloat = false;
    static constexpr bool IsFp16Bf16 = true;
    static constexpr bool IsIntU8 = false;
    static constexpr uint32_t SizeOf = 2;
};
template <>
struct AddrV2TypeTraits<int8_t> {
    static constexpr bool IsFloat = false;
    static constexpr bool IsFp16Bf16 = false;
    static constexpr bool IsIntU8 = true;
    static constexpr uint32_t SizeOf = 1;
};
template <>
struct AddrV2TypeTraits<uint8_t> {
    static constexpr bool IsFloat = false;
    static constexpr bool IsFp16Bf16 = false;
    static constexpr bool IsIntU8 = true;
    static constexpr uint32_t SizeOf = 1;
};
template <>
struct AddrV2TypeTraits<bool> {
    static constexpr bool IsFloat = false;
    static constexpr bool IsFp16Bf16 = false;
    static constexpr bool IsIntU8 = true;
    static constexpr uint32_t SizeOf = 1;
};

// ============================================================================
// Kernel 类 - 封装 NPU 上的计算逻辑
// ============================================================================
template <typename D_T_X>
class AddrV2Kernel {
public:
    // BOOL 底层为 int8_t，所有 tensor 操作使用 int8_t
    using T = typename std::conditional<std::is_same<D_T_X, bool>::value, int8_t, D_T_X>::type;

    __aicore__ inline AddrV2Kernel() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, const AddrV2TilingData* tiling)
    {
        blockIdx_ = GetBlockIdx();

        // 解析 TilingData（依据 §7）
        N_ = tiling->totalRows;
        M_ = tiling->totalCols;
        tileM_ = tiling->tileM;
        tileMLoop_ = tiling->tileMLoop;
        tileMTail_ = tiling->tileMTail;
        tilingKey_ = tiling->tilingKey;
        selfBcastMode_ = tiling->selfBroadcastMode;
        betaFp32_ = tiling->betaValue;
        alphaFp32_ = tiling->alphaValue;

        // 多核行范围（former/tail，依据 §6.4）
        uint32_t formerNum = tiling->formerNum;
        uint32_t formerRows = tiling->formerRows;
        uint32_t tailRows = tiling->tailRows;
        if (blockIdx_ < formerNum) {
            startRow_ = blockIdx_ * formerRows;
            endRow_ = startRow_ + formerRows;
        } else {
            startRow_ = formerNum * formerRows + (blockIdx_ - formerNum) * tailRows;
            endRow_ = startRow_ + tailRows;
        }

        // 设置 Global Tensor: x1=self, x2=vec1, x3=vec2, y=out
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        x3Gm.SetGlobalBuffer((__gm__ T*)x3);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        InitBuffers();
    }

    __aicore__ inline void InitBuffers()
    {
        constexpr uint32_t dtypeSize = AddrV2TypeTraits<T>::SizeOf;
        constexpr bool isFloat = AddrV2TypeTraits<T>::IsFloat;
        constexpr bool isIntU8 = AddrV2TypeTraits<T>::IsIntU8;

        bool needVec2 = (tilingKey_ != ADDR_V2_TILING_KEY_WITHOUT_ALPHA); // key 1,2
        bool needSelf = (tilingKey_ != ADDR_V2_TILING_KEY_WITHOUT_BETA);  // key 0,2

        uint32_t tileBytes = tileM_ * dtypeSize;
        uint32_t fp32Bytes = tileM_ * sizeof(float);
        uint32_t halfBytes = tileM_ * sizeof(half);

        // 输出 TQue（double buffer）
        pipe.InitBuffer(outQueueY, 2, tileBytes);

        // vec2 输入 TBuf（每个 tile 加载一次，跨行复用）
        if (needVec2) {
            pipe.InitBuffer(vec2InBuf, tileBytes);
            if constexpr (!isFloat) {
                pipe.InitBuffer(vec2Fp32Buf, fp32Bytes);
                pipe.InitBuffer(outerFp32Buf, fp32Bytes);
            }
        }

        // self 输入 TBuf
        if (needSelf) {
            pipe.InitBuffer(selfInBuf, tileBytes);
            if constexpr (!isFloat) {
                pipe.InitBuffer(selfFp32Buf, fp32Bytes);
            }
        }

        // fp32 结果 TBuf（非 float 类型）
        if constexpr (!isFloat) {
            pipe.InitBuffer(resultFp32Buf, fp32Bytes);
        }

        // int8/uint8 mod256 和 half 中转 TBuf
        if constexpr (isIntU8) {
            pipe.InitBuffer(modBufA, fp32Bytes);
            pipe.InitBuffer(modBufB, fp32Bytes);
            pipe.InitBuffer(outHalfBuf, halfBytes);
        }

        // 标量加载 TBuf
        pipe.InitBuffer(scalarInBuf, ADDR_V2_SCALAR_BUF_BYTES);
        if constexpr (isIntU8) {
            pipe.InitBuffer(scalarMidBuf, ADDR_V2_SCALAR_BUF_BYTES);
        }
        if constexpr (!isFloat) {
            pipe.InitBuffer(scalarFp32Buf, ADDR_V2_SCALAR_BUF_BYTES);
        }
    }

    // ========================================================================
    // 主计算流程（依据 §9.2）
    // ========================================================================
    __aicore__ inline void Process()
    {
        // 列 tile 外层循环（vec2 每个 tile 加载一次跨行复用，依据 §6.6）
        for (uint32_t tileIdx = 0; tileIdx < tileMLoop_; tileIdx++) {
            uint32_t curTileM = (tileIdx == tileMLoop_ - 1) ? tileMTail_ : tileM_;

            // 加载 vec2 tile（AddrV2WithoutAlpha 不需要 vec2）
            if (tilingKey_ != ADDR_V2_TILING_KEY_WITHOUT_ALPHA) {
                LoadVec2Tile(tileIdx, curTileM);
            }

            // 行内层循环
            for (uint32_t rowIdx = startRow_; rowIdx < endRow_; rowIdx++) {
                switch (tilingKey_) {
                    case ADDR_V2_TILING_KEY_WITHOUT_ALPHA:
                        ProcessAddrV2WithoutAlpha(rowIdx, tileIdx, curTileM);
                        break;
                    case ADDR_V2_TILING_KEY_WITHOUT_BETA:
                        ProcessAddrV2WithoutBeta(rowIdx, tileIdx, curTileM);
                        break;
                    case ADDR_V2_TILING_KEY_WITH_BETA_ALPHA:
                        ProcessAddrV2WithBetaWithAlpha(rowIdx, tileIdx, curTileM);
                        break;
                    default:
                        // 断言兜底（依据 DESIGN.md §6.2）
                        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Invalid tilingKey"); });
                        break;
                }
            }
        }
    }

private:
    // ========================================================================
    // 标量加载: vec1[rowIdx] → float（依据 §6.6 标量加载子流程）
    // ========================================================================
    __aicore__ inline float LoadVec1Scalar(uint32_t rowIdx)
    {
        auto scalarLocal = scalarInBuf.Get<T>();
        DataCopyExtParams copyParams{1, (uint32_t)AddrV2TypeTraits<T>::SizeOf, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(scalarLocal, x2Gm[static_cast<uint64_t>(rowIdx)], copyParams, padParams);
        // TBuf 无 EnQue/DeQue 同步，需手动 PipeBarrier<PIPE_ALL> 确保 MTE2→V 跨流水线同步
        // 注意: PipeBarrier<PIPE_MTE2> 仅同步 MTE2 管线内部，不提供 MTE2→V 跨管线同步
        PipeBarrier<PIPE_ALL>();

        if constexpr (AddrV2TypeTraits<T>::IsFloat) {
            return scalarLocal.GetValue(0);
        } else if constexpr (AddrV2TypeTraits<T>::IsFp16Bf16) {
            auto fp32Local = scalarFp32Buf.Get<float>();
            Cast<float, T>(fp32Local, scalarLocal, RoundMode::CAST_NONE, 1);
            return fp32Local.GetValue(0);
        } else {
            auto halfLocal = scalarMidBuf.Get<half>();
            Cast<half, T>(halfLocal, scalarLocal, RoundMode::CAST_NONE, 1);
            auto fp32Local = scalarFp32Buf.Get<float>();
            Cast<float, half>(fp32Local, halfLocal, RoundMode::CAST_NONE, 1);
            return fp32Local.GetValue(0);
        }
    }

    // ========================================================================
    // 标量加载: self → float（广播模式 2/3，依据 §8.2）
    // ========================================================================
    __aicore__ inline float LoadSelfScalar(uint32_t rowIdx)
    {
        uint64_t gmOffset = (selfBcastMode_ == ADDR_V2_BCAST_SCALAR) ? 0ULL : static_cast<uint64_t>(rowIdx);
        auto scalarLocal = scalarInBuf.Get<T>();
        DataCopyExtParams copyParams{1, (uint32_t)AddrV2TypeTraits<T>::SizeOf, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(scalarLocal, x1Gm[gmOffset], copyParams, padParams);
        // TBuf 无 EnQue/DeQue 同步，需手动 PipeBarrier<PIPE_ALL> 确保 MTE2→V 跨流水线同步
        PipeBarrier<PIPE_ALL>();

        if constexpr (AddrV2TypeTraits<T>::IsFloat) {
            return scalarLocal.GetValue(0);
        } else if constexpr (AddrV2TypeTraits<T>::IsFp16Bf16) {
            auto fp32Local = scalarFp32Buf.Get<float>();
            Cast<float, T>(fp32Local, scalarLocal, RoundMode::CAST_NONE, 1);
            return fp32Local.GetValue(0);
        } else {
            auto halfLocal = scalarMidBuf.Get<half>();
            Cast<half, T>(halfLocal, scalarLocal, RoundMode::CAST_NONE, 1);
            auto fp32Local = scalarFp32Buf.Get<float>();
            Cast<float, half>(fp32Local, halfLocal, RoundMode::CAST_NONE, 1);
            return fp32Local.GetValue(0);
        }
    }

    // ========================================================================
    // 加载 vec2 tile（依据 §6.6 Phase 1）
    // ========================================================================
    __aicore__ inline void LoadVec2Tile(uint32_t tileIdx, uint32_t curTileM)
    {
        auto vec2Local = vec2InBuf.Get<T>();
        DataCopyExtParams copyParams{1, (uint32_t)(curTileM * AddrV2TypeTraits<T>::SizeOf), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(vec2Local, x3Gm[static_cast<uint64_t>(tileIdx) * tileM_], copyParams, padParams);
        // TBuf 无 EnQue/DeQue 同步，需手动 PipeBarrier<PIPE_ALL> 确保 MTE2→V 跨流水线同步
        PipeBarrier<PIPE_ALL>();

        if constexpr (AddrV2TypeTraits<T>::IsFp16Bf16) {
            auto vec2Fp32 = vec2Fp32Buf.Get<float>();
            Cast<float, T>(vec2Fp32, vec2Local, RoundMode::CAST_NONE, curTileM);
        } else if constexpr (AddrV2TypeTraits<T>::IsIntU8) {
            // int8/uint8: T → half → float (依据 §6.7, A2 不支持 int8↔float 直接 Cast)
            auto halfMid = outHalfBuf.Get<half>();
            Cast<half, T>(halfMid, vec2Local, RoundMode::CAST_NONE, curTileM);
            auto vec2Fp32 = vec2Fp32Buf.Get<float>();
            Cast<float, half>(vec2Fp32, halfMid, RoundMode::CAST_NONE, curTileM);
        }
    }

    // ========================================================================
    // 准备 self fp32 数据（所有广播模式，依据 §8.2）
    // ========================================================================
    __aicore__ inline void PrepareSelf(uint32_t rowIdx, uint32_t tileIdx, uint32_t curTileM)
    {
        if (selfBcastMode_ == ADDR_V2_BCAST_NONE || selfBcastMode_ == ADDR_V2_BCAST_ROW) {
            // 模式 0/1: 整行加载
            auto selfLocal = selfInBuf.Get<T>();
            uint32_t rowOffset = (selfBcastMode_ == ADDR_V2_BCAST_ROW) ? 0 : rowIdx;
            uint64_t gmOffset = static_cast<uint64_t>(rowOffset) * M_ + static_cast<uint64_t>(tileIdx) * tileM_;
            DataCopyExtParams copyParams{1, (uint32_t)(curTileM * AddrV2TypeTraits<T>::SizeOf), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(selfLocal, x1Gm[gmOffset], copyParams, padParams);
            // TBuf 无 EnQue/DeQue 同步，需手动 PipeBarrier<PIPE_ALL> 确保 MTE2→V 跨流水线同步
            PipeBarrier<PIPE_ALL>();

            if constexpr (AddrV2TypeTraits<T>::IsFp16Bf16) {
                auto selfFp32 = selfFp32Buf.Get<float>();
                Cast<float, T>(selfFp32, selfLocal, RoundMode::CAST_NONE, curTileM);
            } else if constexpr (AddrV2TypeTraits<T>::IsIntU8) {
                auto halfMid = outHalfBuf.Get<half>();
                Cast<half, T>(halfMid, selfLocal, RoundMode::CAST_NONE, curTileM);
                auto selfFp32 = selfFp32Buf.Get<float>();
                Cast<float, half>(selfFp32, halfMid, RoundMode::CAST_NONE, curTileM);
            }
        } else {
            // 模式 2/3: 标量加载 → Duplicate
            float selfScalar = LoadSelfScalar(rowIdx);
            if constexpr (AddrV2TypeTraits<T>::IsFloat) {
                auto selfLocal = selfInBuf.Get<float>();
                Duplicate<float>(selfLocal, selfScalar, (int32_t)curTileM);
            } else {
                auto selfFp32 = selfFp32Buf.Get<float>();
                Duplicate<float>(selfFp32, selfScalar, (int32_t)curTileM);
            }
        }
    }

    // ========================================================================
    // INT8/UINT8/BOOL mod256 包裹（依据 §6.7）
    // ========================================================================
    __aicore__ inline void ApplyMod256(uint32_t curTileM)
    {
        auto resultFp32 = resultFp32Buf.Get<float>();
        auto modA = modBufA.Get<float>();
        auto modB = modBufB.Get<float>();
        auto modBInt32 = modBufB.Get<int32_t>();

        // modBufA = result / 256
        Muls<float>(modA, resultFp32, 1.0f / 256.0f, (int32_t)curTileM);
        // modBufB(int32) = floor(result / 256)
        Cast<int32_t, float>(modBInt32, modA, RoundMode::CAST_FLOOR, curTileM);
        // modBufA(float) = floor(result / 256)
        Cast<float, int32_t>(modA, modBInt32, RoundMode::CAST_RINT, curTileM);
        // modBufB = -256 * floor(result / 256)
        Muls<float>(modB, modA, -256.0f, (int32_t)curTileM);
        // modBufA = result - 256*floor ∈ [0, 255]
        Add<float>(modA, resultFp32, modB, (int32_t)curTileM);

        // BOOL 归一化: mod256 结果 ∈ [0, 255]，bool 语义要求 0=false(0), 非零=true(1)。
        // 原 A5 在 op_api 层用 bool→int8→bool Cast roundtrip 归一化，但 A2 custom opp
        // 中 bool Cast kernel 不可用 (561103)。改为在 kernel 内用 Mins 钳位到 [0,1]。
        if constexpr (std::is_same<D_T_X, bool>::value) {
            Mins<float>(modA, modA, 1.0f, (int32_t)curTileM);
        }
    }

    // ========================================================================
    // 输出: fp32 结果 → T 并写回 GM
    // ========================================================================
    __aicore__ inline void CastResultAndCopyOut(uint32_t rowIdx, uint32_t tileIdx, uint32_t curTileM)
    {
        auto outLocal = outQueueY.AllocTensor<T>();

        if constexpr (AddrV2TypeTraits<T>::IsFp16Bf16) {
            auto resultFp32 = resultFp32Buf.Get<float>();
            Cast<T, float>(outLocal, resultFp32, RoundMode::CAST_RINT, curTileM);
        } else if constexpr (AddrV2TypeTraits<T>::IsIntU8) {
            // mod256 结果在 modBufA (float) → half → T
            auto modA = modBufA.Get<float>();
            auto outHalf = outHalfBuf.Get<half>();
            Cast<half, float>(outHalf, modA, RoundMode::CAST_RINT, curTileM);
            auto outU8 = outLocal.template ReinterpretCast<uint8_t>();
            Cast<uint8_t, half>(outU8, outHalf, RoundMode::CAST_RINT, curTileM);
        }

        outQueueY.EnQue(outLocal);
        auto outDeq = outQueueY.DeQue<T>();

        uint64_t gmOffset = static_cast<uint64_t>(rowIdx) * M_ + static_cast<uint64_t>(tileIdx) * tileM_;
        DataCopyExtParams copyParams{1, (uint32_t)(curTileM * AddrV2TypeTraits<T>::SizeOf), 0, 0, 0};
        DataCopyPad(yGm[gmOffset], outDeq, copyParams);
        outQueueY.FreeTensor(outDeq);
    }

    // ========================================================================
    // 输出 float 结果写回 GM（float 专用）
    // ========================================================================
    __aicore__ inline void CopyOutFloat(LocalTensor<float>& outLocal, uint32_t rowIdx, uint32_t tileIdx,
                                        uint32_t curTileM)
    {
        outQueueY.EnQue(outLocal);
        auto outDeq = outQueueY.DeQue<float>();
        uint64_t gmOffset = static_cast<uint64_t>(rowIdx) * M_ + static_cast<uint64_t>(tileIdx) * tileM_;
        DataCopyExtParams copyParams{1, (uint32_t)(curTileM * sizeof(float)), 0, 0, 0};
        DataCopyPad(yGm[gmOffset], outDeq, copyParams);
        outQueueY.FreeTensor(outDeq);
    }

    // ========================================================================
    // 分支 0: AddrV2WithoutAlpha (out = beta * self)
    // ========================================================================
    __aicore__ inline void ProcessAddrV2WithoutAlpha(uint32_t rowIdx, uint32_t tileIdx, uint32_t curTileM)
    {
        if constexpr (AddrV2TypeTraits<T>::IsFloat) {
            PrepareSelf(rowIdx, tileIdx, curTileM);
            auto selfLocal = selfInBuf.Get<float>();
            auto outLocal = outQueueY.AllocTensor<float>();
            Muls<float>(outLocal, selfLocal, betaFp32_, (int32_t)curTileM);
            CopyOutFloat(outLocal, rowIdx, tileIdx, curTileM);
        } else {
            PrepareSelf(rowIdx, tileIdx, curTileM);
            auto selfFp32 = selfFp32Buf.Get<float>();
            auto resultFp32 = resultFp32Buf.Get<float>();
            Muls<float>(resultFp32, selfFp32, betaFp32_, (int32_t)curTileM);

            if constexpr (AddrV2TypeTraits<T>::IsIntU8) {
                ApplyMod256(curTileM);
            }
            CastResultAndCopyOut(rowIdx, tileIdx, curTileM);
        }
    }

    // ========================================================================
    // 分支 1: AddrV2WithoutBeta (out = alpha * vec1[rowIdx] * vec2[tile])
    // ========================================================================
    __aicore__ inline void ProcessAddrV2WithoutBeta(uint32_t rowIdx, uint32_t tileIdx, uint32_t curTileM)
    {
        float vec1Scalar = LoadVec1Scalar(rowIdx);
        float vec1TimesAlpha = vec1Scalar * alphaFp32_;

        if constexpr (AddrV2TypeTraits<T>::IsFloat) {
            auto vec2Local = vec2InBuf.Get<float>();
            auto outLocal = outQueueY.AllocTensor<float>();
            Muls<float>(outLocal, vec2Local, vec1TimesAlpha, (int32_t)curTileM);
            CopyOutFloat(outLocal, rowIdx, tileIdx, curTileM);
        } else {
            auto vec2Fp32 = vec2Fp32Buf.Get<float>();
            auto resultFp32 = resultFp32Buf.Get<float>();
            Muls<float>(resultFp32, vec2Fp32, vec1TimesAlpha, (int32_t)curTileM);

            if constexpr (AddrV2TypeTraits<T>::IsIntU8) {
                ApplyMod256(curTileM);
            }
            CastResultAndCopyOut(rowIdx, tileIdx, curTileM);
        }
    }

    // ========================================================================
    // 分支 2: AddrV2WithBetaWithAlpha (out = beta * self + alpha * vec1 * vec2)
    // ========================================================================
    __aicore__ inline void ProcessAddrV2WithBetaWithAlpha(uint32_t rowIdx, uint32_t tileIdx, uint32_t curTileM)
    {
        float vec1Scalar = LoadVec1Scalar(rowIdx);
        float vec1TimesAlpha = vec1Scalar * alphaFp32_;

        if constexpr (AddrV2TypeTraits<T>::IsFloat) {
            PrepareSelf(rowIdx, tileIdx, curTileM);
            auto selfLocal = selfInBuf.Get<float>();
            auto vec2Local = vec2InBuf.Get<float>();
            auto outLocal = outQueueY.AllocTensor<float>();

            // outLocal = beta * self
            Muls<float>(outLocal, selfLocal, betaFp32_, (int32_t)curTileM);
            // selfLocal = vec1TimesAlpha * vec2 (复用 selfInBuf 空间)
            Muls<float>(selfLocal, vec2Local, vec1TimesAlpha, (int32_t)curTileM);
            // outLocal = outLocal + selfLocal
            Add<float>(outLocal, outLocal, selfLocal, (int32_t)curTileM);
            // REVIEW S3: 移除冗余 PipeBarrier<PIPE_ALL>。TQue EnQue/DeQue 机制已处理 V→MTE3 同步。
            // 非float路径 (CastResultAndCopyOut) 同样无此 barrier 且正确。

            CopyOutFloat(outLocal, rowIdx, tileIdx, curTileM);
        } else {
            PrepareSelf(rowIdx, tileIdx, curTileM);
            auto selfFp32 = selfFp32Buf.Get<float>();
            auto vec2Fp32 = vec2Fp32Buf.Get<float>();
            auto outerFp32 = outerFp32Buf.Get<float>();
            auto resultFp32 = resultFp32Buf.Get<float>();

            // resultFp32 = beta * self
            Muls<float>(resultFp32, selfFp32, betaFp32_, (int32_t)curTileM);
            // outerFp32 = vec1TimesAlpha * vec2
            Muls<float>(outerFp32, vec2Fp32, vec1TimesAlpha, (int32_t)curTileM);
            // resultFp32 = resultFp32 + outerFp32
            Add<float>(resultFp32, resultFp32, outerFp32, (int32_t)curTileM);

            if constexpr (AddrV2TypeTraits<T>::IsIntU8) {
                ApplyMod256(curTileM);
            }
            CastResultAndCopyOut(rowIdx, tileIdx, curTileM);
        }
    }

private:
    TPipe pipe;

    GlobalTensor<T> x1Gm; // self
    GlobalTensor<T> x2Gm; // vec1 [N]
    GlobalTensor<T> x3Gm; // vec2 [M]
    GlobalTensor<T> yGm;  // out [N,M]

    // Tiling 参数
    uint32_t blockIdx_;
    uint32_t N_;
    uint32_t M_;
    uint32_t tileM_;
    uint32_t tileMLoop_;
    uint32_t tileMTail_;
    uint32_t tilingKey_;
    uint32_t selfBcastMode_;
    uint32_t startRow_;
    uint32_t endRow_;
    float betaFp32_;
    float alphaFp32_;

    // 输出 TQue（double buffer）
    TQue<TPosition::VECOUT, 2> outQueueY;

    // 输入 TBuf（vec2 跨行复用，self 按行加载）
    TBuf<TPosition::VECCALC> vec2InBuf;
    TBuf<TPosition::VECCALC> selfInBuf;

    // fp32 计算 TBuf（非 float 类型）
    TBuf<TPosition::VECCALC> vec2Fp32Buf;
    TBuf<TPosition::VECCALC> selfFp32Buf;
    TBuf<TPosition::VECCALC> outerFp32Buf;
    TBuf<TPosition::VECCALC> resultFp32Buf;

    // int8/uint8 mod256 和 half 中转 TBuf
    TBuf<TPosition::VECCALC> modBufA;
    TBuf<TPosition::VECCALC> modBufB;
    TBuf<TPosition::VECCALC> outHalfBuf;

    // 标量加载 TBuf
    TBuf<TPosition::VECCALC> scalarInBuf;
    TBuf<TPosition::VECCALC> scalarMidBuf;
    TBuf<TPosition::VECCALC> scalarFp32Buf;
};

// ============================================================================
// 核函数入口 - 按 dtype 编译期实例化（由 ASCENDC_TPL_DATATYPE_DECL 驱动）
// ============================================================================
template <typename D_T_X>
__global__ __aicore__ void addr_v2(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR beta, GM_ADDR alpha, GM_ADDR y,
                                   GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    REGISTER_TILING_DEFAULT(AddrV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(AddrV2TilingData, tilingData, tiling);

    AddrV2Kernel<D_T_X> op;
    op.Init(x1, x2, x3, y, &tilingData);
    op.Process();
}
