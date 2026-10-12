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
 * \file log_space.h
 * \brief LogSpace Kernel 类定义（Ascend950 / Atlas A2/A3）
 *
 * 模板参数：
 *   - T:    输出数据类型（float / half / bfloat16_t / int8_t / int16_t / int32_t / uint8_t）
 *   - MODE: 0 = NORMAL（steps>=2），1 = SINGLE（steps==0/1）
 *
 * 算法（NORMAL，浮点路径）：
 *   idx = ArithProgression(firstValue=base_idx, diffValue=1, count=N)
 *   val_fp32 = idx * stepF + startF
 *   val_fp32 = val_fp32 * logBase
 *   val_fp32 = Exp(val_fp32)
 *   out = Cast<T>(val_fp32)  [T==float 时直接搬出]
 *   DataCopyPad UB -> GM
 *
 * 整型路径（int8/int16/int32/uint8）额外要求 base^x 在整数幂上逐位精确（末步 TRUNC 截断会把
 * 1 ulp 的负向误差放大成整数 -1），故在 arg 空间用 double-float(fp32 pair) 提精度，并对落在
 * 整数幂上的点用 host 预算的精确值覆写，详见 ComputeIntegral / PatchIntegerPowers。
 */
#ifndef LOG_SPACE_H
#define LOG_SPACE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "log_space_tiling_data.h"
#include "log_space_tiling_key.h"

namespace NsLogSpace {

using namespace AscendC;

// 末步 fp32 -> 输出 dtype 的舍入模式：
//   整型(int8/int16/int32/uint8) -> CAST_TRUNC：向零取整，匹配 torch 的整型转换语义
//     （torch.logspace(dtype=int) = 先在 float 计算 base^x，再 .to(int) 截断取整）；
//     注：Cast 对整型溢出按饱和处理，自测 case 取值域需落在 dtype 范围内。
//   fp16/bf16            -> CAST_RINT：四舍六入五成双，与 PyTorch 默认浮点舍入一致。
template <typename U>
__aicore__ inline AscendC::RoundMode LogSpaceCastMode()
{
    if constexpr (std::is_same_v<U, int8_t> || std::is_same_v<U, uint8_t> || std::is_same_v<U, int16_t> ||
                  std::is_same_v<U, int32_t>) {
        return AscendC::RoundMode::CAST_TRUNC;
    } else {
        return AscendC::RoundMode::CAST_RINT;
    }
}

template <typename T, int MODE>
class LogSpace {
public:
    __aicore__ inline LogSpace() {}

    __aicore__ inline void Init(GM_ADDR result, const LogSpaceTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessNormal();
    __aicore__ inline void ProcessNormalDfV();
    __aicore__ inline void ProcessSingle();
    __aicore__ inline void ComputeChunk(int64_t chunkBase, int64_t currentNum);
    __aicore__ inline void ComputeChunkDfV(int64_t chunkBase, int64_t currentNum, float ebH, float ebL);
    __aicore__ inline void DfExpVec(float argHi, float argLo, float& ehi, float& elo);
    __aicore__ inline void DfMulVec(float aH, float aL, float bH, float bL, float& pH, float& pL);
    __aicore__ inline void DfProdErr(LocalTensor<float>& D);
    __aicore__ inline void ComputeValueKeepDfV(LocalTensor<float>& A, LocalTensor<float>& B, LocalTensor<float>& C,
                                               LocalTensor<float>& E, int32_t n, float ebH, float ebL);
    __aicore__ inline void EmitPow(int64_t localPos, float powVal);
    __aicore__ inline void ComputeFp32(LocalTensor<T>& outLocal, int32_t n, int64_t currentNum, float xBase,
                                       bool patchStart_, bool patchEnd_);
    __aicore__ inline void ComputeIntegral(LocalTensor<float>& idxLocal, LocalTensor<float>& valLocal,
                                           int64_t chunkBase, int32_t n);
    __aicore__ inline void PatchIntegerPowers(LocalTensor<float>& valLocal, int32_t n, float fgA, float fgB);
    __aicore__ inline void CastToOut(LocalTensor<T>& outLocal, LocalTensor<float>& valLocal, int32_t n);
    __aicore__ inline void ComputeQuant(LocalTensor<float>& idxLocal, LocalTensor<float>& valLocal, int32_t n,
                                        float xBase);

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> idxBuf_;
    TBuf<TPosition::VECCALC> valBuf_;
    TBuf<TPosition::VECCALC> loBuf_;  // df-V：V_lo
    TBuf<TPosition::VECCALC> tmpBuf_; // df-V：临时
    TBuf<TPosition::VECCALC> dfBuf_;  // df-V：向量 df-exp 工作区（1 元素槽）
    TQue<QuePosition::VECOUT, 2> outQueue_;

    GlobalTensor<T> outGM_;

    uint64_t totalLen_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t tileLen_ = 0;
    uint32_t tailCoreIdx_ = 0;
    uint32_t tailTileLen_ = 0;
    uint32_t ubChunk_ = 0;
    float startF_ = 0.0f;
    float stepF_ = 0.0f;
    float logBase_ = 0.0f;
    float startValF_ = 0.0f; // host 算好的 base^start（端点精确修正用）
    float endValF_ = 0.0f;   // host 算好的 base^end
    // 整型路径 arg 空间 double-float 系数（host 用 double 算好拆 hi/lo）
    float argStartHi_ = 0.0f;
    float argStartLo_ = 0.0f;
    float stepLnHi_ = 0.0f;
    float stepLnLo_ = 0.0f;
    // 整数幂精确覆写表（整型路径用，详见 ComputeChunk）
    float stepLoX_ = 0.0f;
    int32_t nmin_ = 0;
    int32_t nCount_ = 0;
    float baseNTab_[96] = {0.0f};
    int32_t useDfV_ = 0;
    float rfHiTab_[12] = {0.0f};
    float rfLoTab_[12] = {0.0f};
    float constHi_ = 0.0f;
    float constLo_ = 0.0f;

    int64_t idxStart_ = 0;
    int64_t blockLen_ = 0;
};

// ---- double-float(fp32 pair) 标量原语：Veltkamp 拆分(4097=2^12+1) ----
__aicore__ inline void DfTwoSum(float a, float b, float& s, float& e)
{
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
__aicore__ inline void DfTwoProd(float a, float b, float& p, float& e)
{
    p = a * b;
    const float SPL = 4097.0f;
    float ca = SPL * a;
    float aHi = ca - (ca - a);
    float aLo = a - aHi;
    float cb = SPL * b;
    float bHi = cb - (cb - b);
    float bLo = b - bHi;
    e = ((aHi * bHi - p) + aHi * bLo + aLo * bHi) + aLo * bLo;
}

template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::Init(GM_ADDR result, const LogSpaceTilingData* tilingData)
{
    totalLen_ = tilingData->totalLen;
    coreNum_ = tilingData->coreNum;
    tileLen_ = tilingData->tileLen;
    tailCoreIdx_ = tilingData->tailCoreIdx;
    tailTileLen_ = tilingData->tailTileLen;
    ubChunk_ = tilingData->ubChunk;
    startF_ = tilingData->startF;
    stepF_ = tilingData->stepF;
    logBase_ = tilingData->logBase;
    startValF_ = tilingData->startValF;
    endValF_ = tilingData->endValF;
    argStartHi_ = tilingData->argStartHi;
    argStartLo_ = tilingData->argStartLo;
    stepLnHi_ = tilingData->stepLnHi;
    stepLnLo_ = tilingData->stepLnLo;
    stepLoX_ = tilingData->stepLoX;
    nmin_ = tilingData->nmin;
    nCount_ = tilingData->nCount;
    for (int32_t k = 0; k < nCount_; ++k) {
        baseNTab_[k] = tilingData->baseN[k];
    }
    useDfV_ = tilingData->useDfV;
    for (int32_t k = 0; k < 12; ++k) {
        rfHiTab_[k] = tilingData->rfHi[k];
        rfLoTab_[k] = tilingData->rfLo[k];
    }
    constHi_ = tilingData->constHi;
    constLo_ = tilingData->constLo;

    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    idxStart_ = blockIdx * static_cast<int64_t>(tileLen_);
    if (blockIdx == static_cast<int64_t>(tailCoreIdx_)) {
        blockLen_ = static_cast<int64_t>(tailTileLen_);
    } else if (blockIdx < static_cast<int64_t>(coreNum_)) {
        blockLen_ = static_cast<int64_t>(tileLen_);
    } else {
        blockLen_ = 0;
    }

    outGM_.SetGlobalBuffer((__gm__ T*)result + idxStart_, blockLen_);

    // UB 分配：index (fp32) + val (fp32)，out 队列按 T 分配
    pipe.InitBuffer(idxBuf_, ubChunk_ * sizeof(float));
    pipe.InitBuffer(valBuf_, ubChunk_ * sizeof(float));
    if (useDfV_) {
        pipe.InitBuffer(loBuf_, ubChunk_ * sizeof(float));
        pipe.InitBuffer(tmpBuf_, ubChunk_ * sizeof(float));
        pipe.InitBuffer(dfBuf_, 160 * sizeof(float)); // 向量 df-exp：17 个 32B 对齐 1 元素槽
    }
    pipe.InitBuffer(outQueue_, 2, ubChunk_ * sizeof(T));
}

template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::ProcessNormalDfV()
{
    if constexpr (sizeof(T) == 1 && std::is_integral_v<T>) {
        // expBase[0] = exp(argBase of chunk 0)，argBase = argStart + idxStart·stepLn（标量 df 算指数）
        const int64_t g0 = idxStart_;
        const int64_t gA = g0 >> 13;
        const int64_t gB = g0 & 8191;
        const float fgA = static_cast<float>(gA);
        const float fgB = static_cast<float>(gB);
        const float SH = 8192.0f * stepLnHi_;
        float p1, e1;
        DfTwoProd(fgA, SH, p1, e1);
        float p2, e2;
        DfTwoProd(fgB, stepLnHi_, p2, e2);
        float gsHi, gsE;
        DfTwoSum(p1, p2, gsHi, gsE);
        float gsLo = gsE + e1 + e2 + fgA * (8192.0f * stepLnLo_) + fgB * stepLnLo_;
        float argBaseHi, bE;
        DfTwoSum(argStartHi_, gsHi, argBaseHi, bE);
        float argBaseLo = bE + gsLo + argStartLo_;
        DfTwoSum(argBaseHi, argBaseLo, argBaseHi, argBaseLo);
        float ebH, ebL;
        DfExpVec(argBaseHi, argBaseLo, ebH, ebL);
        const int64_t chunk = static_cast<int64_t>(ubChunk_);
        int64_t processed = 0;
        while (processed < blockLen_) {
            int64_t cur = blockLen_ - processed;
            if (cur > chunk)
                cur = chunk;
            ComputeChunkDfV(processed, cur, ebH, ebL);
            processed += cur;
            if (processed < blockLen_) {
                DfMulVec(ebH, ebL, constHi_, constLo_, ebH, ebL);
            } // 递推下一 chunk
        }
    }
}

// 单元素精确写出端点值 base^(start|end)。powVal 由 host 用 double std::pow 算好（与验收 golden 的
// numpy base**linspace 同口径，对整数幂精确），这里只 Duplicate→Cast 落型，不在设备上走 exp。
// 原因：设备 fp32 exp(k·ln base) 对整数幂会不稳定地上溢/下溢（10^1 凑巧上溢得 10、10^2 下溢得 99.9999
// 被 TRUNC 砍成 99）。numpy linspace 强制两端点精确，故首/末元素都用 host pow 覆写以对齐 golden。
// 中间元素仍走 ComputeChunk 累加路径，逐元素不变 → 仅覆写 2 个端点元素，对已通过的 case 不回归。
template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::EmitPow(int64_t localPos, float powVal)
{
    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();
    if constexpr (std::is_same_v<T, float>) {
        AscendC::Duplicate<float>(outLocal, powVal, 1);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        LocalTensor<float> valLocal = valBuf_.Get<float>();
        AscendC::Duplicate<float>(valLocal, powVal, 1);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (sizeof(T) == 1) {
            // int8/uint8 经 half 中转（详见 ComputeChunk）
            LocalTensor<half> halfTmp = idxBuf_.Get<half>();
            AscendC::Cast<half, float>(halfTmp, valLocal, AscendC::RoundMode::CAST_TRUNC, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast<T, half>(outLocal, halfTmp, AscendC::RoundMode::CAST_TRUNC, 1);
        } else {
            AscendC::Cast<T, float>(outLocal, valLocal, LogSpaceCastMode<T>(), 1);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    outQueue_.EnQue(outLocal);
    LocalTensor<T> outDq = outQueue_.template DeQue<T>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outGM_[localPos], outDq, copyParams);
    outQueue_.FreeTensor(outDq);
}

template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::ProcessNormal()
{
    if (blockLen_ <= 0) {
        return;
    }
    if constexpr (sizeof(T) == 1 && std::is_integral_v<T>) {
        if (useDfV_) {
            ProcessNormalDfV();
            return;
        } // 大值 int8/uint8 走 df-V 递推路径
    }
    const int64_t chunk = static_cast<int64_t>(ubChunk_);
    int64_t processed = 0;
    while (processed < blockLen_) {
        int64_t cur = blockLen_ - processed;
        if (cur > chunk)
            cur = chunk;
        ComputeChunk(processed, cur);
        processed += cur;
    }
    // 端点精确修正已下沉到 ComputeChunk（在 UB 内 SetValue 打补丁 + 单次搬出），
    // 替代原先落 GM 后再 EmitPow 覆写——后者的 1B->32B padding 写与相邻 chunk 写重叠且无定序，
    // 存在 ~17% 的 WAW 竞争。此处不再做事后覆写。
}

template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::ProcessSingle()
{
    // MODE=1: 仅核 0 写 1 个元素（steps==1）= base^start；steps==0 时 blockLen_ 为 0，直接返回
    if (AscendC::GetBlockIdx() != 0 || blockLen_ <= 0) {
        return;
    }
    EmitPow(0, startValF_);
}

template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::Process()
{
    if constexpr (MODE == 0) {
        ProcessNormal();
    } else {
        ProcessSingle();
    }
}

} // namespace NsLogSpace

#include "log_space_compute.h"

#endif // LOG_SPACE_H
