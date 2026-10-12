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
 * \file log_space_compute.h
 * \brief LogSpace kernel 计算方法定义（ComputeChunk 及其分支 helper、df-exp 变体）。
 *        从 log_space.h 拆出以控制单文件规模；随 log_space.h 一并包含。
 */
#ifndef LOG_SPACE_COMPUTE_H
#define LOG_SPACE_COMPUTE_H

#include "log_space.h"

namespace NsLogSpace {

using namespace AscendC;

template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::ComputeChunk(int64_t chunkBase, int64_t currentNum)
{
    LocalTensor<float> idxLocal = idxBuf_.Get<float>();
    LocalTensor<float> valLocal = valBuf_.Get<float>();
    // [去量化] 每 chunk 标量基址 xBase = startF + g0*step（g0 = 全局起始索引），配合本地索引 [0..n-1] 使用，
    // 取代原 ArithProgression(firstIdx, 1, n)：当 g0 > 2^24 时 fp32 下 firstIdx+1 的 +1 增量被舍掉，
    // 同一 chunk 内所有元素拿到相同指数（实测 Test_056 连续 V 完全相同）。本地索引 [0..n-1]（n≤ubChunk<2^24）精确，
    // xBase 仅承载大偏移，每元素指数 = xBase + loc*step 互不相同。op 数与原实现一致。
    [[maybe_unused]] const float xBase = startF_ + static_cast<float>(idxStart_ + chunkBase) * stepF_;
    const int32_t n = static_cast<int32_t>(currentNum);

    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();

    // [FIX] 端点补丁标记：该 chunk 是否持有全局首/末元素。numpy/torch linspace 强制两端点精确，
    // 设备 fp32 exp 对整数幂会下溢(如 10^1->9.9999, 整型 TRUNC 成 9)。做法=在 float 缓冲区上 SetValue
    // 打补丁(float->float 合法；aicore 标量域禁止 float->uint8/bf16 直转)，再由既有向量 Cast 落型，
    // 每个 GM 字节只写一次——替代原 EmitPow 落 GM 后覆写(1B->32B padding 与相邻 chunk 写重叠的 WAW 竞争)。
    const int64_t gFirst_ = idxStart_ + chunkBase;
    const bool patchStart_ = (gFirst_ == 0);
    const bool patchEnd_ = (gFirst_ + currentNum == static_cast<int64_t>(totalLen_));

    if constexpr (std::is_same_v<T, float>) {
        ComputeFp32(outLocal, n, currentNum, xBase, patchStart_, patchEnd_);
    } else {
        if constexpr (std::is_integral_v<T>) {
            ComputeIntegral(idxLocal, valLocal, chunkBase, n);
        } else {
            ComputeQuant(idxLocal, valLocal, n, xBase);
        }
        // 端点补丁(在 fp32 valLocal 上，随后由 Cast 落型)：SetValue(标量) -> Cast(向量) 跨 pipe 同步
        if (patchStart_) {
            valLocal.SetValue(0, startValF_);
        }
        if (patchEnd_) {
            valLocal.SetValue(static_cast<uint32_t>(currentNum - 1), endValF_);
        }
        if (patchStart_ || patchEnd_) {
            int32_t eid = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
            AscendC::SetFlag<AscendC::HardEvent::S_V>(eid);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(eid);
        }
        CastToOut(outLocal, valLocal, n);
    }
    outQueue_.EnQue(outLocal);

    LocalTensor<T> outDq = outQueue_.template DeQue<T>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outGM_[chunkBase], outDq, copyParams);
    outQueue_.FreeTensor(outDq);
}

template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::ComputeFp32(LocalTensor<T>& outLocal, int32_t n, int64_t currentNum,
                                                      float xBase, bool patchStart_, bool patchEnd_)
{
    // fp32 路径：直接在 outLocal 上以 fp32 计算，省去中间 valBuf + DataCopy
    AscendC::ArithProgression<float>(outLocal, 0.0f, 1.0f, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls<float>(outLocal, outLocal, stepF_, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds<float>(outLocal, outLocal, xBase, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls<float>(outLocal, outLocal, logBase_, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp<float, false>(outLocal, outLocal, n);
    AscendC::PipeBarrier<PIPE_V>();
    // 端点补丁(fp32 直接落 outLocal)：SetValue(标量) -> DataCopyPad(MTE3) 跨 pipe 同步
    if (patchStart_) {
        outLocal.SetValue(0, startValF_);
    }
    if (patchEnd_) {
        outLocal.SetValue(static_cast<uint32_t>(currentNum - 1), endValF_);
    }
    if (patchStart_ || patchEnd_) {
        int32_t eid = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eid);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eid);
    }
}

template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::ComputeQuant(LocalTensor<float>& idxLocal, LocalTensor<float>& valLocal,
                                                       int32_t n, float xBase)
{
    // fp16 / bf16：当前 de-quant 快路径（lenient 阈值已通过、性能敏感，不引入 df 开销）
    AscendC::ArithProgression<float>(idxLocal, 0.0f, 1.0f, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls<float>(valLocal, idxLocal, stepF_, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds<float>(valLocal, valLocal, xBase, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls<float>(valLocal, valLocal, logBase_, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp<float, false>(valLocal, valLocal, n);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::ComputeIntegral(LocalTensor<float>& idxLocal, LocalTensor<float>& valLocal,
                                                          int64_t chunkBase, int32_t n)
{
    // [整型 double-float] arg 空间(arg=x·ln base)用 fp32-pair 算准指数，消除大 g 抵消丢精度。
    // 等价分解 value = exp(argBaseHi)·exp(loc·stepLnHi + argBaseLo)：
    //   exp(argBaseHi) 每 chunk 标量算一次(承载大值)；exp(小量) 每元素算(参数<0.01、精确)。
    //   每元素 op 数与原路径相同(·logBase 换成 ·expBase)，额外只有每 chunk 标量 df 计算(摊薄)。
    const int64_t g0 = idxStart_ + chunkBase;
    const int64_t gA = g0 >> 13; // < 2^13，(float) 精确
    const int64_t gB = g0 & 8191;
    const float fgA = static_cast<float>(gA);
    const float fgB = static_cast<float>(gB);
    const float SH = 8192.0f * stepLnHi_; // 指数移位，精确
    float p1, e1;
    DfTwoProd(fgA, SH, p1, e1);
    float p2, e2;
    DfTwoProd(fgB, stepLnHi_, p2, e2);
    float gsHi, gsE;
    DfTwoSum(p1, p2, gsHi, gsE);
    float gsLo = gsE + e1 + e2;
    gsLo += fgA * (8192.0f * stepLnLo_) + fgB * stepLnLo_; // g0·stepLnLo 修正
    float argBaseHi, bE;
    DfTwoSum(argStartHi_, gsHi, argBaseHi, bE);
    float argBaseLo = bE + gsLo + argStartLo_;
    DfTwoSum(argBaseHi, argBaseLo, argBaseHi, argBaseLo); // 重规格化
    // expBase = exp(argBaseHi)：1 元素向量 Exp，V_S 同步取回标量
    AscendC::Duplicate<float>(idxLocal, argBaseHi, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp<float, false>(idxLocal, idxLocal, 1);
    int32_t eidVS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eidVS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eidVS);
    const float expBase = idxLocal.GetValue(0);
    // 每元素：delta = loc·stepLnHi + argBaseLo；value = expBase·exp(delta)
    AscendC::ArithProgression<float>(idxLocal, 0.0f, 1.0f, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls<float>(valLocal, idxLocal, stepLnHi_, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds<float>(valLocal, valLocal, argBaseLo, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp<float, false>(valLocal, valLocal, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls<float>(valLocal, valLocal, expBase, n);
    AscendC::PipeBarrier<PIPE_V>();
    // [整数幂覆写] df 的 exp 对落在整数指数(网格点)的元素会下溢到整数下方(10^1->9.9999,
    // 整型 TRUNC 砍成 9)——这是 arg 空间提精后整数幂落到整数下侧的副作用。用 host 精确幂表
    // 对这些元素 SetValue 覆写(在 mod/cast 之前，uint8 的 mod 256 对覆写后的精确值仍正确)。
    // 仅覆写真正落在整数指数的少数元素；非网格点元素(df 与 golden 同侧)不动 → 不回归大 case。
    PatchIntegerPowers(valLocal, n, fgA, fgB);
}

template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::PatchIntegerPowers(LocalTensor<float>& valLocal, int32_t n, float fgA,
                                                             float fgB)
{
    if (nCount_ > 0) {
        if (stepF_ == 0.0f) {
            // start==end：所有元素 = base^start，用 host 精确值覆写全部
            AscendC::Duplicate<float>(valLocal, startValF_, n);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            // xBase(df, x 空间) = startF + g0*step：定位网格点需高精度(大 g0 下 fp32 单精会错位)
            const float SHx = 8192.0f * stepF_;
            float xp1, xe1;
            DfTwoProd(fgA, SHx, xp1, xe1);
            float xp2, xe2;
            DfTwoProd(fgB, stepF_, xp2, xe2);
            float xgsHi, xgsE;
            DfTwoSum(xp1, xp2, xgsHi, xgsE);
            float xgsLo = xgsE + xe1 + xe2 + fgA * (8192.0f * stepLoX_) + fgB * stepLoX_;
            float xBaseHi, xbE;
            DfTwoSum(startF_, xgsHi, xBaseHi, xbE);
            const float xBaseLo = xbE + xgsLo;
            // [perf] 只遍历本 chunk 指数范围 [xLo,xHi] 内的整数 m（大 case chunk 跨度<<1 → 仅 0-3 个），
            // 不再扫全表（否则大 case 每 chunk 白扫 nCount 项、Test_166 实测从 104→237µs 翻倍回归）。
            const float xEnd = xBaseHi + static_cast<float>(n - 1) * stepF_ + xBaseLo;
            const float xLo = (xBaseHi < xEnd) ? xBaseHi : xEnd;
            const float xHi = (xBaseHi < xEnd) ? xEnd : xBaseHi;
            int32_t kLo = static_cast<int32_t>(xLo) - nmin_ - 1; // (int)截断+1 余量覆盖 floor/ceil
            int32_t kHi = static_cast<int32_t>(xHi) - nmin_ + 1;
            if (kLo < 0) {
                kLo = 0;
            }
            if (kHi > nCount_ - 1) {
                kHi = nCount_ - 1;
            }
            for (int32_t k = kLo; k <= kHi; ++k) {
                const float baseM = baseNTab_[k];
                if (!(baseM < 3.0e38f)) {
                    continue;
                } // 溢出(inf)跳过，交给 df 值(同样溢出，一致)
                const float fm = static_cast<float>(nmin_ + k);
                const float locF = (fm - xBaseHi - xBaseLo) / stepF_;
                if (locF < -0.5f) {
                    continue;
                }
                const int32_t loc = static_cast<int32_t>(locF + 0.5f);
                if (loc < 0 || loc >= n) {
                    continue;
                }
                const float xAt = xBaseHi + static_cast<float>(loc) * stepF_ + xBaseLo;
                const float d = xAt - fm;
                const float ad = (d < 0.0f) ? -d : d;
                if (ad < 1.0e-5f) { // 该元素指数确实=m(网格点) → 精确覆写
                    valLocal.SetValue(static_cast<uint32_t>(loc), baseM);
                }
            }
        }
    }
}

template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::CastToOut(LocalTensor<T>& outLocal, LocalTensor<float>& valLocal, int32_t n)
{
    // 末步落输出 dtype：
    //   fp16/bf16     -> 直接 float->T，CAST_RINT（IEEE 就近，PyTorch 默认一致）；
    //   int16/int32   -> 直接 float->T，CAST_TRUNC（向零取整，匹配 torch）；
    //   int8/uint8    -> c220 无 float->1字节直接 intrinsic，经 half 中转：
    //                    float --CAST_TRUNC--> half --CAST_TRUNC--> int8/uint8。
    //                    前提：须先用纯 fp32 把值 mod 到 [0,256)(uint8) / [-128,128)(int8)。half 有效位
    //                    11 bit，|v|<=2048 内整数才精确，落进该区间后中转方能无损；若移除下方的 mod
    //                    逻辑，大值经 half 会静默丢精度。
    if constexpr (std::is_same_v<T, uint8_t>) {
        // [回绕] uint8 溢出对齐 torch：torch 的 float->uint8 取低字节(值 mod 256)而非饱和。
        // 用纯 fp32 算 V mod 256 = V - 256*floor(V/256)：floor 走 float->float CAST_FLOOR（对大值=恒等，
        // 不经 int 不溢出）；mod 后落在 [0,256)，再 float--TRUNC-->half--TRUNC-->uint8（半精确、无溢出）。
        // idxBuf_ 此时空闲（idxLocal 已被 Muls 消费），复用作 modTmp / halfTmp。
        LocalTensor<float> modTmp = idxBuf_.Get<float>();
        AscendC::Muls<float>(modTmp, valLocal, 1.0f / 256.0f, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<float, float>(modTmp, modTmp, AscendC::RoundMode::CAST_FLOOR, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls<float>(modTmp, modTmp, 256.0f, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(valLocal, valLocal, modTmp, n);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<half> halfTmp = idxBuf_.Get<half>();
        AscendC::Cast<half, float>(halfTmp, valLocal, AscendC::RoundMode::CAST_TRUNC, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<T, half>(outLocal, halfTmp, AscendC::RoundMode::CAST_TRUNC, n);
    } else if constexpr (sizeof(T) == 1) {
        // int8 对齐 torch：torch 的 float->int8 = 取整数低字节(有符号回绕)；超 INT32 处给 -1，与全回绕给 0
        // 只差 1，在 int8 容差(diff<=1)内。显式算有符号低字节(纯 fp32，无需 SetSaturationFlag)：
        //   Vf = floor(V)；m8 = Vf - 256*floor((Vf+128)/256) ∈ [-128,128)；再 half->int8 精确(在值域内)。
        LocalTensor<float> tmp = idxBuf_.Get<float>();
        AscendC::Cast<float, float>(valLocal, valLocal, AscendC::RoundMode::CAST_FLOOR, n); // Vf=floor(V)
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds<float>(tmp, valLocal, 128.0f, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls<float>(tmp, tmp, 1.0f / 256.0f, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<float, float>(tmp, tmp, AscendC::RoundMode::CAST_FLOOR, n); // floor((Vf+128)/256)
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls<float>(tmp, tmp, 256.0f, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(valLocal, valLocal, tmp, n); // m8 = Vf - 256*q ∈ [-128,128)
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<half> halfTmp = idxBuf_.Get<half>();
        AscendC::Cast<half, float>(halfTmp, valLocal, AscendC::RoundMode::CAST_TRUNC, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<T, half>(outLocal, halfTmp, AscendC::RoundMode::CAST_TRUNC, n);
    } else {
        AscendC::Cast<T, float>(outLocal, valLocal, LogSpaceCastMode<T>(), n);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

// [df 乘公共块] Veltkamp 两积误差：ee = fl(aHi·bHi − p) + aHi·bLo + aLo·bHi + aLo·bLo（w0/w1 暂存）。
// DfExpVec 的 Horner 迭代与 DfMulVec 共用同一展开，抽出消除重复；缓冲偏移两处一致
// (p=D[32] ee=D[40] aHi=D[56] aLo=D[64] bHi=D[72] bLo=D[80] w0=D[88] w1=D[96])。
template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::DfProdErr(LocalTensor<float>& D)
{
    LocalTensor<float> p = D[32], ee = D[40], aHi = D[56], aLo = D[64];
    LocalTensor<float> bHi = D[72], bLo = D[80], w0 = D[88], w1 = D[96];
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float>(w0, aHi, bHi, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float>(w0, w0, p, 1);
    AscendC::Mul<float>(w1, aHi, bLo, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float>(w0, w0, w1, 1);
    AscendC::Mul<float>(w1, aLo, bHi, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float>(w0, w0, w1, 1);
    AscendC::Mul<float>(w1, aLo, bLo, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float>(ee, w0, w1, 1);
}

// [df-V 每元素值+keep] 由 expBase(ebH,ebL) 与本 chunk 局部索引算 value(fp32,存 B) 及 keep 掩码
// (值≥2^31 → 0，存 E)。从 ComputeChunkDfV 拆出以控函数规模；数值逐运算逐字保持不变。
template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::ComputeValueKeepDfV(LocalTensor<float>& A, LocalTensor<float>& B,
                                                              LocalTensor<float>& C, LocalTensor<float>& E, int32_t n,
                                                              float ebH, float ebL)
{
    // expBase=(ebH,ebL) 由 ProcessNormalDfV 递推给入（整核 df-exp 只算 1 次，之后每 chunk 1 次 df 乘）
    // 每元素：delta=loc·stepLn，s=expm1(delta)，V=ebH·(1+s)（light df：eb·s<2^23 不需 TwoProd）
    AscendC::ArithProgression<float>(A, 0.0f, 1.0f, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls<float>(B, A, stepLnHi_, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls<float>(E, A, stepLnLo_, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float>(A, B, B, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls<float>(C, B, 1.0f / 24.0f, n);
    AscendC::PipeBarrier<PIPE_V>(); // dh/24
    AscendC::Adds<float>(C, C, 1.0f / 6.0f, n);
    AscendC::PipeBarrier<PIPE_V>(); // 1/6 + dh/24
    AscendC::Mul<float>(C, C, B, n);
    AscendC::PipeBarrier<PIPE_V>(); // dh/6 + dh^2/24
    AscendC::Adds<float>(C, C, 0.5f, n);
    AscendC::PipeBarrier<PIPE_V>(); // 0.5 + dh/6 + dh^2/24
    AscendC::Mul<float>(C, A, C, n);
    AscendC::PipeBarrier<PIPE_V>(); // d2*(..) = delta^2/2 + delta^3/6 + delta^4/24
    AscendC::Add<float>(C, C, E, n);
    AscendC::PipeBarrier<PIPE_V>(); // s_lo (含 delta_lo)
    AscendC::Add<float>(C, C, B, n);
    AscendC::PipeBarrier<PIPE_V>(); // s = delta_hi + s_lo = expm1(delta)
    // value = ebH·(1+s) + ebL = ebH + ebH·s + ebL，合成单 fp32（关键：golden = torch 的 fp32 值取低字节，
    // 需匹配 fp32 舍入而非更高精度；ebH/ebL 由向量 df-exp 精确给出 → value 舍入到与 torch 同一个 fp32）。
    AscendC::Muls<float>(C, C, ebH, n);
    AscendC::PipeBarrier<PIPE_V>(); // ebH·s
    AscendC::Adds<float>(C, C, ebL, n);
    AscendC::PipeBarrier<PIPE_V>(); // + ebL
    AscendC::Adds<float>(B, C, ebH, n);
    AscendC::PipeBarrier<PIPE_V>(); // value (fp32)
    // keep = 1 - min(1, floor(value/2^31))：值 ≥2^31 时 fp32 是 256 的倍数 → mod=0，下面 ·keep 清零
    AscendC::Muls<float>(E, B, 1.0f / 2147483648.0f, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast<float, float>(E, E, AscendC::RoundMode::CAST_FLOOR, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mins<float>(E, E, 1.0f, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls<float>(E, E, -1.0f, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds<float>(E, E, 1.0f, n);
    AscendC::PipeBarrier<PIPE_V>(); // keep ∈ {0,1}
}

// [向量 df-exp] 在向量单元上算 exp(argHi+argLo) 到 double-float（标量单元精度不够，故走向量）。
// 范围规约(Cody-Waite，标量但用整数运算+Sterbenz 故精确) + Horner 泰勒(向量 df，1 元素槽 32B 对齐)。
// 仅每 chunk 算一次参考 expBase，慢但摊薄；此版先验证精度，性能后续优化。
template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::DfExpVec(float argHi, float argLo, float& ehi, float& elo)
{
    const float INVLN2 = 1.4426950408889634f, LN2HI = 0.693145751953125f, LN2LO = 1.4286068203094172e-6f;
    float kf = argHi * INVLN2;
    int32_t k = static_cast<int32_t>(kf + (kf >= 0.0f ? 0.5f : -0.5f));
    float fk = static_cast<float>(k);
    float rhi = argHi - fk * LN2HI; // 整数运算 + Sterbenz 相减 → 精确
    float rlo = argLo - fk * LN2LO;

    LocalTensor<float> D = dfBuf_.Get<float>();
    LocalTensor<float> aH = D[0], aL = D[8], rH = D[16], rL = D[24], p = D[32], ee = D[40];
    LocalTensor<float> ca = D[48], aHi = D[56], aLo = D[64], bHi = D[72], bLo = D[80];
    LocalTensor<float> w0 = D[88], w1 = D[96], w2 = D[104], cH = D[112], cL = D[120];
    AscendC::Duplicate<float>(rH, rhi, 1);
    AscendC::Duplicate<float>(rL, rlo, 1);
    AscendC::Duplicate<float>(aH, rfHiTab_[9], 1);
    AscendC::Duplicate<float>(aL, rfLoTab_[9], 1);
    AscendC::PipeBarrier<PIPE_V>();
    for (int32_t i = 8; i >= 0; --i) {
        // acc = acc * r （Veltkamp df 乘，1 元素）
        AscendC::Mul<float>(p, aH, rH, 1);
        AscendC::Muls<float>(ca, aH, 4097.0f, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(w0, ca, aH, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(aHi, ca, w0, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(aLo, aH, aHi, 1);
        AscendC::Muls<float>(ca, rH, 4097.0f, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(w0, ca, rH, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(bHi, ca, w0, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(bLo, rH, bHi, 1);
        DfProdErr(D);
        AscendC::Mul<float>(w0, aH, rL, 1);
        AscendC::Mul<float>(w1, aL, rH, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(w0, w0, w1, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(ee, ee, w0, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(aH, p, ee, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(w0, aH, p, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(aL, ee, w0, 1);
        AscendC::PipeBarrier<PIPE_V>();
        // acc = acc + c_i （df 加）
        AscendC::Duplicate<float>(cH, rfHiTab_[i], 1);
        AscendC::Duplicate<float>(cL, rfLoTab_[i], 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(w0, aH, cH, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(w1, w0, aH, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(w2, w0, w1, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(w2, aH, w2, 1);
        AscendC::Sub<float>(aHi, cH, w1, 1);
        AscendC::PipeBarrier<PIPE_V>(); // aHi 复用为临时
        AscendC::Add<float>(w2, w2, aHi, 1);
        AscendC::Add<float>(aHi, aL, cL, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(w2, w2, aHi, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(aH, w0, w2, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(w1, aH, w0, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(aL, w2, w1, 1);
        AscendC::PipeBarrier<PIPE_V>();
    }
    int32_t eidVS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eidVS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eidVS);
    float aHv = aH.GetValue(0), aLv = aL.GetValue(0);
    float scale = 1.0f;
    int32_t ak = (k >= 0) ? k : -k;
    for (int32_t i = 0; i < ak; ++i) {
        scale *= 2.0f;
    }
    if (k < 0) {
        scale = 1.0f / scale;
    }
    ehi = aHv * scale;
    elo = aLv * scale;
}

// [df-V] 大值 int8/uint8（值 ∈ (2^24,2^32)）：expBase 用向量 df-exp（高精度），每元素 V=expBase·(1+expm1(delta))，
// df 下取 mod256 / 有符号低字节。缓冲 A=idxBuf B=valBuf C=loBuf E=tmpBuf。
template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::ComputeChunkDfV(int64_t chunkBase, int64_t currentNum, float ebH, float ebL)
{
    if constexpr (sizeof(T) == 1 && std::is_integral_v<T>) {
        const int32_t n = static_cast<int32_t>(currentNum);
        LocalTensor<float> A = idxBuf_.Get<float>();
        LocalTensor<float> B = valBuf_.Get<float>();
        LocalTensor<float> C = loBuf_.Get<float>();
        LocalTensor<float> E = tmpBuf_.Get<float>();
        LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();
        // 每元素 value(fp32,→B) 与 keep 掩码(→E)：见 ComputeValueKeepDfV
        ComputeValueKeepDfV(A, B, C, E, n, ebH, ebL);
        // fp32 mod 256（value<2^32 单段精确；value≥2^32 单段是垃圾，靠 ·keep=0 清零）
        AscendC::Muls<float>(A, B, 1.0f / 256.0f, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<float, float>(A, A, AscendC::RoundMode::CAST_FLOOR, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls<float>(A, A, 256.0f, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(B, B, A, n);
        AscendC::PipeBarrier<PIPE_V>(); // value mod 256
        AscendC::Mul<float>(B, B, E, n);
        AscendC::PipeBarrier<PIPE_V>(); // ·keep（值≥2^31 → 0）
        if constexpr (std::is_same_v<T, int8_t>) {
            AscendC::Muls<float>(A, B, 1.0f / 128.0f, n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast<float, float>(A, A, AscendC::RoundMode::CAST_FLOOR, n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls<float>(A, A, 256.0f, n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub<float>(B, B, A, n);
            AscendC::PipeBarrier<PIPE_V>();
        }
        LocalTensor<half> halfTmp = idxBuf_.Get<half>();
        AscendC::Cast<half, float>(halfTmp, B, AscendC::RoundMode::CAST_TRUNC, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<T, half>(outLocal, halfTmp, AscendC::RoundMode::CAST_TRUNC, n);
        AscendC::PipeBarrier<PIPE_V>();
        outQueue_.EnQue(outLocal);
        LocalTensor<T> outDq = outQueue_.template DeQue<T>();
        AscendC::DataCopyExtParams cp;
        cp.blockCount = 1;
        cp.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        cp.srcStride = 0;
        cp.dstStride = 0;
        AscendC::DataCopyPad(outGM_[chunkBase], outDq, cp);
        outQueue_.FreeTensor(outDq);
    }
}

// [向量 df 乘] (aH,aL)·(bH,bL) → (pH,pL)，1 元素槽（递推 expBase 用，标量精度不够走向量）。
template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::DfMulVec(float aH, float aL, float bH, float bL, float& pH, float& pL)
{
    LocalTensor<float> D = dfBuf_.Get<float>();
    LocalTensor<float> A0 = D[0], A1 = D[8], B0 = D[16], B1 = D[24], p = D[32], ee = D[40];
    LocalTensor<float> ca = D[48], aHi = D[56], aLo = D[64], bHi = D[72], bLo = D[80], w0 = D[88], w1 = D[96];
    AscendC::Duplicate<float>(A0, aH, 1);
    AscendC::Duplicate<float>(A1, aL, 1);
    AscendC::Duplicate<float>(B0, bH, 1);
    AscendC::Duplicate<float>(B1, bL, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float>(p, A0, B0, 1);
    AscendC::Muls<float>(ca, A0, 4097.0f, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float>(w0, ca, A0, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float>(aHi, ca, w0, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float>(aLo, A0, aHi, 1);
    AscendC::Muls<float>(ca, B0, 4097.0f, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float>(w0, ca, B0, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float>(bHi, ca, w0, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float>(bLo, B0, bHi, 1);
    DfProdErr(D);
    AscendC::Mul<float>(w0, A0, B1, 1);
    AscendC::Mul<float>(w1, A1, B0, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float>(w0, w0, w1, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float>(ee, ee, w0, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float>(w0, p, ee, 1);
    AscendC::PipeBarrier<PIPE_V>(); // pH
    AscendC::Sub<float>(w1, w0, p, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float>(ee, ee, w1, 1);
    AscendC::PipeBarrier<PIPE_V>(); // pL
    int32_t eidVS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eidVS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eidVS);
    pH = w0.GetValue(0);
    pL = ee.GetValue(0);
}

// [df-V 递推路径] 整核：chunk 0 用向量 df-exp 算 expBase，之后每 chunk 只做 1 次向量 df 乘 ·const
// （const=base^(ubChunk·step)）→ df-exp 整核只算 1 次，大幅降开销。

} // namespace NsLogSpace

#endif // LOG_SPACE_COMPUTE_H
