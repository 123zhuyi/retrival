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
 * \file cdist_broadcast.h
 * \brief 方案2: cdist M==1 Broadcast 融合内核 —— OneDim(④) + 静态 rank UB-BRC(③).
 *
 *   语义 (M==1)：y[b,p,r] = ( |x1[b,p]-x2[b,r]|^p )^(1/p)
 *     - 一般 p (0<p<∞, p∉{1,2})：用 Ln/Muls/Exp 复现 (|d|^p)^(1/p)，与 golden(torch.cdist)
 *       的数值行为对齐（含极端 |d| 下上溢 inf / 下溢 0），避免精度对比 FAIL。
 *     - p∈{1,2,inf}：M==1 数学恒等于 |d|，直接输出（golden 亦精确）。
 *     - p==0        ：min(ceil(|d|), 1)（非零计数）
 *
 *   实现分支：
 *     brcBranch==0 OneDim (P==R==1)：输出塌一维长度 B，逐块 |x1-x2|，M==1 归一恒 |d|/count。
 *     brcBranch==1 UB-BRC          ：每 batch 取 (rowsP, R) tile；x1[rowsP,1]、x2[1,R] 紧凑搬入 UB 后用
 *                                    编译期静态 rank Broadcast<T,2>/GetBroadcastTilingInfo<T,2> 展开到
 *                                    [rowsP, R]，再 Sub->Abs(->p0 归一)。相对方案1(运行期 Broadcast<T>)
 *                                    差异仅在 constRank 编译期固化，减少派发/tiling 开销。
 *
 *   计算走 fp32（与 golden 一致：fp32 计算后 cast back）。x1/x2 均可能广播 → 各配独立 bufSrc（避免 RAW 冒险）。
 */
#ifndef CDIST_BROADCAST_H
#define CDIST_BROADCAST_H

#include "kernel_operator.h" // adv_api Broadcast/GetBroadcastTilingInfo (register-invoke: no cdist_local_deps.h)
#include "../cdist_brc_tilingdata.h"

#if !defined(__NPU_HOST__)

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace NsCdist {
using namespace AscendC;

constexpr int32_t CDIST_BRC_BLOCK = 32;
constexpr int32_t CDIST_STATIC_RANK = 2; // 编译期固定 rank（方案2 核心）
constexpr int32_t CDIST_QUEUE_DEPTH = 2; // TQue 双缓冲深度

template <typename T>
class CdistBroadcast {
public:
    __aicore__ inline CdistBroadcast(TPipe* pipe, const CdistBrcTilingData* tiling) : pipe_(pipe), td_(tiling) {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessOneDim();
    __aicore__ inline void ProcessUbBrc();
    __aicore__ inline void ProcessUbBrcMulti();
    __aicore__ inline void ComputeTile(const LocalTensor<float>& x1exp, const LocalTensor<float>& x2exp,
                                       const LocalTensor<float>& yfp, int32_t count);
    __aicore__ inline int32_t AlignUp(int32_t n, int32_t align) { return (n + align - 1) / align * align; }

    TPipe* pipe_ = nullptr;
    const CdistBrcTilingData* td_ = nullptr;
    GlobalTensor<T> x1Gm_, x2Gm_, yGm_;
    float p_ = 1.0f;
    static constexpr bool IS_FP32 = (sizeof(T) == sizeof(float));

    // Staging buffers (T-typed load region for fp16/bf16 before cast to fp32).
    TQue<TPosition::VECIN, CDIST_QUEUE_DEPTH> qLoad1_, qLoad2_; // OneDim inputs (T)
    // fp32 compute planes
    TBuf<TPosition::VECCALC> bufSrc1_;              // x1 紧凑源 fp32 (rowsP,1) / OneDim x1 fp32
    TBuf<TPosition::VECCALC> bufSrc2_;              // x2 紧凑源 fp32 (1,R)     / OneDim x2 fp32
    TBuf<TPosition::VECCALC> bufX1Exp_;             // x1 展开 (rowsP,R) fp32   / OneDim yfp32
    TBuf<TPosition::VECCALC> bufX2Exp_;             // x2 展开 (rowsP,R) fp32
    TBuf<TPosition::VECCALC> bufStage_;             // T-typed staging for UB-BRC compact load
    TQue<TPosition::VECOUT, CDIST_QUEUE_DEPTH> qY_; // output (T)
};

template <typename T>
__aicore__ inline void CdistBroadcast<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y)
{
    p_ = td_->p;
    x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
    x2Gm_.SetGlobalBuffer((__gm__ T*)x2);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    if (td_->brcBranch == 0) {
        int32_t tn = (int32_t)td_->tileNum;
        int32_t tnAlignF = AlignUp(tn, CDIST_BRC_BLOCK / (int32_t)sizeof(float));
        pipe_->InitBuffer(qLoad1_, 2, tn * (int32_t)sizeof(T));
        pipe_->InitBuffer(qLoad2_, 2, tn * (int32_t)sizeof(T));
        pipe_->InitBuffer(qY_, 2, tn * (int32_t)sizeof(T));
        pipe_->InitBuffer(bufSrc1_, tnAlignF * (int32_t)sizeof(float));  // x1 fp32
        pipe_->InitBuffer(bufSrc2_, tnAlignF * (int32_t)sizeof(float));  // x2 fp32
        pipe_->InitBuffer(bufX1Exp_, tnAlignF * (int32_t)sizeof(float)); // yfp32
    } else if (td_->brcBranch == 2) {
        // multi-batch: pack G=batchPerTile 个 batch/一个 tile。P*R 小(<=64x64)，整块 (P,R) 装得下。
        int64_t en = td_->elemNum; // 每个 fp32 平面容量（>= G*P*R）
        int32_t alignT = CDIST_BRC_BLOCK / (int32_t)sizeof(T);
        int64_t G = td_->batchPerTile;
        int32_t src1Ele = AlignUp((int32_t)(G * td_->P), alignT); // x1 块 G*P
        int32_t src2Ele = AlignUp((int32_t)(G * td_->R), alignT); // x2 块 G*R
        if (src1Ele < alignT)
            src1Ele = alignT;
        if (src2Ele < alignT)
            src2Ele = alignT;
        pipe_->InitBuffer(qLoad1_, 2, src1Ele * (int32_t)sizeof(T));   // x1 batched load (T)
        pipe_->InitBuffer(qLoad2_, 2, src2Ele * (int32_t)sizeof(T));   // x2 batched load (T)
        pipe_->InitBuffer(bufSrc1_, src1Ele * (int32_t)sizeof(float)); // x1 batched fp32
        pipe_->InitBuffer(bufSrc2_, src2Ele * (int32_t)sizeof(float)); // x2 batched fp32
        pipe_->InitBuffer(bufX1Exp_, en * (int64_t)sizeof(float));     // x1 expand fp32 (also result)
        pipe_->InitBuffer(bufX2Exp_, en * (int64_t)sizeof(float));     // x2 expand fp32
        pipe_->InitBuffer(qY_, 2, en * (int64_t)sizeof(T));            // output G*P*R (T)
    } else {
        int64_t en = td_->elemNum;
        int32_t alignT = CDIST_BRC_BLOCK / (int32_t)sizeof(T);
        int32_t src1Ele = AlignUp((int32_t)td_->ubFormerP, alignT);
        int32_t src2Ele = AlignUp((int32_t)td_->rSeg, alignT);
        if (src1Ele < alignT)
            src1Ele = alignT;
        if (src2Ele < alignT)
            src2Ele = alignT;
        // Compact loads go through double-buffered VECIN queues (auto MTE2<->V sync, avoids
        // cross-iteration WAR on a single TBuf staging). qLoad1_=x1(rows), qLoad2_=x2(rlen).
        pipe_->InitBuffer(qLoad1_, 2, src1Ele * (int32_t)sizeof(T));   // x1 compact (T)
        pipe_->InitBuffer(qLoad2_, 2, src2Ele * (int32_t)sizeof(T));   // x2 compact (T)
        pipe_->InitBuffer(bufSrc1_, src1Ele * (int32_t)sizeof(float)); // x1 compact fp32
        pipe_->InitBuffer(bufSrc2_, src2Ele * (int32_t)sizeof(float)); // x2 compact fp32
        pipe_->InitBuffer(bufX1Exp_, en * (int64_t)sizeof(float));     // x1 expand fp32 (also result)
        pipe_->InitBuffer(bufX2Exp_, en * (int64_t)sizeof(float));     // x2 expand fp32
        pipe_->InitBuffer(qY_, 2, en * (int64_t)sizeof(T));
    }
}

// count 个元素的 cdist 单元素归一：p>0 时与 golden(torch.cdist) 对齐，p==0 时非零计数。
template <typename T>
__aicore__ inline void CdistBroadcast<T>::ComputeTile(const LocalTensor<float>& x1exp, const LocalTensor<float>& x2exp,
                                                      const LocalTensor<float>& yfp, int32_t count)
{
    // |x1 - x2|：AbsSub 合并 sub+abs（与 golden 输入语义一致）
    AbsSub(yfp, x1exp, x2exp, count);

    if (p_ == 0.0f) {
        // L0：M==1 => d==0 ? 0 : 1（非零计数）
        Ceil(yfp, yfp, (uint32_t)count);
        Mins(yfp, yfp, (float)1, count);
    } else if (p_ != 1.0f && p_ != 2.0f && p_ != INFINITY) {
        // 一般 p (0<p<∞, p∉{1,2})：与 golden(torch.cdist) 数值行为对齐，复现 (|d|^p)^(1/p)。
        // 中间 |d|^p 在极端 |d| 下上溢(inf)/下溢(0)——与 torch 的 pow 溢出行为一致，
        // 避免 broadcast 直接输出 |d| 导致与 golden 的 inf/0 对比 FAIL。
        // 用 FTZ_FALSE（不 flush denormal）：exp 链中间值 |d|^p 需保住 denormal
        constexpr static ExpConfig expConfig = {ExpAlgo::PRECISION_1ULP_FTZ_FALSE};
        constexpr static LnConfig lnConfig = {LnAlgo::PRECISION_1ULP_FTZ_FALSE};
        Ln<float, lnConfig>(x2exp, yfp, count);   // log|d|
        Muls(x2exp, x2exp, p_, count);            // p·log|d|
        Exp<float, expConfig>(yfp, x2exp, count); // |d|^p   (极端 d 时 inf/0)
        Ln<float, lnConfig>(x2exp, yfp, count);   // log(|d|^p)  (inf→inf, 0→-inf)
        Muls(x2exp, x2exp, 1.0f / p_, count);     // (1/p)·log(...)
        Exp<float, expConfig>(yfp, x2exp, count); // (|d|^p)^(1/p)  (inf→inf, 0→0)
    }
    // p==1 / p==2 / p==inf：M==1 数学上恒等于 |d|，golden 亦精确，保持 |d|。
}

// ---------------- OneDim (P==R==1) ----------------
template <typename T>
__aicore__ inline void CdistBroadcast<T>::ProcessOneDim()
{
    int64_t dimLen = td_->dimLen;
    int64_t tn = td_->tileNum;
    int64_t bn = td_->blockNum;
    if (tn < 1)
        tn = 1;
    if (bn < 1)
        bn = 1;
    // 缺陷① 修复：每核负责一段连续区间 [blockStart, blockEnd)，blockEnd 对 dimLen 做全局 clamp，
    // 每个 tile 的 len 取 min(tn, blockEnd-off)。旧实现按 (blockFormer,blockTail) 独立算每核尾块，
    // 当 ubOuter 不足以填满 bn*blockFormer 时 blockTail 为负、且非末核用满 tn 写超 dimLen（OOB
    // DataCopyPad 踩踏相邻 GM，msprof 多次 launch 下累积成精度 FAIL）。全局 clamp 后 missing/oob/dup 均为 0。
    int64_t ubOuter = (dimLen + tn - 1) / tn;
    int64_t blockFormer = (ubOuter + bn - 1) / bn; // 每核最多承担的 tile 数
    int64_t bid = GetBlockIdx();
    if (bid >= bn)
        return;
    int64_t blockStart = blockFormer * tn * bid;
    if (blockStart >= dimLen)
        return; // 本核无有效数据
    int64_t blockEnd = blockStart + blockFormer * tn;
    if (blockEnd > dimLen)
        blockEnd = dimLen; // 末核 clamp，杜绝越界写
    int32_t alignFp = CDIST_BRC_BLOCK / (int32_t)sizeof(float);

    LocalTensor<float> x1f = bufSrc1_.Get<float>();
    LocalTensor<float> x2f = bufSrc2_.Get<float>();
    LocalTensor<float> yfp = bufX1Exp_.Get<float>();

    for (int64_t off = blockStart; off < blockEnd; off += tn) {
        int64_t len = blockEnd - off;
        if (len > tn)
            len = tn; // 每 tile 长度对本核剩余量 clamp
        if (len <= 0)
            break;
        LocalTensor<T> x1l = qLoad1_.template AllocTensor<T>();
        LocalTensor<T> x2l = qLoad2_.template AllocTensor<T>();
        DataCopyExtParams inExt{1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(x1l, x1Gm_[off], inExt, pad);
        DataCopyPad(x2l, x2Gm_[off], inExt, pad);
        qLoad1_.EnQue(x1l);
        qLoad2_.EnQue(x2l);
        LocalTensor<T> x1v = qLoad1_.template DeQue<T>();
        LocalTensor<T> x2v = qLoad2_.template DeQue<T>();
        if constexpr (!IS_FP32) {
            Cast(x1f, x1v, RoundMode::CAST_NONE, (int32_t)len);
            Cast(x2f, x2v, RoundMode::CAST_NONE, (int32_t)len);
            ComputeTile(x1f, x2f, yfp, (int32_t)len);
        } else {
            ComputeTile(x1v.template ReinterpretCast<float>(), x2v.template ReinterpretCast<float>(), yfp,
                        (int32_t)len);
        }
        qLoad1_.FreeTensor(x1v);
        qLoad2_.FreeTensor(x2v);
        LocalTensor<T> yl = qY_.template AllocTensor<T>();
        if constexpr (!IS_FP32) {
            Cast(yl, yfp, RoundMode::CAST_RINT, (int32_t)len);
        } else {
            DataCopy(yl.template ReinterpretCast<float>(), yfp, AlignUp((int32_t)len, alignFp));
        }
        qY_.EnQue(yl);
        LocalTensor<T> yo = qY_.template DeQue<T>();
        DataCopyExtParams outExt{1, (uint32_t)(len * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[off], yo, outExt);
        qY_.FreeTensor(yo);
    }
}

// ---------------- UB-BRC (静态 rank=2) ----------------
template <typename T>
__aicore__ inline void CdistBroadcast<T>::ProcessUbBrc()
{
    int64_t B = td_->B, P = td_->P, R = td_->R;
    int64_t ubFormerP = td_->ubFormerP, ubOuterP = td_->ubOuterP;
    int64_t rSeg = td_->rSeg, rOuter = td_->rOuter;
    int64_t totalTiles = td_->totalTiles;
    int64_t bn = td_->coreNumT;
    int64_t blockFormer = td_->blockFormerT;
    if (bn < 1)
        bn = 1;
    if (blockFormer < 1)
        blockFormer = 1;
    int64_t bid = GetBlockIdx();
    if (bid >= bn)
        return;
    int64_t tileStart = blockFormer * bid;
    int64_t loops = (bid == bn - 1) ? (totalTiles - blockFormer * (bn - 1)) : blockFormer;

    int32_t alignFp = CDIST_BRC_BLOCK / (int32_t)sizeof(float);

    LocalTensor<float> src1 = bufSrc1_.Get<float>();
    LocalTensor<float> src2 = bufSrc2_.Get<float>();
    LocalTensor<float> x1exp = bufX1Exp_.Get<float>();
    LocalTensor<float> x2exp = bufX2Exp_.Get<float>();

    for (int64_t t = 0; t < loops; t++) {
        int64_t lin = tileStart + t;
        if (lin >= totalTiles)
            break;
        int64_t rTile = lin % rOuter;
        int64_t tmp = lin / rOuter;
        int64_t pTile = tmp % ubOuterP;
        int64_t b = tmp / ubOuterP;

        int64_t p0 = pTile * ubFormerP;
        int64_t rows = (pTile == ubOuterP - 1) ? (P - p0) : ubFormerP;
        int64_t r0 = rTile * rSeg;
        int64_t rlen = (rTile == rOuter - 1) ? (R - r0) : rSeg;
        if (rows <= 0 || rlen <= 0)
            continue;

        DataCopyPadExtParams<T> pad{false, 0, 0, 0};

        // --- x1 compact load (rows) via double-buffered queue (auto MTE2<->V sync) ---
        LocalTensor<T> x1t = qLoad1_.template AllocTensor<T>();
        {
            DataCopyExtParams e1{1, (uint32_t)(rows * sizeof(T)), 0, 0, 0};
            DataCopyPad(x1t, x1Gm_[b * P + p0], e1, pad);
        }
        qLoad1_.EnQue(x1t);
        // --- x2 compact load (rlen) ---
        LocalTensor<T> x2t = qLoad2_.template AllocTensor<T>();
        {
            DataCopyExtParams e2{1, (uint32_t)(rlen * sizeof(T)), 0, 0, 0};
            DataCopyPad(x2t, x2Gm_[b * R + r0], e2, pad);
        }
        qLoad2_.EnQue(x2t);

        LocalTensor<T> x1v = qLoad1_.template DeQue<T>();
        LocalTensor<T> x2v = qLoad2_.template DeQue<T>();
        if constexpr (!IS_FP32) {
            Cast(src1, x1v, RoundMode::CAST_NONE, (int32_t)rows);
            Cast(src2, x2v, RoundMode::CAST_NONE, (int32_t)rlen);
        } else {
            DataCopy(src1, x1v.template ReinterpretCast<float>(), AlignUp((int32_t)rows, alignFp));
            DataCopy(src2, x2v.template ReinterpretCast<float>(), AlignUp((int32_t)rlen, alignFp));
        }
        qLoad1_.FreeTensor(x1v);
        qLoad2_.FreeTensor(x2v);

        // --- Broadcast x1: [rows,1] -> [rows,rlen] (static rank=2) ---
        uint32_t d1Shape[CDIST_STATIC_RANK] = {(uint32_t)rows, (uint32_t)rlen};
        uint32_t s1Shape[CDIST_STATIC_RANK] = {(uint32_t)rows, 1u};
        BroadcastTiling bt1;
        GetBroadcastTilingInfo<float, CDIST_STATIC_RANK>((uint32_t)CDIST_STATIC_RANK, d1Shape, s1Shape, false, bt1);
        Broadcast<float, CDIST_STATIC_RANK>(x1exp, src1, d1Shape, s1Shape, &bt1);

        // --- Broadcast x2: [1,rlen] -> [rows,rlen] (static rank=2) ---
        uint32_t d2Shape[CDIST_STATIC_RANK] = {(uint32_t)rows, (uint32_t)rlen};
        uint32_t s2Shape[CDIST_STATIC_RANK] = {1u, (uint32_t)rlen};
        BroadcastTiling bt2;
        GetBroadcastTilingInfo<float, CDIST_STATIC_RANK>((uint32_t)CDIST_STATIC_RANK, d2Shape, s2Shape, false, bt2);
        Broadcast<float, CDIST_STATIC_RANK>(x2exp, src2, d2Shape, s2Shape, &bt2);

        // --- compute |x1-x2| (p0: count) into x1exp ---
        int32_t count = (int32_t)(rows * rlen);
        ComputeTile(x1exp, x2exp, x1exp, count);

        LocalTensor<T> yl = qY_.template AllocTensor<T>();
        if constexpr (!IS_FP32) {
            Cast(yl, x1exp, RoundMode::CAST_RINT, count);
        } else {
            DataCopy(yl.template ReinterpretCast<float>(), x1exp, AlignUp(count, alignFp));
        }
        qY_.EnQue(yl);
        LocalTensor<T> yo = qY_.template DeQue<T>();
        // Tiling 保证二选一，输出 GM 段恒连续，用单块 DataCopyPad（UB src 也是 packed 连续）：
        //   R-fits (rOuter==1, rlen==R): (rows×R) 行连续 → y[b,p0*R .. +rows*R] 连续。
        //   R-split (rlen<R):            rows==1        → 单行 rlen 连续。
        // 避免 blockCount>1 且 blockLen 非 32B 对齐时对 UB packed 源的错位读取。
        DataCopyExtParams outExt{1, (uint32_t)(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[b * P * R + p0 * R + r0], yo, outExt);
        qY_.FreeTensor(yo);
    }
}

// ---------------- UB-BRC multi-batch (缺陷② 修复, brcBranch==2) ----------------
// 小 tile 场景：P*R 小，把 G=batchPerTile 个 batch 打包进一个 UB tile。
//   一次 DataCopyPad 搬 G*P(x1) / G*R(x2)，逐 batch 静态 rank Broadcast 展开到 (P,R) 平面，
//   计算后一次 DataCopyPad 写 G*P*R。相比每 tile 1 batch，把 per-tile 固定开销摊薄 G 倍。
template <typename T>
__aicore__ inline void CdistBroadcast<T>::ProcessUbBrcMulti()
{
    int64_t B = td_->B, P = td_->P, R = td_->R;
    int64_t G = td_->batchPerTile;
    int64_t totalTiles = td_->totalTiles;
    int64_t bn = td_->coreNumT;
    int64_t blockFormer = td_->blockFormerT;
    if (G < 1)
        G = 1;
    if (bn < 1)
        bn = 1;
    if (blockFormer < 1)
        blockFormer = 1;
    int64_t bid = GetBlockIdx();
    if (bid >= bn)
        return;
    int64_t tileStart = blockFormer * bid;
    int64_t loops = (bid == bn - 1) ? (totalTiles - blockFormer * (bn - 1)) : blockFormer;

    int32_t alignFp = CDIST_BRC_BLOCK / (int32_t)sizeof(float);
    int64_t plane = P * R;

    LocalTensor<float> src1 = bufSrc1_.Get<float>();   // x1 batched fp32 (G*P)
    LocalTensor<float> src2 = bufSrc2_.Get<float>();   // x2 batched fp32 (G*R)
    LocalTensor<float> x1exp = bufX1Exp_.Get<float>(); // (G*P*R) fp32 & result
    LocalTensor<float> x2exp = bufX2Exp_.Get<float>(); // (G*P*R) fp32

    // x2 平面 (1,R)->(P,R) 对每 batch 恒相同 → tiling 外提，仍需逐 batch 调用（packed [g][p][r] 中
    // x2 在 p 维复制，无法用 rank2 一次覆盖）。x1 则可整块一次广播（见下 bt1All）。
    uint32_t dShape[CDIST_STATIC_RANK] = {(uint32_t)P, (uint32_t)R};
    uint32_t s2Shape[CDIST_STATIC_RANK] = {1u, (uint32_t)R};
    BroadcastTiling bt2;
    GetBroadcastTilingInfo<float, CDIST_STATIC_RANK>((uint32_t)CDIST_STATIC_RANK, dShape, s2Shape, false, bt2);

    for (int64_t t = 0; t < loops; t++) {
        int64_t lin = tileStart + t;
        if (lin >= totalTiles)
            break;
        int64_t b0 = lin * G;
        int64_t g = (b0 + G <= B) ? G : (B - b0); // 尾块：本 tile 实际 batch 数
        if (g <= 0)
            break;

        DataCopyPadExtParams<T> pad{false, 0, 0, 0};

        // --- x1 batched load: G*P 连续 (x1 layout (B,P)) ---
        LocalTensor<T> x1t = qLoad1_.template AllocTensor<T>();
        {
            DataCopyExtParams e1{1, (uint32_t)(g * P * sizeof(T)), 0, 0, 0};
            DataCopyPad(x1t, x1Gm_[b0 * P], e1, pad);
        }
        qLoad1_.EnQue(x1t);
        // --- x2 batched load: G*R 连续 (x2 layout (B,R)) ---
        LocalTensor<T> x2t = qLoad2_.template AllocTensor<T>();
        {
            DataCopyExtParams e2{1, (uint32_t)(g * R * sizeof(T)), 0, 0, 0};
            DataCopyPad(x2t, x2Gm_[b0 * R], e2, pad);
        }
        qLoad2_.EnQue(x2t);

        LocalTensor<T> x1v = qLoad1_.template DeQue<T>();
        LocalTensor<T> x2v = qLoad2_.template DeQue<T>();
        if constexpr (!IS_FP32) {
            Cast(src1, x1v, RoundMode::CAST_NONE, (int32_t)(g * P));
            Cast(src2, x2v, RoundMode::CAST_NONE, (int32_t)(g * R));
        } else {
            DataCopy(src1, x1v.template ReinterpretCast<float>(), AlignUp((int32_t)(g * P), alignFp));
            DataCopy(src2, x2v.template ReinterpretCast<float>(), AlignUp((int32_t)(g * R), alignFp));
        }
        qLoad1_.FreeTensor(x1v);
        qLoad2_.FreeTensor(x2v);

        // --- x1: 整块一次广播 (g*P, 1) -> (g*P, R)，packed [g][p][r] 恰为 x1exp（一次调用覆盖全部 g）---
        {
            uint32_t d1All[CDIST_STATIC_RANK] = {(uint32_t)(g * P), (uint32_t)R};
            uint32_t s1All[CDIST_STATIC_RANK] = {(uint32_t)(g * P), 1u};
            BroadcastTiling bt1All;
            GetBroadcastTilingInfo<float, CDIST_STATIC_RANK>((uint32_t)CDIST_STATIC_RANK, d1All, s1All, false, bt1All);
            Broadcast<float, CDIST_STATIC_RANK>(x1exp, src1, d1All, s1All, &bt1All);
        }
        // --- x2: packed [g][p][r] 中 x2 在 p 维复制 → 逐 batch (1,R)->(P,R)（tiling 已外提） ---
        for (int64_t j = 0; j < g; j++) {
            Broadcast<float, CDIST_STATIC_RANK>(x2exp[j * plane], src2[j * R], dShape, s2Shape, &bt2);
        }

        // --- compute |x1-x2| (+p0 归一) 整块 g*P*R ---
        int32_t count = (int32_t)(g * plane);
        ComputeTile(x1exp, x2exp, x1exp, count);

        LocalTensor<T> yl = qY_.template AllocTensor<T>();
        if constexpr (!IS_FP32) {
            Cast(yl, x1exp, RoundMode::CAST_RINT, count);
        } else {
            DataCopy(yl.template ReinterpretCast<float>(), x1exp, AlignUp(count, alignFp));
        }
        qY_.EnQue(yl);
        LocalTensor<T> yo = qY_.template DeQue<T>();
        // 输出 y[b0..b0+g] 的 (P,R) 平面在 GM 连续 (out layout (B,P,R)) → 单块 DataCopyPad。
        DataCopyExtParams outExt{1, (uint32_t)(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[b0 * plane], yo, outExt);
        qY_.FreeTensor(yo);
    }
}

template <typename T>
__aicore__ inline void CdistBroadcast<T>::Process()
{
    if (td_->brcBranch == 0) {
        ProcessOneDim();
    } else if (td_->brcBranch == 2) {
        ProcessUbBrcMulti();
    } else {
        ProcessUbBrc();
    }
}

} // namespace NsCdist

#endif // !defined(__NPU_HOST__)
#endif // CDIST_BROADCAST_H
