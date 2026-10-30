/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file bincount.h
 * \brief bincount kernel impl
 */

#ifndef BINCOUNT_H_
#define BINCOUNT_H_

#include "bincount_tiling_data.h"
#include "bincount_tiling_key.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace NsBincount {
using namespace AscendC;

constexpr int32_t BINCOUNT_BLOCK_BYTES = 32;
constexpr int32_t MIN_SLOT = 4;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BINCOUNT_ZTILE_W = 2048;  // 大 L 路径清零块（int32 元素数 = 8192B,32B 对齐）
constexpr int32_t BINCOUNT_BLOCK_WORDS = 8; // 一个 32B block = 8 个 int32

template <typename T>
struct BincountAccumType {
    using Type = T;
};

template <>
struct BincountAccumType<double> {
    using Type = float;
};

__aicore__ inline uint64_t FloatToDoubleBits(float value)
{
    union FloatBits {
        float f;
        uint32_t u;
    } in;
    in.f = value;

    // IEEE 754 位布局：float32 为 sign[31]、exponent[30:23]、fraction[22:0]；
    // float64 为 sign[63]、exponent[62:52]、fraction[51:0]。
    uint64_t sign = static_cast<uint64_t>(in.u >> 31) << 63;
    uint32_t exp = (in.u >> 23) & 0xFFU;
    uint32_t frac = in.u & 0x7FFFFFU;

    if (exp == 0U) {
        return sign;
    }
    if (exp == 0xFFU) {
        uint64_t frac64 = static_cast<uint64_t>(frac) << 29;
        return sign | (0x7FFULL << 52) | frac64;
    }

    uint64_t exp64 = static_cast<uint64_t>(static_cast<int32_t>(exp) - 127 + 1023);
    uint64_t frac64 = static_cast<uint64_t>(frac) << 29;
    return sign | (exp64 << 52) | frac64;
}

__aicore__ inline float DoubleBitsToFloat(uint64_t bits)
{
    uint32_t sign = static_cast<uint32_t>(bits >> 63) << 31;
    uint32_t exp = static_cast<uint32_t>((bits >> 52) & 0x7FFULL);
    uint64_t frac = bits & 0xFFFFFFFFFFFFFULL;
    union FloatBits {
        uint32_t u;
        float f;
    } out;
    if (exp == 0U) { // 0 或非规格 double -> float 下溢为 ±0
        out.u = sign;
        return out.f;
    }
    if (exp == 0x7FFU) { // inf / nan
        uint32_t f32frac = static_cast<uint32_t>(frac >> 29);
        out.u = sign | 0x7F800000U | (frac != 0ULL ? (f32frac | 1U) : 0U);
        return out.f;
    }
    int32_t e = static_cast<int32_t>(exp) - 1023 + 127;
    if (e >= 0xFF) { // 上溢 -> inf
        out.u = sign | 0x7F800000U;
        return out.f;
    }
    if (e <= 0) { // 下溢 -> ±0（不处理 float 非规格,权重场景足够）
        out.u = sign;
        return out.f;
    }
    uint32_t f32frac = static_cast<uint32_t>(frac >> 29);
    out.u = sign | (static_cast<uint32_t>(e) << 23) | f32frac;
    return out.f;
}

// GM 输出读改写：out[idx] += w。double 输出经位拼接读改写（910B AI Core 无 fp64
// 标量运算）。
template <typename TYPE_OUT, typename TYPE_ACC>
struct BincountGmAccum {
    __aicore__ inline static void Add(AscendC::GlobalTensor<TYPE_OUT>& outGm, AscendC::GlobalTensor<uint64_t>&,
                                      uint64_t idx, TYPE_ACC w)
    {
        outGm.SetValue(idx, static_cast<TYPE_OUT>(static_cast<TYPE_ACC>(outGm.GetValue(idx)) + w));
    }
};

template <typename TYPE_ACC>
struct BincountGmAccum<double, TYPE_ACC> {
    __aicore__ inline static void Add(AscendC::GlobalTensor<double>&, AscendC::GlobalTensor<uint64_t>& outBitsGm,
                                      uint64_t idx, TYPE_ACC w)
    {
        float cur = DoubleBitsToFloat(outBitsGm.GetValue(idx));
        outBitsGm.SetValue(idx, FloatToDoubleBits(cur + static_cast<float>(w)));
    }
};

template <typename TYPE_OUT, typename TYPE_ACC>
struct BincountOutputWriter {
    __aicore__ inline static void Write(AscendC::GlobalTensor<TYPE_OUT>& outGm, AscendC::GlobalTensor<uint64_t>&,
                                        uint64_t idx, TYPE_ACC value)
    {
        outGm.SetValue(idx, static_cast<TYPE_OUT>(value));
    }
};

template <typename TYPE_ACC>
struct BincountOutputWriter<double, TYPE_ACC> {
    __aicore__ inline static void Write(AscendC::GlobalTensor<double>&, AscendC::GlobalTensor<uint64_t>& outBitsGm,
                                        uint64_t idx, TYPE_ACC value)
    {
        outBitsGm.SetValue(idx, FloatToDoubleBits(static_cast<float>(value)));
    }
};

// 单核整段写出：非 double 时 acc 类型即 out 类型，整块 DataCopy(UB->GM)
// 写,尾部不足 一个 32B 块的部分用标量补；显著加速大 outLength（避免逐桶标量 GM
// 写）。double 走位拼接标量。
template <typename TYPE_OUT, typename TYPE_ACC>
struct BincountSingleWrite {
    __aicore__ inline static void Write(AscendC::GlobalTensor<TYPE_OUT>& outGm, AscendC::GlobalTensor<uint64_t>&,
                                        AscendC::LocalTensor<TYPE_ACC>& hist, uint64_t outLen)
    {
        constexpr uint32_t blockElems = BINCOUNT_BLOCK_BYTES / sizeof(TYPE_OUT);
        uint64_t vecCount = (outLen / blockElems) * blockElems;
        if (vecCount > 0) {
            AscendC::DataCopy(outGm, hist, vecCount);
        }
        for (uint64_t k = vecCount; k < outLen; k++) {
            outGm.SetValue(k, static_cast<TYPE_OUT>(hist.GetValue(k)));
        }
    }
};

template <typename TYPE_ACC>
struct BincountSingleWrite<double, TYPE_ACC> {
    __aicore__ inline static void Write(AscendC::GlobalTensor<double>&, AscendC::GlobalTensor<uint64_t>& outBitsGm,
                                        AscendC::LocalTensor<TYPE_ACC>& hist, uint64_t outLen)
    {
        for (uint64_t k = 0; k < outLen; k++) {
            outBitsGm.SetValue(k, FloatToDoubleBits(static_cast<float>(hist.GetValue(k))));
        }
    }
};

template <typename TYPE_SELF, typename TYPE_OUT>
class Bincount {
public:
    using TYPE_ACC = typename BincountAccumType<TYPE_OUT>::Type;
    using TYPE_WEIGHT = TYPE_ACC;

    __aicore__ inline Bincount() {}
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR weights, GM_ADDR out, GM_ADDR workspace,
                                const BincountTilingData* t)
    {
        blockIdx_ = static_cast<uint64_t>(AscendC::GetBlockIdx());
        totalNum_ = t->totalNum;
        outLength_ = t->outLength;
        coreNum_ = t->coreNum;
        tileDataNum_ = t->tileDataNum > 0 ? t->tileDataNum : 4096;
        hasWeights_ = t->hasWeights;

        uint64_t base = (coreNum_ > 0) ? totalNum_ / coreNum_ : totalNum_;
        uint64_t rem = (coreNum_ > 0) ? totalNum_ % coreNum_ : 0;
        if (blockIdx_ < rem) {
            coreData_ = base + 1;
            coreStart_ = blockIdx_ * (base + 1);
        } else {
            coreData_ = base;
            coreStart_ = rem * (base + 1) + (blockIdx_ - rem) * base;
        }

        selfGm_.SetGlobalBuffer((__gm__ TYPE_SELF*)self + coreStart_, coreData_ > 0 ? coreData_ : 1);
        if (hasWeights_) {
            weightGm_.SetGlobalBuffer((__gm__ TYPE_WEIGHT*)weights + coreStart_, coreData_ > 0 ? coreData_ : 1);
        }
        outGm_.SetGlobalBuffer((__gm__ TYPE_OUT*)out, outLength_);
        outBitsGm_.SetGlobalBuffer((__gm__ uint64_t*)out, outLength_);
        minWsGm_.SetGlobalBuffer((__gm__ int64_t*)workspace, coreNum_ * MIN_SLOT);

        largeL_ = t->largeL;
        if (largeL_) {
            // 大 L 路径：直方图放不下 UB,改为 0 号核直接散射到 GM。
            // 不申请 histBuf/tmpHistBuf（L 过大会爆 UB）；准备全量输入视图 +
            // 输出字节视图 + 清零块。
            selfFullGm_.SetGlobalBuffer((__gm__ TYPE_SELF*)self, totalNum_ > 0 ? totalNum_ : 1);
            if (hasWeights_) {
                weightFullGm_.SetGlobalBuffer((__gm__ TYPE_WEIGHT*)weights, totalNum_ > 0 ? totalNum_ : 1);
            }
            outWordGm_.SetGlobalBuffer((__gm__ int32_t*)out,
                                       outLength_ * static_cast<uint64_t>(sizeof(TYPE_OUT)) / sizeof(int32_t));
            pipe_.InitBuffer(zeroBuf_, BINCOUNT_ZTILE_W * sizeof(int32_t));
        } else {
            histWsGm_.SetGlobalBuffer((__gm__ TYPE_ACC*)(workspace + coreNum_ * MIN_SLOT * sizeof(int64_t)),
                                      coreNum_ * AlignedLen());
            pipe_.InitBuffer(inQueueSelf_, BUFFER_NUM, tileDataNum_ * sizeof(TYPE_SELF));
            if (hasWeights_) {
                pipe_.InitBuffer(inQueueW_, BUFFER_NUM, tileDataNum_ * sizeof(TYPE_WEIGHT));
            }
            pipe_.InitBuffer(histBuf_, AlignedLen() * sizeof(TYPE_ACC));
            pipe_.InitBuffer(tmpHistBuf_, AlignedLen() * sizeof(TYPE_ACC));
            pipe_.InitBuffer(minBlkBuf_, MIN_SLOT * sizeof(int64_t));
            pipe_.InitBuffer(minAllBuf_, coreNum_ * MIN_SLOT * sizeof(int64_t));
        }
    }

    // 大 L 回退路径：0 号核求全局 min 做负数检查、清零 GM
    // 输出、再逐元素散射写回。
    // 输出长度只受显存限制；输入很小（典型场景为极端值/大
    // minlength）时开销主要在清零。
    __aicore__ inline void ProcessLargeL()
    {
        if (blockIdx_ != 0) {
            return; // 单核完成,其余核直接退出（不调用 SyncAll,避免死锁）
        }
        constexpr int64_t BIG = static_cast<int64_t>(1) << 62;

        // 1) 全量求 min，用于负数检查
        int64_t gMin = BIG;
        for (uint64_t i = 0; i < totalNum_; i++) {
            int64_t v = static_cast<int64_t>(selfFullGm_.GetValue(i));
            if (v < gMin) {
                gMin = v;
            }
        }
        if (gMin == BIG) {
            gMin = 0;
        }
        // 负数需求改为「检查 + 报错」：含负数则运行期报错中止，不再做偏移映射。
        // 非负输入行为与 torch.bincount / 内置一致（idx = value）。
        ascendc_assert(gMin >= 0, "BinCount: input 'self' contains negative value, "
                                  "which is not supported "
                                  "(non-negative integers required).\n");

        // 2) 清零 GM 输出（按 int32 字,DataCopy 批量 32B 对齐 + 标量尾巴）
        uint64_t totalWords = outLength_ * static_cast<uint64_t>(sizeof(TYPE_OUT)) / sizeof(int32_t);
        AscendC::LocalTensor<int32_t> zt = zeroBuf_.Get<int32_t>();
        AscendC::Duplicate(zt, static_cast<int32_t>(0), BINCOUNT_ZTILE_W);
        uint64_t pos = 0;
        while (pos + BINCOUNT_BLOCK_WORDS <= totalWords) {
            uint64_t chunk = totalWords - pos;
            if (chunk > static_cast<uint64_t>(BINCOUNT_ZTILE_W)) {
                chunk = static_cast<uint64_t>(BINCOUNT_ZTILE_W);
            }
            chunk = (chunk / BINCOUNT_BLOCK_WORDS) * BINCOUNT_BLOCK_WORDS; // 取 32B(8 字)倍数
            if (chunk == 0) {
                break;
            }
            AscendC::DataCopy(outWordGm_[pos], zt, static_cast<int32_t>(chunk));
            pos += chunk;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint64_t p = pos; p < totalWords; p++) {
            outWordGm_.SetValue(p, static_cast<int32_t>(0));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // 3) 逐元素散射：out[value] += (weights or 1)
        for (uint64_t i = 0; i < totalNum_; i++) {
            int64_t idx = static_cast<int64_t>(selfFullGm_.GetValue(i));
            if (idx < 0 || static_cast<uint64_t>(idx) >= outLength_) {
                continue;
            }
            TYPE_ACC w = hasWeights_ ? static_cast<TYPE_ACC>(weightFullGm_.GetValue(i)) : static_cast<TYPE_ACC>(1);
            BincountGmAccum<TYPE_OUT, TYPE_ACC>::Add(outGm_, outBitsGm_, static_cast<uint64_t>(idx), w);
        }
    }

    __aicore__ inline void Process()
    {
        if (largeL_) {
            ProcessLargeL();
            return;
        }
        int64_t alignedLen = static_cast<int64_t>(AlignedLen());
        AscendC::LocalTensor<TYPE_ACC> hist = histBuf_.Get<TYPE_ACC>();
        int64_t localMin = ScatterHist(hist, alignedLen); // 单遍散射 + 跟踪局部 min

        // 单核快路径：无 workspace / 无 SyncAll / 无归并，整段向量化写出（小输入 +
        // 大输出主路径）。
        if (coreNum_ <= 1) {
            ascendc_assert(localMin >= 0, "BinCount: input 'self' contains negative "
                                          "value, which is not supported "
                                          "(non-negative integers required).\n");
            BincountSingleWrite<TYPE_OUT, TYPE_ACC>::Write(outGm_, outBitsGm_, hist, outLength_);
            return;
        }
        MergeMultiCore(hist, localMin,
                       alignedLen); // 多核：单道 SyncAll 后 0 号核归并写出
    }

private:
    __aicore__ inline uint64_t AlignedLen()
    {
        return ((outLength_ * sizeof(TYPE_ACC) + BINCOUNT_BLOCK_BYTES - 1) / BINCOUNT_BLOCK_BYTES *
                BINCOUNT_BLOCK_BYTES) /
               sizeof(TYPE_ACC);
    }

    // 单遍散射到本核私有直方图，同遍跟踪局部 min（用于负数检查）；idx = value。
    // 返回该核的局部 min（无元素时为 0）。
    __aicore__ inline int64_t ScatterHist(AscendC::LocalTensor<TYPE_ACC>& hist, int64_t alignedLen)
    {
        constexpr int64_t BIG = static_cast<int64_t>(1) << 62;
        uint64_t fullTiles = (tileDataNum_ > 0) ? coreData_ / tileDataNum_ : 0;
        uint64_t tailStart = fullTiles * tileDataNum_;
        for (int64_t k = 0; k < alignedLen; k++) {
            hist.SetValue(k, static_cast<TYPE_ACC>(0));
        }
        int64_t localMin = BIG;
        for (uint64_t tIdx = 0; tIdx < fullTiles; tIdx++) {
            AscendC::LocalTensor<TYPE_SELF> s = inQueueSelf_.AllocTensor<TYPE_SELF>();
            AscendC::DataCopy(s, selfGm_[tIdx * tileDataNum_], tileDataNum_);
            inQueueSelf_.EnQue(s);
            AscendC::LocalTensor<TYPE_WEIGHT> wt;
            if (hasWeights_) {
                wt = inQueueW_.AllocTensor<TYPE_WEIGHT>();
                AscendC::DataCopy(wt, weightGm_[tIdx * tileDataNum_], tileDataNum_);
                inQueueW_.EnQue(wt);
                wt = inQueueW_.DeQue<TYPE_WEIGHT>();
            }
            s = inQueueSelf_.DeQue<TYPE_SELF>();
            for (uint64_t k = 0; k < tileDataNum_; k++) {
                AccumOne(hist, static_cast<int64_t>(s.GetValue(k)),
                         hasWeights_ ? static_cast<TYPE_ACC>(wt.GetValue(k)) : static_cast<TYPE_ACC>(1), localMin);
            }
            inQueueSelf_.FreeTensor(s);
            if (hasWeights_) {
                inQueueW_.FreeTensor(wt);
            }
        }
        for (uint64_t i = tailStart; i < coreData_; i++) {
            AccumOne(hist, static_cast<int64_t>(selfGm_.GetValue(i)),
                     hasWeights_ ? static_cast<TYPE_ACC>(weightGm_.GetValue(i)) : static_cast<TYPE_ACC>(1), localMin);
        }
        return (localMin == BIG) ? 0 : localMin;
    }

    // 单元素累加：更新 localMin；非负且落在 [0,outLength) 时 hist[v] += w。
    __aicore__ inline void AccumOne(AscendC::LocalTensor<TYPE_ACC>& hist, int64_t v, TYPE_ACC w, int64_t& localMin)
    {
        if (v < localMin) {
            localMin = v;
        }
        if (v < 0 || static_cast<uint64_t>(v) >= outLength_) {
            return;
        }
        uint64_t u = static_cast<uint64_t>(v);
        hist.SetValue(u, hist.GetValue(u) + w);
    }

    // 多核归并：本核 min + 直方图写 workspace，单道 SyncAll 后 0
    // 号核做负数检查并归并写出。
    __aicore__ inline void MergeMultiCore(AscendC::LocalTensor<TYPE_ACC>& hist, int64_t localMin, int64_t alignedLen)
    {
        constexpr int64_t BIG = static_cast<int64_t>(1) << 62;
        AscendC::LocalTensor<int64_t> minBlk = minBlkBuf_.Get<int64_t>();
        for (int32_t j = 0; j < MIN_SLOT; j++) {
            minBlk.SetValue(j, BIG);
        }
        minBlk.SetValue(0, localMin);
        AscendC::DataCopy(minWsGm_[blockIdx_ * MIN_SLOT], minBlk, MIN_SLOT);
        AscendC::DataCopy(histWsGm_[blockIdx_ * alignedLen], hist, alignedLen);
        AscendC::SyncAll();
        if (blockIdx_ != 0) {
            return;
        }
        AscendC::LocalTensor<int64_t> minAll = minAllBuf_.Get<int64_t>();
        AscendC::DataCopy(minAll, minWsGm_, coreNum_ * MIN_SLOT);
        int64_t gMin = BIG;
        for (uint64_t c = 0; c < coreNum_; c++) {
            int64_t m = minAll.GetValue(c * MIN_SLOT);
            if (m < gMin) {
                gMin = m;
            }
        }
        ascendc_assert(gMin == BIG || gMin >= 0, "BinCount: input 'self' contains negative value, which is "
                                                 "not supported "
                                                 "(non-negative integers required).\n");
        AscendC::LocalTensor<TYPE_ACC> tmp = tmpHistBuf_.Get<TYPE_ACC>();
        for (uint64_t c = 1; c < coreNum_; c++) {
            AscendC::DataCopy(tmp, histWsGm_[c * alignedLen], alignedLen);
            for (uint64_t k = 0; k < outLength_; k++) {
                hist.SetValue(k, hist.GetValue(k) + tmp.GetValue(k));
            }
        }
        for (uint64_t k = 0; k < outLength_; k++) {
            BincountOutputWriter<TYPE_OUT, TYPE_ACC>::Write(outGm_, outBitsGm_, k, hist.GetValue(k));
        }
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueSelf_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueW_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> histBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpHistBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> minBlkBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> minAllBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroBuf_; // 大 L 路径清零块
    AscendC::GlobalTensor<TYPE_SELF> selfGm_;
    AscendC::GlobalTensor<TYPE_WEIGHT> weightGm_;
    AscendC::GlobalTensor<TYPE_SELF> selfFullGm_;     // 大 L 路径：全量输入视图
    AscendC::GlobalTensor<TYPE_WEIGHT> weightFullGm_; // 大 L 路径：全量权重视图
    AscendC::GlobalTensor<TYPE_OUT> outGm_;
    AscendC::GlobalTensor<uint64_t> outBitsGm_;
    AscendC::GlobalTensor<int32_t> outWordGm_; // 大 L 路径：输出 int32 字视图（清零用）
    AscendC::GlobalTensor<int64_t> minWsGm_;
    AscendC::GlobalTensor<TYPE_ACC> histWsGm_;
    uint64_t blockIdx_ = 0;
    uint64_t totalNum_ = 0;
    uint64_t outLength_ = 0;
    uint64_t coreNum_ = 1;
    uint64_t coreStart_ = 0;
    uint64_t coreData_ = 0;
    uint64_t tileDataNum_ = 4096;
    uint32_t hasWeights_ = 0;
    uint32_t largeL_ = 0;
};

template <typename TYPE_SELF, typename TYPE_OUT>
using KernelBincount = Bincount<TYPE_SELF, TYPE_OUT>;
} // namespace NsBincount
#endif // BINCOUNT_H_
