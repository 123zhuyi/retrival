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
 * \file polar.h
 * \brief Polar kernel —— 同 shape input/angle 逐元素：out = input*(cos(angle)+i*sin(angle))。
 *        广播由 op_api 层 BroadcastTo + Contiguous 完成后再调本 kernel；
 *        kernel 假设 input.shape == angle.shape == out.shape；
 *        out 为 complex64（real/imag 交织 fp32，用 Gather + 静态 offset 表组装）。
 *        平台：Atlas A2 (ascend910b) / A3 (ascend910_93)；vector 不支持 complex64，
 *        complex64 当 2*fp32 交织处理。
 *
 *        资源布局（BUFFER_NUM=2 double buffer）：
 *          qIn        VECIN  : 2 buffer × (2×T fp32) = 2 × 16KB —— [abs(0:n) | ang(0:n)] 合并
 *          qOut       VECOUT : 2 buffer × (2×T fp32) = 2 × 16KB —— complex64 交织
 *          bufPacked  VECCALC: 1 buffer × (2×T fp32) = 16KB     —— [cos | sin] 中间结果
 *          bufOff     VECCALC: 1 buffer × (2×T u32 ) = 16KB     —— Gather 静态 offset 表
 *          bufSinCosTmp VECCALC: tilingData.tmpBufferSize       —— Sin/Cos 显式 sharedTmpBuffer
 *                                                                  （由 host GetSinMaxMinTmpSize 算）
 *        TPipe Tensor 数：qIn(2) + qOut(2) + bufPacked(1) + bufOff(1) + bufSinCosTmp(1) = 7 (≤8 上限)。
 *        UB 用量：32 + 32 + 16 + 16 + 24 = 120 KB / 192 KB。
 *
 *        合并 qAbs+qAng → qIn 的关键：每个 qIn buffer 容纳 2T = 4096 fp32；
 *        CopyIn 阶段 DataCopyPad input → qIn[0:n]，DataCopyPad angle → qIn[T:T+n]；
 *        Compute 阶段 Sin/Cos 取 qIn[T:T+n] 作为 source，Mul 取 qIn[0:n] 作为 abs。
 */
#ifndef POLAR_H_
#define POLAR_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/math/sin.h"
#include "lib/math/cos.h"
#include "polar_tiling_data.h"
#include "polar_tiling_key.h"

namespace NsPolar {

constexpr int32_t BUFFER_NUM = 2;

class KernelPolar {
public:
    __aicore__ inline KernelPolar() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR angle, GM_ADDR out, const PolarTilingData& t,
                                AscendC::TPipe* pipe)
    {
        tileLen_ = t.tileLen;
        bcastMode_ = t.bcastMode;
        K_ = t.anN;

        uint32_t bid = AscendC::GetBlockIdx();
        if (bid < t.bigCoreNum) {
            coreLen_ = t.bigCoreLen;
            coreStart_ = bid * t.bigCoreLen;
        } else {
            coreLen_ = t.smallCoreLen;
            coreStart_ = t.bigCoreNum * t.bigCoreLen + (bid - t.bigCoreNum) * t.smallCoreLen;
        }

        // input/out 永远是 outN 长度（aclnn 层已把 input BroadcastTo out）
        inGm_.SetGlobalBuffer((__gm__ float*)input + coreStart_, coreLen_);
        outGmF_.SetGlobalBuffer((__gm__ float*)out + 2u * coreStart_, 2u * coreLen_);
        // angle：bcastMode=1 取整段 [0:K]（不偏移，周期复用）；same-shape 按 coreStart 偏移
        if (bcastMode_ == 1u) {
            anGm_.SetGlobalBuffer((__gm__ float*)angle, K_);
        } else {
            anGm_.SetGlobalBuffer((__gm__ float*)angle + coreStart_, coreLen_);
        }

        // qIn: 每 buffer 容 2T fp32（[abs|ang]），BUFFER_NUM=2 double buffer
        pipe->InitBuffer(qIn_, BUFFER_NUM, 2u * tileLen_ * sizeof(float));
        pipe->InitBuffer(qOut_, BUFFER_NUM, 2u * tileLen_ * sizeof(float));
        pipe->InitBuffer(bufPacked_, 2u * tileLen_ * sizeof(float)); // [cos | sin]
        pipe->InitBuffer(bufOff_, 2u * tileLen_ * sizeof(uint32_t)); // 交织 offset 表
        pipe->InitBuffer(bufSinCosTmp_, t.tmpBufferSize); // Sin/Cos sharedTmpBuffer / bcast 常驻 cos/sin tile

        BuildOffsetTable();
    }

    __aicore__ inline void Process()
    {
        if (bcastMode_ == 1u) {
            // angle inner-broadcast：周期 K，每 tile 整数个 period → cos/sin tile pattern 恒定，只 precompute 一次
            PrecomputeBcastCosSin();
            uint32_t done = 0;
            while (done < coreLen_) {
                uint32_t rem = coreLen_ - done;
                uint32_t n = rem < tileLen_ ? rem : tileLen_;
                if (n >= K_)
                    n = (n / K_) * K_; // n 取 K 整数倍，保证 period 对齐
                ProcessTileBcast(done, n);
                done += n;
            }
        } else {
            uint32_t done = 0;
            while (done < coreLen_) {
                uint32_t n = (coreLen_ - done) < tileLen_ ? (coreLen_ - done) : tileLen_;
                ProcessTile(done, n);
                done += n;
            }
        }
    }

private:
    // 交织 offset 表：off[j] = 4·(j>>1) + (j&1)·4T （元素 j 选自 cos 部分还是 sin 部分的字节地址）
    //   = 4T·j + (4 - 8T)·(j>>1)，纯 int32 矢量构造，O(1) 一次 Init 后整 kernel 周期复用
    __aicore__ inline void BuildOffsetTable()
    {
        const int32_t M = (int32_t)(2u * tileLen_);
        const int32_t T = (int32_t)tileLen_;
        AscendC::LocalTensor<int32_t> A = bufOff_.Get<uint32_t>().ReinterpretCast<int32_t>();
        AscendC::LocalTensor<int32_t> B = bufPacked_.Get<float>().ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(A, (int32_t)0, M);
        AscendC::ShiftRight(B, A, (int32_t)1, M);
        AscendC::Muls(A, A, (int32_t)(4 * T), M);
        AscendC::Muls(B, B, (int32_t)(4 - 8 * T), M);
        AscendC::Add(A, A, B, M);
    }

    // 交织 + 写出：Gather 静态 offset 表把 [real|imag] 重排为 [r0,i0,r1,i1,...]，complex64 视作 2*fp32、
    // DataCopyPad 写 2n 个 fp32。same-shape 与 inner-bcast 两路共用（packed 均为 real[0:n]/imag[T:T+n]）。
    __aicore__ inline void InterleaveAndStore(AscendC::LocalTensor<float>& packed, uint32_t off, uint32_t n)
    {
        AscendC::LocalTensor<uint32_t> off32 = bufOff_.Get<uint32_t>();
        AscendC::LocalTensor<float> outL = qOut_.AllocTensor<float>();
        AscendC::Gather(outL, packed, off32, (uint32_t)0, 2u * n);
        qOut_.EnQue(outL);
        outL = qOut_.DeQue<float>();
        AscendC::DataCopyExtParams cpo{1, 2u * n * (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(outGmF_[2u * off], outL, cpo);
        qOut_.FreeTensor(outL);
    }

    __aicore__ inline void ProcessTile(uint32_t off, uint32_t n)
    {
        // CopyIn —— input → qIn[0:n], angle → qIn[T:T+n]
        AscendC::LocalTensor<float> inL = qIn_.AllocTensor<float>();
        AscendC::DataCopyExtParams cpi{1, n * (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(inL, inGm_[off], cpi, pad);
        AscendC::DataCopyPad(inL[tileLen_], anGm_[off], cpi, pad);
        qIn_.EnQue(inL);

        inL = qIn_.DeQue<float>();
        AscendC::LocalTensor<float> absSeg = inL;           // abs at offset 0
        AscendC::LocalTensor<float> angSeg = inL[tileLen_]; // ang at offset T

        // Compute —— Cos/Sin 用显式 sharedTmpBuffer（4-arg 重载），避免 PopStackBuffer 占 TPipe stack
        AscendC::LocalTensor<float> packed = bufPacked_.Get<float>();
        AscendC::LocalTensor<uint8_t> sharedTmp = bufSinCosTmp_.Get<uint8_t>();
        AscendC::Cos<float, false>(packed, angSeg, sharedTmp, n);           // packed[0:n] = cos(ang)
        AscendC::Sin<float, false>(packed[tileLen_], angSeg, sharedTmp, n); // packed[T:T+n] = sin(ang)
        // 原地 *abs 得 real/imag
        AscendC::Mul(packed, packed, absSeg, n);                     // real = abs * cos
        AscendC::Mul(packed[tileLen_], packed[tileLen_], absSeg, n); // imag = abs * sin
        qIn_.FreeTensor(inL);

        InterleaveAndStore(packed, off, n);
    }

    // ===== bcastMode=1 inner-broadcast 路径 =====
    // angle 周期 K，每 tile = 整数个 period → cos/sin tile pattern 对所有 tile 相同。
    // Init 阶段 pre Cos/Sin angle[0:K] 并 broadcast 填满 bufSinCosTmp_ 的 cosTile[0:T]/sinTile[T:2T] 常驻；
    // main loop 只读 input → Mul×2 → Gather 交织，省掉每 tile 的 angle 搬运与 Sin/Cos 重算。
    __aicore__ inline void PrecomputeBcastCosSin()
    {
        // 1) angle[0:K] → qIn buffer
        AscendC::LocalTensor<float> aL = qIn_.AllocTensor<float>();
        AscendC::DataCopyExtParams cpi{1, K_ * (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(aL, anGm_, cpi, pad);
        qIn_.EnQue(aL);
        aL = qIn_.DeQue<float>();

        // 2) Cos/Sin 写 bufPacked_[0:K]/[K:2K]，scratch 用 bufPacked_ 尾部 [2K:2T]（N=K 所需 tmp 远小于该区）
        AscendC::LocalTensor<float> packed = bufPacked_.Get<float>();
        AscendC::LocalTensor<uint8_t> scratch = bufPacked_.Get<uint8_t>()[2u * K_ * sizeof(float)];
        AscendC::Cos<float, false>(packed[0], aL, scratch, K_);
        AscendC::Sin<float, false>(packed[K_], aL, scratch, K_);
        qIn_.FreeTensor(aL);

        // 3) broadcast cosT[0:K]→cosTile[0:T]、sinT[0:K]→sinTile[0:T]，常驻 bufSinCosTmp_
        //    （一次性 m=T/K 次 K-段复制，不在 hot loop）
        AscendC::LocalTensor<float> tile = bufSinCosTmp_.Get<uint8_t>().ReinterpretCast<float>();
        const uint32_t mFull = tileLen_ / K_;
        for (uint32_t i = 0; i < mFull; ++i) {
            AscendC::DataCopy(tile[i * K_], packed[0], K_);             // cosTile period i
            AscendC::DataCopy(tile[tileLen_ + i * K_], packed[K_], K_); // sinTile period i
        }
    }

    // bcastMode=1 单 tile：input only HBM read；cos/sin tile 从 bufSinCosTmp_ 常驻区取（前 n 个即正确 period）。
    // off 必 K 对齐、n 是 K 整数倍（host K-block 分核 + main loop 保证）。
    __aicore__ inline void ProcessTileBcast(uint32_t off, uint32_t n)
    {
        // CopyIn input[off:off+n] only
        AscendC::LocalTensor<float> inL = qIn_.AllocTensor<float>();
        AscendC::DataCopyExtParams cpi{1, n * (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(inL, inGm_[off], cpi, pad);
        qIn_.EnQue(inL);
        inL = qIn_.DeQue<float>();

        // Mul：real = cosTile * input → packed[0:n]；imag = sinTile * input → packed[T:T+n]
        AscendC::LocalTensor<float> packed = bufPacked_.Get<float>();
        AscendC::LocalTensor<float> tile = bufSinCosTmp_.Get<uint8_t>().ReinterpretCast<float>();
        AscendC::LocalTensor<float> cosTile = tile;           // [0:T]
        AscendC::LocalTensor<float> sinTile = tile[tileLen_]; // [T:2T]
        AscendC::Mul(packed[0], cosTile, inL, n);
        AscendC::Mul(packed[tileLen_], sinTile, inL, n);
        qIn_.FreeTensor(inL);

        InterleaveAndStore(packed, off, n);
    }

    AscendC::GlobalTensor<float> inGm_, anGm_, outGmF_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> qIn_; // 合并 abs + ang，buffer 大小 2T
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> qOut_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufPacked_, bufOff_, bufSinCosTmp_;
    uint32_t tileLen_{0}, coreLen_{0}, coreStart_{0};
    uint32_t bcastMode_{0}, K_{0};
};

} // namespace NsPolar
#endif // POLAR_H_
