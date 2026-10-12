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
 * \file truncate_div.h
 * \brief TruncateDiv kernel class (A2/A3, ascend910b / ascend910_93).
 *
 * 语义与官方 ops-math TruncateDiv 对齐：对所有 dtype 计算 y = trunc(x1 / x2)，
 * 向零取整（截断取整）。浮点类型同样截断（这是与旧 TBE ops_legacy 的关键差异：
 * ops_legacy 浮点是普通除法不截断，新实现浮点也要 trunc）。
 *
 * 实现：除 int64 外统一提升到 float32 计算：
 *   t = x1 / x2                                （float32 高精度除）
 *   trunc(t) = ceil(min(t,0)) + floor(max(t,0))（向零取整，结果仍为 float，
 *              正数取 floor、负数取 ceil）
 *   y = cast(trunc(t), TY)
 * int64：float32 无法精确表示，走标量 GM 读写，C++ int64 除法本身向零取整。
 *
 * 支持 (TX1, TX2, TY) 组合（与 def 索引一致）：
 *   bf16/bf16/bf16, half/half/half, half/float/float, float/half/float,
 *   float/float/float, float/int32/float, int32/int32/int32, int32/float/float,
 *   uint8/uint8/uint8, int8/int8/int8, int64/int64/int64, int16/int16/int16
 * 除零为未定义行为（int64 标量路径对 0 除数返回 0 以避免标量异常）。
 */

#ifndef TRUNCATEDIV_H
#define TRUNCATEDIV_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "truncate_div_tiling_data.h"

namespace NsTruncateDiv {

using namespace AscendC;

constexpr uint64_t VEC_ALIGN_F32 = 64u; // 256B / sizeof(float)

__aicore__ inline uint64_t CeilAlign(uint64_t v, uint64_t a) { return (v + a - 1u) & ~(a - 1u); }

// 提升到 float32 计算的通用 element-wise 内核（int64 除外）。
template <typename TX1, typename TX2, typename TY>
class TruncateDivKernel {
public:
    __aicore__ inline TruncateDivKernel(const TruncateDivTilingData* tilingData)
        : totalLength(tilingData->totalLength), coreLength(tilingData->coreLength), tileLength(tilingData->tileLength)
    {
        uint64_t blockIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        this->start = blockIdx * this->coreLength;
        if (this->start >= this->totalLength) {
            this->myLength = 0u;
        } else {
            uint64_t remain = this->totalLength - this->start;
            this->myLength = (this->coreLength < remain) ? this->coreLength : remain;
        }
        uint64_t alignedTile = CeilAlign(this->tileLength, VEC_ALIGN_F32);
        this->pipe.InitBuffer(this->inQue0, 1, this->tileLength * sizeof(TX1));
        this->pipe.InitBuffer(this->inQue1, 1, this->tileLength * sizeof(TX2));
        this->pipe.InitBuffer(this->outQue, 1, this->tileLength * sizeof(TY));
        this->pipe.InitBuffer(this->calcBuf0, alignedTile * sizeof(float));
        this->pipe.InitBuffer(this->calcBuf1, alignedTile * sizeof(float));
        this->pipe.InitBuffer(this->tmpBuf, alignedTile * sizeof(float));
        this->pipe.InitBuffer(this->halfBuf, alignedTile * sizeof(half));
    }

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y)
    {
        this->src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ TX1*>(x1) + this->start);
        this->src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ TX2*>(x2) + this->start);
        this->dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ TY*>(y) + this->start);
    }

    __aicore__ inline void Process()
    {
        if (this->myLength == 0u) {
            return;
        }
        AscendC::LocalTensor<float> calc0 = this->calcBuf0.template Get<float>();
        AscendC::LocalTensor<float> calc1 = this->calcBuf1.template Get<float>();
        AscendC::LocalTensor<float> tmp = this->tmpBuf.template Get<float>();

        uint64_t off = 0u;
        while (off < this->myLength) {
            uint64_t len = this->myLength - off;
            if (len > this->tileLength) {
                len = this->tileLength;
            }
            uint64_t clen = CeilAlign(len, VEC_ALIGN_F32);
            this->CopyInAndCast(calc0, calc1, off, len, clen);
            // t = x1 / x2 ; trunc(t) = ceil(min(t,0)) + floor(max(t,0))
            AscendC::Div<float>(tmp, calc0, calc1, clen);
            AscendC::Mins<float>(calc0, tmp, 0.0f, clen);
            AscendC::Ceil<float>(calc0, calc0, clen);
            AscendC::Maxs<float>(calc1, tmp, 0.0f, clen);
            AscendC::Floor<float>(calc1, calc1, clen);
            AscendC::Add<float>(calc0, calc0, calc1, clen);
            this->CastAndCopyOut(calc0, off, len);
            off += len;
        }
    }

private:
    // x1(TX1)->float32 calc0, x2(TX2)->float32 calc1（读满 clen，尾部为无效值不落盘）。
    __aicore__ inline void CopyInAndCast(const AscendC::LocalTensor<float>& calc0,
                                         const AscendC::LocalTensor<float>& calc1, uint64_t off, uint64_t len,
                                         uint64_t clen)
    {
        AscendC::LocalTensor<TX1> s0 = this->inQue0.template AllocTensor<TX1>();
        AscendC::LocalTensor<TX2> s1 = this->inQue1.template AllocTensor<TX2>();
        AscendC::DataCopyExtParams p0{1u, static_cast<uint32_t>(len * sizeof(TX1)), 0u, 0u, 0u};
        AscendC::DataCopyExtParams p1{1u, static_cast<uint32_t>(len * sizeof(TX2)), 0u, 0u, 0u};
        AscendC::DataCopyPadExtParams<TX1> pad0{false, 0u, 0u, static_cast<TX1>(0)};
        AscendC::DataCopyPadExtParams<TX2> pad1{false, 0u, 0u, static_cast<TX2>(0)};
        AscendC::DataCopyPad(s0, this->src0Global[off], p0, pad0);
        AscendC::DataCopyPad(s1, this->src1Global[off], p1, pad1);
        this->inQue0.template EnQue(s0);
        this->inQue1.template EnQue(s1);
        s0 = this->inQue0.template DeQue<TX1>();
        s1 = this->inQue1.template DeQue<TX2>();
        AscendC::LocalTensor<half> htmp = this->halfBuf.template Get<half>();
        this->template CastToFloat<TX1>(calc0, s0, htmp, clen);
        this->template CastToFloat<TX2>(calc1, s1, htmp, clen);
        this->inQue0.template FreeTensor(s0);
        this->inQue1.template FreeTensor(s1);
    }

    template <typename TX>
    __aicore__ inline void CastToFloat(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<TX>& src,
                                       const AscendC::LocalTensor<half>& htmp, uint64_t clen)
    {
        if constexpr (std::is_same_v<TX, float>) {
            AscendC::DataCopy(dst, src, clen); // clen 是 64 的倍数，UB->UB 对齐拷贝
        } else if constexpr (std::is_same_v<TX, int8_t> || std::is_same_v<TX, uint8_t>) {
            // AICORE 无 int8/uint8 <-> float 直转，经 half 过渡（值域 <=255 half 精确）
            AscendC::Cast(htmp, src, AscendC::RoundMode::CAST_NONE, clen);
            AscendC::Cast(dst, htmp, AscendC::RoundMode::CAST_NONE, clen);
        } else {
            // half / bfloat16 / int32 / int16 -> float 直转
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, clen);
        }
    }

    // float32 trunc 结果 -> TY 落盘（trunc 结果为整数值 float，转整型精确）。
    __aicore__ inline void CastAndCopyOut(const AscendC::LocalTensor<float>& calc0, uint64_t off, uint64_t len)
    {
        AscendC::LocalTensor<TY> dstLocal = this->outQue.template AllocTensor<TY>();
        uint64_t clen = CeilAlign(len, VEC_ALIGN_F32);
        if constexpr (std::is_same_v<TY, float>) {
            AscendC::DataCopy(dstLocal, calc0, clen);
        } else if constexpr (std::is_same_v<TY, int8_t> || std::is_same_v<TY, uint8_t>) {
            AscendC::LocalTensor<half> htmp = this->halfBuf.template Get<half>();
            AscendC::Cast(htmp, calc0, AscendC::RoundMode::CAST_NONE, clen);
            AscendC::Cast(dstLocal, htmp, AscendC::RoundMode::CAST_RINT, clen);
        } else {
            // half / bfloat16 / int32 / int16
            AscendC::Cast(dstLocal, calc0, AscendC::RoundMode::CAST_RINT, clen);
        }
        this->outQue.template EnQue(dstLocal);
        dstLocal = this->outQue.template DeQue<TY>();
        AscendC::DataCopyExtParams pOut{1u, static_cast<uint32_t>(len * sizeof(TY)), 0u, 0u, 0u};
        AscendC::DataCopyPad(this->dstGlobal[off], dstLocal, pOut);
        this->outQue.template FreeTensor(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQue0;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQue1;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf0;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> halfBuf;

    AscendC::GlobalTensor<TX1> src0Global;
    AscendC::GlobalTensor<TX2> src1Global;
    AscendC::GlobalTensor<TY> dstGlobal;

    uint64_t totalLength = 0u;
    uint64_t coreLength = 0u;
    uint64_t tileLength = 0u;
    uint64_t start = 0u;
    uint64_t myLength = 0u;
};

// int64：标量 GM 读写，C++ int64 除法本身向零取整；除零返回 0（未定义行为，避免标量异常）。
__aicore__ inline void RunInt64(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TruncateDivTilingData* tilingData)
{
    uint64_t totalLength = tilingData->totalLength;
    uint64_t coreLength = tilingData->coreLength;
    uint64_t start = static_cast<uint64_t>(AscendC::GetBlockIdx()) * coreLength;
    if (start >= totalLength) {
        return;
    }
    uint64_t remain = totalLength - start;
    uint64_t myLength = (coreLength < remain) ? coreLength : remain;

    AscendC::GlobalTensor<int64_t> gx1;
    AscendC::GlobalTensor<int64_t> gx2;
    AscendC::GlobalTensor<int64_t> gy;
    gx1.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(x1));
    gx2.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(x2));
    gy.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(y));
    for (uint64_t i = 0u; i < myLength; i++) {
        uint64_t idx = start + i;
        int64_t a = gx1.GetValue(idx);
        int64_t b = gx2.GetValue(idx);
        int64_t r = (b == 0) ? 0 : (a / b);
        gy.SetValue(idx, r);
    }
}

template <typename TX1, typename TX2, typename TY>
__aicore__ inline void Run(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TruncateDivTilingData* tilingData)
{
    if constexpr (std::is_same_v<TY, int64_t>) {
        RunInt64(x1, x2, y, tilingData);
    } else {
        TruncateDivKernel<TX1, TX2, TY> op(tilingData);
        op.Init(x1, x2, y);
        op.Process();
    }
}

} // namespace NsTruncateDiv

#endif
