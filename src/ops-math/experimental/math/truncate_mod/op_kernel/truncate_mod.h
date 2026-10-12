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
 * \file truncate_mod.h
 * \brief TruncateMod kernel class.
 *
 * element-wise 二元算子。所有 dtype 在 UB 上统一提升到 float 计算：
 *   y = x1 - trunc(x1 / x2) * x2
 * 其中 trunc 向零取整，用 ceil(min(t,0)) + floor(max(t,0)) 实现，无 int32 溢出风险。
 */

#ifndef TRUNCATEMOD_H
#define TRUNCATEMOD_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "truncate_mod_tiling_data.h"

namespace NsTruncateMod {

using namespace AscendC;

constexpr uint64_t VEC_REPEAT_F32 = 64u; // 256B vector repeat / sizeof(float)

template <typename T>
class TruncateModKernel {
public:
    __aicore__ inline TruncateModKernel(const TruncateModTilingData* tilingData)
        : bufferNum(tilingData->bufferNum),
          epochs(tilingData->epochs),
          tileLength(tilingData->tileLength),
          tailTileLength(tilingData->tailTileLength)
    {
        this->globalOffset = tilingData->coreLength * AscendC::GetBlockIdx();
        this->isLastCore = (AscendC::GetBlockIdx() == tilingData->coreNum - 1u);
        if (this->isLastCore) {
            this->epochs = tilingData->epochsForLastCore;
            this->tailTileLength = tilingData->tailTileLengthForLastCore;
            this->tailElems = tilingData->tailElems;
        }
        uint64_t alignedTile = (this->tileLength + VEC_REPEAT_F32 - 1u) & ~(VEC_REPEAT_F32 - 1u);
        this->pipe.InitBuffer(this->inQue0, this->bufferNum, this->tileLength * sizeof(T));
        this->pipe.InitBuffer(this->inQue1, this->bufferNum, this->tileLength * sizeof(T));
        this->pipe.InitBuffer(this->outQue, this->bufferNum, this->tileLength * sizeof(T));
        this->pipe.InitBuffer(this->calcBuf0, alignedTile * sizeof(float));
        this->pipe.InitBuffer(this->calcBuf1, alignedTile * sizeof(float));
        this->pipe.InitBuffer(this->tmpBuf, alignedTile * sizeof(float));
        this->pipe.InitBuffer(this->tmp2Buf, alignedTile * sizeof(float));
        this->pipe.InitBuffer(this->halfBuf, alignedTile * sizeof(half));
    }

    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR src0, GM_ADDR src1, GM_ADDR /*workspace*/)
    {
        this->src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0) + this->globalOffset);
        this->src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1) + this->globalOffset);
        this->dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst) + this->globalOffset);
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> calc0 = this->calcBuf0.template Get<float>();
        AscendC::LocalTensor<float> calc1 = this->calcBuf1.template Get<float>();

        for (uint64_t i = 0u; i < this->epochs; i++) {
            this->CopyIn(calc0, calc1, i * this->tileLength, this->tileLength, this->tileLength);
            this->Compute(calc0, calc1, this->tileLength);
            this->CopyOut(calc0, i * this->tileLength, this->tileLength, this->tileLength);
        }

        if (this->tailTileLength || (this->isLastCore && this->tailElems)) {
            // 用局部变量表示尾块有效元素数，避免改写成员 tailTileLength 影响后续定位/调试。
            uint64_t tailLength = this->tailTileLength;
            if (this->isLastCore && this->tailElems) {
                tailLength += this->tailElems;
            }
            // UB 计算按 block 对齐；GM 搬运用实际长度(见 DataCopyPad)以避免尾块越界读/写。
            uint64_t tailAligned = (tailLength + ELEM_PER_BLOCK - 1u) & ~(ELEM_PER_BLOCK - 1u);
            this->CopyIn(calc0, calc1, this->epochs * this->tileLength, tailLength, tailAligned);
            this->Compute(calc0, calc1, tailAligned);
            this->CopyOut(calc0, this->epochs * this->tileLength, tailLength, tailAligned);
        }
    }

private:
    __aicore__ inline void CopyIn(const AscendC::LocalTensor<float>& calc0, const AscendC::LocalTensor<float>& calc1,
                                  uint64_t offset, uint64_t copyLength, uint64_t ubLength)
    {
        // GM -> UB 按实际长度搬运，尾块非对齐时用 DataCopyPad 避免越界读(仅 A2/A3 支持)。
        AscendC::DataCopyExtParams copyParams{1u, static_cast<uint32_t>(copyLength * sizeof(T)), 0u, 0u, 0u};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0u, 0u, static_cast<T>(0)};

        AscendC::LocalTensor<T> src0Local = this->inQue0.template AllocTensor<T>();
        AscendC::DataCopyPad(src0Local, this->src0Global[offset], copyParams, padParams);
        this->inQue0.template EnQue<T>(src0Local);
        src0Local = this->inQue0.template DeQue<T>();

        AscendC::LocalTensor<T> src1Local = this->inQue1.template AllocTensor<T>();
        AscendC::DataCopyPad(src1Local, this->src1Global[offset], copyParams, padParams);
        this->inQue1.template EnQue<T>(src1Local);
        src1Local = this->inQue1.template DeQue<T>();

        if constexpr (std::is_same_v<T, float>) {
            AscendC::DataCopy<float>(calc0, src0Local.template ReinterpretCast<float>(), ubLength);
            AscendC::DataCopy<float>(calc1, src1Local.template ReinterpretCast<float>(), ubLength);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            // int8 / uint8 -> half -> float（AICORE 无 int8<->float 直转指令，值域 <=255 用 half 精确表示）
            AscendC::LocalTensor<half> htmp = this->halfBuf.template Get<half>();
            AscendC::Cast<half, T>(htmp, src0Local, AscendC::RoundMode::CAST_NONE, ubLength);
            AscendC::Cast<float, half>(calc0, htmp, AscendC::RoundMode::CAST_NONE, ubLength);
            AscendC::Cast<half, T>(htmp, src1Local, AscendC::RoundMode::CAST_NONE, ubLength);
            AscendC::Cast<float, half>(calc1, htmp, AscendC::RoundMode::CAST_NONE, ubLength);
        } else {
            // float16 / bfloat16 / int32 -> float
            AscendC::Cast<float, T>(calc0, src0Local, AscendC::RoundMode::CAST_NONE, ubLength);
            AscendC::Cast<float, T>(calc1, src1Local, AscendC::RoundMode::CAST_NONE, ubLength);
        }
        this->inQue0.template FreeTensor<T>(src0Local);
        this->inQue1.template FreeTensor<T>(src1Local);
    }

    // calc0 = x1, calc1 = x2 (both float). Result written to calc0.
    //   y = x1 - trunc(x1/x2) * x2
    __aicore__ inline void Compute(const AscendC::LocalTensor<float>& calc0, const AscendC::LocalTensor<float>& calc1,
                                   const uint64_t length)
    {
        uint64_t clen = (length + VEC_REPEAT_F32 - 1u) & ~(VEC_REPEAT_F32 - 1u);
        AscendC::LocalTensor<float> tmp = this->tmpBuf.template Get<float>();
        AscendC::LocalTensor<float> tmp2 = this->tmp2Buf.template Get<float>();
        AscendC::Div<float>(tmp, calc0, calc1, clen); // t = x1 / x2
        // trunc(t) = ceil(min(t,0)) + floor(max(t,0))
        AscendC::Mins<float>(tmp2, tmp, 0.0f, clen);   // min(t, 0)
        AscendC::Ceil<float>(tmp2, tmp2, clen);        // ceil(min(t,0))
        AscendC::Maxs<float>(tmp, tmp, 0.0f, clen);    // max(t, 0)
        AscendC::Floor<float>(tmp, tmp, clen);         // floor(max(t,0))
        AscendC::Add<float>(tmp2, tmp2, tmp, clen);    // trunc(t)
        AscendC::Mul<float>(tmp2, tmp2, calc1, clen);  // trunc(t) * x2
        AscendC::Sub<float>(calc0, calc0, tmp2, clen); // x1 - trunc(t)*x2
    }

    __aicore__ inline void CopyOut(const AscendC::LocalTensor<float>& calc0, uint64_t offset, uint64_t copyLength,
                                   uint64_t ubLength)
    {
        AscendC::LocalTensor<T> dstLocal = this->outQue.template AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) {
            AscendC::DataCopy<float>(dstLocal.template ReinterpretCast<float>(), calc0, ubLength);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            AscendC::Cast<int32_t, float>(dstLocal, calc0, AscendC::RoundMode::CAST_ROUND, ubLength);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            // mod 结果为整数值：float -> half -> int8/uint8
            AscendC::LocalTensor<half> htmp = this->halfBuf.template Get<half>();
            AscendC::Cast<half, float>(htmp, calc0, AscendC::RoundMode::CAST_NONE, ubLength);
            AscendC::Cast<T, half>(dstLocal, htmp, AscendC::RoundMode::CAST_ROUND, ubLength);
        } else {
#if __CCE_AICORE__ == 200
            AscendC::Cast<T, float>(dstLocal, calc0, AscendC::RoundMode::CAST_NONE, ubLength);
#else
            AscendC::Cast<T, float>(dstLocal, calc0, AscendC::RoundMode::CAST_RINT, ubLength);
#endif
        }
        this->outQue.template EnQue<T>(dstLocal);
        dstLocal = this->outQue.template DeQue<T>();
        // UB -> GM 按实际长度写回，尾块非对齐时用 DataCopyPad 避免越界写。
        AscendC::DataCopyExtParams copyParams{1u, static_cast<uint32_t>(copyLength * sizeof(T)), 0u, 0u, 0u};
        AscendC::DataCopyPad(this->dstGlobal[offset], dstLocal, copyParams);
        this->outQue.template FreeTensor<T>(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQue0;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQue1;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf0;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp2Buf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> halfBuf;

    AscendC::GlobalTensor<T> src0Global;
    AscendC::GlobalTensor<T> src1Global;
    AscendC::GlobalTensor<T> dstGlobal;

    bool isLastCore = false;
    uint64_t tailElems = 0u;
    uint64_t bufferNum = 1u;
    uint64_t epochs = 0u;
    uint64_t globalOffset = 0u;
    uint64_t tileLength = 0u;
    uint64_t tailTileLength = 0u;
    constexpr static uint64_t ELEM_PER_BLOCK = 32u / sizeof(T);
};

template <typename T>
__aicore__ inline void Run(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace,
                           const TruncateModTilingData* tilingData)
{
    TruncateModKernel<T> op(tilingData);
    op.Init(y, x1, x2, workspace);
    op.Process();
}

} // namespace NsTruncateMod

#endif
