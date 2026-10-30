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
/*!
 * \file ndtri_kernel.h
 * \brief Ndtri Kernel 实现（arch35 / Ascend950）
 *
 * 公式：
 *   y = ndtri(p) = sqrt(2) * erf^{-1}(2p - 1)
 *   对齐 PyTorch torch.special.ndtri / SciPy scipy.special.ndtri（Cephes 算法）。
 *
 * 计算流（单 tile）：
 *   Step 0: 输入 Cast→fp32 (fp16/bf16: CAST_NONE, fp32: ReinterpretCast)
 *   Step 1: 构造 maskTail / maskNeg / maskSpecial
 *   Step 2: pSafe = clamp(p, FLT_MIN, 1 - FLT_MIN)
 *   Step 3: yTail = cal_tail(pSafe, maskNeg)
 *   Step 4: yCenter = cal_p0(pSafe)
 *   Step 5: y = select(maskTail, yTail, yCenter)
 *   Step 6: y = select(maskSpecial, ySpecial, y)
 *   Step 7: 输出 Cast→T (fp16/bf16: CAST_RINT, fp32: ReinterpretCast)
 *
 * 迭代二范围（本次整合）：
 *   - FP32 / FP16 / BF16 × 对齐/非对齐 共 6 个 TilingKey 真实实现
 *   - FP16 路径：Cast fp16→fp32 → 统一算法 → Cast fp32→fp16 (穿刺 P-2 已验证 bit-exact)
 *   - BF16 路径：Cast bf16→fp32 → 统一算法 → Cast fp32→bf16 (穿刺 P-3 已验证 bit-exact)
 *   - 非对齐路径：DataCopyPad 处理尾块（本来就用 DataCopyPad，天然兼容）
 *
 * TilingKey 矩阵：{fp32, fp16, bf16} × {对齐, 非对齐} = 6 个
 */

#ifndef NDTRI_KERNEL_H
#define NDTRI_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ndtri_tiling_data.h"
#include "ndtri_tiling_key.h"
#include "ndtri_coeffs.h"
#include "ndtri_compute.h"

namespace NsNdtri {

using namespace AscendC;

template <typename T, int K_ALIGN>
class Ndtri {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr bool IS_FP32 = AscendC::IsSameType<T, float>::value;

    // Compare/CompareScalar API 对 `count` 所占空间有 256B 对齐硬约束（见
    // ascendc-api-best-practices §2.1）。FP32 下等价 count 为 64 元素倍数。
    // 稳态 tile（ubFactor）已由 Host Tiling 的 FloorAlign(_, 256) 对齐；
    // 尾块 currentNum 可能任意非 64 倍数，需在 Kernel 层向上取 64 对齐并在
    // padding 区域填入中性值 0.5f（中心区值，既不触发 tail 分支也不触发
    // special 分支），由 DataCopyPad 的 blockLen=currentNum*sizeof(T) 保证
    // padding 位置最终不会被写回 GM。
    static constexpr int32_t CMP_ALIGN_ELEM = 64;
    __aicore__ inline static int32_t AlignCmpLen(int32_t len)
    {
        return (len + CMP_ALIGN_ELEM - 1) / CMP_ALIGN_ELEM * CMP_ALIGN_ELEM;
    }

public:
    __aicore__ inline Ndtri() = default;

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR out, const NdtriTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

    // 构造三个 mask：tail / neg / special
    __aicore__ inline void BuildMasks(
        const LocalTensor<float>& p, const LocalTensor<uint8_t>& maskTail, const LocalTensor<uint8_t>& maskNeg,
        const LocalTensor<uint8_t>& maskSpecial, const LocalTensor<float>& scratch, int32_t len);

    // 构造 y_special：p==0 -> -inf, p==1 -> +inf, otherwise -> NaN
    __aicore__ inline void BuildSpecialY(
        const LocalTensor<float>& ySpecial, const LocalTensor<float>& p, const LocalTensor<float>& scratch,
        int32_t len);

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueSelf;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueY;

    // fp32 域：p / y（所有 dtype 统一使用，FP32 路径通过 Cast(CAST_NONE) 等价搬运）
    TBuf<TPosition::VECCALC> pBuf;
    TBuf<TPosition::VECCALC> yBuf;

    // fp32 计算 Buffer
    TBuf<TPosition::VECCALC> tmpBuf0;
    TBuf<TPosition::VECCALC> tmpBuf1;
    TBuf<TPosition::VECCALC> tmpBuf2;
    TBuf<TPosition::VECCALC> tmpBuf3;
    TBuf<TPosition::VECCALC> tmpBuf4;
    TBuf<TPosition::VECCALC> tmpBuf5;
    TBuf<TPosition::VECCALC> tmpBuf6;
    TBuf<TPosition::VECCALC> tmpBuf7;
    TBuf<TPosition::VECCALC> tmpBuf8;
    TBuf<TPosition::VECCALC> tmpBuf9;  // CalTail tmp4（避免与 q 别名）
    TBuf<TPosition::VECCALC> tmpBuf10; // CalTail tmp5（避免与 x 别名）

    // mask buffer（uint8）
    TBuf<TPosition::VECCALC> maskBuf0;
    TBuf<TPosition::VECCALC> maskBuf1;
    TBuf<TPosition::VECCALC> maskBuf2;
    TBuf<TPosition::VECCALC> maskBuf3; // scratch mask for cal_p12 / BuildSpecialY

    GlobalTensor<T> selfGm;
    GlobalTensor<T> outGm;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// ---------------------------------------------------------------
// Init
// ---------------------------------------------------------------
template <typename T, int K_ALIGN>
__aicore__ inline void Ndtri<T, K_ALIGN>::Init(GM_ADDR self, GM_ADDR out, const NdtriTilingData* tilingData)
{
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * blockIdx;
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    int64_t offset = tilingData->blockFactor * blockIdx;
    selfGm.SetGlobalBuffer((__gm__ T*)self + offset, blockLength_);
    outGm.SetGlobalBuffer((__gm__ T*)out + offset, blockLength_);

    // InQue / OutQue（DB）
    pipe.InitBuffer(inQueSelf, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outQueY, BUFFER_NUM, ubLength_ * sizeof(T));

    // fp32 域 p / y buffer（独立分配，保证 fp16/bf16 Cast 链路有足够空间）
    pipe.InitBuffer(pBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(yBuf, ubLength_ * sizeof(float));

    // fp32 工作 Buffer
    pipe.InitBuffer(tmpBuf0, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf1, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf2, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf3, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf4, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf5, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf6, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf7, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf8, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf9, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf10, ubLength_ * sizeof(float));

    // uint8 mask：按 bit 存储，大小 = ceil(len/8)；保守分配 len/8 + 32 字节冗余
    int64_t maskBytes = (ubLength_ + 7) / 8 + 32;
    pipe.InitBuffer(maskBuf0, maskBytes);
    pipe.InitBuffer(maskBuf1, maskBytes);
    pipe.InitBuffer(maskBuf2, maskBytes);
    pipe.InitBuffer(maskBuf3, maskBytes);
}

// ---------------------------------------------------------------
// Process
// ---------------------------------------------------------------
template <typename T, int K_ALIGN>
__aicore__ inline void Ndtri<T, K_ALIGN>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

// ---------------------------------------------------------------
// CopyIn
//   DataCopyPad 天然支持对齐/非对齐两种路径，K_ALIGN 仅影响 Host 的 TilingKey 派发。
// ---------------------------------------------------------------
template <typename T, int K_ALIGN>
__aicore__ inline void Ndtri<T, K_ALIGN>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inLocal = inQueSelf.template AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    int64_t gmOffset = progress * ubLength_;
    DataCopyPad(inLocal, selfGm[gmOffset], copyParams, padParams);
    inQueSelf.EnQue(inLocal);
}

// ---------------------------------------------------------------
// BuildMasks
//   - maskSpecial = (p <= 0) | (p >= 1) | (p != p)   [NaN 也视为特殊值]
//   - maskTail    = |p - 0.5| >= (0.5 - VAL_SUB)
//                 <=>  p <= VAL_SUB or p >= RES_EXP
//   - maskNeg     = p >= 0.5
//
//   NaN 处理：CompareScalar(NaN, x, any) 返回 false；因此单纯 (p<=0)|(p>=1)
//   不会把 NaN 放入 maskSpecial。使用 CompareScalar(p, p, NE) 等价 isnan：
//   浮点 NaN 有 NaN != NaN，正常数有 x == x。将 isnan 与 (p<=0)|(p>=1) 合并。
//
//   ISSUE-001：调用者传入的 len 必须是 64 倍数（FP32 下 256B 对齐），
//   由 Compute 层的 lenAligned 保证。
// ---------------------------------------------------------------
template <typename T, int K_ALIGN>
__aicore__ inline void Ndtri<T, K_ALIGN>::BuildMasks(
    const LocalTensor<float>& p, const LocalTensor<uint8_t>& maskTail, const LocalTensor<uint8_t>& maskNeg,
    const LocalTensor<uint8_t>& maskSpecial, const LocalTensor<float>& scratch, int32_t len)
{
    // maskTail = (|p - 0.5| >= 0.5 - VAL_SUB)
    Adds(scratch, p, -0.5f, len);
    Abs(scratch, scratch, len);
    CompareScalar(maskTail, scratch, 0.5f - NDTRI_VAL_SUB, CMPMODE::GE, len);

    // maskSpecial：由 3 个条件组合：
    //   c1 = (p <= 0)
    //   c2 = (p >= 1)
    //   c3 = isnan(p)  → 通过 (p != p) 判断（等价：Compare(p, p, NE)）
    // 但 AscendC 的 CompareScalar 不支持 src1 == src2 同一张量的比较（要求不同 tensor）；
    // 等价：isnan(p) == (Max(Min(p, +INF), -INF) != p)，这在大多数 NaN bit pattern
    // 下等价（因 Max/Min 返回非 NaN 值，而 NaN != 任意值）。
    //   简化实现：NaN 经过 Adds/Muls/Max/Min 后仍为 NaN，但 Max(NaN, -INF) 在 arch35
    //   通常返回 -INF（取非 NaN 操作数）。因此 scratch = Mins(Maxs(p, -INF), +INF) 得到非 NaN
    //   正常值，Compare(p, scratch, NE) 即可捕获 NaN。
    //
    // 本实现策略：先在 maskSpecial 存 (p <= 0)，然后 OR 进 (p >= 1)（借 maskNeg 中转），
    // 最后 OR 进 isnan（借 maskNeg 中转），最后重建 maskNeg。

    constexpr uint32_t NEG_INF_BITS_U = 0xFF800000U;
    constexpr uint32_t POS_INF_BITS_U = 0x7F800000U;
    float negInf, posInf;
    {
        union {
            uint32_t u;
            float f;
        } cvt;
        cvt.u = NEG_INF_BITS_U;
        negInf = cvt.f;
        cvt.u = POS_INF_BITS_U;
        posInf = cvt.f;
    }

    // maskSpecial = (p <= 0)
    CompareScalar(maskSpecial, p, 0.0f, CMPMODE::LE, len);

    // maskNeg_tmp = (p >= 1)
    CompareScalar(maskNeg, p, 1.0f, CMPMODE::GE, len);
    Or(maskSpecial, maskSpecial, maskNeg, len);

    // isnan(p)：scratch = Mins(Maxs(p, -INF), +INF)  （NaN 经此回填为非 NaN）
    //   然后 maskNeg_tmp = Compare(p, scratch, NE) → 对 NaN 位置置 1
    Maxs(scratch, p, negInf, len);
    Mins(scratch, scratch, posInf, len);
    Compare(maskNeg, p, scratch, CMPMODE::NE, len);
    Or(maskSpecial, maskSpecial, maskNeg, len);

    // 重建 maskNeg = (p >= 0.5)
    CompareScalar(maskNeg, p, 0.5f, CMPMODE::GE, len);
}

// ---------------------------------------------------------------
// BuildSpecialY
//   - p == 0  →  -inf
//   - p == 1  →  +inf
//   - 其他（包括 NaN / p < 0 / p > 1）  →  NaN
// ---------------------------------------------------------------
template <typename T, int K_ALIGN>
__aicore__ inline void Ndtri<T, K_ALIGN>::BuildSpecialY(
    const LocalTensor<float>& ySpecial, const LocalTensor<float>& p, const LocalTensor<float>& scratch, int32_t len)
{
    constexpr uint32_t NAN_BITS = 0x7FC00000U;
    constexpr uint32_t POS_INF_BITS = 0x7F800000U;
    constexpr uint32_t NEG_INF_BITS = 0xFF800000U;

    float nanVal, posInf, negInf;
    {
        union {
            uint32_t u;
            float f;
        } cvt;
        cvt.u = NAN_BITS;
        nanVal = cvt.f;
        cvt.u = POS_INF_BITS;
        posInf = cvt.f;
        cvt.u = NEG_INF_BITS;
        negInf = cvt.f;
    }

    // ISSUE-001：调用者传入的 len 由 Compute 层已对齐到 256B（64 元素）。

    // 默认 NaN
    Duplicate(ySpecial, nanVal, len);

    // 临时复用 maskBuf3 scratch 存 mask_eq0 / mask_eq1
    LocalTensor<uint8_t> maskEq = maskBuf3.Get<uint8_t>();

    // p == 0 -> -inf
    CompareScalar(maskEq, p, 0.0f, CMPMODE::EQ, len);
    Duplicate(scratch, negInf, len);
    Select(ySpecial, maskEq, scratch, ySpecial, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);

    // p == 1 -> +inf
    CompareScalar(maskEq, p, 1.0f, CMPMODE::EQ, len);
    Duplicate(scratch, posInf, len);
    Select(ySpecial, maskEq, scratch, ySpecial, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);
}

// ---------------------------------------------------------------
// Compute
// ---------------------------------------------------------------
template <typename T, int K_ALIGN>
__aicore__ inline void Ndtri<T, K_ALIGN>::Compute(int64_t currentNum)
{
    LocalTensor<T> inLocal = inQueSelf.template DeQue<T>();
    LocalTensor<T> outLocal = outQueY.template AllocTensor<T>();
    int32_t len = static_cast<int32_t>(currentNum);
    // ISSUE-001：Compare/CompareScalar API 对 count 所占空间要求 256B 对齐
    // （FP32: 64 元素倍数）。尾块 currentNum 可能任意非 64 倍数，此处向上
    // 取 64 对齐；padding 区域 [len, lenAligned) 需要在 Step 0 填入中性值
    // 0.5f，以保证 BuildMasks / BuildSpecialY 在 padding 位置计算结果不会
    // 触发越界异常或产生 NaN/Inf 干扰流水线。padding 位置的输出不会被
    // CopyOut 写回 GM（DataCopyPad.blockLen = len * sizeof(T)）。
    int32_t lenAligned = AlignCmpLen(len);

    // fp32 域别名
    LocalTensor<float> p = pBuf.Get<float>();
    LocalTensor<float> y = yBuf.Get<float>();

    // ISSUE-001：先对 p 的整个对齐区间 [0, lenAligned) 填入中性值 0.5f，
    // 然后再用有效数据覆盖前 len 个位置。这样避免 LocalTensor 切片
    // `p[len]` 可能产生的非 32B 对齐起始地址问题（Duplicate 对起始地址
    // 有 32B 对齐要求）。中性值 0.5f 在中心区，既不触发 tail 分支也不
    // 触发 special 分支。
    if (lenAligned > len) {
        Duplicate(p, 0.5f, lenAligned);
    }

    // Step 0: 输入 Cast → fp32
    //   - FP32: ReinterpretCast 零拷贝（inLocal 本身就是 fp32 bit pattern）
    //     但为了保持与 fp16/bf16 路径的语义一致（p 是独立 fp32 buffer），
    //     此处统一使用 Cast(CAST_NONE)。对于 float→float，Cast(CAST_NONE) 等价于 Copy。
    //   - FP16: Cast(fp16→fp32, CAST_NONE) 无损
    //   - BF16: Cast(bf16→fp32, CAST_NONE) 无损
    if constexpr (IS_FP32) {
        // float → float：用 Muls(x, 1.0f) 拷贝到独立 buffer（等价 Copy）
        // AscendC 没有裸 Copy API，使用 DataCopy 或 Adds(x, 0.0f) 均可；
        // 此处 Adds(p, inLocal_fp32, 0.0f) 最直观。
        LocalTensor<float> inFp32 = inLocal.template ReinterpretCast<float>();
        Adds(p, inFp32, 0.0f, len);
    } else {
        // half / bf16 → fp32
        Cast(p, inLocal, RoundMode::CAST_NONE, len);
    }

    // Buffer 别名
    LocalTensor<float> tmpPm = tmpBuf0.Get<float>();
    LocalTensor<float> tmpZ = tmpBuf1.Get<float>();
    LocalTensor<float> tmpP = tmpBuf2.Get<float>();
    LocalTensor<float> tmpQ = tmpBuf3.Get<float>();
    LocalTensor<float> scratch = tmpBuf4.Get<float>();
    LocalTensor<float> pSafe = tmpBuf5.Get<float>();
    LocalTensor<float> yCenter = tmpBuf6.Get<float>();
    LocalTensor<float> yTail = tmpBuf7.Get<float>();
    LocalTensor<float> ySpecial = tmpBuf8.Get<float>();

    LocalTensor<uint8_t> maskTail = maskBuf0.Get<uint8_t>();
    LocalTensor<uint8_t> maskNeg = maskBuf1.Get<uint8_t>();
    LocalTensor<uint8_t> maskSpecial = maskBuf2.Get<uint8_t>();

    // ISSUE-001：后续所有 Vector 计算统一使用 lenAligned 长度运行。
    // 理由：
    //   1) Compare/CompareScalar 要求 count 所占空间 256B 对齐（64 元素）
    //   2) p 在 [len, lenAligned) 已填 0.5f 中性值，计算中不会产生 NaN/Inf
    //   3) 所有 UB buffer 均已分配 ubLength_ * sizeof(float)，且 ubLength_
    //      已由 Host Tiling FloorAlign(_, 256) 保证 ≥ lenAligned
    //   4) CopyOut 仍按 len 字节写回 GM，padding 位置的输出被自然丢弃

    // Step 1: 构造 mask_tail / mask_neg / mask_special
    BuildMasks(p, maskTail, maskNeg, maskSpecial, scratch, lenAligned);

    // Step 2: pSafe = clamp(p, FLT_MIN, 1 - FLT_MIN)
    Maxs(pSafe, p, NDTRI_SAFE_LO, lenAligned);
    Mins(pSafe, pSafe, 1.0f - NDTRI_SAFE_LO, lenAligned);

    // Step 3: 计算 yTail（cal_tail 内部调用 cal_sub + cal_p12）
    //   采用"先 yTail 再 yCenter"策略，以使 yCenter (tmpBuf6) 可作为 CalTail 的 tmp3 使用。
    //   buffer 映射（CalTail 参数名 → tmpBuf 来源）：
    //     tmpQ(q)  = tmpPm    (tmpBuf0)
    //     tmpX     = tmpZ     (tmpBuf1)
    //     tmpX0    = tmpP     (tmpBuf2)
    //     tmpCorr  = tmpQ     (tmpBuf3)
    //     tmp1     = scratch  (tmpBuf4)
    //     tmp2     = ySpecial (tmpBuf8)  [Step 6 前 ySpecial 可借用]
    //     tmp3     = yCenter  (tmpBuf6)  [随后会被 Step 4 覆盖]
    //     tmp4     = tmpBuf9
    //     tmp5     = tmpBuf10
    CalTail(
        yTail, pSafe, maskNeg,
        /*tmpQ   */ tmpPm,
        /*tmpX   */ tmpZ,
        /*tmpX0  */ tmpP,
        /*tmpCorr*/ tmpQ,
        /*tmp1   */ scratch,
        /*tmp2   */ ySpecial,
        /*tmp3   */ yCenter,
        /*tmp4   */ tmpBuf9.Get<float>(),
        /*tmp5   */ tmpBuf10.Get<float>(),
        /*maskX  */ maskBuf3.Get<uint8_t>(), lenAligned);

    // Step 4: 计算 yCenter（覆盖 yCenter 暂借值）
    CalP0(yCenter, pSafe, tmpPm, tmpZ, tmpP, tmpQ, scratch, lenAligned);

    // Step 5: y = select(maskTail, yTail, yCenter)
    Select(y, maskTail, yTail, yCenter, SELMODE::VSEL_TENSOR_TENSOR_MODE, lenAligned);

    // Step 6: y = select(maskSpecial, ySpecial, y)
    BuildSpecialY(ySpecial, p, scratch, lenAligned);
    Select(y, maskSpecial, ySpecial, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, lenAligned);

    // Step 7: 输出 Cast → T
    //   - FP32: Adds(outLocal_fp32, y, 0.0f) 等价 Copy（只处理前 len 位置，
    //     padding 位置不需要写回 GM）
    //   - FP16: Cast(fp32 → fp16, CAST_RINT)
    //   - BF16: Cast(fp32 → bf16, CAST_RINT)
    if constexpr (IS_FP32) {
        LocalTensor<float> outFp32 = outLocal.template ReinterpretCast<float>();
        Adds(outFp32, y, 0.0f, len);
    } else {
        Cast(outLocal, y, RoundMode::CAST_RINT, len);
    }

    outQueY.template EnQue<T>(outLocal);
    inQueSelf.FreeTensor(inLocal);
}

// ---------------------------------------------------------------
// CopyOut
//   DataCopyPad 天然支持对齐/非对齐两种路径。
// ---------------------------------------------------------------
template <typename T, int K_ALIGN>
__aicore__ inline void Ndtri<T, K_ALIGN>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outLocal = outQueY.template DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    int64_t gmOffset = progress * ubLength_;
    DataCopyPad(outGm[gmOffset], outLocal, copyParams);
    outQueY.FreeTensor(outLocal);
}

} // namespace NsNdtri

#endif // NDTRI_KERNEL_H
