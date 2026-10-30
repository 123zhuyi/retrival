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
 * \file asinh.h
 * \brief Asinh 算子 Kernel 类（arch35 / Ascend950）
 *
 * 与 DESIGN.md v1.1 §3.5 对齐：
 *   - 19 步数值稳定 asinh 公式（含 small-arg / large-arg 分段选择 + 符号恢复）
 *   - FP32 直通 / FP16/BF16 入口 CAST_NONE → 19 步 FP32 → 出口 CAST_RINT
 *   - Double Buffer (BUFFER_NUM=2)
 *   - Buffer 复用：absX (step 1 写) / b (data_b → u-1 → clipped s → 1/x²) /
 *                 r (data_b² → data_r → log(|x|)+ln(2)+1/x² → neg_output) /
 *                 s (u → log(u) → res → result_2 → output)
 *   - Log natural 三参数调用（不接收 sharedTmpBuffer），由框架自动从未 InitBuffer 的剩余 UB 申请
 *   - **v1.1 关键约束**：Compare/Select 调用使用 nAligned = CeilAlign(currentNum, 64)
 *     满足 Compare.md line 114/125 的 count 256B 对齐强约束
 *
 * 迭代一范围（FP32 主线骨架）：
 *   - FP32 路径完整实现（19 步原地分段算法）；FP16/BF16 路径已包含 Cast 骨架，
 *     完整端到端验证在迭代二落地
 */
#ifndef NSASINH_ASINH_H
#define NSASINH_ASINH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "asinh_tiling_data.h"
#include "asinh_tiling_key.h"

namespace NsAsinh {
using namespace AscendC;

// ============================================================
// 19 步流程的 FP32 常量（REQUIREMENTS §8.1 强约束 FP32 字面量精度）
// ============================================================
constexpr float CONST_ONE              =  1.0f;
constexpr float CONST_NEG_ONE          = -1.0f;
constexpr float CONST_ZERO             =  0.0f;
constexpr float CONST_S_MIN            =  1.0e-45f;        // clip 下界（REQUIREMENTS §8.1）
constexpr float CONST_S_MAX            =  3.4028235e34f;   // clip 上界（REQUIREMENTS §8.1）
// ln(2) FP32 近似（避免 acosh 笔误）。
// 字面量截断到 FP32 实际可表示精度（mantissa 24 bit，约 7 位有效数字）。
// 完整精度参考: ln(2) ≈ 0.69314718055994530941723212145818...，编译期 round 到最近 FP32。
constexpr float CONST_LN2              =  0.6931472f;
constexpr float CONST_BRANCH_THRESHOLD =  0.00024414063f;  // 2^-12，小参数分支阈值

// Double Buffer 固定为 2（19 步含 Log/Sqrt/Div/Compare 计算密集，双缓冲收益显著）
static constexpr int32_t BUFFER_NUM = 2;

// **v1.1 关键约束**：Compare/Select count 必须 256B 对齐 → FP32 视角 64 元素
static constexpr uint32_t COMPARE_ALIGN_ELEMS_KERNEL = 64;

template <typename T>
class Asinh {
public:
    __aicore__ inline Asinh() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR out, const AsinhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

    // 19 步 FP32 计算（保留 xOrigFp32 全程，最终结果写入 yFp32）
    __aicore__ inline void ComputeFp32Pipeline(LocalTensor<float>& xOrigFp32,
                                                LocalTensor<float>& yFp32,
                                                int64_t count);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inputQue;     // GM → UB 输入队列（dtype = T）
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue;    // UB → GM 输出队列（dtype = T）

    // FP32 工作 Buffer（所有 dtype 路径都使用）
    TBuf<QuePosition::VECCALC> xOrigBuf;    // 原始 x FP32 备份（保留符号位，step 19 符号恢复用）
    TBuf<QuePosition::VECCALC> absXBuf;     // |x| (data_a)；FP16/BF16 路径下也作 fp32 输出区
    TBuf<QuePosition::VECCALC> bBuf;        // 1/|x| (data_b)；后续 step 复用为 u-1 / clipped s / 1/x²
    TBuf<QuePosition::VECCALC> rBuf;        // 通用 R Buffer
    TBuf<QuePosition::VECCALC> sBuf;        // 通用 S Buffer（u / log(u) / res / result_2 / output）
    TBuf<QuePosition::VECCALC> selMaskBuf;  // Compare 输出 mask (uint8_t)
    // Log 隐式 tmpBuffer 由框架从未 InitBuffer 的剩余 UB 自动申请，
    // Host Tiling 已通过 GetLogMaxMinTmpSize 在 ubFactor 计算中扣除字节

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t blockLength_ = 0;   // 本核要处理的元素总数（可能 < blockFactor，例如尾核）
    int64_t ubFactor_    = 0;   // 单次 UB 循环处理元素数（已经 64 元素对齐）
};

// ============================================================
// Init: 设置 GM 偏移 + 申请 UB Buffer
//   - inputQue / outputQue：dtype = T，DoubleBuffer
//   - 5 个 FP32 工作 Buffer（xOrigBuf / absXBuf / bBuf / rBuf / sBuf）
//   - 1 个 uint8_t selMaskBuf
// ============================================================
template <typename T>
__aicore__ inline void Asinh<T>::Init(GM_ADDR input, GM_ADDR out, const AsinhTilingData* tilingData)
{
    int64_t blockIdx = AscendC::GetBlockIdx();
    // MED-2 防御性强转：tilingData->blockFactor 为 int64_t、blockIdx 为 int64_t，
    // 显式 static_cast<int64_t> 提示并保证乘积在 int64_t 域内运算（实际 blockFactor*blockIdx ≤ totalNum + blockFactor 不会溢出）。
    int64_t blockOffset = static_cast<int64_t>(tilingData->blockFactor) * blockIdx;
    int64_t remainder = tilingData->totalNum - blockOffset;
    blockLength_ = (remainder > tilingData->blockFactor) ? tilingData->blockFactor : remainder;
    ubFactor_    = tilingData->ubFactor;

    inputGM.SetGlobalBuffer((__gm__ T*)input + blockOffset, blockLength_);
    outputGM.SetGlobalBuffer((__gm__ T*)out   + blockOffset, blockLength_);

    pipe.InitBuffer(inputQue,   BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe.InitBuffer(outputQue,  BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe.InitBuffer(xOrigBuf,   ubFactor_ * sizeof(float));  // 原始 x（保留符号位）
    pipe.InitBuffer(absXBuf,    ubFactor_ * sizeof(float));  // |x|（FP16/BF16 路径也作 fp32Out）
    pipe.InitBuffer(bBuf,       ubFactor_ * sizeof(float));  // 通用 b Buffer
    pipe.InitBuffer(rBuf,       ubFactor_ * sizeof(float));  // 通用 R Buffer
    pipe.InitBuffer(sBuf,       ubFactor_ * sizeof(float));  // 通用 S Buffer
    // selMask：1 字节/元素，按 256B 对齐保守分配 (ubFactor 字节本身就 64 元素对齐 → 64 字节，不足 256B
    // 保险起见再 round up 到 256B)
    pipe.InitBuffer(selMaskBuf, ((ubFactor_ + 255) / 256 * 256) * sizeof(uint8_t));
}

// ============================================================
// Process: 多次循环，每次处理 ubFactor 个元素
// ============================================================
template <typename T>
__aicore__ inline void Asinh<T>::Process()
{
    if (blockLength_ <= 0) {
        return;   // 尾核可能没有数据
    }
    int64_t loopCount = (blockLength_ + ubFactor_ - 1) / ubFactor_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubFactor_ * i) : ubFactor_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

// ============================================================
// CopyIn: GM → UB（DataCopyPad 自动处理非对齐尾块）
// ============================================================
template <typename T>
__aicore__ inline void Asinh<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQue.template AllocTensor<T>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen   = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride  = 0;
    copyParams.dstStride  = 0;
    AscendC::DataCopyPad(xLocal, inputGM[progress * ubFactor_], copyParams, {false, 0, 0, 0});
    inputQue.EnQue(xLocal);
}

// ============================================================
// CopyOut: UB → GM
// CopyOut 仅按 currentNum * sizeof(T) blockLen 写回 GM；超出 currentNum 的对齐冗余
// 区间（[currentNum, nAligned)）即使是脏值，也被 DataCopyPad 自动截断不写出。
// ============================================================
template <typename T>
__aicore__ inline void Asinh<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQue.template DeQue<T>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen   = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride  = 0;
    copyParams.dstStride  = 0;
    AscendC::DataCopyPad(outputGM[progress * ubFactor_], yLocal, copyParams);
    outputQue.FreeTensor(yLocal);
}

// ============================================================
// Compute: 入口/出口 Cast + 19 步 FP32 流水
//   - FP32 路径：xLocal → DataCopy → xOrigBuf；ComputeFp32Pipeline 写入 yLocal（FP32）
//   - FP16/BF16 路径：Cast(xLocal → xOrigBuf, CAST_NONE)；
//     ComputeFp32Pipeline 写入 absXBuf 作为 fp32Out；Cast(absXBuf → yLocal, CAST_RINT)
// ============================================================
template <typename T>
__aicore__ inline void Asinh<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQue.template DeQue<T>();
    LocalTensor<T> yLocal = outputQue.template AllocTensor<T>();
    LocalTensor<float> xOrig = xOrigBuf.Get<float>();
    uint32_t n = static_cast<uint32_t>(currentNum);

    // **精度修复（FP32-tail-precision）**：DataCopy / Cast 的 count 参数必须 32B 对齐
    // （FP32 视角 8 元素 / FP16 视角 16 元素 / BF16 视角 16 元素），否则非对齐尾部元素
    // 不会被正确搬运 → xOrigFp32 末尾位置遗留旧值，导致 Step 19 sign-select 选择错误。
    // 用 nAligned (= CeilAlign(n, 64)，256B 对齐，已是 32B 倍数) 作为 count，
    // 安全覆盖 [0..n) 有效区间；超出区间 [n..nAligned) 的脏值不影响最终 CopyOut
    // （DataCopyPad 按 currentNum * sizeof(T) 截断写回，详见 ComputeFp32Pipeline 注释）。
    uint32_t nAligned = (n + COMPARE_ALIGN_ELEMS_KERNEL - 1) & ~(COMPARE_ALIGN_ELEMS_KERNEL - 1);

    if constexpr (std::is_same_v<T, float>) {
        // ----------------- FP32 主线（迭代一目标） -----------------
        // DataCopy(xLocal → xOrig)：保留原始 x 到 xOrig（符号位保留），用于 step 19 符号恢复
        // 必须使用 nAligned 满足 DataCopy 32B 对齐约束；xLocal 由 DataCopyPad 搬入，
        // [n..nAligned) 区间可能为 0 填充或脏值，xOrig 同区间也被覆盖为相同值，不影响 [0..n) 正确性
        AscendC::DataCopy(xOrig, xLocal, nAligned);
        // LOW-1 修复：模板特化 T=float 时 LocalTensor<T> 与 LocalTensor<float> 是相同类型，
        // 直接将 yLocal 作为 LocalTensor<float>& 传入 ComputeFp32Pipeline 即可，
        // 无需 reinterpret_cast 提升可读性（cpp-secure 9.1）。
        ComputeFp32Pipeline(xOrig, yLocal, currentNum);
    } else {
        // ----------------- FP16/BF16 路径（迭代二完整实现） -----------------
        // Cast(xLocal → xOrig, CAST_NONE)：Ascend950PR/DT half/bfloat16_t → float 仅 CAST_NONE
        // 同样使用 nAligned 作为 count，避免尾部元素未 Cast；FP16/BF16 32B = 16 元素，nAligned (64 倍数) 必然满足
        AscendC::Cast(xOrig, xLocal, AscendC::RoundMode::CAST_NONE, nAligned);

        // 借 absXBuf 作为 fp32 工作输出区（19 步内部 absXBuf 自 step 1 起用作 |x|，
        // 在 step 19 完成后写入最终输出；FP16/BF16 路径下 absXBuf 同时承担「|x| 工作区」与
        // 「fp32 输出区」双重角色，时序无冲突，详见 DESIGN §3.5.3 Buffer 生命周期表）
        LocalTensor<float> fp32Out = absXBuf.Get<float>();
        ComputeFp32Pipeline(xOrig, fp32Out, currentNum);

        // Cast(fp32Out → yLocal, CAST_RINT)：Ascend950PR/DT float → half/bfloat16_t 不支持 CAST_NONE
        // 使用 nAligned 满足 Cast 32B 对齐；yLocal[n..nAligned) 为脏值但 CopyOut 按 currentNum 截断
        AscendC::Cast(yLocal, fp32Out, AscendC::RoundMode::CAST_RINT, nAligned);
    }

    outputQue.template EnQue<T>(yLocal);
    inputQue.FreeTensor(xLocal);
}

// ============================================================
// ComputeFp32Pipeline: 严格按 DESIGN v1.1 §3.5.3 19 步流程实现
//
// 19 步映射（详见 DESIGN §3.5.1 + §3.5.4）：
//   step 1:    absX = |xOrigFp32|
//   step 3:    b = 1/|x|
//   step 4-9:  r = |x| + |x| / (sqrt(1/x²+1) + 1/x)
//   step 10:   r = r (data_r)
//   step 11-13: s = clip(u-1, 1e-45, 3.4e34)（借 bBuf 存）
//   step 14:   s = log(u)
//   step 15:   s = log(u)*r / clipped_s → 主路径 res
//   step 16:   r = log(|x|)+ln(2)+1/|x|²； s = min(res, r) → result_2
//   step 17:   output = select(|x| < 2^-12, |x|, result_2)
//   step 18:   r = -output
//   step 19:   yFp32 = select(x ≥ 0, output, -output)
//
// v1.1 关键约束：step 17 / step 19 的 Compare/Select 调用必须使用 nAligned (= CeilAlign(n, 64))
// 作为 count，满足 Compare.md line 114/125 的 256B 对齐强约束。
// ============================================================
template <typename T>
__aicore__ inline void Asinh<T>::ComputeFp32Pipeline(
    LocalTensor<float>& xOrigFp32,
    LocalTensor<float>& yFp32,
    int64_t count)
{
    uint32_t n = static_cast<uint32_t>(count);
    // **v1.1 关键**：Compare/Select 必须 256B 对齐 → FP32 视角 64 元素
    //   非尾 tile：ubFactor 本身 64 对齐 → n == ubFactor 即 64 倍数，nAligned == n
    //   尾 tile：n 可能不对齐（如 80、56），向上对齐到 64 倍数；
    //   超出 currentNum 的对齐区间内 yFp32 为脏值，但 CopyOut DataCopyPad 按字节 blockLen 截断写回
    uint32_t nAligned = (n + COMPARE_ALIGN_ELEMS_KERNEL - 1) & ~(COMPARE_ALIGN_ELEMS_KERNEL - 1);

    LocalTensor<float>   absX    = absXBuf.Get<float>();
    LocalTensor<float>   b       = bBuf.Get<float>();
    LocalTensor<float>   r       = rBuf.Get<float>();
    LocalTensor<float>   s       = sBuf.Get<float>();
    LocalTensor<uint8_t> selMask = selMaskBuf.Get<uint8_t>();

    // ===== Step 1: absX = |xOrigFp32| =====
    AscendC::Abs(absX, xOrigFp32, n);

    // Step 2「const_one_tensor = broadcast(1.0)」不需要单独 Buffer，
    // 后续 (1/|x|² + 1) 通过 Adds(scalar=1.0) 直接处理

    // ===== Step 3: b = 1 / |x| =====
    AscendC::Duplicate(b, CONST_ONE, n);
    AscendC::Div(b, b, absX, n);

    // ===== Step 4: r = b² = 1/|x|² =====
    AscendC::Mul(r, b, b, n);

    // Step 5「result_1 = data_a」隐含 — step 17 使用 absX 替代 result_1，不需单独 Buffer

    // ===== Step 6: r = 1/|x|² + 1 =====
    AscendC::Adds(r, r, CONST_ONE, n);

    // ===== Step 7: r = sqrt(1/|x|² + 1) =====
    AscendC::Sqrt(r, r, n);

    // ===== Step 8: r = sqrt(1/|x|² + 1) + 1/|x| =====
    AscendC::Add(r, r, b, n);   // bBuf 之后释放可复用

    // ===== Step 9: r = |x| / (sqrt + 1/|x|) =====
    AscendC::Div(r, absX, r, n);

    // ===== Step 10: r = |x| + |x|/(sqrt + 1/|x|) → data_r =====
    AscendC::Add(r, absX, r, n);

    // ===== Step 11: s = r + 1 → data_u =====
    AscendC::Adds(s, r, CONST_ONE, n);

    // ===== Step 12: b = s - 1 → data_u_sub_1（借用 bBuf 暂存 u-1，bBuf 自 step 8 已释放） =====
    AscendC::Adds(b, s, CONST_NEG_ONE, n);

    // ===== Step 13: b = clip(u-1, 1e-45, 3.4e34) → clipped_s =====
    AscendC::Maxs(b, b, CONST_S_MIN, n);
    AscendC::Mins(b, b, CONST_S_MAX, n);

    // ===== Step 14: s = log(u)（原地覆盖 u）=====
    // Log natural 三参数版本，框架从未 InitBuffer 的剩余 UB 自动申请 tmpBuffer
    AscendC::Log(s, s, n);

    // ===== Step 15: s = log(u) * r / clipped_s → 主路径 res =====
    AscendC::Mul(s, s, r, n);   // s = log(u) * r
    AscendC::Div(s, s, b, n);   // s = log(u) * r / clipped_s
    // bBuf 此后再次释放

    // ===== Step 16: 大参数修正 result_2 = min(res, log(|x|) + ln(2) + 1/|x|²) =====
    // 16a: r = log(|x|)
    AscendC::Log(r, absX, n);
    // 16b: r += ln(2)
    AscendC::Adds(r, r, CONST_LN2, n);
    // 16c: 重新计算 1/|x|² 存入 b（bBuf 自 step 15 后已释放）
    // LOW-4 说明：step 4 曾计算过 r = 1/|x|²，但 step 6 立即覆盖 r 为 (1/|x|²+1)；
    // step 3 的 b = 1/|x| 也在 step 12-15 中被覆盖。按 DESIGN v1.1 §3.5.3 的 6 个 Buffer
    // 复用顺序约束（absX/b/r/s/selMask 严格 in-order），此处必须重算 1/|x|²。
    // 若要消除重算需新增独立 Buffer 保留 1/|x|²，将打破 5×FP32 + 1 selMask = 37 字节/元素
    // 的预算（参考 DESIGN §3.8.1），不在本次防御性修复范围。
    AscendC::Duplicate(b, CONST_ONE, n);
    AscendC::Div(b, b, absX, n);
    AscendC::Mul(b, b, b, n);                 // b = 1/|x|²
    AscendC::Add(r, r, b, n);                 // r = log(|x|) + ln(2) + 1/|x|²
    // 16d: s = min(s, r) → result_2
    AscendC::Min(s, s, r, n);

    // ===== Step 17: output = select(|x| < 2^-12, |x|, result_2) =====
    // 17a: selMask = Compare(absX, 2^-12, LT)
    //   AscendC::Compare 函数原型为 LocalTensor src0/src1，标量比较需用 Duplicate 构造
    //   **v1.1 关键**：count 使用 nAligned（FP32 64 元素 = 256B 对齐）
    //   对齐区间内 r 被填入 2^-12，与 absX 比较对应位为脏值（absX 末尾元素未在 step 1 写入），
    //   但 selMask 末几比特随机化的语义后果由 step 17b 接管，且 CopyOut 截断不写出对齐区间
    AscendC::Duplicate(r, CONST_BRANCH_THRESHOLD, nAligned);
    AscendC::Compare(selMask, absX, r, AscendC::CMPMODE::LT, nAligned);
    // 17b: output = select(selMask, |x|, result_2) → s（覆盖 result_2）
    //   模式 2 VSEL_TENSOR_TENSOR_MODE：src0=|x|（小参数路径），src1=result_2（大参数路径）
    AscendC::Select(s, selMask, absX, s, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, nAligned);
    // 此时 s = output

    // ===== Step 18: r = -output =====
    // Muls 无 256B 对齐约束，使用 n 即可（nAligned 也安全）
    AscendC::Muls(r, s, CONST_NEG_ONE, n);

    // ===== Step 19: yFp32 = select(x ≥ 0, output, -output) =====
    // 19a: selMask = Compare(xOrigFp32, 0, GE)
    //   借 absXBuf 临时存 0 tensor（absX 之前的角色 step 1~17 都已完成读取，可重新写入）
    //   **v1.1 关键**：count 使用 nAligned 满足 256B 对齐
    AscendC::Duplicate(absX, CONST_ZERO, nAligned);
    AscendC::Compare(selMask, xOrigFp32, absX, AscendC::CMPMODE::GE, nAligned);
    // 19b: yFp32 = select(selMask, output(=s), neg_output(=r))
    //   yFp32[0..currentNum) = 最终 asinh(x) 有效结果
    //   yFp32[currentNum..nAligned) = 脏值（CopyOut DataCopyPad 按 currentNum * sizeof(T) 截断不写出）
    AscendC::Select(yFp32, selMask, s, r, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, nAligned);
}

} // namespace NsAsinh
#endif // NSASINH_ASINH_H
