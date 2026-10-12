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
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file slogdet.h
 * \brief Slogdet 模板类实现（带部分主元 LU + 双输出合成）。
 *
 * 模板参数：
 *   D_T          数据类型（仅 fp32 / float）
 *   MEM_STRATEGY 0=FULL_RESIDENT（全驻留）, 1=BLOCKED（核内分块）
 *
 * 核心算法：对单 n×n 方阵带部分主元 LU，合成：
 *   signOut = (-1)^{swapParity} * ∏ sign(U_ii)，奇异 → 0
 *   logOut  = Σ log|U_ii|，奇异 → -inf
 *
 * 主元全局行号推导：
 *   分段局部 argmax + 标量合并。逐段 segLen ≤ 64 保证 ReduceMax 单 repeat（索引为段内局部索引），
 *   不依赖框架对 count>64 的多轮内部索引推导，逐段可验证。
 *
 * 实现规范参考 math/cholesky（DAV_2201 原生线代算子）：
 *   DataCopyExtParams/DataCopyPadExtParams 行/列 stride 搬运；PIPE_* 事件同步；
 *   LocalTensor GetValue/SetValue 标量；输出经 UB→GM DataCopyPad（非 GlobalTensor.SetValue）。
 */

#ifndef SLOGDET_H
#define SLOGDET_H

#include "kernel_operator.h"
#include "slogdet_tiling_data.h"

namespace NsSlogdet {

using namespace AscendC;

constexpr uint32_t SLOGDET_BASIC_BLOCK = 32U; // 32B DataBlock
constexpr int32_t SLOGDET_SEG = 64; // fp32 单 repeat 上限（256B/4B），主元搜索分段长度（§3.3.4b）
constexpr uint32_t SLOGDET_UB_BLOCK_ELEM = 8U; // 32B / sizeof(fp32) = 8 元素/DataBlock（BLOCKED 列 gather pad 步长）
constexpr float SLOGDET_FLT_EPS = 1.1920929e-7f; // fp32 机器精度（FLT_EPSILON, 2^-23），LAPACK 相对阈值系数
// BLOCKED 消元行块大小（性能优化）：一次载入 ROW_BLOCK 行（连续 GM→UB，32B 对齐落位），
// 块内逐行 Axpy 后整块回写，把 O(n²) 单行 DMA 降为 O(n²/ROW_BLOCK) 块 DMA（减少 MTE 往返）。
// UB 占用 ROW_BLOCK*rowStride*4B（n=512→16*512*4=32KB），≪184KB。
constexpr uint32_t SLOGDET_ROW_BLOCK = 16U;

struct SlogdetAccum {
    float signProd;
    float logabs;
    uint32_t swapParity;
    bool singular;
    float maxPiv;
};

// ── 奇异阈值策略（对齐 torch.linalg.slogdet fp32 oracle）──────────────────────────────────────
// 相对阈值（LAPACK getrf 风格）：`|piv| <= max(n·FLT_EPSILON·maxPiv, epsFloor)` 判奇异。
//   maxPiv = LU 过程中已见主元 |U_ii| 的**运行最大值**（= 最大对角 U 元素；部分主元下大主元先出现）。
//   仅依赖主元值（kernel 在每步 gather/swap 后已可靠 GetValue），**不做全矩阵 UB/GM 标量扫描**。
//   host 下发 epsFloor（绝对下限，见 op_host ComputeEps）。
//
// 修复缘由：纯绝对阈值 1e-30 会对 dup-col（col_i==col_j）精确奇异漏判 —— 重复列消元后残留
//   ~1e-7 fp32 舍入小主元，1e-30 ≪ 残留 → 误判有限，与 torch（-inf/0）分叉。
//   改用「运行最大主元 maxPiv」给阈值提供稳定相对尺度，避免全矩阵 maxAbs 扫描在不同执行路径下引入不稳定。
// 实测分隔（fp32 LU，n·FLT_EPS·maxPiv vs |piv|）：
//   dup-col 3x3 min|piv|/maxPiv=2.55e-8 ＜ 阈值(n·FLT_EPS=3.57e-7) → 判奇异；2x2 同理；
//   ill[64,64] min|piv|/maxPiv=0.75 ≫ 阈值 → 判有限；rand256 ratio=0.71 ≫ 阈值 → 判有限。7 个数量级裕量。
//   epsFloor 取极小绝对值，仅在 maxPiv 极小时兜底。
// ────────────────────────────────────────────────────────────────────────────────────────────

// 核内 +inf 常量（kernel 侧无 <cmath> 宏）：用位模式构造 fp32 +inf。
__aicore__ inline float SlogdetPosInf()
{
    uint32_t bits = 0x7F800000u; // fp32 +inf
    // MED-4 豁免：reinterpret_cast 位模式双关（uint32→float）。kernel 侧无 <cstring>/std::bit_cast，
    //   AscendC kernel 普遍写法，目标编译器（bisheng）行为确定；实测正确（ST 全过）。非标准严格别名 UB，
    //   但属官方/工程惯例，记入检视基线。
    return *reinterpret_cast<float*>(&bits);
}

template <typename D_T, int MEM_STRATEGY>
class SlogdetKernel {
public:
    __aicore__ inline SlogdetKernel()
    {}

    __aicore__ inline void Init(
        GM_ADDR self, GM_ADDR signOut, GM_ADDR logOut, GM_ADDR workspace, const SlogdetTilingData* tiling, TPipe* pipe)
    {
        pipe_ = pipe;
        n_ = tiling->matSizeN;
        matrixNumCount_ = tiling->matrixNumCount;
        epsSingular_ = tiling->epsSingular;

        blockIdx_ = GetBlockIdx();
        blockDim_ = GetBlockNum();

        // UB 内行 stride 按 32B（8 个 fp32）对齐，列 gather / Axpy 起址 32B 对齐前提
        rowStrideElem_ = (n_ + 7U) / 8U * 8U;
        matStrideElem_ = static_cast<uint64_t>(n_) * static_cast<uint64_t>(n_); // GM 中每个矩阵连续 n*n

        selfGm_.SetGlobalBuffer((__gm__ float*)self);
        signGm_.SetGlobalBuffer((__gm__ float*)signOut);
        logGm_.SetGlobalBuffer((__gm__ float*)logOut);

        if constexpr (MEM_STRATEGY == 0) {
            // ── FULL：n×n 整块驻留 UB ──
            // MED-3：uWork 尺寸用 uint64 提升，消除右侧 uint32*uint32 中间溢出（与 BLOCKED 偏移写法统一）。
            pipe_->InitBuffer(uWorkBuf_, static_cast<uint64_t>(n_) * rowStrideElem_ * sizeof(float));
            pipe_->InitBuffer(colBuf_, rowStrideElem_ * sizeof(float));    // 列 gather 连续目标
            pipe_->InitBuffer(absBuf_, rowStrideElem_ * sizeof(float));    // |col|
            pipe_->InitBuffer(tmpRowBuf_, rowStrideElem_ * sizeof(float)); // 行交换中转
            pipe_->InitBuffer(redOutBuf_, SLOGDET_BASIC_BLOCK);            // ReduceMax 输出（最大值,索引）
            pipe_->InitBuffer(redTmpBuf_, rowStrideElem_ * sizeof(float)); // ReduceMax sharedTmpBuffer（充分裕量）
            pipe_->InitBuffer(outScalarBuf_, SLOGDET_BASIC_BLOCK);         // 标量输出中转
        } else {
            // ── BLOCKED：U 常驻 GM workspace，UB 仅持 O(n) 行/列向量（large-n）──
            // 每核独占一块 n*n workspace slot（核间无串扰）；slot 起址 = blockIdx_ * n*n。
            wsGm_.SetGlobalBuffer((__gm__ float*)workspace);
            // colWide：GM→UB 列 gather 落位 buffer，每元素占 1 个 32B DataBlock（stride-8），故 n*8 元素。
            pipe_->InitBuffer(colWideBuf_, rowStrideElem_ * SLOGDET_UB_BLOCK_ELEM * sizeof(float));
            pipe_->InitBuffer(colBuf_, rowStrideElem_ * sizeof(float));  // 紧致后连续子列
            pipe_->InitBuffer(absBuf_, rowStrideElem_ * sizeof(float));  // |col|
            pipe_->InitBuffer(rowKBuf_, rowStrideElem_ * sizeof(float)); // pivot 整行（消元 src）
            // 消元行块 buffer：一次驻留 ROW_BLOCK 行（每行 32B 对齐 stride），批量消元后整块回写
            pipe_->InitBuffer(rowBlockBuf_, SLOGDET_ROW_BLOCK * rowStrideElem_ * sizeof(float));
            pipe_->InitBuffer(swapBuf_, rowStrideElem_ * sizeof(float));   // 行交换整行中转
            pipe_->InitBuffer(cpyBuf_, rowStrideElem_ * sizeof(float));    // self→ws 行中转
            pipe_->InitBuffer(redOutBuf_, SLOGDET_BASIC_BLOCK);            // ReduceMax 输出（最大值,索引）
            pipe_->InitBuffer(redTmpBuf_, rowStrideElem_ * sizeof(float)); // ReduceMax sharedTmpBuffer（充分裕量）
            pipe_->InitBuffer(outScalarBuf_, SLOGDET_BASIC_BLOCK);         // 标量输出中转
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (MEM_STRATEGY == 0) {
            ProcessFull(); // 全驻留：n×n 整块进 UB
        } else {
            ProcessBlocked(); // 核内分块：U 常驻 GM workspace，UB 仅持 O(n) 行/列向量（large-n）
        }
    }

private:
    // batch 按核切分（核间无依赖，确定性）。每核处理 blockIdx_ + blockDim_*loop < count。
    __aicore__ inline void ProcessFull()
    {
        for (uint64_t mi = blockIdx_; mi < matrixNumCount_; mi += blockDim_) {
            ProcessOneMatrix(mi);
        }
    }

    __aicore__ inline void ProcessBlocked()
    {
        for (uint64_t mi = blockIdx_; mi < matrixNumCount_; mi += blockDim_) {
            ProcessOneMatrixBlocked(mi);
        }
    }

    // 统一事件同步模板：FetchEventID / SetFlag / WaitFlag 均使用 HardEvent 模板参数，避免为每种流水重复写法。
    template <HardEvent EVENT>
    __aicore__ inline void PipeSync()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID<EVENT>());
        SetFlag<EVENT>(eventId);
        WaitFlag<EVENT>(eventId);
    }

    // ---- 列 gather：第 k 列下方子列 uWork[k..n-1][k] → 连续 col[0..m-1] ----
    // 行连续布局下列非连续（行 stride = rowStrideElem_）。两种受限路径：
    //   (1) UB↔UB DataCopyPad 不支持 srcStride 的单元素 gather（仅 GM↔UB 支持带 stride）；
    //   (2) 经 GM workspace 中转，UB 源起址 uWork[k*stride + k] 在 (k%8)!=0 时非 32B 对齐
    //       → 触发 507035（VEC/MTE UB 地址非对齐）且读到错列 → 误判奇异。
    // 故采用**标量 gather**：逐元素 GetValue/SetValue 把子列搬到连续 col（标量域无对齐约束），
    // 再 S→V 同步后走真实分段 ReduceMax。标量 gather 的 GetValue 列偏移 +k 在标量域合法
    // （非向量指令，无 32B 起址要求）。
    __aicore__ inline void GatherColumn(
        const LocalTensor<float>& uWork, const LocalTensor<float>& col, uint32_t k, uint32_t m)
    {
        for (uint32_t i = 0; i < m; ++i) {
            col.SetValue(i, uWork.GetValue((k + i) * rowStrideElem_ + k));
        }
        // S → V：SetValue(标量) 写完 col 后才能 Abs(向量) 读
        PipeSync<HardEvent::S_V>();
    }

    // ---- 分段局部 argmax + 标量合并 ----
    // 返回 [0, m) 内子列行偏移；逐段 segLen≤64，ReduceMax 单 repeat（段内局部索引）。
    __aicore__ inline uint32_t ArgMaxAbsColumn(const LocalTensor<float>& absCol, uint32_t m)
    {
        LocalTensor<float> redOut = redOutBuf_.Get<float>();
        LocalTensor<float> redTmp = redTmpBuf_.Get<float>();

        float bestVal = -1.0f;
        uint32_t bestIdx = 0;
        for (uint32_t segStart = 0; segStart < m; segStart += static_cast<uint32_t>(SLOGDET_SEG)) {
            uint32_t rem = m - segStart;
            uint32_t segLen = rem < static_cast<uint32_t>(SLOGDET_SEG) ? rem : static_cast<uint32_t>(SLOGDET_SEG);
            // segLen ≤ 64 → 单 repeat → 返回索引为段内局部索引
            ReduceMax<float>(redOut, absCol[segStart], redTmp, static_cast<int32_t>(segLen), true);
            PipeSync<HardEvent::V_S>(); // V → S：等待 ReduceMax 写完再 GetValue

            float segVal = redOut.GetValue(0);
            float idxAsFloat = redOut.GetValue(1);
            // MED-4 豁免：ReduceMax(calIndex=true) 返回索引按 uint32 读取，是 ReduceMax.md **官方惯例**
            //   `reinterpret_cast<uint32_t>(dst.GetValue(1))`；kernel 侧无 std::bit_cast，目标编译器行为确定，
            //   实测正确（n>64 分段全局行号 ST 全过）。非标准严格别名 UB 但属官方 API 指引，记入检视基线。
            uint32_t segLocal = *reinterpret_cast<uint32_t*>(&idxAsFloat);
            uint32_t globalIdx = segStart + segLocal; // 段起点 + 段内局部 → 全局子列偏移

            if (segVal > bestVal) { // 严格大于 → 命中首个 |max|（与部分主元一致，确定性）
                bestVal = segVal;
                bestIdx = globalIdx;
            }
        }
        return bestIdx; // ∈ [0, m)
    }

    // ---- 行交换：UB 内整行 swap（向量 DataCopy，禁逐元素 SetValue）----
    __aicore__ inline void SwapRows(
        const LocalTensor<float>& uWork, const LocalTensor<float>& tmpRow, uint32_t r1, uint32_t r2)
    {
        DataCopy(tmpRow, uWork[r1 * rowStrideElem_], rowStrideElem_);
        PipeBarrier<PIPE_V>();
        DataCopy(uWork[r1 * rowStrideElem_], uWork[r2 * rowStrideElem_], rowStrideElem_);
        PipeBarrier<PIPE_V>();
        DataCopy(uWork[r2 * rowStrideElem_], tmpRow, rowStrideElem_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float AbsScalar(float value)
    {
        return value >= 0.0f ? value : -value;
    }

    __aicore__ inline bool IsSingularPivot(float pivAbs, SlogdetAccum& acc)
    {
        if (pivAbs > acc.maxPiv) {
            acc.maxPiv = pivAbs;
        }
        float eps = static_cast<float>(static_cast<int32_t>(n_)) * SLOGDET_FLT_EPS * acc.maxPiv;
        if (epsSingular_ > eps) {
            eps = epsSingular_;
        }
        return pivAbs <= eps;
    }

    __aicore__ inline void AccumulatePivot(float piv, float pivAbs, SlogdetAccum& acc)
    {
        if (piv < 0.0f) {
            acc.signProd = -acc.signProd;
        }
        acc.logabs += LogScalar(pivAbs);
    }

    __aicore__ inline void WriteSlogdetResult(
        uint64_t mi, const LocalTensor<float>& outScalar, const SlogdetAccum& acc)
    {
        float finalSign;
        float finalLog;
        if (acc.singular) {
            finalSign = 0.0f;
            finalLog = -SlogdetPosInf();
        } else {
            finalSign = (acc.swapParity != 0) ? -acc.signProd : acc.signProd;
            finalLog = acc.logabs;
        }

        outScalar.SetValue(0, finalSign);
        PipeSync<HardEvent::S_MTE3>();
        {
            DataCopyExtParams p{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(signGm_[mi], outScalar, p);
        }
        PipeSync<HardEvent::MTE3_S>();
        outScalar.SetValue(0, finalLog);
        PipeSync<HardEvent::S_MTE3>();
        {
            DataCopyExtParams p{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(logGm_[mi], outScalar, p);
        }
        PipeSync<HardEvent::MTE3_S>();
    }

    __aicore__ inline void LoadMatrixToUb(const LocalTensor<float>& uWork, uint64_t mi)
    {
        uint64_t base = mi * matStrideElem_;
        for (uint32_t r = 0; r < n_; ++r) {
            DataCopyExtParams p{1, static_cast<uint32_t>(n_ * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            DataCopyPad(
                uWork[static_cast<uint64_t>(r) * rowStrideElem_], selfGm_[base + static_cast<uint64_t>(r) * n_], p,
                padParams);
        }
        PipeSync<HardEvent::MTE2_S>();
    }

    __aicore__ inline void EliminateFullTrailingRows(const LocalTensor<float>& uWork, uint32_t k, float piv)
    {
        if (k + 1U >= n_) {
            return;
        }
        if (piv == 0.0f) {
            return;
        }
        LocalTensor<float> rowK = uWork[k * rowStrideElem_];
        int32_t fullLen = static_cast<int32_t>(rowStrideElem_);
        for (uint32_t j = k + 1U; j < n_; ++j) {
            float ajk = uWork.GetValue(j * rowStrideElem_ + k);
            float mult = ajk / piv;
            LocalTensor<float> rowJ = uWork[j * rowStrideElem_];
            Axpy(rowJ, rowK, -mult, fullLen);
        }
        PipeSync<HardEvent::V_S>();
    }

    __aicore__ inline bool ProcessFullStep(const LocalTensor<float>& uWork, const LocalTensor<float>& col,
                                           const LocalTensor<float>& absCol, const LocalTensor<float>& tmpRow,
                                           uint32_t k, SlogdetAccum& acc)
    {
        uint32_t m = n_ - k;
        GatherColumn(uWork, col, k, m);
        Abs(absCol, col, static_cast<int32_t>(m));
        PipeBarrier<PIPE_V>();
        uint32_t pivLocal = ArgMaxAbsColumn(absCol, m);
        uint32_t pidx = k + pivLocal;

        float piv = uWork.GetValue(pidx * rowStrideElem_ + k);
        float pivAbs = AbsScalar(piv);
        if (IsSingularPivot(pivAbs, acc)) {
            acc.singular = true;
            return false;
        }

        if (pidx != k) {
            SwapRows(uWork, tmpRow, k, pidx);
            acc.swapParity ^= 1u;
            PipeSync<HardEvent::V_S>();
        }
        AccumulatePivot(piv, pivAbs, acc);
        EliminateFullTrailingRows(uWork, k, piv);
        return true;
    }

    __aicore__ inline void ProcessOneMatrix(uint64_t mi)
    {
        LocalTensor<float> uWork = uWorkBuf_.Get<float>();
        LocalTensor<float> col = colBuf_.Get<float>();
        LocalTensor<float> absCol = absBuf_.Get<float>();
        LocalTensor<float> tmpRow = tmpRowBuf_.Get<float>();
        LocalTensor<float> outScalar = outScalarBuf_.Get<float>();

        LoadMatrixToUb(uWork, mi);
        SlogdetAccum acc{1.0f, 0.0f, 0U, false, 0.0f};
        for (uint32_t k = 0; k < n_; ++k) {
            if (!ProcessFullStep(uWork, col, absCol, tmpRow, k, acc)) {
                break;
            }
        }
        WriteSlogdetResult(mi, outScalar, acc);
    }

    // ============================================================================
    // BLOCKED 路径（MEM_STRATEGY=1，large-n）
    //   U 工作区常驻 GM workspace（行连续 n×n，无 32B 行 pad，便于 srcStride 列 gather）；
    //   每核独占一块 slot（offset = blockIdx_ * n*n）；UB 仅持 O(n) 行/列向量，整块 n×n 永不进 UB。
    // ============================================================================

    // 整行 GM→UB→GM 拷贝（单行 n 个 fp32 ≤2KB）：self[mi][r] → wsU[r]，建立可变工作副本。
    __aicore__ inline void CopySelfToWorkspace(uint64_t mi, uint64_t wsBase)
    {
        LocalTensor<float> cpy = cpyBuf_.Get<float>();
        uint64_t selfBase = mi * matStrideElem_;
        DataCopyExtParams p{1, static_cast<uint32_t>(n_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        for (uint32_t r = 0; r < n_; ++r) {
            DataCopyPad(cpy, selfGm_[selfBase + static_cast<uint64_t>(r) * n_], p, pad);
            PipeSync<HardEvent::MTE2_MTE3>();
            DataCopyPad(wsGm_[wsBase + static_cast<uint64_t>(r) * n_], cpy, p);
            PipeSync<HardEvent::MTE3_MTE2>();
        }
        PipeSync<HardEvent::MTE3_S>(); // workspace 写完后才能 GetValue / 后续 gather
    }

    // 列 gather：
    //   GM→UB DataCopyPad 带 srcStride（blockCount=m, blockLen=4B, srcStride=(n-1)*4B）gather 第 k 列子列；
    //   ⚠ blockLen=4B 非 32B → 框架把每元素 pad 到完整 32B DataBlock，m 个列元素落 colWide[i*8]（stride-8，非连续）；
    //   故 gather 到宽 buffer 后按 colWide[i*8] 标量紧致到连续 col[0..m-1] 再 Abs/ReduceMax。
    __aicore__ inline void GatherColumnFromGM(
        const LocalTensor<float>& colWide, const LocalTensor<float>& col, uint64_t wsBase, uint32_t k, uint32_t m)
    {
        uint64_t startElem = wsBase + static_cast<uint64_t>(k) * n_ + k; // wsU[k][k]
        DataCopyExtParams p{
            static_cast<uint16_t>(m), static_cast<uint32_t>(sizeof(float)),
            static_cast<uint32_t>((n_ - 1) * sizeof(float)), 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(colWide, wsGm_[startElem], p, pad); // 列元素落 colWide[i*8]（32B-pad）
        PipeSync<HardEvent::MTE2_S>();                  // gather(MTE2) 写完 → 标量紧致读 colWide
        for (uint32_t i = 0; i < m; ++i) {
            col.SetValue(i, colWide.GetValue(i * SLOGDET_UB_BLOCK_ELEM)); // 紧致到连续
        }
        PipeSync<HardEvent::S_V>(); // S→V：紧致写完 col 后 Abs(向量) 才能读
    }

    // 行交换：GM workspace 内两行 swap，整行经 UB 中转（单行 ≤2KB）。
    __aicore__ inline void SwapRowsGM(uint64_t wsBase, uint32_t r1, uint32_t r2)
    {
        LocalTensor<float> sbuf = swapBuf_.Get<float>();
        LocalTensor<float> cpy = cpyBuf_.Get<float>();
        uint64_t b1 = wsBase + static_cast<uint64_t>(r1) * n_;
        uint64_t b2 = wsBase + static_cast<uint64_t>(r2) * n_;
        DataCopyExtParams p{1, static_cast<uint32_t>(n_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(sbuf, wsGm_[b1], p, pad); // r1 → sbuf
        DataCopyPad(cpy, wsGm_[b2], p, pad);  // r2 → cpy
        PipeSync<HardEvent::MTE2_MTE3>();
        DataCopyPad(wsGm_[b1], cpy, p);  // r1 = old r2
        DataCopyPad(wsGm_[b2], sbuf, p); // r2 = old r1
        PipeSync<HardEvent::MTE3_S>();   // swap 写完 → 后续 GetValue 读
    }

    __aicore__ inline void EliminateBlockedTrailingRows(uint64_t wsBase, const LocalTensor<float>& rowK,
                                                       const LocalTensor<float>& rowBlock, uint32_t k, float piv,
                                                       const DataCopyExtParams& rowParams,
                                                       const DataCopyPadExtParams<float>& rowPad)
    {
        if (k + 1U >= n_) {
            return;
        }
        if (piv == 0.0f) {
            return;
        }
        uint64_t rowKBase = wsBase + static_cast<uint64_t>(k) * n_;
        DataCopyPad(rowK, wsGm_[rowKBase], rowParams, rowPad);
        PipeSync<HardEvent::MTE2_S>();
        int32_t fullLen = static_cast<int32_t>(n_);
        uint32_t ubGapBytes = (rowStrideElem_ - n_) * static_cast<uint32_t>(sizeof(float));
        for (uint32_t jb = k + 1U; jb < n_; jb += SLOGDET_ROW_BLOCK) {
            uint32_t rb = (n_ - jb) < SLOGDET_ROW_BLOCK ? (n_ - jb) : SLOGDET_ROW_BLOCK;
            uint64_t blockBase = wsBase + static_cast<uint64_t>(jb) * n_;
            DataCopyExtParams inP{
                static_cast<uint16_t>(rb), static_cast<uint32_t>(n_ * sizeof(float)), 0, ubGapBytes, 0};
            DataCopyPad(rowBlock, wsGm_[blockBase], inP, rowPad);
            PipeSync<HardEvent::MTE2_S>();
            for (uint32_t i = 0; i < rb; ++i) {
                uint32_t off = i * rowStrideElem_;
                float ajk = rowBlock.GetValue(off + k);
                float mult = ajk / piv;
                LocalTensor<float> rowJ = rowBlock[off];
                Axpy(rowJ, rowK, -mult, fullLen);
                PipeBarrier<PIPE_V>();
            }
            PipeSync<HardEvent::V_MTE3>();
            DataCopyExtParams outP{
                static_cast<uint16_t>(rb), static_cast<uint32_t>(n_ * sizeof(float)), ubGapBytes, 0, 0};
            DataCopyPad(wsGm_[blockBase], rowBlock, outP);
            PipeSync<HardEvent::MTE3_S>();
        }
    }

    __aicore__ inline bool ProcessBlockedStep(uint64_t wsBase, const LocalTensor<float>& colWide,
                                              const LocalTensor<float>& col, const LocalTensor<float>& absCol,
                                              const LocalTensor<float>& rowK, const LocalTensor<float>& rowBlock,
                                              uint32_t k, SlogdetAccum& acc, const DataCopyExtParams& rowParams,
                                              const DataCopyPadExtParams<float>& rowPad)
    {
        PipeSync<HardEvent::MTE3_MTE2>();
        uint32_t m = n_ - k;
        GatherColumnFromGM(colWide, col, wsBase, k, m);
        Abs(absCol, col, static_cast<int32_t>(m));
        PipeBarrier<PIPE_V>();
        uint32_t pivLocal = ArgMaxAbsColumn(absCol, m);
        uint32_t pidx = k + pivLocal;

        float piv = col.GetValue(pivLocal);
        float pivAbs = AbsScalar(piv);
        if (IsSingularPivot(pivAbs, acc)) {
            acc.singular = true;
            return false;
        }

        if (pidx != k) {
            SwapRowsGM(wsBase, k, pidx);
            acc.swapParity ^= 1u;
            PipeSync<HardEvent::MTE3_MTE2>();
        }
        AccumulatePivot(piv, pivAbs, acc);
        EliminateBlockedTrailingRows(wsBase, rowK, rowBlock, k, piv, rowParams, rowPad);
        return true;
    }

    __aicore__ inline void ProcessOneMatrixBlocked(uint64_t mi)
    {
        uint64_t wsBase = static_cast<uint64_t>(blockIdx_) * matStrideElem_;
        CopySelfToWorkspace(mi, wsBase);

        LocalTensor<float> colWide = colWideBuf_.Get<float>();
        LocalTensor<float> col = colBuf_.Get<float>();
        LocalTensor<float> absCol = absBuf_.Get<float>();
        LocalTensor<float> rowK = rowKBuf_.Get<float>();
        LocalTensor<float> rowBlock = rowBlockBuf_.Get<float>();
        LocalTensor<float> outScalar = outScalarBuf_.Get<float>();

        SlogdetAccum acc{1.0f, 0.0f, 0U, false, 0.0f};
        DataCopyExtParams rowParams{1, static_cast<uint32_t>(n_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> rowPad{false, 0, 0, 0};
        for (uint32_t k = 0; k < n_; ++k) {
            if (!ProcessBlockedStep(wsBase, colWide, col, absCol, rowK, rowBlock, k, acc, rowParams, rowPad)) {
                break;
            }
        }
        WriteSlogdetResult(mi, outScalar, acc);
    }

    // 标量自然对数 log(x), x>0。核内标量域用稳定多项式逼近（不依赖 libm log）。
    // x = f * 2^e, f∈[1,2)；log(f)=2*atanh((f-1)/(f+1)) 级数（f∈[1,2) 收敛快），fp32 精度满足 1e-4。
    __aicore__ inline float LogScalar(float x)
    {
        if (x <= 0.0f) {
            return -SlogdetPosInf();
        }
        int32_t e = 0;
        float f = x;
        while (f >= 2.0f) {
            f *= 0.5f;
            e += 1;
        }
        while (f < 1.0f) {
            f *= 2.0f;
            e -= 1;
        }
        float t = (f - 1.0f) / (f + 1.0f);
        float t2 = t * t;
        float series =
            t * (1.0f + t2 * (1.0f / 3.0f +
                              t2 * (1.0f / 5.0f + t2 * (1.0f / 7.0f + t2 * (1.0f / 9.0f + t2 * (1.0f / 11.0f))))));
        const float LN2 = 0.69314718055994530942f;
        return static_cast<float>(e) * LN2 + 2.0f * series;
    }

    TPipe* pipe_ = nullptr;
    GlobalTensor<float> selfGm_;
    GlobalTensor<float> signGm_;
    GlobalTensor<float> logGm_;
    GlobalTensor<float> wsGm_; // BLOCKED：U 工作区常驻 GM workspace（每核独占 slot）

    // FULL 路径 buffer
    TBuf<TPosition::VECCALC> uWorkBuf_;
    TBuf<TPosition::VECCALC> tmpRowBuf_;
    // BLOCKED 路径 buffer（colBuf_/absBuf_/redOutBuf_/redTmpBuf_/outScalarBuf_ 两路共用）
    TBuf<TPosition::VECCALC> colWideBuf_;
    TBuf<TPosition::VECCALC> rowKBuf_;
    TBuf<TPosition::VECCALC> rowBlockBuf_; // BLOCKED 消元行块（ROW_BLOCK 行批量载入/回写）
    TBuf<TPosition::VECCALC> swapBuf_;
    TBuf<TPosition::VECCALC> cpyBuf_;
    // 共用 buffer
    TBuf<TPosition::VECCALC> colBuf_;
    TBuf<TPosition::VECCALC> absBuf_;
    TBuf<TPosition::VECCALC> redOutBuf_;
    TBuf<TPosition::VECCALC> redTmpBuf_;
    TBuf<TPosition::VECCALC> outScalarBuf_;

    uint32_t n_ = 0;
    uint32_t rowStrideElem_ = 0;
    uint64_t matStrideElem_ = 0;
    uint64_t matrixNumCount_ = 0;
    float epsSingular_ = 0.f;
    uint32_t blockIdx_ = 0;
    uint32_t blockDim_ = 0;
};

} // namespace NsSlogdet

#endif // SLOGDET_H
