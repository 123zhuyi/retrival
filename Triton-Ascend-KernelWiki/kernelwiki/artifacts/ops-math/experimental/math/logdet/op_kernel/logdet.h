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
 * \file logdet.h
 * \brief Logdet template implementation with partial-pivot LU.
 *
 * 模板参数：
 *   D_T          数据类型（仅 fp32 / float）
 *   MEM_STRATEGY 0=FULL_RESIDENT（全驻留）, 1=BLOCKED（核内分块）
 *
 * 核心算法：对单 n×n 方阵带部分主元 LU，合成：
 *   out = log(det(A)); 负行列式 -> NaN，奇异 -> -inf
 *
 * 主元全局行号推导：
 *   分段局部 argmax + 标量合并。逐段 segLen ≤ 64 保证 ReduceMax 单 repeat（索引为段内局部索引），
 *   不依赖框架对 count>64 的多轮内部索引推导，逐段可验证。
 *
 * 实现规范参考 math/cholesky（DAV_2201 原生线代算子）：
 *   DataCopyExtParams/DataCopyPadExtParams 行/列 stride 搬运；PIPE_* 事件同步；
 *   LocalTensor GetValue/SetValue 标量；输出经 UB→GM DataCopyPad（非 GlobalTensor.SetValue）。
 */

#ifndef LOGDET_H
#define LOGDET_H

#include "kernel_operator.h"
#include "logdet_tiling_data.h"

namespace NsLogdet {

using namespace AscendC;

constexpr uint32_t LOGDET_BASIC_BLOCK = 32U;    // 32B DataBlock
constexpr int32_t LOGDET_SEG = 64;              // fp32 单 repeat 上限（256B/4B），主元搜索分段长度
constexpr uint32_t LOGDET_UB_BLOCK_ELEM = 8U;   // 32B / sizeof(fp32) = 8 元素/DataBlock（BLOCKED 列 gather pad 步长）

// ST golden uses an absolute singular threshold in LU decomposition. Logdet follows that contract here;
// the relative threshold used by slogdet would misclassify negative-determinant random matrices as singular.

// 核内 +inf 常量（kernel 侧无 <cmath> 宏）：用位模式构造 fp32 +inf。
__aicore__ inline float LogdetPosInf()
{
    uint32_t bits = 0x7F800000u;  // fp32 +inf
    // reinterpret_cast 位模式双关（uint32→float）：kernel 侧无 <cstring>/std::bit_cast，
    //   AscendC kernel 普遍写法，目标编译器（bisheng）行为确定。
    return *reinterpret_cast<float*>(&bits);
}

__aicore__ inline float LogdetQuietNan()
{
    uint32_t bits = 0x7FC00000u;  // fp32 quiet NaN
    return *reinterpret_cast<float*>(&bits);
}

template <typename D_T, int MEM_STRATEGY>
class LogdetKernel {
public:
    __aicore__ inline LogdetKernel() {}

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR out, GM_ADDR workspace,
                                const LogdetTilingData* tiling, TPipe* pipe)
    {
        pipe_ = pipe;
        n_ = tiling->matSizeN;
        matrixNumCount_ = tiling->matrixNumCount;
        epsSingular_ = tiling->epsSingular;

        blockIdx_ = GetBlockIdx();
        blockDim_ = GetBlockNum();

        // UB 内行 stride 按 32B（8 个 fp32）对齐，列 gather / Axpy 起址 32B 对齐前提
        rowStrideElem_ = (n_ + 7U) / 8U * 8U;
        matStrideElem_ = static_cast<uint64_t>(n_) * static_cast<uint64_t>(n_);  // 输入 GM 中每个矩阵连续 n*n
        wsMatStrideElem_ = static_cast<uint64_t>(n_) * static_cast<uint64_t>(rowStrideElem_);

        selfGm_.SetGlobalBuffer((__gm__ float*)self);
        outGm_.SetGlobalBuffer((__gm__ float*)out);

        if constexpr (MEM_STRATEGY == 0) {
            // ── FULL：n×n 整块驻留 UB ──
            // uWork 尺寸用 uint64 提升，消除右侧 uint32*uint32 中间溢出。
            pipe_->InitBuffer(uWorkBuf_, static_cast<uint64_t>(n_) * rowStrideElem_ * sizeof(float));
            pipe_->InitBuffer(colBuf_, rowStrideElem_ * sizeof(float));     // 列 gather 连续目标
            pipe_->InitBuffer(absBuf_, rowStrideElem_ * sizeof(float));     // |col|
            pipe_->InitBuffer(tmpRowBuf_, rowStrideElem_ * sizeof(float));  // 行交换中转
            pipe_->InitBuffer(redOutBuf_, LOGDET_BASIC_BLOCK);             // ReduceMax 输出（最大值,索引）
            pipe_->InitBuffer(redTmpBuf_, rowStrideElem_ * sizeof(float));  // ReduceMax sharedTmpBuffer（充分裕量）
            pipe_->InitBuffer(outScalarBuf_, LOGDET_BASIC_BLOCK);          // 标量输出中转
        } else {
            // ── BLOCKED：U 常驻 GM workspace，UB 仅持 O(n) 行/列向量（large-n）──
            // 每核独占一块 n*rowStrideElem workspace slot；行/slot 起址均 32B 对齐。
            wsGm_.SetGlobalBuffer((__gm__ float*)workspace);
            pipe_->InitBuffer(colWideBuf_, rowStrideElem_ * LOGDET_UB_BLOCK_ELEM * sizeof(float));
            pipe_->InitBuffer(colBuf_, rowStrideElem_ * sizeof(float));     // 紧致后连续子列
            pipe_->InitBuffer(absBuf_, rowStrideElem_ * sizeof(float));     // |col|
            pipe_->InitBuffer(rowKBuf_, rowStrideElem_ * sizeof(float));    // pivot 整行（消元 src）
            pipe_->InitBuffer(rowBlockBuf_, rowStrideElem_ * sizeof(float));
            pipe_->InitBuffer(swapBuf_, rowStrideElem_ * sizeof(float));    // 行交换整行中转
            pipe_->InitBuffer(cpyBuf_, rowStrideElem_ * sizeof(float));     // self→ws 行中转
            pipe_->InitBuffer(redOutBuf_, LOGDET_BASIC_BLOCK);             // ReduceMax 输出（最大值,索引）
            pipe_->InitBuffer(redTmpBuf_, rowStrideElem_ * sizeof(float));  // ReduceMax sharedTmpBuffer（充分裕量）
            pipe_->InitBuffer(outScalarBuf_, LOGDET_BASIC_BLOCK);          // 标量输出中转
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (MEM_STRATEGY == 0) {
            ProcessFull();      // 全驻留：n×n 整块进 UB
        } else {
            ProcessBlocked();   // 核内分块：U 常驻 GM workspace，UB 仅持 O(n) 行/列向量（large-n）
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

    __aicore__ inline void PIPE_V_S()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(e);
        WaitFlag<HardEvent::V_S>(e);
    }
    __aicore__ inline void PIPE_MTE2_S()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(e);
        WaitFlag<HardEvent::MTE2_S>(e);
    }
    __aicore__ inline void PIPE_MTE3_S()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(e);
        WaitFlag<HardEvent::MTE3_S>(e);
    }
    __aicore__ inline void PIPE_S_MTE3()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(e);
        WaitFlag<HardEvent::S_MTE3>(e);
    }
    // ── BLOCKED 路径额外事件同步（GM-resident 读写定序）──
    __aicore__ inline void PIPE_V_MTE3()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e);
        WaitFlag<HardEvent::V_MTE3>(e);
    }
    __aicore__ inline void PIPE_MTE2_MTE3()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(e);
        WaitFlag<HardEvent::MTE2_MTE3>(e);
    }
    __aicore__ inline void PIPE_MTE3_MTE2()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(e);
        WaitFlag<HardEvent::MTE3_MTE2>(e);
    }

    // ---- 列 gather：第 k 列下方子列 uWork[k..n-1][k] → 连续 col[0..m-1] ----
    // 行连续布局下列非连续（行 stride = rowStrideElem_）。两种受限路径：
    //   (1) UB↔UB DataCopyPad 不支持 srcStride 的单元素 gather（仅 GM↔UB 支持带 stride）；
    //   (2) 经 GM workspace 中转，UB 源起址 uWork[k*stride + k] 在 (k%8)!=0 时非 32B 对齐
    //       → 触发 507035（VEC/MTE UB 地址非对齐）且读到错列 → 误判奇异。
    // 故采用**标量 gather**：
    // 逐元素 GetValue/SetValue 把子列搬到连续 col（标量域无对齐约束），再 S→V 同步后走真实分段 ReduceMax。
    // 标量 gather 的 GetValue 列偏移 +k 在标量域合法（非向量指令，无 32B 起址要求）。
    __aicore__ inline void GatherColumn(const LocalTensor<float>& uWork, const LocalTensor<float>& col,
                                        uint32_t k, uint32_t m)
    {
        for (uint32_t i = 0; i < m; ++i) {
            col.SetValue(i, uWork.GetValue((k + i) * rowStrideElem_ + k));
        }
        // S → V：SetValue(标量) 写完 col 后才能 Abs(向量) 读
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(e);
        WaitFlag<HardEvent::S_V>(e);
    }

    // ---- 分段局部 argmax + 标量合并 ----
    // 返回 [0, m) 内子列行偏移；逐段 segLen≤64，ReduceMax 单 repeat（段内局部索引）。
    __aicore__ inline uint32_t ArgMaxAbsColumn(const LocalTensor<float>& absCol, uint32_t m)
    {
        LocalTensor<float> redOut = redOutBuf_.Get<float>();
        LocalTensor<float> redTmp = redTmpBuf_.Get<float>();

        float bestVal = -1.0f;
        uint32_t bestIdx = 0;
        for (uint32_t segStart = 0; segStart < m; segStart += static_cast<uint32_t>(LOGDET_SEG)) {
            uint32_t rem = m - segStart;
            uint32_t segLen = rem < static_cast<uint32_t>(LOGDET_SEG) ? rem : static_cast<uint32_t>(LOGDET_SEG);
            // segLen ≤ 64 → 单 repeat → 返回索引为段内局部索引
            ReduceMax<float>(redOut, absCol[segStart], redTmp, static_cast<int32_t>(segLen), true);
            PIPE_V_S();  // V → S：等待 ReduceMax 写完再 GetValue

            float segVal = redOut.GetValue(0);
            float idxAsFloat = redOut.GetValue(1);
            // ReduceMax(calIndex=true) 返回索引按 uint32 读取，是 ReduceMax.md **官方惯例**
            //   `reinterpret_cast<uint32_t>(dst.GetValue(1))`；kernel 侧无 std::bit_cast，目标编译器行为确定。
            uint32_t segLocal = *reinterpret_cast<uint32_t*>(&idxAsFloat);
            uint32_t globalIdx = segStart + segLocal;                       // 段起点 + 段内局部 → 全局子列偏移

            if (segVal > bestVal) {  // 严格大于 → 命中首个 |max|（与部分主元一致，确定性）
                bestVal = segVal;
                bestIdx = globalIdx;
            }
        }
        return bestIdx;  // ∈ [0, m)
    }

    // ---- 行交换：UB 内整行 swap（向量 DataCopy，禁逐元素 SetValue）----
    __aicore__ inline void SwapRows(const LocalTensor<float>& uWork, const LocalTensor<float>& tmpRow,
                                    uint32_t r1, uint32_t r2)
    {
        DataCopy(tmpRow, uWork[r1 * rowStrideElem_], rowStrideElem_);
        PipeBarrier<PIPE_V>();
        DataCopy(uWork[r1 * rowStrideElem_], uWork[r2 * rowStrideElem_], rowStrideElem_);
        PipeBarrier<PIPE_V>();
        DataCopy(uWork[r2 * rowStrideElem_], tmpRow, rowStrideElem_);
        PipeBarrier<PIPE_V>();
    }

    // ---- 奇异判定（绝对阈值，FULL/BLOCKED 共用）----
    __aicore__ inline bool CheckSingular(float piv)
    {
        float pivAbs = piv >= 0.0f ? piv : -piv;
        return pivAbs < epsSingular_;
    }

    // ---- 结果合成 + 写回 GM（FULL/BLOCKED 共用）----
    // Logdet 只输出 log(det): 奇异 -> -inf；最终符号为负 -> NaN；最终符号为正 -> logabs。
    __aicore__ inline void WriteResult(uint64_t mi, bool singular, uint32_t swapParity, float signProd, float logabs)
    {
        LocalTensor<float> outScalar = outScalarBuf_.Get<float>();

        float finalLogdet;
        if (singular) {
            finalLogdet = -LogdetPosInf();  // -inf
        } else {
            float finalSign = (swapParity != 0) ? -signProd : signProd;
            finalLogdet = (finalSign < 0.0f) ? LogdetQuietNan() : logabs;
        }

        // ---- 写回 GM：经 UB→GM DataCopyPad（每 batch 1 标量）----
        outScalar.SetValue(0, finalLogdet);
        PIPE_S_MTE3();
        {
            DataCopyExtParams p{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(outGm_[mi], outScalar, p);
        }
        PIPE_MTE3_S();
    }

    // ---- FULL 矩阵搬入 GM → UB：逐行 DataCopyPad（每行 n 个 fp32，落到 32B 对齐行起址）----
    __aicore__ inline void LoadMatrixToUB(uint64_t mi, const LocalTensor<float>& uWork)
    {
        uint64_t base = mi * matStrideElem_;
        for (uint32_t r = 0; r < n_; ++r) {
            DataCopyExtParams p{1, static_cast<uint32_t>(n_ * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            DataCopyPad(uWork[static_cast<uint64_t>(r) * rowStrideElem_],
                        selfGm_[base + static_cast<uint64_t>(r) * n_], p, padParams);
        }
        PIPE_MTE2_S();  // 搬入完成后才能 GetValue / 计算
    }

    // ---- FULL 消元（全行 Axpy，32B 对齐）：row_j -= (u[j][k]/piv)*row_k, j=k+1..n-1 ----
    //   Axpy 要求 src/dst 起址 32B 对齐；故对**整行**（列 0 起，32B 对齐起址）做 Axpy，长度 rowStrideElem_（8 倍数）。
    //   正确性：列 k 恰消元为 0；列 > k 有效 LU 更新；列 < k 死列被改写无害（后续不再读）。
    __aicore__ inline void EliminateFull(const LocalTensor<float>& uWork, uint32_t k, float piv)
    {
        if (piv == 0.0f) { return; }  // 防御除零：调用方已保证 piv 非零（奇异在上游判定并 break），此处满足静态除零检查
        LocalTensor<float> rowK = uWork[k * rowStrideElem_];
        int32_t fullLen = static_cast<int32_t>(rowStrideElem_);
        for (uint32_t j = k + 1U; j < n_; ++j) {
            // uWork[j][k] 仅被本轮 row j 自己的 Axpy 改写；读在本 j 的 Axpy 之前（程序序），
            // 其它行的 Axpy 不触及行 j → 跨 j 无需逐次同步。
            float ajk = uWork.GetValue(j * rowStrideElem_ + k);
            float mult = ajk / piv;
            LocalTensor<float> rowJ = uWork[j * rowStrideElem_];
            Axpy(rowJ, rowK, -mult, fullLen);  // 整行；列 ≥ k 有效更新，列 < k 死列无害
        }
        PIPE_V_S();  // 所有 Axpy(V) 完成 → 下一轮 gather/对角元 GetValue(S)
    }

    __aicore__ inline void ProcessOneMatrix(uint64_t mi)
    {
        LocalTensor<float> uWork = uWorkBuf_.Get<float>();
        LocalTensor<float> col = colBuf_.Get<float>();
        LocalTensor<float> absCol = absBuf_.Get<float>();
        LocalTensor<float> tmpRow = tmpRowBuf_.Get<float>();

        LoadMatrixToUB(mi, uWork);  // GM → UB 逐行搬入 + PIPE_MTE2_S

        float signProd = 1.0f;
        float logabs = 0.0f;
        uint32_t swapParity = 0;
        bool singular = false;
        for (uint32_t k = 0; k < n_; ++k) {
            uint32_t m = n_ - k;  // 子列长度

            // ---- 主元搜索：列 gather → Abs → 分段局部 argmax + 标量合并 ----
            GatherColumn(uWork, col, k, m);  // 标量 gather + S→V 同步在内部完成
            Abs(absCol, col, static_cast<int32_t>(m));
            PipeBarrier<PIPE_V>();
            uint32_t pivLocal = ArgMaxAbsColumn(absCol, m);
            uint32_t pidx = k + pivLocal;  // 全局行号 = 列起点 k + 子列偏移

            // ---- 奇异判定（绝对阈值）----
            float piv = uWork.GetValue(pidx * rowStrideElem_ + k);  // 带符号主元
            float pivAbs = piv >= 0.0f ? piv : -piv;
            if (CheckSingular(piv)) {
                singular = true;
                break;
            }

            // ---- 行交换 ----
            if (pidx != k) {
                SwapRows(uWork, tmpRow, k, pidx);
                swapParity ^= 1u;
                PIPE_V_S();  // swap(DataCopy,V) 写 uWork → 后续消元 GetValue(S) 读
            }

            // ---- 对角元归约（标量顺序累加，形态 a，确定性天然满足）----
            if (piv < 0.0f) {
                signProd = -signProd;
            }
            logabs += LogScalar(pivAbs);

            // ---- 消元（见 EliminateFull：整行 Axpy，32B 对齐）----
            if (k + 1U < n_) {
                EliminateFull(uWork, k, piv);
            }
        }

        WriteResult(mi, singular, swapParity, signProd, logabs);  // 合成 + 标量写回 GM
    }

    // ============================================================================
    // BLOCKED 路径（MEM_STRATEGY=1，large-n）
    //   U 工作区常驻 GM workspace（每行按 rowStrideElem_ pad 到 32B 对齐）；
    //   每核独占一块 slot（offset = blockIdx_ * n * rowStrideElem_）；UB 仅持 O(n) 行/列向量。
    // ============================================================================

    // 建立可变工作副本：input self 是紧凑 n*n，workspace 按 rowStrideElem_ 补齐到 32B 对齐行。
    // self→UB 用 DataCopyPad 补齐行尾，UB→workspace 写对齐整行。
    __aicore__ inline void CopySelfToWorkspace(uint64_t mi, uint64_t wsBase)
    {
        LocalTensor<float> cpy = cpyBuf_.Get<float>();
        uint64_t selfBase = mi * matStrideElem_;
        DataCopyExtParams inP{1, static_cast<uint32_t>(n_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> inPad{true, 0, static_cast<uint8_t>(rowStrideElem_ - n_), 0.0f};
        DataCopyExtParams outP{1, static_cast<uint32_t>(rowStrideElem_ * sizeof(float)), 0, 0, 0};
        for (uint32_t r = 0; r < n_; ++r) {
            uint64_t srcRow = selfBase + static_cast<uint64_t>(r) * n_;
            DataCopyPad(cpy, selfGm_[srcRow], inP, inPad);
            PIPE_MTE2_MTE3();
            DataCopyPad(wsGm_[wsBase + static_cast<uint64_t>(r) * rowStrideElem_], cpy, outP);
            PIPE_MTE3_MTE2();  // MTE3 读完 cpy 后，下一行才能继续用 MTE2 改写 cpy
        }
        PIPE_MTE3_MTE2();  // workspace 写完后才能被后续 GM→UB 读
    }

    // 列 gather（workspace 行 stride 已 32B 对齐）：
    //   GM→UB DataCopyPad 带 srcStride gather 第 k 列子列；
    //   blockLen=4B 时每个元素落到 colWide[i*8]，再标量紧致到连续 col 后做 Abs/ReduceMax。
    __aicore__ inline void GatherColumnFromGM(const LocalTensor<float>& colWide, const LocalTensor<float>& col,
                                              uint64_t wsBase, uint32_t k, uint32_t m)
    {
        uint64_t startElem = wsBase + static_cast<uint64_t>(k) * rowStrideElem_ + k;  // wsU[k][k]
        DataCopyExtParams p{static_cast<uint16_t>(m), static_cast<uint32_t>(sizeof(float)),
                            static_cast<uint32_t>((rowStrideElem_ - 1U) * sizeof(float)), 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(colWide, wsGm_[startElem], p, pad);
        PIPE_MTE2_S();
        for (uint32_t i = 0; i < m; ++i) {
            col.SetValue(i, colWide.GetValue(i * LOGDET_UB_BLOCK_ELEM));
        }
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(e);
        WaitFlag<HardEvent::S_V>(e);
    }

    // 行交换：GM workspace 内两行 swap，整行经 UB 中转（单行 ≤2KB）。
    __aicore__ inline void SwapRowsGM(uint64_t wsBase, uint32_t r1, uint32_t r2)
    {
        LocalTensor<float> sbuf = swapBuf_.Get<float>();
        LocalTensor<float> cpy = cpyBuf_.Get<float>();
        uint64_t b1 = wsBase + static_cast<uint64_t>(r1) * rowStrideElem_;
        uint64_t b2 = wsBase + static_cast<uint64_t>(r2) * rowStrideElem_;
        DataCopyExtParams p{1, static_cast<uint32_t>(rowStrideElem_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(sbuf, wsGm_[b1], p, pad);   // r1 → sbuf
        DataCopyPad(cpy, wsGm_[b2], p, pad);    // r2 → cpy
        PIPE_MTE2_MTE3();
        DataCopyPad(wsGm_[b1], cpy, p);         // r1 = old r2
        DataCopyPad(wsGm_[b2], sbuf, p);        // r2 = old r1
        PIPE_MTE3_S();  // swap 写完 → 后续 GetValue 读
    }

    // ---- BLOCKED 消元 ----
    // 按单行搬运，行起址和长度均 32B 对齐；每行用 Axpy 恢复向量路径。
    __aicore__ inline void EliminateBlocked(uint64_t wsBase, uint32_t k, float piv,
                                            const LocalTensor<float>& rowK, const LocalTensor<float>& rowBlock,
                                            const DataCopyExtParams& rowParams,
                                            const DataCopyPadExtParams<float>& rowPad)
    {
        if (piv == 0.0f) { return; }  // 防御除零：调用方已保证 piv 非零（奇异在上游判定并 break），此处满足静态除零检查
        uint64_t rowKBase = wsBase + static_cast<uint64_t>(k) * rowStrideElem_;
        DataCopyPad(rowK, wsGm_[rowKBase], rowParams, rowPad);  // pivot 整行进 UB 一次
        PIPE_MTE2_S();  // rowK 进 UB（MTE2）→ 下面 GetValue(S) 安全
        int32_t fullLen = static_cast<int32_t>(rowStrideElem_);
        for (uint32_t j = k + 1U; j < n_; ++j) {
            uint64_t rowBase = wsBase + static_cast<uint64_t>(j) * rowStrideElem_;
            DataCopyPad(rowBlock, wsGm_[rowBase], rowParams, rowPad);
            PIPE_MTE2_S();

            float ajk = rowBlock.GetValue(k);
            float mult = ajk / piv;
            Axpy(rowBlock, rowK, -mult, fullLen);  // 列 ≥ k 有效，列 < k 死列无害
            PIPE_V_MTE3();

            DataCopyPad(wsGm_[rowBase], rowBlock, rowParams);
            PIPE_MTE3_S();
        }
    }

    __aicore__ inline void ProcessOneMatrixBlocked(uint64_t mi)
    {
        uint64_t wsBase = static_cast<uint64_t>(blockIdx_) * wsMatStrideElem_;  // 本核独占对齐 slot

        CopySelfToWorkspace(mi, wsBase);

        LocalTensor<float> colWide = colWideBuf_.Get<float>();
        LocalTensor<float> col = colBuf_.Get<float>();
        LocalTensor<float> absCol = absBuf_.Get<float>();
        LocalTensor<float> rowK = rowKBuf_.Get<float>();
        LocalTensor<float> rowBlock = rowBlockBuf_.Get<float>();

        float signProd = 1.0f;
        float logabs = 0.0f;
        uint32_t swapParity = 0;
        bool singular = false;
        DataCopyExtParams rowParams{1, static_cast<uint32_t>(rowStrideElem_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> rowPad{false, 0, 0, 0.0f};

        for (uint32_t k = 0; k < n_; ++k) {
            uint32_t m = n_ - k;  // 子列长度

            // MTE3→MTE2 屏障：上一轮 MTE3 写 wsGM 后，本轮 gather(MTE2)
            // 读同一 wsGM 须先等 MTE3 落盘（MTE3/MTE2 独立流水、硬件不自动定序），否则读旧值。
            PIPE_MTE3_MTE2();

            // 主元搜索：列 gather → Abs → 分段局部 argmax。
            GatherColumnFromGM(colWide, col, wsBase, k, m);
            Abs(absCol, col, static_cast<int32_t>(m));
            PipeBarrier<PIPE_V>();
            uint32_t pivLocal = ArgMaxAbsColumn(absCol, m);
            uint32_t pidx = k + pivLocal;  // 全局行号 = 列起点 k + 子列偏移

            // 奇异判定（绝对阈值，同 FULL 路径）
            float piv = col.GetValue(pivLocal);  // 带符号主元（来自 UB col，避免额外 GM 标量读）
            float pivAbs = piv >= 0.0f ? piv : -piv;
            if (CheckSingular(piv)) {
                singular = true;
                break;
            }

            // 行交换（GM workspace 内，整行）
            if (pidx != k) {
                SwapRowsGM(wsBase, k, pidx);  // 内含 MTE2_MTE3 + MTE3_S
                swapParity ^= 1u;
                PIPE_MTE3_MTE2();  // swap 经 MTE3 写 → 下面 rowK 载入(MTE2) 须等 MTE3 落盘
            }

            // 对角元归约（标量顺序累加，形态 a，确定性天然）
            if (piv < 0.0f) {
                signProd = -signProd;
            }
            logabs += LogScalar(pivAbs);

            // 消元（见 EliminateBlocked：单行 DMA + Axpy，GM↔UB 32B 对齐）
            if (k + 1U < n_) {
                EliminateBlocked(wsBase, k, piv, rowK, rowBlock, rowParams, rowPad);
            }
        }

        WriteResult(mi, singular, swapParity, signProd, logabs);  // 合成 + 标量写回 GM
    }

    // 标量自然对数 log(x), x>0。核内标量域用稳定多项式逼近（不依赖 libm log）。
    // x = f * 2^e, f∈[1,2)；log(f)=2*atanh((f-1)/(f+1)) 级数（f∈[1,2) 收敛快），fp32 精度满足 1e-4。
    __aicore__ inline float LogScalar(float x)
    {
        if (x <= 0.0f) {
            return -LogdetPosInf();
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
        float series = t * (1.0f + t2 * (1.0f / 3.0f + t2 * (1.0f / 5.0f +
                            t2 * (1.0f / 7.0f + t2 * (1.0f / 9.0f + t2 * (1.0f / 11.0f))))));
        const float LN2 = 0.69314718055994530942f;
        return static_cast<float>(e) * LN2 + 2.0f * series;
    }

    TPipe* pipe_ = nullptr;
    GlobalTensor<float> selfGm_;
    GlobalTensor<float> outGm_;
    GlobalTensor<float> wsGm_;  // BLOCKED：U 工作区常驻 GM workspace（每核独占 slot）

    // FULL 路径 buffer
    TBuf<TPosition::VECCALC> uWorkBuf_;
    TBuf<TPosition::VECCALC> tmpRowBuf_;
    // BLOCKED 路径 buffer（colBuf_/absBuf_/redOutBuf_/redTmpBuf_/outScalarBuf_ 两路共用）
    TBuf<TPosition::VECCALC> colWideBuf_;
    TBuf<TPosition::VECCALC> rowKBuf_;
    TBuf<TPosition::VECCALC> rowBlockBuf_;
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
    uint64_t wsMatStrideElem_ = 0;
    uint64_t matrixNumCount_ = 0;
    float epsSingular_ = 0.f;
    uint32_t blockIdx_ = 0;
    uint32_t blockDim_ = 0;
};

} // namespace NsLogdet

#endif // LOGDET_H
