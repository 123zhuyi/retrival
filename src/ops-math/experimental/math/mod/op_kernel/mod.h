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
 * \file mod.h
 * \brief Mod operator implementation
 */

#ifndef MOD_H
#define MOD_H

#include "kernel_operator.h"
#include "mod_tiling_data.h"
#include "mod_tiling_key.h"
#include <limits>

// arch22 (A2/910b, DAV_C220 向量核) 守卫开关 MOD_ENH_ARCH22。主判据 __NPU_ARCH__==2201
//   (ccec -dM -E 实测：ascend910b device 编译里 __NPU_ARCH__==2201 / __DAV_C220_VEC__==1 /
//   __CCE_AICORE__==220 同时定义 -> 以下 OR 守卫对 opbuild 任一 c220 arch flag 都鲁棒)。
//   仅此守卫下走 AlgoA (Sign + CAST_TRUNC)；非 c220 走上游朴素核 (零回归)。
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201) || (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || \
    defined(__DAV_C220_VEC__)
#define MOD_ENH_ARCH22 1
#include "adv_api/math/sign.h"
#else
#define MOD_ENH_ARCH22 0
#endif

namespace ModNs {
using namespace AscendC;

// Mod<T, ST, OT> keeps the historical template axes, but every registered/instantiated lane is same-dtype
// (ST==OT==T). Cross-dtype inputs are normalized by aclnn before entering this kernel.
template <typename T, typename ST = T, typename OT = T>
class Mod {
public:
    __aicore__ inline Mod(){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, const ModTilingData* tilingData);
    __aicore__ inline void Process();

    uint8_t bufferNum = 2;
    constexpr static uint8_t DATA_BLOCK = 64;
    constexpr static int QUEUE_DEPTH = 2;

    // x1/x2 fp32 intermediate buffers for the same-dtype half/bf16/int16 conversion paths.
    static constexpr bool NEED_FP32_IO_BUF = std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> ||
                                             std::is_same_v<T, int16_t>;
    // K2: int16 same-dtype (T==int16) 强制走整数域 naive 4-op (|self|<=32767<2^24 精确, 永不 AlgoA —
    //   独立于 K1 adaptive, 防 |int16|>=256 被误路由到 AlgoA 型病态慢)。
    //   fp lanes (T ∈ {float,half,bfloat16}) -> K1 自适应 (naive|AlgoA per-tile max|a|)。
    static constexpr bool USE_ALGO_A = !std::is_same_v<T, int16_t>;
    // 连续路精简计算核：仅 same-dtype fp32/fp16/bf16 的连续派发 (isInput2Scalar || isInput2SameShape)
    //   走精简核 —— 直接 RemainderAdaptive (mod_algoa_impl.h), 绕过 ComputeCore/ComputeFPCore 的
    //   inf/nan/zero Select 收尾 (6 op/tile) + 常驻 inf/nan/zero/mask 常量 + tmp, 把 UB 69/65 -> 48 B/elem
    //   (tile 更宽 -> tile 数更少 -> 摊薄 per-tile RemainderAdaptive 探针同步)。
    //   inf/nan 由 naive/AlgoA 自然传播 (匹配内置 nan 语义；上游通用核的 Select 收尾把 fmod(x,inf) 映到 x,
    //   与内置 nan 语义相反 —— 精简核回避该收尾)。int16 / int32 USE_LEAN_CONTIG=false ->
    //   保持 ProcessContigPipeline<false>/ComputeCore 原路 (.o 不变)。general broadcast (非融合) 走
    //   ComputeCore -> 需常驻常量 -> host 保持 69/65 divider (不精简)。
    static constexpr bool USE_LEAN_CONTIG = std::is_same_v<T, float> || std::is_same_v<T, half> ||
                                            std::is_same_v<T, bfloat16_t>;

private:
    __aicore__ inline void CopyIn(uint64_t offset, int32_t calCount, bool isConstantX2 = false);
    __aicore__ inline void CopyOut(uint64_t offset, int32_t calCount);
    __aicore__ inline void ParseTilingData(const ModTilingData* tilingData);
    __aicore__ inline void InitConstants();
    __aicore__ inline void ProcessBroadcast(uint64_t inOffset, uint64_t outOffset);
    __aicore__ inline void ProcessContiguous(uint64_t inOffset, uint64_t outOffset);
    __aicore__ inline void Compute(int32_t computeCount);
    __aicore__ inline uint64_t GetInput2Offset(uint64_t outputOffset);
    // `isConstantRun` (out-param) 表示返回的 `count` 整段落在一段纯广播尾轴内
    // (input2Stride[i]==0, 即 input2Shape[i]==1) -> x2 在整段内是单一常量。仅 tiling 粒度改动
    // (见 GetInput2ContiguousCopyCount)，不影响 K1/K2 数值路由。
    __aicore__ inline uint32_t GetInput2ContiguousCopyCount(uint64_t outputOffset, uint32_t remainingCount,
                                                            bool& isConstantRun);

    __aicore__ inline void InitBuffers();

    // 共享数学主体：从原 Compute() 抽出 (逻辑不变，仅参数化替原隐式 TQue DeQue/AllocTensor)，
    // 使 legacy TQue Compute() (ProcessBroadcast / 非 arch22 ProcessContiguous 仍用) 与 arch22
    // flat-buffer 路径 (ProcessContiguousFlat) 调用同一份 K1/K2/int32 路由真值源，
    // 消除两套 buffer 管理机制间的漂移风险。
    __aicore__ inline void ComputeCore(const int32_t calCount, LocalTensor<ST>& x1Tensor, LocalTensor<OT>& x2Tensor,
                                       LocalTensor<T>& dstTensor, LocalTensor<uint8_t>& sharedTmpBuffer);

#if MOD_ENH_ARCH22
    // flat-path：常驻 FLAT_SLOTS 深度 ping-pong buffer + 预分配 manual event，替代每 tile 的
    // TQue Alloc/EnQue/DeQue/Free 簿记。即 2-slot ping-pong (FLAT_SLOTS=2)。仅用于 CONTIGUOUS 派发
    // (isInput2Scalar || isInput2SameShape，== ProcessContiguous 域)。general broadcast (ProcessBroadcast)
    // 不受影响，保留既有 TQue 机制。
    static constexpr int32_t FLAT_SLOTS = 2;

    __aicore__ inline void InitFlatBuffers();
    // flat (ComputeCore) 与 lean (ComputeContigLean) 连续派发原是两个近似相同的 Process 循环
    // (重复代码)，现合并为一个按编译期 LEAN flag 参数化的模板：被丢弃的 `if constexpr` 分支保证
    // 每个实例化体都是原路径循环 (LEAN=true == 原 ProcessContigLean；LEAN=false == 原
    // ProcessContiguousFlat)。定义在 mod_flat_impl.h。
    template <bool LEAN>
    __aicore__ inline void ProcessContigPipeline(uint64_t inOffset, uint64_t outOffset);
    __aicore__ inline void CopyInFlat(int32_t slot, uint64_t offset, uint64_t n, bool reuse);
    __aicore__ inline void CopyOutFlat(int32_t slot, uint64_t offset, uint64_t n);
    // ROUND8-style: read the GM scalar divisor ONCE per core and Duplicate it across the full tile width
    // ONCE (the legacy TQue CopyIn's isInput2Scalar branch re-reads + re-Duplicates EVERY tile — redundant,
    // since the value is tile-invariant; this is a genuine additional lever beyond the TQue->flat swap).
    __aicore__ inline void LoadScalarOtherFlat();
    __aicore__ inline LocalTensor<ST> FlatSelfSlot(int32_t slot);
    __aicore__ inline LocalTensor<OT> FlatOtherSlot(int32_t slot);
    __aicore__ inline LocalTensor<T> FlatOutSlot(int32_t slot);

    // 融合广播 (arch22-only)：OUTER build-once-per-core + INNER per-row Duplicate 反冗余。仅当 host 判定资格
    //   (fp32/fp16/bf16/int16 同 dtype + collapse-2D) 置 bcastFusedMode_ != 0 才进入 -> 通用 ProcessBroadcast /
    //   mixed / int32 广播 全部走不到融合 (独立运行时分支、风险隔离)。核心：OUTER 行广播每核只把 x2 的 [INNER]
    //   行读一次 -> UB 常驻复用 (消 mte2 冗余)；INNER 列广播每行 Duplicate (避 GatherMask)。0811 起 INNER 不再
    //   要求 32B 对齐：UB 内统一 [rows, bcIpad_] padding 行布局 (行首恒 32B 对齐)，pad 车道 x1=0/x2=1.0 消毒
    //   (商=0 良性、结果被 2D CopyOut 丢弃、maxAbsA 探针不受污染)。
    //   物化后跑 RemainderAdaptive (与连续精简核同一 fp32 数学主体)。SetBlockDim 仍取通用路 needCoreNum
    //   (融合按 OUTER 行切到同一批核，越界核 coreRows_==0 空转) -> 非 arch22 回落通用路 block dim 一致、零回归。
    __aicore__ inline void InitFusedBcastBuffers();
    __aicore__ inline void ProcessFusedBcast();
    __aicore__ inline void BuildOuterOtherFused();
    __aicore__ inline void CopyInFusedBcast(uint64_t rowOffset, uint64_t rows);
    __aicore__ inline void ComputeFusedBcast(uint64_t rows);
    __aicore__ inline void CopyOutFusedBcast(uint64_t rowOffset, uint64_t rows);

    // same-dtype fp32/fp16/bf16 连续路精简核 (USE_LEAN_CONTIG，定义在 mod_leancontig_impl.h)。复用
    //   mod_flat_impl.h 的 flat ping-pong (CopyInFlat/CopyOutFlat/LoadScalarOtherFlat/events + 共享
    //   ProcessContigPipeline<LEAN=true> 骨架)，仅把 per-tile 计算换成精简 ComputeContigLean (RemainderAdaptive
    //   直算 fp32 域，无 inf/nan 收尾 / 无 tmp)。buffer 只含 flat self/out/other + 6 个 fp32 工作块 (+cast:
    //   selfF32/otherF32/rF32) -> 48 B/elem。
    __aicore__ inline void InitLeanWorkBuffers();
    __aicore__ inline void ComputeContigLean(int32_t calCount, LocalTensor<ST>& x1Tensor, LocalTensor<OT>& x2Tensor,
                                             LocalTensor<T>& dstTensor);
#endif

    __aicore__ inline void ComputeInt32(const int32_t calCount, const int32_t alignedCalCount,
                                        LocalTensor<T>& dstTensor, LocalTensor<ST>& x1Tensor, LocalTensor<OT>& x2Tensor,
                                        LocalTensor<uint8_t>& sharedTmpBuffer);

    __aicore__ inline void ComputeFPCore(const int32_t calCount, const int32_t alignedCalCount,
                                         LocalTensor<float>& x1Float, LocalTensor<float>& x2Float,
                                         LocalTensor<float>& resRem, LocalTensor<float>& resQuot,
                                         LocalTensor<uint8_t>& sharedTmpBuffer);

    // AlgoA/Dekker 数学辅助在此类内声明、在 mod_algoa_impl.h 外置定义。
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlign(T1 a, T2 b);

    // Dekker TwoProduct (Veltkamp split，全 fp32)：pOut = round(x*y)，eOut = x*y - pOut
    // (fp32 乘积的精确残差)。x/y 不变；s0/s1/s2 为与 x/y/pOut/eOut 不相交的暂存块。
    // 仅用 Mul/Muls/Sub/Add；split 常量 4097 = 2^12+1 (float 24-bit 尾数)。
    __aicore__ inline void TwoProd(const LocalTensor<float>& pOut, const LocalTensor<float>& eOut,
                                   const LocalTensor<float>& x, const LocalTensor<float>& y,
                                   const LocalTensor<float>& s0, const LocalTensor<float>& s1,
                                   const LocalTensor<float>& s2, int32_t cnt);

#if MOD_ENH_ARCH22
    // 大商补偿路径：rOut = aIn - trunc(aIn/bIn)*bIn (torch.fmod 大商语义)，aIn/bIn 保留。
    //   6 个不相交 fp32 工作块 w0..w5 (均 != rOut/aIn/bIn)；含 Sign 符号修正与 CAST_TRUNC。
    __aicore__ inline void RemainderAlgoA(const LocalTensor<float>& rOut, const LocalTensor<float>& aIn,
                                          const LocalTensor<float>& bIn, const LocalTensor<float>& w0,
                                          const LocalTensor<float>& w1, const LocalTensor<float>& w2,
                                          const LocalTensor<float>& w3, const LocalTensor<float>& w4,
                                          const LocalTensor<float>& w5, int32_t cnt);

    // K2 整数域/自适应小值分支共用的 4-op 余数计算。
    __aicore__ inline void RemainderNaive(const LocalTensor<float>& rOut, const LocalTensor<float>& aIn,
                                          const LocalTensor<float>& bIn, const LocalTensor<float>& work, int32_t cnt);

    // 自适应路由：per-tile max|a| < naiveThresh_ 走朴素 4-op (小商省算力)，否则走 RemainderAlgoA
    //   (大商补偿)；tile 内任一 |a|>=thresh 即整 tile 走 AlgoA。maxAbsA = Abs(aIn)->ReduceMax->GetValue。
    __aicore__ inline void RemainderAdaptive(const LocalTensor<float>& rOut, const LocalTensor<float>& aIn,
                                             const LocalTensor<float>& bIn, const LocalTensor<float>& w0,
                                             const LocalTensor<float>& w1, const LocalTensor<float>& w2,
                                             const LocalTensor<float>& w3, const LocalTensor<float>& w4,
                                             const LocalTensor<float>& w5, int32_t cnt);

    // per-core 路由包装 (0811 深夜)：coreRoute_==1 -> 整核 RemainderNaive；==2 -> 整核 RemainderAlgoA；
    //   否则 -> per-tile RemainderAdaptive 现状回落。5 个原 RemainderAdaptive 调用点全部改调本函数。
    __aicore__ inline void RemainderRouted(const LocalTensor<float>& rOut, const LocalTensor<float>& aIn,
                                           const LocalTensor<float>& bIn, const LocalTensor<float>& w0,
                                           const LocalTensor<float>& w1, const LocalTensor<float>& w2,
                                           const LocalTensor<float>& w3, const LocalTensor<float>& w4,
                                           const LocalTensor<float>& w5, int32_t cnt);

    // 每核一次置位 coreRoute_ (Process() 内 InitConstants 后、派发前调用一次)。V5.1 收敛形态：
    //   仅 int16 置 1 (K2 整核 naive 锁)；fp lanes 恒 0 (fp 预扫已实证否决，见 mod_algoa_impl.h
    //   函数头记录)；int32 恒 0。不做任何 GM 预扫读写。
    __aicore__ inline void PreScanCoreRoute();
#endif

private:
    TPipe pipe;
    // TQue framework: used by ProcessBroadcast (general broadcast, all archs) AND by ProcessContiguous on
    // non-arch22 builds. On arch22, the CONTIGUOUS dispatch uses the flat buffers below instead (InitBuffers
    // allocates ONE OR THE OTHER at runtime, never both -> no UB double-booking).
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inputx1Queue;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inputx2Queue;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> outputQueue;

#if MOD_ENH_ARCH22
    // flat-path（arch22 CONTIGUOUS 派发专用，见上方 FLAT_SLOTS 注释）。
    TBuf<TPosition::VECCALC> flatSelfBuf_;
    TBuf<TPosition::VECCALC> flatOtherBuf_;       // per-tile tensor-other (isInput2SameShape && !isInput2Scalar)
    TBuf<TPosition::VECCALC> flatOtherScalarBuf_; // once-per-core scalar-other (isInput2Scalar)
    TBuf<TPosition::VECCALC> flatOutBuf_;
    event_t flatEvMte2V_[FLAT_SLOTS] = {EVENT_ID0, EVENT_ID0}; // MTE2->V: read slot inputs landed
    event_t flatEvVMte2_[FLAT_SLOTS] = {EVENT_ID0, EVENT_ID0}; // V->MTE2: read slot reusable
    event_t flatEvVMte3_[FLAT_SLOTS] = {EVENT_ID0, EVENT_ID0}; // V->MTE3: out slot ready for drain
    event_t flatEvMte3V_[FLAT_SLOTS] = {EVENT_ID0, EVENT_ID0}; // MTE3->V: out slot reusable

    // Path B 融合广播: 物化后的广播 other (OT 型), [ubFormer*INNER] 满 tile。OUTER 每核建一次常驻;
    // INNER 每 tile 逐行 Duplicate。self/out 复用 inputx1Queue/outputQueue, 原始 other 读用 inputx2Queue(小)。
    TBuf<TPosition::VECCALC> bcastOtherBuf_;
#endif

    TBuf<TPosition::VECCALC> tmpBuff;

    // Computational Buffers
    TBuf<TPosition::VECCALC> ResQuotTensorBuff;
    TBuf<TPosition::VECCALC> ResRemTensorBuff;

    // AlgoA 的 5 个额外 fp32 工作块 (w1..w5; w0 复用 ResQuotTensor)
    TBuf<TPosition::VECCALC> A1Buff;
    TBuf<TPosition::VECCALC> A2Buff;
    TBuf<TPosition::VECCALC> A3Buff;
    TBuf<TPosition::VECCALC> A4Buff;
    TBuf<TPosition::VECCALC> A5Buff;

    // Auxiliary Buffers (Constants & Temps)
    TBuf<TPosition::VECCALC> ZeroTensorBuff;
    TBuf<TPosition::VECCALC> InfTensorBuff;
    TBuf<TPosition::VECCALC> NanTensorBuff;
    TBuf<TPosition::VECCALC> MaskTensorBuff;
    TBuf<TPosition::VECCALC> EpsilonTensorBuff;

    // Type Conversion Buffers
    TBuf<TPosition::VECCALC> x1TensorFP32Buff;
    TBuf<TPosition::VECCALC> x2TensorFP32Buff;

    // Int32 High Precision Buffers
    TBuf<TPosition::VECCALC> FP32MaxValidBuff;
    TBuf<TPosition::VECCALC> INT32MaxValidBuff;
    TBuf<TPosition::VECCALC> SplitQuotInt32Buff;
    TBuf<TPosition::VECCALC> SplitRemInt32Buff;

    // Local Tensors (Members to hold handles across functions)
    LocalTensor<float> ResQuotTensor;
    LocalTensor<float> ResRemTensor;
    LocalTensor<float> ZeroTensor;
    LocalTensor<float> InfTensor;
    LocalTensor<float> NanTensor;
    LocalTensor<uint8_t> MaskTensor;
    LocalTensor<float> EpsilonTensor;

    LocalTensor<float> x1TensorFP32Tensor;
    LocalTensor<float> x2TensorFP32Tensor;

    // Int32 Specific Tensors
    LocalTensor<float> FP32MaxValidTensor;
    LocalTensor<int32_t> INT32MaxValidTensor;
    LocalTensor<int32_t> SplitQuotInt32Tensor;
    LocalTensor<int32_t> SplitRemInt32Tensor;

    GlobalTensor<ST> inputx1GM;
    GlobalTensor<OT> inputx2GM;
    GlobalTensor<T> outputGM;

    // Tiling Parameters
    uint32_t coreNum = 0;
    uint64_t tailCoreNum = 0;
    uint64_t perCoreDataCount = 0;
    uint64_t blockOffset = 0;
    uint32_t blockIdx = 0;
    uint32_t maxDataCount = 0;
    uint32_t actualMaxDataCount = 0;
    uint32_t usableUbSize = 0;
    bool isInput2Scalar = false;
    bool isInput2SameShape = false;
    uint32_t dimNum = 0;
    uint64_t input1Shape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint64_t input2Shape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint64_t input2Stride[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // K1 自适应路由阈值 (host 下发, 默认 256)。tile max|a| < naiveThresh_ -> naive, >= -> RemainderAlgoA。
    float naiveThresh_ = 256.0f;
    // per-core 路由 (V5.1 实证收口)：0=per-tile 现状 (RemainderAdaptive 逐 tile 探针) / 1=整核 naive /
    //   2=整核 AlgoA (理论死路，见 mod_algoa_impl.h RemainderRouted 注释)。由 PreScanCoreRoute 每核一次
    //   置位；V5.1 起仅 int16 置 1 (K2 锁 naive，纯赚)，fp lanes 恒 0 —— fp 的 per-core 预扫已被真机
    //   A/B 证伪净亏 (预扫 chunk 成本 ≳ 探针本身 + 白付一遍全量 GM 读；老套件 52 例/评委 26 例回退
    //   在案)，扫描循环已删除。int32/非 arch22 恒 0 -> 行为与现状逐位一致。
    uint32_t coreRoute_ = 0;

    // Path B 融合广播 tiling 参数 (host 下发; 非融合/非 arch22 恒 0)。arch22-only 消费。
    uint32_t bcastFusedMode_ = 0; // 0=off, 1=OUTER 行广播, 2=INNER 列广播
    uint64_t bcOuter_ = 0;
    uint64_t bcInner_ = 0;
    uint64_t bcUbFormer_ = 0;
    uint64_t bcBlockFactor_ = 0;
    // 0811 tile 塌陷修复：padding 行步长 ceil(inner*sizeof(dtype)/32)*32/sizeof(dtype) (UB 行首恒 32B 对齐
    //   -> 任意 inner 可融合, int16 same-dtype 入列)。bcIpad_==bcInner_ 时退化原 1D 平铺。pad 车道：x1 恒 0
    //   (ProcessFusedBcast 每核 priming 两个 ping-pong 槽), x2 恒 1.0 (OUTER 建块前/INNER 每 tile 预填) ->
    //   商=0 良性, 结果被 2D CopyOut 丢弃。
    uint64_t bcIpad_ = 0;
    uint64_t coreRowBase_ = 0; // 本核 OUTER 行范围 [coreRowBase_, coreRowBase_+coreRows_)
    uint64_t coreRows_ = 0;

    // Constants
    int32_t ShiftParam = 24;
    float infValue = std::numeric_limits<float>::infinity();
    float nanValue = std::numeric_limits<float>::quiet_NaN();
    float Zero = float(0);
    float Epsilon = 1e-20f;
    float FP32MaxValid = 16777216.0f; // 2^24
    int32_t INT32MaxValid = 16777216; // 2^24
};

// AlgoA/Dekker 数学辅助定义 (类内声明见上) 拆出到 mod_algoa_impl.h (声明/定义拆分，无逻辑改动)。
#include "mod_algoa_impl.h"

template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::InitBuffers()
{
    // arch22 上 CONTIGUOUS 派发 (isInput2Scalar || isInput2SameShape) 用 flat ping-pong buffer
    // (InitFlatBuffers)；general-broadcast 派发全平台保留 TQue。每核只分配其中一套 (不重复占用 UB)；
    // isInput2Scalar / isInput2SameShape 此时已有效 (ParseTilingData 在 InitBuffers 前执行)。
#if MOD_ENH_ARCH22
    if (bcastFusedMode_ != 0) {
        // 融合广播：自带整套精简 buffer (self/out 双缓 + 原始 other 小队列 + otherF32 常驻块 + 6 个 fp32
        // 工作块 + cast scratch)。直接 return，不落入下方通用 tmp/ResQuot/A1..A5/int32 分配路 (避免双分配、
        // UB 精简到 perElem ~44)。InitConstants 对融合路亦跳过 (精简核不用常量)。
        InitFusedBcastBuffers();
        return;
    } else if (isInput2Scalar || isInput2SameShape) {
        if constexpr (USE_LEAN_CONTIG) {
            // same-dtype fp32/fp16/bf16 连续路精简核。flat 双缓 self/other/out (InitFlatBuffers) + 6 个 fp32
            //   工作块 (InitLeanWorkBuffers, RemainderAdaptive w0..w5)。不分配 tmp/Zero/Inf/Nan/Mask (精简核
            //   不用 inf/nan 收尾) -> UB 69/65 -> 48 -> early return。int32 (非 USE_LEAN_CONTIG) 走 else 分支不变。
            InitFlatBuffers();
            InitLeanWorkBuffers();
            return;
        } else {
            InitFlatBuffers();
        }
    } else {
        pipe.InitBuffer(inputx1Queue, bufferNum, actualMaxDataCount * sizeof(ST));
        pipe.InitBuffer(inputx2Queue, bufferNum, actualMaxDataCount * sizeof(OT));
        pipe.InitBuffer(outputQueue, bufferNum, actualMaxDataCount * sizeof(T));
    }
#else
    pipe.InitBuffer(inputx1Queue, bufferNum, actualMaxDataCount * sizeof(ST));
    pipe.InitBuffer(inputx2Queue, bufferNum, actualMaxDataCount * sizeof(OT));
    pipe.InitBuffer(outputQueue, bufferNum, actualMaxDataCount * sizeof(T));
#endif
    pipe.InitBuffer(tmpBuff, maxDataCount * sizeof(float));

    // 2. Non-Int32 types (Float16, Bfloat16, Float32 and Int16).
    if constexpr (!std::is_same_v<T, int>) {
        pipe.InitBuffer(ResQuotTensorBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(ResRemTensorBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(ZeroTensorBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(InfTensorBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(NanTensorBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(MaskTensorBuff, maxDataCount * sizeof(uint8_t));

        // AlgoA 的 5 个 fp32 工作块 (w1..w5) 仅 arch22 (K1 adaptive/AlgoA 路) 需要；非 arch22 朴素路
        // (Div/Trunc/Mul/Sub) 不用 -> 守卫掉省 UB。host UB_DIVIDER 全平台统一取 AlgoA 大值 -> 非 arch22 仅
        // maxDataCount 偏小 (tile 略多)，保守安全，绝不 UB 溢出。
        // K2: int16 same-dtype (USE_ALGO_A=false) 走整数域 naive，不用 A1..A5 -> 不分配 (-20 B/elem)。
        //   host UB_DIVIDER_INT16 与之锁步下调 (mod_tiling.cpp)。
#if MOD_ENH_ARCH22
        if constexpr (USE_ALGO_A) {
            pipe.InitBuffer(A1Buff, maxDataCount * sizeof(float));
            pipe.InitBuffer(A2Buff, maxDataCount * sizeof(float));
            pipe.InitBuffer(A3Buff, maxDataCount * sizeof(float));
            pipe.InitBuffer(A4Buff, maxDataCount * sizeof(float));
            pipe.InitBuffer(A5Buff, maxDataCount * sizeof(float));
        }
#endif

        // half / bf16 / int16 same-dtype paths need fp32 intermediate buffers.
        if constexpr (NEED_FP32_IO_BUF) {
            pipe.InitBuffer(x1TensorFP32Buff, maxDataCount * sizeof(float));
            pipe.InitBuffer(x2TensorFP32Buff, maxDataCount * sizeof(float));
        }
    }

    // 3. Int32
    if constexpr (std::is_same_v<T, int>) {
        pipe.InitBuffer(x1TensorFP32Buff, maxDataCount * sizeof(float));
        pipe.InitBuffer(x2TensorFP32Buff, maxDataCount * sizeof(float));

        pipe.InitBuffer(ResQuotTensorBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(ResRemTensorBuff, maxDataCount * sizeof(float));

#if defined(HIGH_PRECISION) && HIGH_PRECISION == 1
        pipe.InitBuffer(FP32MaxValidBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(INT32MaxValidBuff, maxDataCount * sizeof(int32_t));
        pipe.InitBuffer(SplitQuotInt32Buff, maxDataCount * sizeof(int32_t));
        pipe.InitBuffer(SplitRemInt32Buff, maxDataCount * sizeof(int32_t));

        pipe.InitBuffer(EpsilonTensorBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(ZeroTensorBuff, maxDataCount * sizeof(float));
        pipe.InitBuffer(MaskTensorBuff, maxDataCount * sizeof(uint8_t));
#endif
    }
}

// arch22 flat-buffer 成员定义 (InitFlatBuffers/Flat*Slot/LoadScalarOtherFlat/CopyInFlat/CopyOutFlat +
// 共享 ProcessContigPipeline<LEAN> ping-pong 循环) 拆出到 mod_flat_impl.h (同 class/namespace/守卫，
// 无逻辑改动)。ComputeCore 留在此处：它是 flat 路径与下方 TQue Compute() 路径共享的单一真值源。
#include "mod_flat_impl.h"

template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace,
                                            const ModTilingData* tilingData)
{
    inputx1GM.SetGlobalBuffer((__gm__ ST*)x1);
    inputx2GM.SetGlobalBuffer((__gm__ OT*)x2);
    outputGM.SetGlobalBuffer((__gm__ T*)y);

    ParseTilingData(tilingData);

    // -------------------------------------------------------------
    // Configure Data Tiling Strategy
    // -------------------------------------------------------------
    uint32_t targetMaxData = tilingData->usableUbSize;

    if (perCoreDataCount < targetMaxData) {
        maxDataCount = perCoreDataCount;
        bufferNum = 1;
    } else {
        maxDataCount = targetMaxData;
        bufferNum = 2;
    }

    // Align to DATA_BLOCK
    if (maxDataCount < DATA_BLOCK) {
        maxDataCount = DATA_BLOCK;
    }
    maxDataCount = (maxDataCount + DATA_BLOCK - 1) / DATA_BLOCK * DATA_BLOCK;
    actualMaxDataCount = maxDataCount;

#if MOD_ENH_ARCH22
    // Path B 融合广播: 覆盖 tile 尺寸 = [ubFormer*INNER] (host 已保证 <= UB 预算, 见 mod_tiling.cpp
    // ModTryFusedBroadcast 的实测 fit 校验)。所有 maxDataCount-尺寸 buffer (self/out/other/ResQuot/A1..A5...)
    // 随之定尺, 与 host lockstep。并算出本核 OUTER 行范围 (按 bcBlockFactor_ 切, blockIdx 直取)。
    if (bcastFusedMode_ != 0) {
        uint64_t tileElems = bcUbFormer_ * bcIpad_; // 0811: tile 按 padding 行步长定尺 (host lockstep)
        if (tileElems < DATA_BLOCK) {
            tileElems = DATA_BLOCK;
        }
        maxDataCount = static_cast<uint32_t>((tileElems + DATA_BLOCK - 1) / DATA_BLOCK * DATA_BLOCK);
        actualMaxDataCount = maxDataCount;
        bufferNum = 2;
        coreRowBase_ = bcBlockFactor_ * static_cast<uint64_t>(blockIdx);
        if (coreRowBase_ >= bcOuter_) {
            coreRows_ = 0;
        } else {
            uint64_t remain = bcOuter_ - coreRowBase_;
            coreRows_ = (remain < bcBlockFactor_) ? remain : bcBlockFactor_;
        }
    }
#endif

    // -------------------------------------------------------------
    // Initialize Buffers
    // -------------------------------------------------------------
    InitBuffers();
}

template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::ParseTilingData(const ModTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    coreNum = tilingData->needCoreNum;
    usableUbSize = tilingData->usableUbSize;
    perCoreDataCount = tilingData->perCoreDataCount;
    tailCoreNum = tilingData->tailDataCoreNum;
    isInput2Scalar = tilingData->isInput2Scalar;
    isInput2SameShape = tilingData->isInput2SameShape;
    dimNum = tilingData->dimNum;
    naiveThresh_ = (tilingData->naiveThresh > 0.0f) ? tilingData->naiveThresh : 256.0f; // K3 自适应路由阈值
    // Path B 融合广播 tiling 参数 (非融合恒 0; 非 arch22 读到亦不用)。
    bcastFusedMode_ = tilingData->bcastFusedMode;
    bcOuter_ = tilingData->bcOuter;
    bcInner_ = tilingData->bcInner;
    bcUbFormer_ = tilingData->bcUbFormer;
    bcBlockFactor_ = tilingData->bcBlockFactor;
    bcIpad_ = tilingData->bcIpad;
    for (uint32_t i = 0; i < 8; ++i) {
        input1Shape[i] = tilingData->input1Shape[i];
        input2Shape[i] = tilingData->input2Shape[i];
        input2Stride[i] = tilingData->input2Stride[i];
    }

    if (tailCoreNum == 0) {
        blockOffset = perCoreDataCount * blockIdx;
    } else {
        if ((blockIdx + 1) <= tailCoreNum) {
            perCoreDataCount += DATA_BLOCK;
            blockOffset = perCoreDataCount * blockIdx;
        } else {
            blockOffset = ((perCoreDataCount + DATA_BLOCK) * tailCoreNum) +
                          (perCoreDataCount * (blockIdx - tailCoreNum));
        }
    }

    if (blockIdx == coreNum - 1) {
        perCoreDataCount = tilingData->lastCoreDataCount;
    }
}

template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::InitConstants()
{
#if MOD_ENH_ARCH22
    // 融合广播：精简核不用 inf/nan/zero 常量 (naive/AlgoA 自然传播)，且未分配对应常量 buffer ->
    // 直接跳过，否则会 Get 未分配的 Zero/Inf/Nan/Mask buffer 而崩。
    if (bcastFusedMode_ != 0) {
        return;
    }
    // same-dtype fp32/fp16/bf16 连续路精简核同样不用 inf/nan/zero 常量，且 InitLeanWorkBuffers 未分配
    //   Zero/Inf/Nan/Mask buffer -> 连续派发时跳过 (否则 Get 未分配 buffer 而崩)。general broadcast fp
    //   (非连续) 仍走通用 ComputeFPCore -> 需要常量 -> 不跳过。int32 (非 USE_LEAN_CONTIG) 恒不跳过。
    if constexpr (USE_LEAN_CONTIG) {
        if (isInput2Scalar || isInput2SameShape) {
            return;
        }
    }
#endif
    if constexpr (std::is_same_v<T, int>) {
#if defined(HIGH_PRECISION) && HIGH_PRECISION == 1
        FP32MaxValidTensor = FP32MaxValidBuff.Get<float>();
        Duplicate(FP32MaxValidTensor, FP32MaxValid, maxDataCount);

        INT32MaxValidTensor = INT32MaxValidBuff.Get<int32_t>();
        Duplicate(INT32MaxValidTensor, INT32MaxValid, maxDataCount);

        ZeroTensor = ZeroTensorBuff.Get<float>();
        Duplicate(ZeroTensor, Zero, maxDataCount);

        EpsilonTensor = EpsilonTensorBuff.Get<float>();
        Duplicate(EpsilonTensor, Epsilon, maxDataCount);
#endif
    } else {
        ZeroTensor = ZeroTensorBuff.Get<float>();
        InfTensor = InfTensorBuff.Get<float>();
        NanTensor = NanTensorBuff.Get<float>();
        MaskTensor = MaskTensorBuff.Get<uint8_t>();

        Duplicate(ZeroTensor, Zero, maxDataCount);
        Duplicate(InfTensor, infValue, maxDataCount);
        Duplicate(NanTensor, nanValue, maxDataCount);
    }
}

template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::ProcessBroadcast(uint64_t inOffset, uint64_t outOffset)
{
    uint64_t remainingDataCount = perCoreDataCount;
    while (remainingDataCount > 0) {
        uint32_t currentCount = maxDataCount;
        if (currentCount > remainingDataCount) {
            currentCount = static_cast<uint32_t>(remainingDataCount);
        }
        bool isConstantX2 = false;
        currentCount = GetInput2ContiguousCopyCount(inOffset, currentCount, isConstantX2);
        CopyIn(inOffset, currentCount, isConstantX2);
        Compute(currentCount);
        CopyOut(outOffset, currentCount);
        inOffset += currentCount;
        outOffset += currentCount;
        remainingDataCount -= currentCount;
    }
}

template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::ProcessContiguous(uint64_t inOffset, uint64_t outOffset)
{
    uint32_t loopCount = perCoreDataCount / maxDataCount;
    uint32_t tailDataCount = perCoreDataCount % maxDataCount;
    for (uint32_t i = 0; i < loopCount; i++) {
        CopyIn(inOffset, maxDataCount);
        Compute(maxDataCount);
        CopyOut(outOffset, maxDataCount);
        inOffset += maxDataCount;
        outOffset += maxDataCount;
    }
    if (tailDataCount > 0) {
        CopyIn(inOffset, tailDataCount);
        Compute(tailDataCount);
        CopyOut(outOffset, tailDataCount);
    }
}

template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::Process()
{
    InitConstants();
#if MOD_ENH_ARCH22
    // per-core 路由置位：V5.1 起仅锁 int16 整核 naive (K2，纯赚)；fp lanes 恒 0 保持 per-tile 现状
    //   (fp 预扫真机 A/B 证伪净亏已删除)。非 arch22 编译期整块移除，行为零变化。
    PreScanCoreRoute();
#endif
    if (!isInput2Scalar && !isInput2SameShape) {
#if MOD_ENH_ARCH22
        // 资格命中 (host 置 bcastFusedMode_ != 0) -> 融合广播；否则通用 ProcessBroadcast。
        if (bcastFusedMode_ != 0) {
            ProcessFusedBcast();
        } else {
            ProcessBroadcast(blockOffset, blockOffset);
        }
#else
        ProcessBroadcast(blockOffset, blockOffset);
#endif
    } else {
#if MOD_ENH_ARCH22
        // arch22 CONTIGUOUS 派发用 flat ping-pong 路径 (InitBuffers 已为本运行时分支分配匹配的 flat buffer)。
        // same-dtype fp32/fp16/bf16 走精简连续核 (LEAN=true -> ComputeContigLean，无 inf/nan 收尾/无 tmp)；
        //   int32 (非 USE_LEAN_CONTIG) 走 LEAN=false -> ComputeCore。同一份 ProcessContigPipeline flat ping-pong
        //   骨架，编译期 LEAN 择路。
        ProcessContigPipeline<USE_LEAN_CONTIG>(blockOffset, blockOffset);
#else
        ProcessContiguous(blockOffset, blockOffset);
#endif
    }
}

// TQue 拷入/出 + 通用广播 offset 辅助 (GetInput2Offset / GetInput2ContiguousCopyCount / CopyIn / CopyOut，
// 类内声明见上) 拆出到 mod_copy_impl.h (同惯例，纯物理迁移，无语句/操作数/分支条件改动)。
#include "mod_copy_impl.h"

// ComputeInt32 为上游所有 (int32 2^24 高精度 split-multiply 路径，K1/K2 不触及)，自包含，
// 拆出到 mod_int32_impl.h (纯物理迁移，无逻辑改动)。
#include "mod_int32_impl.h"

// 每 tile 计算成员定义 (ComputeFPCore / ComputeCore / Compute，类内声明见上) 拆出到 mod_compute_impl.h
// (同惯例，纯物理迁移，无语句/操作数/RoundMode/分支条件改动)。
#include "mod_compute_impl.h"

// 融合广播成员定义 (InitFusedBcastBuffers / ProcessFusedBcast / BuildOuterOtherFused /
// CopyInFusedBcast / ComputeFusedBcast / CopyOutFusedBcast)，全 `#if MOD_ENH_ARCH22` 守卫。精简 fp32 域核
// 直接调 RemainderAdaptive (mod_algoa_impl.h) -> 放在其后。与 mod_flat_impl.h 同惯例：本文件不自带
// namespace，从 mod.h 的 `namespace ModNs { ... }` 内 #include -> 定义附着到 ModNs::Mod<T,ST,OT>。
// 仅追加、arch22 守卫、独立运行时分支 (自带整套精简 buffer) -> 对通用广播/上游路零改动。
#include "mod_bcast_impl.h"

// same-dtype fp32/fp16/bf16 连续路精简核成员定义 (InitLeanWorkBuffers / ComputeContigLean；共享 Process 循环
// = mod_flat_impl.h 的 ProcessContigPipeline<LEAN=true>)，全 `#if MOD_ENH_ARCH22` 守卫。复用 mod_flat_impl.h
// 的 flat ping-pong + mod_algoa_impl.h 的 RemainderAdaptive -> 放在其后 #include。同上惯例：本文件不自带
// namespace，从 mod.h 的 `namespace ModNs { ... }` 内 #include -> 定义附着到 ModNs::Mod<T,ST,OT>。
// 仅追加、arch22 守卫、独立于 int32/int16 派发 (USE_LEAN_CONTIG 编译期择路) -> 对上游/非目标 lane 零改动。
#include "mod_leancontig_impl.h"

// Five same-dtype SEL lanes dispatch from mod_dispatch_impl.h.
#include "mod_dispatch_impl.h"

} // namespace ModNs
#endif // MOD_H
