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
 * \file top_k_small_size_bitonic_sort.h
 * \brief BITONIC-compatible candidate finalize helpers for sorted TopK with 2 <= k <= 32.
 *
 * 文件整体架构概述：
 * 本文件实现了 TopK 算子在 k 较小 (2 <= k <= 32) 时的"收尾排序"逻辑。
 * 双调排序网络 (Bitonic Sort Network) 规模固定为 32，利用 SIMD 向量寄存器或
 * SIMT warp 级通信完成并行 compare-swap，将候选元素排成有序序列。
 *
 * 两条实现路线：
 *   1. Reg-based (SIMD) 路线：基于 Reg:: 向量寄存器 API，直接操作 UB 内存，
 *      针对不同位宽 (B16/B32/B64) 做特化优化，性能最优。
 *   2. SIMT (warp 级) 路线：基于 asc_shfl_xor/asc_ballot/__popc 标量线程模型，
 *      作为 1 字节类型或非 uint32 索引类型的回退路径。
 *
 * sizeof(T) == 1U (int8/uint8) 不能走 Reg 路径的原因：
 *   arch35 的 SIMD Reg:: API (Gather/Compare/Select/LoadAlign 等) 最小操作位宽为
 *   16 位，不支持 8 位 RegTensor。因此 1 字节类型只能回退到 SIMT 标量线程路径，
 *   用 asc_shfl_xor 做 warp 内通信替代 Reg::Gather。这是硬件 ISA 的功能限制，
 *   而非性能选择。
 */

#ifndef TOP_K_SMALL_SIZE_BITONIC_SORT_H
#define TOP_K_SMALL_SIZE_BITONIC_SORT_H

#include <type_traits>

#include "kernel_operator.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/math_functions.h"
#include "top_k_constant_var_simd.h"
#include "top_k_util_type_simd.h"

namespace topkV2 {
using namespace AscendC;

constexpr uint32_t BITONIC_SMALL_TOPK_MAX_ROWS = 32U;
constexpr uint32_t BITONIC_SMALL_TOPK_THREADS = BITONIC_SMALL_TOPK_SIZE * BITONIC_SMALL_TOPK_MAX_ROWS;

template <typename T>
using BitonicSmallGatherIndexType = std::conditional_t<sizeof(T) == 1, uint8_t,
                                                       std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>>;

template <typename T>
using BitonicSmallGatherSignedIndexType = std::conditional_t<sizeof(T) == 1, int8_t,
                                                             std::conditional_t<sizeof(T) == 2, int16_t, int32_t>>;

template <typename T>
using BitonicSmallRegType = std::conditional_t<
    std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>, int32_t,
    std::conditional_t<std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t>,
                       uint32_t, T>>;

template <typename T>
constexpr bool IsBitonicFloatType = std::is_same_v<T, float> || std::is_same_v<T, half> ||
                                    std::is_same_v<T, bfloat16_t>;

constexpr Reg::CastTrait BITONIC_SMALL_CAST_U16_TO_U32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr Reg::CastTrait BITONIC_SMALL_CAST_U16_TO_U32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr Reg::CastTrait BITONIC_SMALL_CAST_U32_TO_U16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

/*!
 * \brief 将 16 位数据从 UB 加载到 32 位向量寄存器，并把位模式展开为 uint32。
 *
 * 作用：16 位数据 (half/bf16/int16/uint16) 无法直接用 uint32 寄存器比较，
 * 需要把每个 16 位元素的位模式提取到独立的 32 位寄存器通道中，后续才能统一
 * 用 uint32 整数比较做排序。
 *
 * 步骤：
 *   1. 将 16 位数据按 BLOCK 模式加载到 rawValue 寄存器（每通道含 2 个 16 位元素）。
 *   2. Cast EVEN：提取偶数位 16 位元素，零扩展为 uint32。
 *   3. Cast ODD：提取奇数位 16 位元素，零扩展为 uint32。
 *   4. Interleave：将 even/odd 交错排列，使每个通道恰好含一个 16 位元素的位模式。
 *
 * \param[out] valueBits  输出的 32 位位模式寄存器，每通道一个 16 位元素的 bits
 * \param[in]  valueAddr  UB 中 16 位数据的起始地址
 * \param[in]  validCount 有效元素个数 (<= 32)
 */
template <typename T>
__simd_callee__ inline void BitonicSmallRegLoadB16Bits(Reg::RegTensor<uint32_t>& valueBits, __ubuf__ T* valueAddr,
                                                       uint32_t validCount)
{
    uint32_t valueCount = validCount;
    Reg::MaskReg valueMask = Reg::UpdateMask<T>(valueCount);
    Reg::RegTensor<uint16_t> rawValue;
    Reg::RegTensor<uint32_t> evenValue;
    Reg::RegTensor<uint32_t> oddValue;
    Reg::RegTensor<uint32_t> unusedValue;
    Reg::LoadAlign<T, Reg::DataCopyMode::DATA_BLOCK_COPY>((Reg::RegTensor<T>&)rawValue, valueAddr, 1U, valueMask);
    Reg::Cast<uint32_t, uint16_t, BITONIC_SMALL_CAST_U16_TO_U32_EVEN>(evenValue, rawValue, valueMask);
    Reg::Cast<uint32_t, uint16_t, BITONIC_SMALL_CAST_U16_TO_U32_ODD>(oddValue, rawValue, valueMask);
    Reg::Interleave<uint32_t>(valueBits, unusedValue, evenValue, oddValue);
}

/*!
 * \brief 将 32 位位模式寄存器还原为 16 位数据并存回 UB。
 *
 * 这是 BitonicSmallRegLoadB16Bits 的逆操作：把每通道一个 16 位元素位模式的
 * uint32 寄存器，重新打包为 16 位连续存储后写回 UB。
 *
 * 步骤：
 *   1. Cast U32→U16：将 32 位位模式截断回 16 位。
 *   2. Pack：将两个相邻的 16 位结果打包到一个 32 位寄存器通道中。
 *   3. StoreAlign：按 BLOCK 模式写回 UB。
 *
 * \param[out] valueAddr  UB 目标地址
 * \param[in]  valueBits  排序后的 32 位位模式寄存器
 * \param[in]  validCount 有效元素个数
 * \param[in]  validMask  有效通道掩码
 */
template <typename T>
__simd_callee__ inline void BitonicSmallRegStoreB16Bits(__ubuf__ T* valueAddr, Reg::RegTensor<uint32_t>& valueBits,
                                                        uint32_t validCount, Reg::MaskReg& validMask)
{
    uint32_t valueCount = validCount;
    Reg::MaskReg valueMask = Reg::UpdateMask<T>(valueCount);
    Reg::RegTensor<uint16_t> rawValue;
    Reg::Cast<uint16_t, uint32_t, BITONIC_SMALL_CAST_U32_TO_U16>(rawValue, valueBits, validMask);
    Reg::Pack(rawValue, (Reg::RegTensor<uint32_t>&)rawValue);
    Reg::StoreAlign<T, Reg::DataCopyMode::DATA_BLOCK_COPY>(valueAddr, (Reg::RegTensor<T>&)rawValue, 1U, valueMask);
}

/*!
 * \brief 将任意类型数据的位模式转换为可单调比较的 32 位无符号整数 key。
 *
 * 这是用整数比较替代浮点/有符号数比较的核心技术。将各种类型的数据位模式映射
 * 为 uint32，使得"整数大小比较"等价于"原始数据的大小比较"，从而在双调网络中
 * 统一用 uint32 比较指令完成排序。
 *
 * 映射规则：
 *   - 无符号整数：key = rawBits (天然单调)
 *   - 有符号整数：key = rawBits XOR signBit (翻转符号位，使负数排在正数前)
 *   - 浮点数 (half/bf16/float)：
 *     正数：key = rawBits XOR signBit (仅翻转符号位)
 *     负数：key = rawBits XOR allBits (全位取反，使绝对值大的负数排更后)
 *     这样浮点位模式变为单调整数序。
 *   - 求 TopK 最小值 (!IsLargest)：key 再整体 XOR allBits 取反，反转排序方向。
 *
 * 浮点 NaN 特殊处理：
 *   NaN 的指数全 1 且尾数非 0。IsLargest 时 NaN key 设为 allBits (排在最后)，
 *   否则设为 1 (排在最前)，保证 NaN 总在结果末尾。
 *   浮点 -0 会被清零，与 +0 统一处理。
 *
 * \param[out] key        生成的 32 位排序 key
 * \param[in]  valueBits  原始数据的 32 位位模式
 * \param[in]  activeMask 活跃通道掩码
 */
template <typename T, bool IsLargest>
__simd_callee__ inline void BitonicSmallRegBuildKey32(Reg::RegTensor<uint32_t>& key,
                                                      Reg::RegTensor<uint32_t>& valueBits, Reg::MaskReg& activeMask)
{
    constexpr uint32_t bitWidth = sizeof(T) * 8U;
    constexpr uint32_t signBit = 1U << (bitWidth - 1U);
    constexpr uint32_t allBits = sizeof(T) == sizeof(uint32_t) ? UINT32_MAX : ((1U << bitWidth) - 1U);
    Reg::RegTensor<uint32_t> rawBits;
    Reg::RegTensor<uint32_t> allBitsReg;
    Reg::RegTensor<uint32_t> xorMask;
    Reg::RegTensor<uint32_t> signMask;
    Reg::Duplicate(allBitsReg, allBits);
    Reg::And(rawBits, valueBits, allBitsReg, activeMask);
    if constexpr (IsBitonicFloatType<T>) {
        constexpr uint32_t exponentMask = std::is_same_v<T, half> ?
                                              0x7c00U :
                                              (std::is_same_v<T, bfloat16_t> ? 0x7f80U : 0x7f800000U);
        constexpr uint32_t fractionMask = std::is_same_v<T, half> ?
                                              0x03ffU :
                                              (std::is_same_v<T, bfloat16_t> ? 0x007fU : 0x007fffffU);
        Reg::RegTensor<uint32_t> absoluteBits;
        Reg::RegTensor<uint32_t> exponentBits;
        Reg::RegTensor<uint32_t> fractionBits;
        Reg::RegTensor<uint32_t> positiveXor;
        Reg::RegTensor<uint32_t> nanKey;
        Reg::RegTensor<uint32_t> zeroReg;
        Reg::MaskReg zeroMask;
        Reg::MaskReg negativeMask;
        Reg::MaskReg exponentMaskReg;
        Reg::MaskReg fractionMaskReg;
        Reg::MaskReg nanMask;
        Reg::Duplicate(zeroReg, 0U);
        Reg::Duplicate(signMask, signBit);
        Reg::And(absoluteBits, rawBits, allBitsReg, activeMask);
        Reg::And(exponentBits, rawBits, signMask, activeMask);
        Reg::Xor(absoluteBits, rawBits, exponentBits, activeMask);
        Reg::Compares<uint32_t, CMPMODE::EQ>(zeroMask, absoluteBits, 0U, activeMask);
        Reg::Select<uint32_t>(rawBits, zeroReg, rawBits, zeroMask);
        Reg::And(exponentBits, rawBits, signMask, activeMask);
        Reg::Compares<uint32_t, CMPMODE::NE>(negativeMask, exponentBits, 0U, activeMask);
        Reg::Duplicate(positiveXor, signBit);
        Reg::Select<uint32_t>(xorMask, allBitsReg, positiveXor, negativeMask);
        Reg::Xor(key, rawBits, xorMask, activeMask);
        if constexpr (!IsLargest) {
            Reg::Xor(key, key, allBitsReg, activeMask);
        }

        Reg::Duplicate(signMask, exponentMask);
        Reg::And(exponentBits, absoluteBits, signMask, activeMask);
        Reg::Duplicate(signMask, fractionMask);
        Reg::And(fractionBits, absoluteBits, signMask, activeMask);
        Reg::Compares<uint32_t, CMPMODE::EQ>(exponentMaskReg, exponentBits, exponentMask, activeMask);
        Reg::Compares<uint32_t, CMPMODE::NE>(fractionMaskReg, fractionBits, 0U, activeMask);
        Reg::And(nanMask, exponentMaskReg, fractionMaskReg, activeMask);
        Reg::Duplicate(nanKey, IsLargest ? allBits : 1U);
        Reg::Select<uint32_t>(key, nanKey, key, nanMask);
    } else {
        Reg::Duplicate(xorMask, std::is_signed_v<T> ? signBit : 0U);
        Reg::Xor(key, rawBits, xorMask, activeMask);
        if constexpr (!IsLargest) {
            Reg::Xor(key, key, allBitsReg, activeMask);
        }
    }
}

/*!
 * \brief 32 位双调网络的单次 compare-swap 阶段 (Reg/SIMD 路径)。
 *
 * 对 32 通道的向量寄存器，按给定 Stride 计算 peer lane (lane XOR stride)，
 * 通过 Reg::Gather 获取对端数据，将本地和对端分为 low/high 两部分，
 * 再根据比较结果和排序方向决定是否交换。
 *
 * 模板参数：
 *   - CompareGroup：true 时按 (group, index) 字典序比较（用于按分组排序）；
 *                   false 时直接按 key 大小比较（用于最终值排序）。
 *   - Stride：当前阶段的步长，决定 peer lane 的计算方式。
 *   - Size：当前双调子网的规模 (2/4/8/16/32)，影响排序方向的判定。
 *
 * 交换逻辑：
 *   swapMask = (compare && lowValid) || !highValid
 *     - 低位比高位"差"且低位有效 → 交换
 *     - 高位无效 → 强制交换（把无效项挤到低位）
 *   takePeerMask = !(swapMask XOR directionMask)
 *     - 决定本通道是否取对端值
 */
template <bool CompareGroup, uint32_t Stride, uint32_t Size>
__simd_callee__ inline void BitonicSmallReg32SwapStage(Reg::RegTensor<uint32_t>& valueBits,
                                                       Reg::RegTensor<uint32_t>& key, Reg::RegTensor<uint32_t>& index,
                                                       Reg::RegTensor<uint32_t>& group, Reg::RegTensor<uint32_t>& lane,
                                                       Reg::MaskReg& activeMask)
{
    Reg::RegTensor<uint32_t> strideReg;
    Reg::RegTensor<uint32_t> peerLane;
    Reg::Duplicate(strideReg, Stride);
    Reg::Xor(peerLane, lane, strideReg, activeMask);
    Reg::RegTensor<uint32_t> peerValueBits;
    Reg::RegTensor<uint32_t> peerKey;
    Reg::RegTensor<uint32_t> peerIndex;
    Reg::RegTensor<uint32_t> peerGroup;
    Reg::Gather(peerValueBits, valueBits, peerLane);
    Reg::Gather(peerKey, key, peerLane);
    Reg::Gather(peerIndex, index, peerLane);
    if constexpr (CompareGroup) {
        Reg::Gather(peerGroup, group, peerLane);
    }
    Reg::MaskReg lowLaneMask;
    Reg::Compare<uint32_t, CMPMODE::LT>(lowLaneMask, lane, peerLane, activeMask);
    Reg::RegTensor<uint32_t> lowKey;
    Reg::RegTensor<uint32_t> highKey;
    Reg::RegTensor<uint32_t> lowIndex;
    Reg::RegTensor<uint32_t> highIndex;
    Reg::Select<uint32_t>(lowKey, key, peerKey, lowLaneMask);
    Reg::Select<uint32_t>(highKey, peerKey, key, lowLaneMask);
    Reg::Select<uint32_t>(lowIndex, index, peerIndex, lowLaneMask);
    Reg::Select<uint32_t>(highIndex, peerIndex, index, lowLaneMask);
    Reg::MaskReg compareMask;
    if constexpr (CompareGroup) {
        Reg::RegTensor<uint32_t> lowGroup;
        Reg::RegTensor<uint32_t> highGroup;
        Reg::MaskReg groupEqualMask;
        Reg::MaskReg indexLessMask;
        Reg::Select<uint32_t>(lowGroup, group, peerGroup, lowLaneMask);
        Reg::Select<uint32_t>(highGroup, peerGroup, group, lowLaneMask);
        Reg::Compare<uint32_t, CMPMODE::LT>(compareMask, lowGroup, highGroup, activeMask);
        Reg::Compare<uint32_t, CMPMODE::EQ>(groupEqualMask, lowGroup, highGroup, activeMask);
        Reg::Compare<uint32_t, CMPMODE::LT>(indexLessMask, lowIndex, highIndex, activeMask);
        Reg::And(groupEqualMask, groupEqualMask, indexLessMask, activeMask);
        Reg::Or(compareMask, compareMask, groupEqualMask, activeMask);
    } else {
        Reg::Compare<uint32_t, CMPMODE::GT>(compareMask, lowKey, highKey, activeMask);
    }
    Reg::MaskReg lowValidMask;
    Reg::MaskReg highValidMask;
    Reg::MaskReg highInvalidMask;
    Reg::MaskReg swapMask;
    Reg::Compares<uint32_t, CMPMODE::NE>(lowValidMask, lowIndex, UINT32_MAX, activeMask);
    Reg::Compares<uint32_t, CMPMODE::NE>(highValidMask, highIndex, UINT32_MAX, activeMask);
    Reg::And(swapMask, compareMask, lowValidMask, activeMask);
    Reg::Not(highInvalidMask, highValidMask, activeMask);
    Reg::Or(swapMask, swapMask, highInvalidMask, activeMask);
    Reg::MaskReg directionMask;
    if constexpr (Size == BITONIC_SMALL_TOPK_SIZE) {
        Reg::Compares<uint32_t, CMPMODE::LT>(directionMask, lane, 0U, activeMask);
    } else {
        Reg::Duplicate(strideReg, Size);
        Reg::And(peerLane, lane, strideReg, activeMask);
        Reg::Compares<uint32_t, CMPMODE::NE>(directionMask, peerLane, 0U, activeMask);
    }
    Reg::MaskReg takePeerMask;
    Reg::Xor(takePeerMask, swapMask, directionMask, activeMask);
    Reg::Not(takePeerMask, takePeerMask, activeMask);
    Reg::Select<uint32_t>(valueBits, peerValueBits, valueBits, takePeerMask);
    Reg::Select<uint32_t>(key, peerKey, key, takePeerMask);
    Reg::Select<uint32_t>(index, peerIndex, index, takePeerMask);
    if constexpr (CompareGroup) {
        Reg::Select<uint32_t>(group, peerGroup, group, takePeerMask);
    }
}

/*!
 * \brief 32 位双调排序网络完整序列 (Reg/SIMD 路径)。
 *
 * 标准双调排序网络对 32 个元素展开 15 个 SwapStage，按 (Stride, Size) 序列：
 *   size=2:  (1,2)
 *   size=4:  (2,4) (1,4)
 *   size=8:  (4,8) (2,8) (1,8)
 *   size=16: (8,16) (4,16) (2,16) (1,16)
 *   size=32: (16,32) (8,32) (4,32) (2,32) (1,32)
 * 先构建双调序列（size 递增），再做双调合并（stride 递减），最终全排序。
 */
template <bool CompareGroup>
__simd_callee__ inline void BitonicSmallReg32BitonicNetwork(Reg::RegTensor<uint32_t>& valueBits,
                                                            Reg::RegTensor<uint32_t>& key,
                                                            Reg::RegTensor<uint32_t>& index,
                                                            Reg::RegTensor<uint32_t>& group,
                                                            Reg::RegTensor<uint32_t>& lane, Reg::MaskReg& activeMask)
{
    BitonicSmallReg32SwapStage<CompareGroup, 1U, 2U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 2U, 4U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 1U, 4U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 4U, 8U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 2U, 8U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 1U, 8U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 8U, 16U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 4U, 16U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 2U, 16U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 1U, 16U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 16U, 32U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 8U, 32U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 4U, 32U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 2U, 32U>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32SwapStage<CompareGroup, 1U, 32U>(valueBits, key, index, group, lane, activeMask);
}

/*!
 * \brief 16 位类型的收尾选择函数 (Reg/SIMD 路径)。
 *
 * 将 UB 中的 k 个候选元素排序后写回。流程：
 *   1. 加载 value (转 32 位 bits) 和 index，越界 index 填 UINT32_MAX 标记无效。
 *   2. 构建 32 位排序 key。
 *   3. 取第 k 个元素的 key 作为阈值，分三组：
 *      - strict (key > 阈值) → group=0
 *      - 其余 valid → group=1
 *      - 无效 → group=2
 *   4. 先按 (group, index) 排序 (BitonicNetwork<true>)，再按 key 全排序 (BitonicNetwork<false>)。
 *   5. 将排序结果存回 UB。
 */
template <typename T, bool IsLargest>
__simd_callee__ inline void BitonicSmallRegFinalizeSelectionB16(__ubuf__ T* valueAddr, __ubuf__ uint32_t* indexAddr,
                                                                uint32_t k)
{
    uint32_t activeCount = BITONIC_SMALL_TOPK_SIZE;
    uint32_t validCount = k;
    Reg::MaskReg activeMask = Reg::UpdateMask<uint32_t>(activeCount);
    Reg::MaskReg validMask = Reg::UpdateMask<uint32_t>(validCount);
    Reg::RegTensor<uint32_t> valueBits;
    Reg::RegTensor<uint32_t> key;
    Reg::RegTensor<uint32_t> index;
    Reg::RegTensor<uint32_t> invalidIndex;
    Reg::RegTensor<uint32_t> lane;
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    BitonicSmallRegLoadB16Bits<T>(valueBits, valueAddr, k);
    Reg::RegTensor<uint32_t> zeroValueBits;
    Reg::Duplicate(zeroValueBits, 0U);
    Reg::Select<uint32_t>(valueBits, valueBits, zeroValueBits, validMask);
    Reg::LoadAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(index, indexAddr, 1U, validMask);
    Reg::Duplicate(invalidIndex, UINT32_MAX);
    Reg::Select<uint32_t>(index, index, invalidIndex, validMask);
    Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
    BitonicSmallRegBuildKey32<T, IsLargest>(key, valueBits, activeMask);
    Reg::RegTensor<uint32_t> thresholdIndex;
    Reg::RegTensor<uint32_t> thresholdKey;
    Reg::Duplicate(thresholdIndex, k - 1U);
    Reg::Gather(thresholdKey, key, thresholdIndex);
    Reg::MaskReg strictMask;
    Reg::Compare<uint32_t, CMPMODE::GT>(strictMask, key, thresholdKey, activeMask);
    Reg::And(strictMask, strictMask, validMask, activeMask);
    Reg::RegTensor<uint32_t> group;
    Reg::RegTensor<uint32_t> strictGroup;
    Reg::RegTensor<uint32_t> invalidGroup;
    Reg::Duplicate(group, 1U);
    Reg::Duplicate(strictGroup, 0U);
    Reg::Duplicate(invalidGroup, 2U);
    Reg::Select<uint32_t>(group, strictGroup, group, strictMask);
    Reg::Select<uint32_t>(group, group, invalidGroup, validMask);
    BitonicSmallReg32BitonicNetwork<true>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallReg32BitonicNetwork<false>(valueBits, key, index, group, lane, activeMask);
    BitonicSmallRegStoreB16Bits<T>(valueAddr, valueBits, k, validMask);
    Reg::StoreAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(indexAddr, index, 1U, validMask);
}

/*!
 * \brief 64 位双调网络的单次 compare-swap 阶段 (Reg/SIMD 路径)。
 *
 * 64 位类型无法放入单个 32 位寄存器，需拆成 valueLow/valueHigh 两半处理。
 * 每次 Gather/Select 都要处理 low/high 两个寄存器。
 * 比较逻辑：先比 keyHigh（高位），相等再比 keyLow（低位），即 64 位比较的拆分实现。
 * 其余 swap 逻辑与 32 位版本 (BitonicSmallReg32SwapStage) 一致。
 */
template <bool CompareGroup, uint32_t Stride, uint32_t Size>
__simd_callee__ inline void BitonicSmallReg64SwapStage(Reg::RegTensor<uint32_t>& valueLow,
                                                       Reg::RegTensor<uint32_t>& valueHigh,
                                                       Reg::RegTensor<uint32_t>& keyLow,
                                                       Reg::RegTensor<uint32_t>& keyHigh,
                                                       Reg::RegTensor<uint32_t>& index, Reg::RegTensor<uint32_t>& group,
                                                       Reg::RegTensor<uint32_t>& lane, Reg::MaskReg& activeMask)
{
    Reg::RegTensor<uint32_t> strideReg;
    Reg::RegTensor<uint32_t> peerLane;
    Reg::Duplicate(strideReg, Stride);
    Reg::Xor(peerLane, lane, strideReg, activeMask);
    Reg::RegTensor<uint32_t> peerValueLow;
    Reg::RegTensor<uint32_t> peerValueHigh;
    Reg::RegTensor<uint32_t> peerKeyLow;
    Reg::RegTensor<uint32_t> peerKeyHigh;
    Reg::RegTensor<uint32_t> peerIndex;
    Reg::RegTensor<uint32_t> peerGroup;
    Reg::Gather(peerValueLow, valueLow, peerLane);
    Reg::Gather(peerValueHigh, valueHigh, peerLane);
    Reg::Gather(peerKeyLow, keyLow, peerLane);
    Reg::Gather(peerKeyHigh, keyHigh, peerLane);
    Reg::Gather(peerIndex, index, peerLane);
    if constexpr (CompareGroup) {
        Reg::Gather(peerGroup, group, peerLane);
    }
    Reg::MaskReg lowLaneMask;
    Reg::Compare<uint32_t, CMPMODE::LT>(lowLaneMask, lane, peerLane, activeMask);
    Reg::RegTensor<uint32_t> lowKeyLow;
    Reg::RegTensor<uint32_t> lowKeyHigh;
    Reg::RegTensor<uint32_t> highKeyLow;
    Reg::RegTensor<uint32_t> highKeyHigh;
    Reg::RegTensor<uint32_t> lowIndex;
    Reg::RegTensor<uint32_t> highIndex;
    Reg::Select<uint32_t>(lowKeyLow, keyLow, peerKeyLow, lowLaneMask);
    Reg::Select<uint32_t>(lowKeyHigh, keyHigh, peerKeyHigh, lowLaneMask);
    Reg::Select<uint32_t>(highKeyLow, peerKeyLow, keyLow, lowLaneMask);
    Reg::Select<uint32_t>(highKeyHigh, peerKeyHigh, keyHigh, lowLaneMask);
    Reg::Select<uint32_t>(lowIndex, index, peerIndex, lowLaneMask);
    Reg::Select<uint32_t>(highIndex, peerIndex, index, lowLaneMask);
    Reg::MaskReg compareMask;
    if constexpr (CompareGroup) {
        Reg::RegTensor<uint32_t> lowGroup;
        Reg::RegTensor<uint32_t> highGroup;
        Reg::MaskReg groupEqualMask;
        Reg::MaskReg indexLessMask;
        Reg::Select<uint32_t>(lowGroup, group, peerGroup, lowLaneMask);
        Reg::Select<uint32_t>(highGroup, peerGroup, group, lowLaneMask);
        Reg::Compare<uint32_t, CMPMODE::LT>(compareMask, lowGroup, highGroup, activeMask);
        Reg::Compare<uint32_t, CMPMODE::EQ>(groupEqualMask, lowGroup, highGroup, activeMask);
        Reg::Compare<uint32_t, CMPMODE::LT>(indexLessMask, lowIndex, highIndex, activeMask);
        Reg::And(groupEqualMask, groupEqualMask, indexLessMask, activeMask);
        Reg::Or(compareMask, compareMask, groupEqualMask, activeMask);
    } else {
        Reg::MaskReg highEqualMask;
        Reg::MaskReg lowGreaterMask;
        Reg::Compare<uint32_t, CMPMODE::GT>(compareMask, lowKeyHigh, highKeyHigh, activeMask);
        Reg::Compare<uint32_t, CMPMODE::EQ>(highEqualMask, lowKeyHigh, highKeyHigh, activeMask);
        Reg::Compare<uint32_t, CMPMODE::GT>(lowGreaterMask, lowKeyLow, highKeyLow, activeMask);
        Reg::And(highEqualMask, highEqualMask, lowGreaterMask, activeMask);
        Reg::Or(compareMask, compareMask, highEqualMask, activeMask);
    }
    Reg::MaskReg lowValidMask;
    Reg::MaskReg highValidMask;
    Reg::MaskReg highInvalidMask;
    Reg::MaskReg swapMask;
    Reg::Compares<uint32_t, CMPMODE::NE>(lowValidMask, lowIndex, UINT32_MAX, activeMask);
    Reg::Compares<uint32_t, CMPMODE::NE>(highValidMask, highIndex, UINT32_MAX, activeMask);
    Reg::And(swapMask, compareMask, lowValidMask, activeMask);
    Reg::Not(highInvalidMask, highValidMask, activeMask);
    Reg::Or(swapMask, swapMask, highInvalidMask, activeMask);
    Reg::MaskReg directionMask;
    if constexpr (Size == BITONIC_SMALL_TOPK_SIZE) {
        Reg::Compares<uint32_t, CMPMODE::LT>(directionMask, lane, 0U, activeMask);
    } else {
        Reg::Duplicate(strideReg, Size);
        Reg::And(peerLane, lane, strideReg, activeMask);
        Reg::Compares<uint32_t, CMPMODE::NE>(directionMask, peerLane, 0U, activeMask);
    }
    Reg::MaskReg takePeerMask;
    Reg::Xor(takePeerMask, swapMask, directionMask, activeMask);
    Reg::Not(takePeerMask, takePeerMask, activeMask);
    Reg::Select<uint32_t>(valueLow, peerValueLow, valueLow, takePeerMask);
    Reg::Select<uint32_t>(valueHigh, peerValueHigh, valueHigh, takePeerMask);
    Reg::Select<uint32_t>(keyLow, peerKeyLow, keyLow, takePeerMask);
    Reg::Select<uint32_t>(keyHigh, peerKeyHigh, keyHigh, takePeerMask);
    Reg::Select<uint32_t>(index, peerIndex, index, takePeerMask);
    if constexpr (CompareGroup) {
        Reg::Select<uint32_t>(group, peerGroup, group, takePeerMask);
    }
}

/*!
 * \brief 64 位双调排序网络完整序列 (Reg/SIMD 路径)。
 *
 * 与 BitonicSmallReg32BitonicNetwork 结构相同，展开 15 个 SwapStage。
 * 区别在于每个 stage 处理 valueLow/valueHigh 和 keyLow/keyHigh 双寄存器。
 */
template <bool CompareGroup>
__simd_callee__ inline void BitonicSmallReg64BitonicNetwork(
    Reg::RegTensor<uint32_t>& valueLow, Reg::RegTensor<uint32_t>& valueHigh, Reg::RegTensor<uint32_t>& keyLow,
    Reg::RegTensor<uint32_t>& keyHigh, Reg::RegTensor<uint32_t>& index, Reg::RegTensor<uint32_t>& group,
    Reg::RegTensor<uint32_t>& lane, Reg::MaskReg& activeMask)
{
    BitonicSmallReg64SwapStage<CompareGroup, 1U, 2U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                     activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 2U, 4U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                     activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 1U, 4U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                     activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 4U, 8U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                     activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 2U, 8U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                     activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 1U, 8U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                     activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 8U, 16U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                      activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 4U, 16U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                      activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 2U, 16U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                      activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 1U, 16U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                      activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 16U, 32U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                       activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 8U, 32U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                      activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 4U, 32U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                      activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 2U, 32U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                      activeMask);
    BitonicSmallReg64SwapStage<CompareGroup, 1U, 32U>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane,
                                                      activeMask);
}

/*!
 * \brief 64 位类型的收尾选择函数 (Reg/SIMD 路径)。
 *
 * 与 BitonicSmallRegFinalizeSelectionB16 流程类似，但处理 64 位整数类型。
 * key 构建：有符号则 high 异或 0x80000000，low 直接复制；求最小值则两半都取反。
 * 通过 DeInterleave 拆分交错存储的 64 位数据为 low/high 两寄存器，
 * 排序后用 Interleave 重新交错存回。
 * 阈值比较也是双寄存器：先比 keyHigh，相等再比 keyLow。
 */
template <typename T, bool IsLargest>
__simd_callee__ inline void BitonicSmallRegFinalizeSelectionB64(__ubuf__ T* valueAddr, __ubuf__ uint32_t* indexAddr,
                                                                uint32_t k)
{
    static_assert(std::is_integral_v<T> && sizeof(T) == sizeof(uint64_t));
    uint32_t activeCount = BITONIC_SMALL_TOPK_SIZE;
    uint32_t validCount = k;
    uint32_t rawCount = k * 2U;
    Reg::MaskReg activeMask = Reg::UpdateMask<uint32_t>(activeCount);
    Reg::MaskReg validMask = Reg::UpdateMask<uint32_t>(validCount);
    Reg::MaskReg rawMask = Reg::UpdateMask<uint32_t>(rawCount);
    Reg::RegTensor<uint32_t> rawValue;
    Reg::RegTensor<uint32_t> valueLow;
    Reg::RegTensor<uint32_t> valueHigh;
    Reg::RegTensor<uint32_t> keyLow;
    Reg::RegTensor<uint32_t> keyHigh;
    Reg::RegTensor<uint32_t> index;
    Reg::RegTensor<uint32_t> invalidIndex;
    Reg::RegTensor<uint32_t> lane;
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    Reg::LoadAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(rawValue, (__ubuf__ uint32_t*)valueAddr, 1U, rawMask);
    Reg::DeInterleave(valueLow, valueHigh, rawValue, rawValue);
    Reg::RegTensor<uint32_t> zeroValue;
    Reg::Duplicate(zeroValue, 0U);
    Reg::Select<uint32_t>(valueLow, valueLow, zeroValue, validMask);
    Reg::Select<uint32_t>(valueHigh, valueHigh, zeroValue, validMask);
    Reg::LoadAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(index, indexAddr, 1U, validMask);
    Reg::Duplicate(invalidIndex, UINT32_MAX);
    Reg::Select<uint32_t>(index, index, invalidIndex, validMask);
    Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
    Reg::Duplicate(keyHigh, std::is_signed_v<T> ? 0x80000000U : 0U);
    Reg::Xor(keyHigh, valueHigh, keyHigh, activeMask);
    Reg::Or(keyLow, valueLow, valueLow, activeMask);
    if constexpr (!IsLargest) {
        Reg::Duplicate(rawValue, UINT32_MAX);
        Reg::Xor(keyHigh, keyHigh, rawValue, activeMask);
        Reg::Xor(keyLow, keyLow, rawValue, activeMask);
    }

    Reg::RegTensor<uint32_t> thresholdIndex;
    Reg::RegTensor<uint32_t> thresholdLow;
    Reg::RegTensor<uint32_t> thresholdHigh;
    Reg::Duplicate(thresholdIndex, k - 1U);
    Reg::Gather(thresholdLow, keyLow, thresholdIndex);
    Reg::Gather(thresholdHigh, keyHigh, thresholdIndex);
    Reg::MaskReg highGreaterMask;
    Reg::MaskReg highEqualMask;
    Reg::MaskReg lowGreaterMask;
    Reg::MaskReg strictMask;
    Reg::Compare<uint32_t, CMPMODE::GT>(highGreaterMask, keyHigh, thresholdHigh, activeMask);
    Reg::Compare<uint32_t, CMPMODE::EQ>(highEqualMask, keyHigh, thresholdHigh, activeMask);
    Reg::Compare<uint32_t, CMPMODE::GT>(lowGreaterMask, keyLow, thresholdLow, activeMask);
    Reg::And(strictMask, highEqualMask, lowGreaterMask, activeMask);
    Reg::Or(strictMask, strictMask, highGreaterMask, activeMask);
    Reg::And(strictMask, strictMask, validMask, activeMask);

    Reg::RegTensor<uint32_t> group;
    Reg::RegTensor<uint32_t> strictGroup;
    Reg::RegTensor<uint32_t> invalidGroup;
    Reg::Duplicate(group, 1U);
    Reg::Duplicate(strictGroup, 0U);
    Reg::Duplicate(invalidGroup, 2U);
    Reg::Select<uint32_t>(group, strictGroup, group, strictMask);
    Reg::Select<uint32_t>(group, group, invalidGroup, validMask);
    BitonicSmallReg64BitonicNetwork<true>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane, activeMask);
    BitonicSmallReg64BitonicNetwork<false>(valueLow, valueHigh, keyLow, keyHigh, index, group, lane, activeMask);

    Reg::RegTensor<uint32_t> unusedValue;
    Reg::Interleave<uint32_t>(rawValue, unusedValue, valueLow, valueHigh);
    Reg::StoreAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>((__ubuf__ uint32_t*)valueAddr, rawValue, 1U, rawMask);
    Reg::StoreAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(indexAddr, index, 1U, validMask);
}

/*!
 * \brief 判断 lhs 是否"优于" rhs (Reg/SIMD 路径)。
 *
 * IsLargest=true 时求最大值，lhs > rhs 则 lhs 优于 rhs；
 * IsLargest=false 时求最小值，lhs < rhs 则 lhs 优于 rhs。
 * 浮点类型有 NaN 特殊语义：IsLargest 时 NaN 视为最大，否则最小。
 */
template <typename T, bool IsLargest>
__simd_callee__ inline void BitonicSmallRegValueBetter(Reg::MaskReg& betterMask, Reg::RegTensor<T>& lhs,
                                                       Reg::RegTensor<T>& rhs, Reg::MaskReg& activeMask)
{
    if constexpr (IsBitonicFloatType<T>) {
        Reg::MaskReg lhsNanMask;
        Reg::MaskReg rhsNanMask;
        Reg::MaskReg notNanMask;
        Reg::MaskReg strictMask;
        Reg::Compare<T, CMPMODE::NE>(lhsNanMask, lhs, lhs, activeMask);
        Reg::Compare<T, CMPMODE::NE>(rhsNanMask, rhs, rhs, activeMask);
        if constexpr (IsLargest) {
            Reg::Compare<T, CMPMODE::GT>(strictMask, lhs, rhs, activeMask);
            Reg::Not(notNanMask, rhsNanMask, activeMask);
            Reg::And(lhsNanMask, lhsNanMask, notNanMask, activeMask);
        } else {
            Reg::Compare<T, CMPMODE::LT>(strictMask, lhs, rhs, activeMask);
            Reg::Not(notNanMask, lhsNanMask, activeMask);
            Reg::And(lhsNanMask, rhsNanMask, notNanMask, activeMask);
        }
        Reg::Or(betterMask, strictMask, lhsNanMask, activeMask);
    } else if constexpr (IsLargest) {
        Reg::Compare<T, CMPMODE::GT>(betterMask, lhs, rhs, activeMask);
    } else {
        Reg::Compare<T, CMPMODE::LT>(betterMask, lhs, rhs, activeMask);
    }
}

/*!
 * \brief 判断 lhs 和 rhs 是否等价 (Reg/SIMD 路径)。
 *
 * 整数类型直接比较相等；浮点类型额外将 NaN==NaN 视为等价（通过
 * "两者都为 NaN" 的掩码与相等掩码取或实现）。
 */
template <typename T>
__simd_callee__ inline void BitonicSmallRegValueEquivalent(Reg::MaskReg& equivalentMask, Reg::RegTensor<T>& lhs,
                                                           Reg::RegTensor<T>& rhs, Reg::MaskReg& activeMask)
{
    Reg::Compare<T, CMPMODE::EQ>(equivalentMask, lhs, rhs, activeMask);
    if constexpr (IsBitonicFloatType<T>) {
        Reg::MaskReg lhsNanMask;
        Reg::MaskReg rhsNanMask;
        Reg::Compare<T, CMPMODE::NE>(lhsNanMask, lhs, lhs, activeMask);
        Reg::Compare<T, CMPMODE::NE>(rhsNanMask, rhs, rhs, activeMask);
        Reg::And(lhsNanMask, lhsNanMask, rhsNanMask, activeMask);
        Reg::Or(equivalentMask, equivalentMask, lhsNanMask, activeMask);
    }
}

/*!
 * \brief 通用类型的双调网络单次 compare-swap 阶段 (Reg/SIMD 路径)。
 *
 * 与 BitonicSmallReg32SwapStage 逻辑相同，但 value 寄存器类型为原始 T（而非统一 uint32 bits），
 * 因此 Gather 索引类型需根据 sizeof(T) 选择匹配宽度的 GatherIndexT。
 * CompareGroup=false 时用 BitonicSmallRegValueBetter 做值比较（支持浮点 NaN 语义）。
 * takePeerMask 逻辑用 NOT+AND+OR 组合实现（非交换且同向 / 交换且反方向）。
 */
template <typename T, bool IsLargest, bool CompareGroup, uint32_t Stride, uint32_t Size>
__simd_callee__ inline void BitonicSmallRegSwapStage(Reg::RegTensor<T>& value, Reg::RegTensor<uint32_t>& index,
                                                     Reg::RegTensor<uint32_t>& group, Reg::RegTensor<uint32_t>& lane,
                                                     Reg::MaskReg& activeMask)
{
    using GatherIndexT = BitonicSmallGatherIndexType<T>;
    using GatherSignedIndexT = BitonicSmallGatherSignedIndexType<T>;
    Reg::RegTensor<GatherSignedIndexT> valueLane;
    Reg::RegTensor<GatherSignedIndexT> valueStride;
    Reg::RegTensor<GatherSignedIndexT> valuePeerIndex;
    Reg::RegTensor<uint32_t> indexStride;
    Reg::RegTensor<uint32_t> indexPeerIndex;
    Reg::RegTensor<uint32_t> directionBit;
    Reg::Arange(valueLane, static_cast<GatherSignedIndexT>(0));
    Reg::Duplicate(valueStride, static_cast<GatherSignedIndexT>(Stride));
    Reg::Xor(valuePeerIndex, valueLane, valueStride, activeMask);
    Reg::Duplicate(indexStride, Stride);
    Reg::Xor(indexPeerIndex, lane, indexStride, activeMask);

    Reg::RegTensor<T> peerValue;
    Reg::RegTensor<uint32_t> peerIndex;
    Reg::RegTensor<uint32_t> peerGroup;
    Reg::Gather(peerValue, value, (Reg::RegTensor<GatherIndexT>&)valuePeerIndex);
    Reg::Gather(peerIndex, index, indexPeerIndex);
    if constexpr (CompareGroup) {
        Reg::Gather(peerGroup, group, indexPeerIndex);
    }

    Reg::MaskReg lowLaneMask;
    Reg::Compare<uint32_t, CMPMODE::LT>(lowLaneMask, lane, indexPeerIndex, activeMask);
    Reg::RegTensor<T> lowValue;
    Reg::RegTensor<T> highValue;
    Reg::RegTensor<uint32_t> lowIndex;
    Reg::RegTensor<uint32_t> highIndex;
    Reg::Select<T>(lowValue, value, peerValue, lowLaneMask);
    Reg::Select<T>(highValue, peerValue, value, lowLaneMask);
    Reg::Select<uint32_t>(lowIndex, index, peerIndex, lowLaneMask);
    Reg::Select<uint32_t>(highIndex, peerIndex, index, lowLaneMask);

    Reg::MaskReg compareMask;
    if constexpr (CompareGroup) {
        Reg::RegTensor<uint32_t> lowGroup;
        Reg::RegTensor<uint32_t> highGroup;
        Reg::MaskReg groupEqualMask;
        Reg::MaskReg indexLessMask;
        Reg::Select<uint32_t>(lowGroup, group, peerGroup, lowLaneMask);
        Reg::Select<uint32_t>(highGroup, peerGroup, group, lowLaneMask);
        Reg::Compare<uint32_t, CMPMODE::LT>(compareMask, lowGroup, highGroup, activeMask);
        Reg::Compare<uint32_t, CMPMODE::EQ>(groupEqualMask, lowGroup, highGroup, activeMask);
        Reg::Compare<uint32_t, CMPMODE::LT>(indexLessMask, lowIndex, highIndex, activeMask);
        Reg::And(groupEqualMask, groupEqualMask, indexLessMask, activeMask);
        Reg::Or(compareMask, compareMask, groupEqualMask, activeMask);
    } else {
        BitonicSmallRegValueBetter<T, IsLargest>(compareMask, lowValue, highValue, activeMask);
    }

    Reg::MaskReg lowValidMask;
    Reg::MaskReg highValidMask;
    Reg::MaskReg highInvalidMask;
    Reg::MaskReg swapMask;
    Reg::Compares<uint32_t, CMPMODE::NE>(lowValidMask, lowIndex, UINT32_MAX, activeMask);
    Reg::Compares<uint32_t, CMPMODE::NE>(highValidMask, highIndex, UINT32_MAX, activeMask);
    Reg::And(swapMask, compareMask, lowValidMask, activeMask);
    Reg::Not(highInvalidMask, highValidMask, activeMask);
    Reg::Or(swapMask, swapMask, highInvalidMask, activeMask);

    Reg::MaskReg directionMask;
    if constexpr (Size == BITONIC_SMALL_TOPK_SIZE) {
        Reg::Compares<uint32_t, CMPMODE::LT>(directionMask, lane, 0U, activeMask);
    } else {
        Reg::Duplicate(indexStride, Size);
        Reg::And(directionBit, lane, indexStride, activeMask);
        Reg::Compares<uint32_t, CMPMODE::NE>(directionMask, directionBit, 0U, activeMask);
    }

    Reg::MaskReg notSwapMask;
    Reg::MaskReg notDirectionMask;
    Reg::MaskReg takePeerMask;
    Reg::MaskReg keepDirectionMask;
    Reg::Not(notSwapMask, swapMask, activeMask);
    Reg::Not(notDirectionMask, directionMask, activeMask);
    Reg::And(takePeerMask, swapMask, directionMask, activeMask);
    Reg::And(keepDirectionMask, notSwapMask, notDirectionMask, activeMask);
    Reg::Or(takePeerMask, takePeerMask, keepDirectionMask, activeMask);
    Reg::Select<T>(value, peerValue, value, takePeerMask);
    Reg::Select<uint32_t>(index, peerIndex, index, takePeerMask);
    if constexpr (CompareGroup) {
        Reg::Select<uint32_t>(group, peerGroup, group, takePeerMask);
    }
}

/*!
 * \brief 通用类型的双调排序网络完整序列 (Reg/SIMD 路径)。
 *
 * 展开 15 个 SwapStage，结构与 BitonicSmallReg32BitonicNetwork 相同。
 * 额外携带 IsLargest 模板参数，用于 CompareGroup=false 时的值比较方向。
 */
template <typename T, bool IsLargest, bool CompareGroup>
__simd_callee__ inline void BitonicSmallRegBitonicNetwork(Reg::RegTensor<T>& value, Reg::RegTensor<uint32_t>& index,
                                                          Reg::RegTensor<uint32_t>& group,
                                                          Reg::RegTensor<uint32_t>& lane, Reg::MaskReg& activeMask)
{
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 1U, 2U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 2U, 4U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 1U, 4U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 4U, 8U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 2U, 8U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 1U, 8U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 8U, 16U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 4U, 16U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 2U, 16U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 1U, 16U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 16U, 32U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 8U, 32U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 4U, 32U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 2U, 32U>(value, index, group, lane, activeMask);
    BitonicSmallRegSwapStage<T, IsLargest, CompareGroup, 1U, 32U>(value, index, group, lane, activeMask);
}

/*!
 * \brief Reg/SIMD 路径收尾选择总分发函数。
 *
 * 按 sizeof(T) 分发到不同特化路径：
 *   - 8 字节 (uint64/int64) → BitonicSmallRegFinalizeSelectionB64 (拆双 32 位寄存器)
 *   - 2 字节 (half/bf16/int16/uint16) → BitonicSmallRegFinalizeSelectionB16 (位模式展开)
 *   - 其他 (1/4 字节) → 通用路径，直接按 value 比较，不转 key
 * 各路径流程一致：加载 → 分组 → 双轮排序 → 存回。
 */
template <typename T, bool IsLargest>
__simd_callee__ inline void BitonicSmallRegFinalizeSelection(__ubuf__ T* valueAddr, __ubuf__ uint32_t* indexAddr,
                                                             uint32_t k)
{
    if constexpr (sizeof(T) == sizeof(uint64_t)) {
        BitonicSmallRegFinalizeSelectionB64<T, IsLargest>(valueAddr, indexAddr, k);
    } else if constexpr (sizeof(T) == 2U) {
        BitonicSmallRegFinalizeSelectionB16<T, IsLargest>(valueAddr, indexAddr, k);
    } else {
        uint32_t activeCount = BITONIC_SMALL_TOPK_SIZE;
        uint32_t validCount = k;
        Reg::MaskReg activeMask = Reg::UpdateMask<T>(activeCount);
        Reg::MaskReg validMask = Reg::UpdateMask<T>(validCount);
        Reg::RegTensor<T> value;
        Reg::RegTensor<uint32_t> index;
        Reg::RegTensor<uint32_t> invalidIndex;
        Reg::RegTensor<uint32_t> lane;
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::LoadAlign<T, Reg::DataCopyMode::DATA_BLOCK_COPY>(value, valueAddr, 1U, validMask);
        Reg::RegTensor<T> zeroValue;
        Reg::Duplicate(zeroValue, static_cast<T>(0));
        Reg::Select<T>(value, value, zeroValue, validMask);
        Reg::LoadAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(index, indexAddr, 1U, validMask);
        Reg::Duplicate(invalidIndex, UINT32_MAX);
        Reg::Select<uint32_t>(index, index, invalidIndex, validMask);
        Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);

        using GatherIndexT = BitonicSmallGatherIndexType<T>;
        Reg::RegTensor<GatherIndexT> thresholdIndex;
        Reg::RegTensor<T> threshold;
        Reg::Duplicate(thresholdIndex, static_cast<GatherIndexT>(k - 1U));
        Reg::Gather(threshold, value, thresholdIndex);
        Reg::MaskReg strictMask;
        BitonicSmallRegValueBetter<T, IsLargest>(strictMask, value, threshold, activeMask);
        Reg::And(strictMask, strictMask, validMask, activeMask);

        // Reorder the selected candidates by (strict/equal group, original index) to match BITONIC gather.
        Reg::RegTensor<uint32_t> group;
        Reg::RegTensor<uint32_t> strictGroup;
        Reg::RegTensor<uint32_t> invalidGroup;
        Reg::Duplicate(group, 1U);
        Reg::Duplicate(strictGroup, 0U);
        Reg::Duplicate(invalidGroup, 2U);
        Reg::Select<uint32_t>(group, strictGroup, group, strictMask);
        Reg::Select<uint32_t>(group, group, invalidGroup, validMask);
        BitonicSmallRegBitonicNetwork<T, IsLargest, true>(value, index, group, lane, activeMask);
        BitonicSmallRegBitonicNetwork<T, IsLargest, false>(value, index, group, lane, activeMask);

        Reg::StoreAlign<T, Reg::DataCopyMode::DATA_BLOCK_COPY>(valueAddr, value, 1U, validMask);
        Reg::StoreAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(indexAddr, index, 1U, validMask);
    }
}

/*!
 * \brief 判断 lhs 是否"优于" rhs (SIMT 标量路径)。
 *
 * 与 Reg 路径的 BitonicSmallRegValueBetter 语义一致，但用标量比较实现。
 * 浮点类型统一转 float 后比较，NaN 语义：isLargest 时 NaN 视为最大，否则最小。
 */
template <typename T>
__simt_callee__ inline bool BitonicSmallValueBetter(T lhs, T rhs, bool isLargest)
{
    if constexpr (IsBitonicFloatType<T>) {
        float lhsFloat = static_cast<float>(lhs);
        float rhsFloat = static_cast<float>(rhs);
        bool lhsNan = isnan(lhsFloat);
        bool rhsNan = isnan(rhsFloat);
        if (isLargest) {
            return (lhsNan && !rhsNan) || (lhsFloat > rhsFloat);
        }
        return (rhsNan && !lhsNan) || (lhsFloat < rhsFloat);
    }
    return isLargest ? (lhs > rhs) : (lhs < rhs);
}

/*!
 * \brief 判断 lhs 和 rhs 是否等价 (SIMT 标量路径)。
 *
 * 浮点类型将 NaN==NaN 视为等价；整数类型直接比较相等。
 */
template <typename T>
__simt_callee__ inline bool BitonicSmallValueEquivalent(T lhs, T rhs)
{
    if constexpr (IsBitonicFloatType<T>) {
        float lhsFloat = static_cast<float>(lhs);
        float rhsFloat = static_cast<float>(rhs);
        return (isnan(lhsFloat) && isnan(rhsFloat)) || (lhsFloat == rhsFloat);
    }
    return lhs == rhs;
}

/*!
 * \brief 返回无效索引标记值 (SIMT 标量路径)。
 *
 * 用 -1 (全 1 位模式) 表示无效候选元素，排序时自然被挤到末尾。
 */
template <typename IndexT>
__simt_callee__ inline IndexT BitonicSmallInvalidIndex()
{
    return static_cast<IndexT>(-1);
}

/*!
 * \brief 双调网络的单次 compare-swap 阶段 (SIMT warp 级路径)。
 *
 * 每个 lane 线程持有一个元素，通过 asc_shfl_xor 获取对端 lane 的数据。
 * CompareValue=true 时按值比较（含 NaN 语义），false 时按 index 比较（用于分组排序）。
 * 交换逻辑与 Reg 路径一致，但用标量条件判断 + 直接赋值实现。
 */
template <typename T, typename IndexT, bool CompareValue>
__simt_callee__ inline void BitonicSmallSwapStage(T& value, IndexT& index, uint32_t stride, uint32_t size,
                                                  bool isLargest)
{
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    T peerValue = asc_shfl_xor(value, static_cast<int32_t>(stride), BITONIC_SMALL_TOPK_SIZE);
    IndexT peerIndex = asc_shfl_xor(index, static_cast<int32_t>(stride), BITONIC_SMALL_TOPK_SIZE);

    bool isLow = (lane & stride) == 0U;
    T valueA = isLow ? value : peerValue;
    T valueB = isLow ? peerValue : value;
    IndexT indexA = isLow ? index : peerIndex;
    IndexT indexB = isLow ? peerIndex : index;
    IndexT invalid = BitonicSmallInvalidIndex<IndexT>();
    bool validA = indexA != invalid;
    bool validB = indexB != invalid;
    bool comp = CompareValue ? BitonicSmallValueBetter<T>(valueA, valueB, isLargest) : (indexA < indexB);
    bool swap = (comp && validA) || !validB;

    uint32_t lowLane = lane & ~stride;
    uint32_t comparatorLane = (lowLane / (stride * 2U)) * stride + (lowLane % stride);
    bool dir = size != BITONIC_SMALL_TOPK_SIZE && ((comparatorLane & (size / 2U)) != 0U);
    if (swap == dir) {
        value = peerValue;
        index = peerIndex;
    }
}

/*!
 * \brief 双调排序网络完整序列 (SIMT warp 级路径)。
 *
 * 用两层 for 循环展开 15 个 SwapStage，比 Reg 路径的模板展开更紧凑。
 * CompareValue=true 时按值排序，false 时按 index 排序（用于恢复原始顺序）。
 */
template <typename T, typename IndexT, bool CompareValue>
__simt_callee__ inline void BitonicSmallBitonicNetwork(T& value, IndexT& index, bool isLargest)
{
    for (uint32_t size = 2U; size < BITONIC_SMALL_TOPK_SIZE; size *= 2U) {
        for (uint32_t stride = size / 2U; stride > 0U; stride /= 2U) {
            BitonicSmallSwapStage<T, IndexT, CompareValue>(value, index, stride, size, isLargest);
        }
    }
    for (uint32_t stride = BITONIC_SMALL_TOPK_SIZE / 2U; stride > 0U; stride /= 2U) {
        BitonicSmallSwapStage<T, IndexT, CompareValue>(value, index, stride, BITONIC_SMALL_TOPK_SIZE, isLargest);
    }
}

/*!
 * \brief 全等场景的单次 compare-swap (SIMT warp 级路径)。
 *
 * 当所有候选值相等时，无需值比较，仅按 index 排序。
 * 交换条件：对端 index 无效时交换，将无效项挤到末尾。
 */
template <typename IndexT>
__simt_callee__ inline void BitonicSmallAllEqualSwapStage(IndexT& index, uint32_t stride, uint32_t size)
{
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    IndexT peerIndex = asc_shfl_xor(index, static_cast<int32_t>(stride), BITONIC_SMALL_TOPK_SIZE);
    bool isLow = (lane & stride) == 0U;
    IndexT indexB = isLow ? peerIndex : index;
    bool swap = indexB == BitonicSmallInvalidIndex<IndexT>();

    uint32_t lowLane = lane & ~stride;
    uint32_t comparatorLane = (lowLane / (stride * 2U)) * stride + (lowLane % stride);
    bool dir = size != BITONIC_SMALL_TOPK_SIZE && ((comparatorLane & (size / 2U)) != 0U);
    if (swap == dir) {
        index = peerIndex;
    }
}

/*!
 * \brief 全等场景的双调排序网络 (SIMT warp 级路径)。
 *
 * 当所有候选值相等时，仅按 index 排序，是 BitonicSmallBitonicNetwork 的轻量特化版。
 */
template <typename IndexT>
__simt_callee__ inline void BitonicSmallAllEqualBitonicNetwork(IndexT& index)
{
    for (uint32_t size = 2U; size < BITONIC_SMALL_TOPK_SIZE; size *= 2U) {
        for (uint32_t stride = size / 2U; stride > 0U; stride /= 2U) {
            BitonicSmallAllEqualSwapStage<IndexT>(index, stride, size);
        }
    }
    for (uint32_t stride = BITONIC_SMALL_TOPK_SIZE / 2U; stride > 0U; stride /= 2U) {
        BitonicSmallAllEqualSwapStage<IndexT>(index, stride, BITONIC_SMALL_TOPK_SIZE);
    }
}

/*!
 * \brief 在位掩码中找到第 rank 个置位的位置 (SIMT 标量路径)。
 *
 * 用二分法在 32 位掩码中查找：每次将掩码分为高低半部，统计低位置位数，
 * 若 rank 落在低位则保留低位掩码，否则减去低位计数并右移。
 * 用于 SIMT finalize 中重构 BITONIC 兼容的 gather 源 lane 顺序。
 */
__simt_callee__ inline uint32_t BitonicSmallNthSetBit(uint32_t mask, uint32_t rank)
{
    uint32_t base = 0U;
    for (uint32_t halfSize = 16U; halfSize > 0U; halfSize /= 2U) {
        uint32_t lowerMask = mask & ((1U << halfSize) - 1U);
        uint32_t lowerCount = static_cast<uint32_t>(__popc(lowerMask));
        if (rank < lowerCount) {
            mask = lowerMask;
        } else {
            rank -= lowerCount;
            mask >>= halfSize;
            base += halfSize;
        }
    }
    return base;
}

/*!
 * \brief 收尾选择函数 (SIMT warp 级路径)，处理已按值排序的候选集。
 *
 * 流程：
 *   1. 取阈值：threshold = 第 k-1 个元素的值（已排序，直接取）。
 *   2. 全等快速路径：若首元素 == 阈值，所有候选相等，仅按 index 排序即可返回。
 *   3. 重复值检测：检查是否存在与前驱相等的元素，若无重复直接返回。
 *   4. BITONIC 兼容性重构：
 *      - 先按 index 排序恢复原始 lane 序 (BitonicNetwork<false>)。
 *      - 计算 strict/equal 掩码，用 BitonicSmallNthSetBit 找到每个输出 lane 的源 lane。
 *      - asc_shfl 按源 lane 重新收集数据。
 *      - 最后按值排序 (BitonicNetwork<true>)。
 *   5. 仅 lane < k 的线程写回结果。
 */
template <typename T, typename IndexT, bool IsLargest>
__simt_callee__ inline void BitonicSmallFinalizeSelection(T& value, IndexT& index, bool valid, uint32_t k)
{
    using RegT = BitonicSmallRegType<T>;
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    RegT regValue = valid ? static_cast<RegT>(value) : static_cast<RegT>(0);
    IndexT regIndex = valid ? index : BitonicSmallInvalidIndex<IndexT>();

    // Selection candidates are value-sorted, so the last valid lane is the kth threshold.
    RegT threshold = asc_shfl(regValue, static_cast<int32_t>(k - 1U), BITONIC_SMALL_TOPK_SIZE);

    if constexpr (std::is_integral_v<T>) {
        RegT firstValue = asc_shfl(regValue, 0, BITONIC_SMALL_TOPK_SIZE);
        if (BitonicSmallValueEquivalent<RegT>(firstValue, threshold)) {
            BitonicSmallAllEqualBitonicNetwork<IndexT>(regIndex);
            if (lane < k) {
                index = regIndex;
            }
            return;
        }

        RegT previous = asc_shfl(regValue, static_cast<int32_t>(lane == 0U ? 0U : lane - 1U), BITONIC_SMALL_TOPK_SIZE);
        bool duplicate = valid && lane > 0U && BitonicSmallValueEquivalent<RegT>(regValue, previous);
        uint32_t duplicateMask = asc_ballot(static_cast<int32_t>(duplicate));
        if (duplicateMask == 0U) {
            return;
        }
    }

    // Reconstruct BITONIC's strict-better/equal gather order inside the selected k candidates.
    BitonicSmallBitonicNetwork<RegT, IndexT, false>(regValue, regIndex, false);
    bool strict = valid && BitonicSmallValueBetter<RegT>(regValue, threshold, IsLargest);
    bool equal = valid && BitonicSmallValueEquivalent<RegT>(regValue, threshold);
    uint32_t strictMask = asc_ballot(static_cast<int32_t>(strict));
    uint32_t equalMask = asc_ballot(static_cast<int32_t>(equal));
    uint32_t strictCount = static_cast<uint32_t>(__popc(strictMask));
    uint32_t sourceLane = 0U;
    if (lane < k) {
        uint32_t rank = lane < strictCount ? lane : lane - strictCount;
        sourceLane = BitonicSmallNthSetBit(lane < strictCount ? strictMask : equalMask, rank);
    }
    regValue = asc_shfl(regValue, static_cast<int32_t>(sourceLane), BITONIC_SMALL_TOPK_SIZE);
    regIndex = asc_shfl(regIndex, static_cast<int32_t>(sourceLane), BITONIC_SMALL_TOPK_SIZE);
    if (lane >= k) {
        regIndex = BitonicSmallInvalidIndex<IndexT>();
    }

    BitonicSmallBitonicNetwork<RegT, IndexT, true>(regValue, regIndex, IsLargest);
    if (lane < k) {
        value = static_cast<T>(regValue);
        index = regIndex;
    }
}

/*!
 * \brief 收尾选择函数 (SIMT warp 级路径)，处理未排序的精确候选集。
 *
 * 与 BitonicSmallFinalizeSelection 的区别：候选集未按值排序，需要先归约求阈值。
 * 流程：
 *   1. 用蝶形归约 (asc_shfl_xor 树) 找到候选中的最差值作为阈值。
 *   2. 其余 BITONIC 兼容性重构逻辑与 BitonicSmallFinalizeSelection 相同。
 *   3. 仅 valid 线程写回结果。
 */
template <typename T, typename IndexT, bool IsLargest>
__simt_callee__ inline void BitonicSmallFinalizeExactSelection(T& value, IndexT& index, bool valid, uint32_t k)
{
    using RegT = BitonicSmallRegType<T>;
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    RegT regValue = valid ? static_cast<RegT>(value) : static_cast<RegT>(0);
    IndexT regIndex = valid ? index : BitonicSmallInvalidIndex<IndexT>();

    // The gathered candidates are exact but not value-sorted. Reduce their worst value to recover the kth threshold.
    RegT firstValue = asc_shfl(regValue, 0, BITONIC_SMALL_TOPK_SIZE);
    RegT threshold = valid ? regValue : firstValue;
    for (uint32_t stride = BITONIC_SMALL_TOPK_SIZE / 2U; stride > 0U; stride /= 2U) {
        RegT peer = asc_shfl_xor(threshold, static_cast<int32_t>(stride), BITONIC_SMALL_TOPK_SIZE);
        if (BitonicSmallValueBetter<RegT>(threshold, peer, IsLargest)) {
            threshold = peer;
        }
    }

    // Restore BITONIC's source-order strict/equal gather before applying SmallBitonicSort<32>.
    BitonicSmallBitonicNetwork<RegT, IndexT, false>(regValue, regIndex, false);
    bool strict = valid && BitonicSmallValueBetter<RegT>(regValue, threshold, IsLargest);
    bool equal = valid && BitonicSmallValueEquivalent<RegT>(regValue, threshold);
    uint32_t strictMask = asc_ballot(static_cast<int32_t>(strict));
    uint32_t equalMask = asc_ballot(static_cast<int32_t>(equal));
    uint32_t strictCount = static_cast<uint32_t>(__popc(strictMask));
    uint32_t sourceLane = 0U;
    if (lane < k) {
        uint32_t rank = lane < strictCount ? lane : lane - strictCount;
        sourceLane = BitonicSmallNthSetBit(lane < strictCount ? strictMask : equalMask, rank);
    }
    regValue = asc_shfl(regValue, static_cast<int32_t>(sourceLane), BITONIC_SMALL_TOPK_SIZE);
    regIndex = asc_shfl(regIndex, static_cast<int32_t>(sourceLane), BITONIC_SMALL_TOPK_SIZE);
    if (!valid) {
        regIndex = BitonicSmallInvalidIndex<IndexT>();
    }
    BitonicSmallBitonicNetwork<RegT, IndexT, true>(regValue, regIndex, IsLargest);
    if (valid) {
        value = static_cast<T>(regValue);
        index = regIndex;
    }
}

/*!
 * \brief 按阈值 key 从输入中 gather 候选元素 (SIMT Kernel)。
 *
 * 两趟扫描：
 *   pass 0：收集 key < threshold（严格优于阈值）的元素。
 *   pass 1：收集 key == threshold（等于阈值）的元素。
 * 用 asc_ballot + __popc 做 warp 内前缀和，确定每个选中元素的输出位置。
 * outputPos = written + rank，确保不超出 quota。
 */
template <typename T, typename KeyT, typename IndexT>
__simt_vf__ LAUNCH_BOUND(BITONIC_SMALL_TOPK_SIZE) __aicore__
    void BitonicGatherThresholdTileKernel(uint32_t axisSize, uint32_t quota, uint64_t indexBase, KeyT threshold,
                                          __ubuf__ T* inputValues, __ubuf__ KeyT* keys, __ubuf__ T* outputValues,
                                          __ubuf__ IndexT* outputIndices)
{
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    uint32_t written = 0U;
    for (uint32_t pass = 0U; pass < 2U && written < quota; ++pass) {
        for (uint32_t base = 0U; base < axisSize && written < quota; base += BITONIC_SMALL_TOPK_SIZE) {
            uint32_t col = base + lane;
            bool inRange = col < axisSize;
            KeyT key = inRange ? keys[col] : static_cast<KeyT>(0);
            bool take = inRange && (pass == 0U ? key < threshold : key == threshold);
            uint32_t takeMask = asc_ballot(static_cast<int32_t>(take));
            uint32_t lowerMask = lane == 0U ? 0U : ((1U << lane) - 1U);
            uint32_t rank = static_cast<uint32_t>(__popc(takeMask & lowerMask));
            uint32_t outputPos = written + rank;
            if (take && outputPos < quota) {
                outputValues[outputPos] = inputValues[col];
                outputIndices[outputPos] = static_cast<IndexT>(indexBase + col);
            }
            written += static_cast<uint32_t>(__popc(takeMask));
        }
    }
}

/*!
 * \brief BitonicGatherThresholdTileKernel 的 LocalTensor 封装。
 *
 * 将 LocalTensor 参数转换为 __ubuf__ 指针后调用 Kernel。
 */
template <typename T, typename KeyT, typename IndexT>
__aicore__ inline void RunBitonicGatherThresholdTile(LocalTensor<T> inputValues, LocalTensor<KeyT> keys,
                                                     LocalTensor<T> outputValues, LocalTensor<IndexT> outputIndices,
                                                     uint32_t axisSize, uint32_t quota, uint64_t indexBase,
                                                     KeyT threshold)
{
    asc_vf_call<BitonicGatherThresholdTileKernel<T, KeyT, IndexT>>(
        dim3(BITONIC_SMALL_TOPK_SIZE), axisSize, quota, indexBase, threshold, (__ubuf__ T*)inputValues.GetPhyAddr(),
        (__ubuf__ KeyT*)keys.GetPhyAddr(), (__ubuf__ T*)outputValues.GetPhyAddr(),
        (__ubuf__ IndexT*)outputIndices.GetPhyAddr());
}

/*!
 * \brief 精确选择收尾 Kernel (SIMT)，封装 BitonicSmallFinalizeExactSelection。
 *
 * 单行处理：从 UB 读取 k 个候选，调用 BitonicSmallFinalizeExactSelection 排序后写回。
 */
template <typename T, typename IndexT, bool IsLargest>
__simt_vf__ LAUNCH_BOUND(BITONIC_SMALL_TOPK_SIZE) __aicore__
    void BitonicFinalizeExactSelectionKernel(uint32_t k, __ubuf__ T* values, __ubuf__ IndexT* indices)
{
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    bool valid = lane < k;
    T value = valid ? values[lane] : static_cast<T>(0);
    IndexT index = valid ? indices[lane] : BitonicSmallInvalidIndex<IndexT>();
    BitonicSmallFinalizeExactSelection<T, IndexT, IsLargest>(value, index, valid, k);
    if (valid) {
        values[lane] = value;
        indices[lane] = index;
    }
}

/*!
 * \brief BitonicFinalizeExactSelectionKernel 的 LocalTensor 封装。
 */
template <typename T, typename IndexT, bool IsLargest>
__aicore__ inline void RunBitonicFinalizeExactSelection(LocalTensor<T> values, LocalTensor<IndexT> indices, uint32_t k)
{
    asc_vf_call<BitonicFinalizeExactSelectionKernel<T, IndexT, IsLargest>>(
        dim3(BITONIC_SMALL_TOPK_SIZE), k, (__ubuf__ T*)values.GetPhyAddr(), (__ubuf__ IndexT*)indices.GetPhyAddr());
}

/*!
 * \brief 小源行收尾 Kernel (SIMT)，处理 axisLen <= 32 的特殊场景。
 *
 * 从已排序输出读入 k 个候选，取阈值后判断：
 *   - 全等路径：所有值相等 → 仅按 index 排序。
 *   - 重复值路径：从原始输入 inputValues 重新 gather 候选（两趟：strict + equal），
 *     用 BitonicSmallNthSetBit 重构顺序后按值排序。
 * 多行并行：threadIdx.y = 行索引，threadIdx.x = lane。
 */
template <typename T, typename IndexT, bool IsLargest>
__simt_vf__ LAUNCH_BOUND(BITONIC_SMALL_TOPK_THREADS) __aicore__
    void BitonicFinalizeSmallSourceRowsKernel(uint32_t inputCount, uint32_t k, uint32_t inputStride,
                                              uint32_t valueStride, uint32_t indexStride, __ubuf__ T* inputValues,
                                              __ubuf__ T* outputValues, __ubuf__ IndexT* outputIndices)
{
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    uint32_t row = static_cast<uint32_t>(threadIdx.y);
    uint32_t valueOffset = row * valueStride + lane;
    uint32_t indexOffset = row * indexStride + lane;
    bool valid = lane < k;
    T value = valid ? outputValues[valueOffset] : static_cast<T>(0);
    IndexT index = valid ? outputIndices[indexOffset] : BitonicSmallInvalidIndex<IndexT>();
    T threshold = asc_shfl(value, static_cast<int32_t>(k - 1U), BITONIC_SMALL_TOPK_SIZE);

    T firstValue = asc_shfl(value, 0, BITONIC_SMALL_TOPK_SIZE);
    if (BitonicSmallValueEquivalent<T>(firstValue, threshold)) {
        BitonicSmallAllEqualBitonicNetwork<IndexT>(index);
    } else {
        T previous = asc_shfl(value, static_cast<int32_t>(lane == 0U ? 0U : lane - 1U), BITONIC_SMALL_TOPK_SIZE);
        bool duplicate = valid && lane > 0U && BitonicSmallValueEquivalent<T>(value, previous);
        uint32_t duplicateMask = asc_ballot(static_cast<int32_t>(duplicate));
        if (duplicateMask != 0U) {
            T gatheredValue = static_cast<T>(0);
            IndexT gatheredIndex = BitonicSmallInvalidIndex<IndexT>();
            uint32_t gatheredCount = 0U;
            uint32_t inputIndex = lane;
            bool inputValid = inputIndex < inputCount;
            T inputValue = inputValid ? inputValues[row * inputStride + inputIndex] : static_cast<T>(0);
            for (uint32_t pass = 0U; pass < 2U && gatheredCount < k; ++pass) {
                bool selected = inputValid &&
                                (pass == 0U ? BitonicSmallValueBetter<T>(inputValue, threshold, IsLargest) :
                                              BitonicSmallValueEquivalent<T>(inputValue, threshold));
                uint32_t selectedMask = asc_ballot(static_cast<int32_t>(selected));
                uint32_t selectedCount = static_cast<uint32_t>(__popc(selectedMask));
                if (lane >= gatheredCount && lane < gatheredCount + selectedCount && lane < k) {
                    uint32_t sourceLane = BitonicSmallNthSetBit(selectedMask, lane - gatheredCount);
                    gatheredValue = asc_shfl(inputValue, static_cast<int32_t>(sourceLane), BITONIC_SMALL_TOPK_SIZE);
                    gatheredIndex = static_cast<IndexT>(sourceLane);
                }
                gatheredCount += selectedCount;
            }
            value = gatheredValue;
            index = gatheredIndex;
            BitonicSmallBitonicNetwork<T, IndexT, true>(value, index, IsLargest);
        }
    }

    if (valid) {
        outputValues[valueOffset] = value;
        outputIndices[indexOffset] = index;
    }
}

/*!
 * \brief BitonicFinalizeSmallSourceRowsKernel 的多行批量调度封装。
 *
 * 对 rowCount 行分批调度，每批最多 BITONIC_SMALL_TOPK_MAX_ROWS (32) 行。
 */
template <typename T, typename IndexT, bool IsLargest>
__aicore__ inline void RunBitonicFinalizeSmallSourceRows(LocalTensor<T> inputValues, LocalTensor<T> outputValues,
                                                         LocalTensor<IndexT> outputIndices, uint32_t inputCount,
                                                         uint32_t k, uint32_t rowCount, uint32_t inputStride,
                                                         uint32_t valueStride, uint32_t indexStride)
{
    for (uint32_t rowStart = 0U; rowStart < rowCount; rowStart += BITONIC_SMALL_TOPK_MAX_ROWS) {
        uint32_t rows = rowCount - rowStart;
        rows = rows > BITONIC_SMALL_TOPK_MAX_ROWS ? BITONIC_SMALL_TOPK_MAX_ROWS : rows;
        asc_vf_call<BitonicFinalizeSmallSourceRowsKernel<T, IndexT, IsLargest>>(
            dim3(BITONIC_SMALL_TOPK_SIZE, rows), inputCount, k, inputStride, valueStride, indexStride,
            (__ubuf__ T*)inputValues[rowStart * inputStride].GetPhyAddr(),
            (__ubuf__ T*)outputValues[rowStart * valueStride].GetPhyAddr(),
            (__ubuf__ IndexT*)outputIndices[rowStart * indexStride].GetPhyAddr());
    }
}

/*!
 * \brief 行批量收尾 Kernel (SIMT)，封装 BitonicSmallFinalizeSelection。
 *
 * 多行并行处理，每行调用 BitonicSmallFinalizeSelection 对 k 个已排序候选做收尾排序。
 * threadIdx.y = 行索引，threadIdx.x = lane (0~31)。
 */
template <typename T, typename IndexT, bool IsLargest>
__simt_vf__ LAUNCH_BOUND(BITONIC_SMALL_TOPK_THREADS) __aicore__
    void BitonicFinalizeSelectionRowsKernel(uint32_t k, uint32_t valueStride, uint32_t indexStride, __ubuf__ T* values,
                                            __ubuf__ IndexT* indices)
{
    uint32_t lane = static_cast<uint32_t>(threadIdx.x);
    uint32_t row = static_cast<uint32_t>(threadIdx.y);
    uint32_t valueOffset = row * valueStride + lane;
    uint32_t indexOffset = row * indexStride + lane;
    bool valid = lane < k;
    T value = valid ? values[valueOffset] : static_cast<T>(0);
    IndexT index = valid ? indices[indexOffset] : BitonicSmallInvalidIndex<IndexT>();
    BitonicSmallFinalizeSelection<T, IndexT, IsLargest>(value, index, valid, k);
    if (valid) {
        values[valueOffset] = value;
        indices[indexOffset] = index;
    }
}

/*!
 * \brief BitonicFinalizeSelectionRowsKernel 的多行批量调度封装。
 *
 * 对 rowCount 行分批调度，每批最多 BITONIC_SMALL_TOPK_MAX_ROWS (32) 行。
 */
template <typename T, typename IndexT, bool IsLargest>
__aicore__ inline void RunBitonicFinalizeSelectionRows(LocalTensor<T> values, LocalTensor<IndexT> indices, uint32_t k,
                                                       uint32_t rowCount, uint32_t valueStride, uint32_t indexStride)
{
    for (uint32_t rowStart = 0U; rowStart < rowCount; rowStart += BITONIC_SMALL_TOPK_MAX_ROWS) {
        uint32_t rows = rowCount - rowStart;
        rows = rows > BITONIC_SMALL_TOPK_MAX_ROWS ? BITONIC_SMALL_TOPK_MAX_ROWS : rows;
        asc_vf_call<BitonicFinalizeSelectionRowsKernel<T, IndexT, IsLargest>>(
            dim3(BITONIC_SMALL_TOPK_SIZE, rows), k, valueStride, indexStride,
            (__ubuf__ T*)values[rowStart * valueStride].GetPhyAddr(),
            (__ubuf__ IndexT*)indices[rowStart * indexStride].GetPhyAddr());
    }
}

/*!
 * \brief 最终收尾主入口：按类型分发到 Reg 或 SIMT 路径。
 *
 * 分发逻辑：
 *   - sizeof(T) != 1 && sizeof(IndexT) == 4：走高性能 Reg 路径
 *     (BitonicSmallRegFinalizeSelection)，逐 batch 调用。
 *   - 否则：走 SIMT 行批量路径 (RunBitonicFinalizeSelectionRows)。
 *
 * sizeof(T) == 1U (int8/uint8) 被排除在 Reg 路径之外的原因：
 *   arch35 的 SIMD Reg:: API (Gather/Compare/Select/LoadAlign 等) 最小操作位宽为
 *   16 位，不支持 8 位 RegTensor。1 字节类型无法加载进 Reg 寄存器体系，也无法
 *   用 Reg::Gather 做 warp 内通信。因此只能回退到 SIMT 标量线程路径，用
 *   asc_shfl_xor 替代 Reg::Gather。这是硬件 ISA 的功能限制，非性能选择。
 */
template <typename T, typename T_INDEX_TO, bool IS_LARGEST>
__aicore__ inline void RunBitonicSmallTopKFinalize(LocalTensor<T> values, LocalTensor<T_INDEX_TO> indices, uint32_t k,
                                                   uint32_t batchNum, uint32_t valueStride, uint32_t indexStride)
{
    if constexpr (sizeof(T) != 1U && sizeof(T_INDEX_TO) == sizeof(uint32_t)) {
        for (uint32_t i = 0U; i < batchNum; ++i) {
            __ubuf__ T* valueAddr = (__ubuf__ T*)values[i * valueStride].GetPhyAddr();
            __ubuf__ uint32_t* indexAddr = (__ubuf__ uint32_t*)indices[i * indexStride].GetPhyAddr();
            __VEC_SCOPE__ { BitonicSmallRegFinalizeSelection<T, IS_LARGEST>(valueAddr, indexAddr, k); }
        }
    } else {
        RunBitonicFinalizeSelectionRows<T, T_INDEX_TO, IS_LARGEST>(values, indices, k, batchNum, valueStride,
                                                                   indexStride);
    }
}
/*!
 * \brief 非尾轴场景调用AscendC::Topk高阶api的双调排序入口。
 *
 * 带 IS_BITONIC_SORT 模板控制，分发逻辑：
 *   - 1 字节 + axisLen <= 32：走小源行路径 RunBitonicFinalizeSmallSourceRows。
 *   - 非 1 字节：走 Reg 路径 BitonicSmallRegFinalizeSelection。
 *   - 1 字节其他：走 SIMT 行批量路径 RunBitonicFinalizeSelectionRows。
 *
 * sizeof(SortT) == 1U (int8/uint8) 不能走 Reg 路径的原因：
 *   arch35 的 SIMD Reg:: API (Gather/Compare/Select/LoadAlign 等) 最小操作位宽为
 *   16 位，不支持 8 位 RegTensor。1 字节类型无法加载进 Reg 寄存器体系，也无法
 *   用 Reg::Gather 做 warp 内通信。因此只能回退到 SIMT 标量线程路径，用
 *   asc_shfl_xor 替代 Reg::Gather。这是硬件 ISA 的功能限制，非性能选择。
 */
template <typename SortT, bool IsLargest, bool IsBitonicSort>
__aicore__ inline void RunBitonicSmallTopKFinalizeNonLast(LocalTensor<SortT> values, LocalTensor<uint32_t> indices,
                                                          LocalTensor<SortT> sortInput, uint32_t axisLen, uint32_t k,
                                                          uint32_t batchNum, uint32_t axisRowElems,
                                                          uint32_t valueStride, uint32_t indexStride)
{
    if constexpr (IsBitonicSort) {
        if constexpr (sizeof(SortT) == 1U) {
            if (axisLen <= BITONIC_SMALL_TOPK_SIZE) {
                RunBitonicFinalizeSmallSourceRows<SortT, uint32_t, IsLargest>(
                    sortInput, values, indices, axisLen, k, batchNum, axisRowElems, valueStride, indexStride);
                return;
            }
        }
        if constexpr (sizeof(SortT) != 1U) {
            for (uint32_t i = 0U; i < batchNum; ++i) {
                __ubuf__ SortT* valueAddr = (__ubuf__ SortT*)values[i * valueStride].GetPhyAddr();
                __ubuf__ uint32_t* indexAddr = (__ubuf__ uint32_t*)indices[i * indexStride].GetPhyAddr();
                __VEC_SCOPE__ { BitonicSmallRegFinalizeSelection<SortT, IsLargest>(valueAddr, indexAddr, k); }
            }
        } else {
            RunBitonicFinalizeSelectionRows<SortT, uint32_t, IsLargest>(values, indices, k, batchNum, valueStride,
                                                                        indexStride);
        }
    }
}

/*!
 * \brief 非尾轴场景调用Ascend::Sort高阶api的进行双调排序入口。
 *
 * 分发逻辑：
 *   - sizeof(SortT) > 4 || sizeof(SortT) == 1：走 SIMT 行批量路径。
 *   - 否则 (2/4 字节)：走 Reg 路径 BitonicSmallRegFinalizeSelection。
 *
 * sizeof(SortT) == 1U (int8/uint8) 走 SIMT 路径的原因：
 *   arch35 的 SIMD Reg:: API (Gather/Compare/Select/LoadAlign 等) 最小操作位宽为
 *   16 位，不支持 8 位 RegTensor。1 字节类型无法加载进 Reg 寄存器体系，也无法
 *   用 Reg::Gather 做 warp 内通信。因此只能回退到 SIMT 标量线程路径，用
 *   asc_shfl_xor 替代 Reg::Gather。这是硬件 ISA 的功能限制，非性能选择。
 *
 * sizeof(SortT) > 4 走 SIMT 路径的原因：
 *   64 位类型在 merge sort 场景走 SIMT 标量路径实现更简单，避免 Reg 路径的
 *   双寄存器拆分复杂度。
 */
template <typename SortT, bool IsLargest, bool IsBitonicSort>
__aicore__ inline void RunBitonicSmallMergeSortFinalizeNonLast(LocalTensor<SortT> values, LocalTensor<uint32_t> indices,
                                                               uint32_t k, uint32_t valueStride, uint32_t indexStride)
{
    if constexpr (IsBitonicSort) {
        if constexpr (sizeof(SortT) > sizeof(uint32_t) || sizeof(SortT) == 1U) {
            RunBitonicFinalizeSelectionRows<SortT, uint32_t, IsLargest>(values, indices, k, 1U, valueStride,
                                                                        indexStride);
        } else {
            __ubuf__ SortT* valueAddr = (__ubuf__ SortT*)values.GetPhyAddr();
            __ubuf__ uint32_t* indexAddr = (__ubuf__ uint32_t*)indices.GetPhyAddr();
            __VEC_SCOPE__ { BitonicSmallRegFinalizeSelection<SortT, IsLargest>(valueAddr, indexAddr, k); }
        }
    }
}

} // namespace topkV2

#endif // TOP_K_SMALL_SIZE_BITONIC_SORT_H
