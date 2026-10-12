/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file transpose_base.h
 * \brief Transpose 算子 Kernel 侧基类定义及多核范围解析
 *
 * 本文件定义了所有 Transpose 策略类的公共基类 TransposeBase，以及全局多核
 * 范围解析函数 ParseMultiCoreRange。基类提供 reverseArray 工具方法，
 * 用于将5维 NDDMA 参数数组逆序排列（NDDMA 硬件要求参数以特定顺序排列）。
 */

#ifndef TRANSPOSE_BASE_H
#define TRANSPOSE_BASE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace Transpose {
using namespace AscendC;

/* 以下常量定义了 Transpose Kernel 侧的核心约束参数 */
constexpr int64_t TRANSPOSE_MAX_AXIS_NUM = 8; // 最大支持维度数（与 Host 侧 TRANSPOSE_MAX_AXIS_NUM 一致）
constexpr int64_t NDDMA_MAX_DIM_NUM = 5;      // NDDMA 硬件支持的最大维度数（5维搬运格式）
constexpr int64_t BLOCK_SIZE_BYTE = 32;       // 硬件 Block 大小（32字节），所有 UB 对齐以此为单位
constexpr int64_t DIM_TWO = 2;                // 维度常量：2维
constexpr int64_t DIM_THREE = 3;              // 维度常量：3维
constexpr int64_t DIM_FOUR = 4;               // 维度常量：4维
constexpr int64_t DIM_FIVE = 5;               // 维度常量：5维
constexpr int64_t DIM_SIX = 6;                // 维度常量：6维
constexpr int64_t DIM_SEVEN = 7;              // 维度常量：7维
constexpr int64_t DIM_EIGHT = 8;              // 维度常量：8维
constexpr NdDmaConfig config = {false, 0, 0, false}; // NDDMA DataCopy 默认配置（非重复模式）
constexpr int64_t TWO = 2;
constexpr int64_t BUFFER_NUM = 2; // 双缓冲数量，用于 TENSOR_MOVE 和 N_LAST 策略

/**
 * @brief Transpose 算子 Kernel 侧基类
 *
 * 所有 9 种转置策略类（TransposeTensorMove、TransposeSmallShape 等）均继承此基类。
 * 基类仅提供 reverseArray 工具方法，用于将 NDDMA 5维参数数组逆序排列。
 * NDDMA 硬件要求 loopSize/loopSrcStride/loopDstStride 以逆序排列（dim4→dim0），
 * 而 Host 侧 Tiling 计算以正序（dim0→dim4）生成，因此需要 Kernel 侧逆序。
 */
template <typename T>
class TransposeBase {
public:
    __aicore__ inline TransposeBase(){};

protected:
    /**
     * @brief 将5维数组元素逆序排列
     *
     * NDDMA 硬件参数（loopSize/loopSrcStride/loopDstStride）需要逆序排列，
     * 即 array[0] 对应 dim4，array[4] 对应 dim0。
     * 此函数将正序数组就地翻转为逆序。
     *
     * @tparam T1 数组元素类型
     * @param array 待逆序的5维数组
     */
    template <typename T1>
    __aicore__ inline void reverseArray(T1 array[])
    {
        for (uint8_t i = 0; i < NDDMA_MAX_DIM_NUM / 2; i++) {
            uint32_t temp = array[i];
            array[i] = array[NDDMA_MAX_DIM_NUM - 1 - i];
            array[NDDMA_MAX_DIM_NUM - 1 - i] = temp;
        }
    };
};

/**
 * @brief 计算当前核处理的多核范围（全局函数，所有策略共用）
 *
 * 将总循环数均匀分配到 realCoreNum 个核。前 blkTailFactor 个核多处理1个 block，
 * 确保负载均衡（总 block 数可能不被核数整除）。
 *
 * 负载均衡策略示例（假设 blkFactor=3, blkTailFactor=2, realCoreNum=4）：
 *   - 核0: 处理4个block (blkFactor + 1), 起始偏移 = 0*3 + 0 = 0
 *   - 核1: 处理4个block (blkFactor + 1), 起始偏移 = 1*3 + 1 = 4
 *   - 核2: 处理3个block (blkFactor),     起始偏移 = 2*3 + 2 = 8
 *   - 核3: 处理3个block (blkFactor),     起始偏移 = 3*3 + 2 = 11
 *
 * @param blockIdx         当前核索引
 * @param realCoreNum      实际使用核数
 * @param blkFactor        每核基础 block 数
 * @param blkTailFactor    尾核额外 block 数（前 blkTailFactor 个核多处理1个 block）
 * @param blkProcessNum    [out] 当前核需处理的 block 数
 * @param blkProcessIdxStart [out] 当前核起始 block 索引
 * @param blkProcessIdxEnd   [out] 当前核结束 block 索引（不含）
 * @return true 当前核有工作分配；false 当前核无需工作（blockIdx >= realCoreNum）
 */
__aicore__ inline bool ParseMultiCoreRange(int64_t blockIdx, int64_t realCoreNum, int64_t blkFactor,
                                           int64_t blkTailFactor, int64_t& blkProcessNum, int64_t& blkProcessIdxStart,
                                           int64_t& blkProcessIdxEnd)
{
    if (blockIdx >= realCoreNum) {
        return false; // 当前核无工作，直接返回
    }
    blkProcessNum = blkFactor;
    blkProcessIdxStart = blockIdx * blkFactor;
    if (blockIdx < blkTailFactor) {
        // 前 blkTailFactor 个核：多处理1个 block
        blkProcessNum += 1;
        blkProcessIdxStart += blockIdx; // 补偿前面核多处理的偏移
    } else {
        // 后续核：只处理 blkFactor 个 block
        blkProcessIdxStart += blkTailFactor; // 加上前面核多处理的总偏移
    }
    blkProcessIdxEnd = blkProcessIdxStart + blkProcessNum;
    return true;
}

} // namespace Transpose

#endif // TRANSPOSE_BASE_H
