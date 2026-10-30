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
 * \file stride_add_simt.h
 * \brief StrideAdd SIMT kernel implementation
 */

#ifndef __STRIDE_ADD_SIMT_H__
#define __STRIDE_ADD_SIMT_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_bf16.h"
#include "stride_add_tiling_data.h"
#include "stride_add_tiling_key.h"

namespace NsStrideAdd {

using namespace AscendC;

// R006: INDEX_SIZE_T 位宽 → 线程数映射
// uint32_t路径：寄存器压力减半，可开 1024 线程提升带宽利用率
// uint64_t路径：寄存器压力大，保持 512 线程避免溢出
constexpr uint32_t INDEX_SIZE_32BIT_BYTES = 4;
constexpr uint32_t THREAD_NUM_32BIT = 1024;
constexpr uint32_t THREAD_NUM_64BIT = 512;

// R003+R006: INDEX_SIZE_T → 线程数映射规则
// uint32_t路径：寄存器压力减半，可开1024线程提升带宽利用率
// uint64_t路径：寄存器压力大，保持512线程避免溢出
// 注：__aicore__ 上下文禁止调用 constexpr host 函数，线程数用 sizeof 内联表达式计算

template <typename T, typename INDEX_SIZE_T = uint64_t>
__simt_vf__ __aicore__ __launch_bounds__(sizeof(INDEX_SIZE_T) <= INDEX_SIZE_32BIT_BYTES ? THREAD_NUM_32BIT : THREAD_NUM_64BIT)
inline void OpStrideAddSimtKernel(
    INDEX_SIZE_T totalElements, INDEX_SIZE_T perCoreElements,
    INDEX_SIZE_T hwC0Size, INDEX_SIZE_T c1Len,
    INDEX_SIZE_T hwC0Magic, INDEX_SIZE_T hwC0Shift,
    INDEX_SIZE_T c1LenMagic, INDEX_SIZE_T c1LenShift,
    INDEX_SIZE_T x1NStride, INDEX_SIZE_T x2NStride,
    int32_t x1C1Offset, int32_t x2C1Offset,
    __gm__ T* x1, __gm__ T* x2, __gm__ T* y)
{
    // 每核起始和结束索引
    INDEX_SIZE_T coreStart = static_cast<INDEX_SIZE_T>(blockIdx.x) * perCoreElements;
    INDEX_SIZE_T coreEnd = coreStart + perCoreElements;
    if (coreEnd > totalElements) {
        coreEnd = totalElements;
    }

    // Grid-Stride 循环（block 内线程按 stride 间隔遍历）
    // R003: 索引变量模板化为 INDEX_SIZE_T，uint32路径寄存器压力减半
    for (INDEX_SIZE_T idx = coreStart + static_cast<INDEX_SIZE_T>(threadIdx.x);
         idx < coreEnd;
         idx += static_cast<INDEX_SIZE_T>(blockDim.x))
    {
        // R002: 用 UintDiv 替代 VF 内无符号定数除法，被除数先转为无符号再快除
        // 坐标分解: idx → (n_coord, c_coord, inner_idx)
        // idx / hwC0Size → outerIdx
        INDEX_SIZE_T outerIdx = Simt::UintDiv<INDEX_SIZE_T>(idx, hwC0Magic, hwC0Shift);
        // idx % hwC0Size → innerIdx = idx - outerIdx * hwC0Size
        INDEX_SIZE_T innerIdx = idx - outerIdx * hwC0Size;

        // outerIdx / c1Len → nCoord
        INDEX_SIZE_T nCoord = Simt::UintDiv<INDEX_SIZE_T>(outerIdx, c1LenMagic, c1LenShift);
        // outerIdx % c1Len → cCoord = outerIdx - nCoord * c1Len
        INDEX_SIZE_T cCoord = outerIdx - nCoord * c1Len;

        // 计算输入地址偏移
        INDEX_SIZE_T x1Offset = nCoord * x1NStride
            + (static_cast<INDEX_SIZE_T>(x1C1Offset) + cCoord) * hwC0Size + innerIdx;
        INDEX_SIZE_T x2Offset = nCoord * x2NStride
            + (static_cast<INDEX_SIZE_T>(x2C1Offset) + cCoord) * hwC0Size + innerIdx;

        // 读取、计算、写入
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            // bf16 路径: 类型提升 bf16→fp32→add→bf16
            float val1 = __bfloat162float(x1[x1Offset]);
            float val2 = __bfloat162float(x2[x2Offset]);
            y[idx] = __float2bfloat16(val1 + val2);
        } else {
            // fp16/fp32 路径: 直接运算符加法
            y[idx] = x1[x1Offset] + x2[x2Offset];
        }
    }
}

template <typename T, typename INDEX_SIZE_T = uint64_t>
__aicore__ inline void Process(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling,
    const StrideAddTilingData* tilingData)
{
    __gm__ T* x1_gm = (__gm__ T*) x1;
    __gm__ T* x2_gm = (__gm__ T*) x2;
    __gm__ T* y_gm = (__gm__ T*) y;

    // R002: 在 __aicore__ 作用域预计算 UintDiv 参数（scalar 指令，禁止在 VF 内调用）
    // R003: UintDiv 参数类型随 INDEX_SIZE_T 模板化
    INDEX_SIZE_T hwC0Magic = 0;
    INDEX_SIZE_T hwC0Shift = 0;
    GetUintDivMagicAndShift<INDEX_SIZE_T>(hwC0Magic, hwC0Shift,
                                           static_cast<INDEX_SIZE_T>(tilingData->hwC0Size));

    INDEX_SIZE_T c1LenMagic = 0;
    INDEX_SIZE_T c1LenShift = 0;
    GetUintDivMagicAndShift<INDEX_SIZE_T>(c1LenMagic, c1LenShift,
                                           static_cast<INDEX_SIZE_T>(tilingData->c1Len));

    // R006: 线程数按 INDEX_SIZE_T 位宽编译期选择（sizeof 在 __aicore__ 上下文可用）
    const uint32_t threadNum = sizeof(INDEX_SIZE_T) <= INDEX_SIZE_32BIT_BYTES ? THREAD_NUM_32BIT : THREAD_NUM_64BIT;

    asc_vf_call<OpStrideAddSimtKernel<T, INDEX_SIZE_T>>(
        dim3(threadNum),
        static_cast<INDEX_SIZE_T>(tilingData->totalElements),
        static_cast<INDEX_SIZE_T>(tilingData->perCoreElements),
        static_cast<INDEX_SIZE_T>(tilingData->hwC0Size),
        static_cast<INDEX_SIZE_T>(tilingData->c1Len),
        hwC0Magic,
        hwC0Shift,
        c1LenMagic,
        c1LenShift,
        static_cast<INDEX_SIZE_T>(tilingData->x1NStride),
        static_cast<INDEX_SIZE_T>(tilingData->x2NStride),
        tilingData->x1C1Offset,
        tilingData->x2C1Offset,
        x1_gm, x2_gm, y_gm
    );
}

} // namespace NsStrideAdd

#endif // __STRIDE_ADD_SIMT_H__