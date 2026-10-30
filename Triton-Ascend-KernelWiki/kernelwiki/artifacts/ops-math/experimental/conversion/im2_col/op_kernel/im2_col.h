/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Qiu Zhuang <@qiu-zhuang>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file im2_col.h
 * \brief
 */
#ifndef IM_2_COL_H
#define IM_2_COL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "im2_col_tiling_data.h"
#include "im2_col_tiling_key.h"

namespace NsIm2Col {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Im2Col {
public:
    __aicore__ inline Im2Col(){};

    __aicore__ inline uint32_t AlignUp(uint32_t a, uint32_t b);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, const Im2ColTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalculateCoreAllocation();
    __aicore__ inline void CopyIn(int32_t progress, int32_t current_tile_size);
    __aicore__ inline void Compute(int32_t progress, int32_t current_tile_size);
    __aicore__ inline void CopyOut(int32_t progress, int32_t current_tile_size);
    __aicore__ inline int32_t CalculateInputIndex(int32_t outputIdx);

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;

    GlobalTensor<T> xGm;
    GlobalTensor<T> zGm;

    int32_t core_id;
    int32_t aligned_tile_size;

    // 动态计算的核间划分参数
    int32_t core_element_start;
    int32_t core_element_end;
    int32_t core_element_count;
    int32_t loop_times;
    int32_t tail_elements;

    Im2ColTilingData tiling;
};

template <typename T>
__aicore__ inline uint32_t Im2Col<T>::AlignUp(uint32_t a, uint32_t b)
{
    if (b == 0)
        return a;
    return ((a + 32 / b - 1) / (32 / b)) * (32 / b);
}

template <typename T>
__aicore__ inline void Im2Col<T>::Init(GM_ADDR x, GM_ADDR z, const Im2ColTilingData& tilingData)
{
    this->tiling = tilingData;
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    this->core_id = AscendC::GetBlockIdx();

    // 在kernel中动态计算核间划分参数
    CalculateCoreAllocation();

    if (this->core_element_count <= 0) {
        return;
    }

    // 设置GM地址
    xGm.SetGlobalBuffer((__gm__ T*)x, tiling.input_elements);
    zGm.SetGlobalBuffer((__gm__ T*)z + this->core_element_start, this->core_element_count);

    // 初始化管道和队列
    this->aligned_tile_size = AlignUp((uint32_t)tiling.tile_element_num, sizeof(T));
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->aligned_tile_size * sizeof(T));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->aligned_tile_size * sizeof(T));
}

template <typename T>
__aicore__ inline void Im2Col<T>::CalculateCoreAllocation()
{
    int32_t base = tiling.base_elements_per_core;
    int32_t big_core_num = tiling.big_core_num;

    if (this->core_id < big_core_num) {
        // 大核
        this->core_element_count = base + 1;
        this->core_element_start = this->core_id * (base + 1);
    } else {
        // 小核
        this->core_element_count = base;
        this->core_element_start = big_core_num * (base + 1) + (this->core_id - big_core_num) * base;
    }

    this->core_element_end = this->core_element_start + this->core_element_count - 1;
}

template <typename T>
__aicore__ inline void Im2Col<T>::CopyIn(int32_t progress, int32_t current_tile_size)
{
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();

    // 计算输入数据范围 - 预先gather可能需要的输入数据
    int32_t outputStart = this->core_element_start + progress * tiling.tile_element_num;

    // 简化处理：为当前tile的所有输出元素预取可能需要的输入
    int32_t max_input_range = current_tile_size * (tiling.kernel_h * tiling.kernel_w);

    // 由于im2col的复杂映射关系，这里采用按需读取策略
    // 将xLocal作为临时缓存使用
    AscendC::Duplicate(xLocal, static_cast<T>(0.0f), this->aligned_tile_size);

    inQueueX.EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void Im2Col<T>::CopyOut(int32_t progress, int32_t current_tile_size)
{
    AscendC::LocalTensor<T> zLocal = outQueueZ.DeQue<T>();

    int32_t outputOffset = progress * tiling.tile_element_num;

    // 边界检查
    int32_t remaining = this->core_element_count - outputOffset;
    if (current_tile_size > remaining) {
        current_tile_size = remaining;
    }

    // 向量化拷贝
    uint32_t bytes_to_copy = current_tile_size * sizeof(T);
    AscendC::DataCopyExtParams copyParams{1, bytes_to_copy, 0, 0, 0};

    AscendC::DataCopyPad(zGm[outputOffset], zLocal, copyParams);

    outQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void Im2Col<T>::Compute(int32_t progress, int32_t current_tile_size)
{
    AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

    int32_t outputStart = this->core_element_start + progress * tiling.tile_element_num;

    // 使用向量化处理
    int32_t vec_size = 8; // 向量长度
    int32_t vec_loop = current_tile_size / vec_size;
    int32_t vec_tail = current_tile_size % vec_size;

    // 向量化循环处理
    for (int32_t v = 0; v < vec_loop; v++) {
        int32_t base_idx = v * vec_size;

// 批量计算8个输出索引对应的输入索引
#pragma unroll
        for (int32_t i = 0; i < vec_size; i++) {
            int32_t outputIdx = outputStart + base_idx + i;

            if (outputIdx >= tiling.total_output_elements) {
                zLocal.SetValue(base_idx + i, static_cast<T>(0.0f));
                continue;
            }

            // 计算输入索引
            int32_t inputIdx = CalculateInputIndex(outputIdx);

            if (inputIdx >= 0 && inputIdx < tiling.input_elements) {
                T value = xGm.GetValue(inputIdx);
                zLocal.SetValue(base_idx + i, static_cast<T>(value));
            } else {
                zLocal.SetValue(base_idx + i, static_cast<T>(0.0f));
            }
        }
    }

    // 处理尾部
    for (int32_t i = 0; i < vec_tail; i++) {
        int32_t idx = vec_loop * vec_size + i;
        int32_t outputIdx = outputStart + idx;

        if (outputIdx >= tiling.total_output_elements) {
            zLocal.SetValue(idx, static_cast<T>(0.0f));
            continue;
        }

        int32_t inputIdx = CalculateInputIndex(outputIdx);

        if (inputIdx >= 0 && inputIdx < tiling.input_elements) {
            T value = xGm.GetValue(inputIdx);
            zLocal.SetValue(idx, static_cast<T>(value));
        } else {
            zLocal.SetValue(idx, static_cast<T>(0.0f));
        }
    }

    inQueueX.FreeTensor(xLocal);
    outQueueZ.EnQue<T>(zLocal);
}

template <typename T>
// 计算输出索引对应的输入索引
__aicore__ inline int32_t Im2Col<T>::CalculateInputIndex(int32_t outputIdx)
{
    // 输出索引 -> [N, output_channels, L]
    int32_t n_idx = outputIdx / (tiling.output_channels * tiling.L);
    int32_t remaining = outputIdx % (tiling.output_channels * tiling.L);
    int32_t channel_block = remaining / tiling.L;
    int32_t pos_idx = remaining % tiling.L;

    // 分解channel_block -> (c, kh, kw)
    int32_t kernel_area = tiling.kernel_h * tiling.kernel_w;
    int32_t c_idx = channel_block / kernel_area;
    int32_t kernel_idx = channel_block % kernel_area;
    int32_t kh_idx = kernel_idx / tiling.kernel_w;
    int32_t kw_idx = kernel_idx % tiling.kernel_w;

    // 分解pos_idx -> (out_h, out_w)
    int32_t out_h = pos_idx / tiling.out_W;
    int32_t out_w = pos_idx % tiling.out_W;

    // 计算输入位置（考虑dilation）
    int32_t in_h = out_h * tiling.stride_h + kh_idx * tiling.dilation_h - tiling.pad_h;
    int32_t in_w = out_w * tiling.stride_w + kw_idx * tiling.dilation_w - tiling.pad_w;

    // 边界检查
    if (in_h < 0 || in_h >= tiling.H || in_w < 0 || in_w >= tiling.W) {
        return -1;
    }

    // 计算输入索引: [N, C, H, W]
    return n_idx * (tiling.C * tiling.H * tiling.W) + c_idx * (tiling.H * tiling.W) + in_h * tiling.W + in_w;
}

template <typename T>
__aicore__ inline void Im2Col<T>::Process()
{
    if (this->core_element_count <= 0) {
        return;
    }

    // 计算循环次数
    this->loop_times = (this->core_element_count + tiling.tile_element_num - 1) / tiling.tile_element_num;
    this->tail_elements = this->core_element_count % tiling.tile_element_num;
    if (this->tail_elements == 0) {
        this->tail_elements = tiling.tile_element_num;
    }

    for (int32_t i = 0; i < this->loop_times; i++) {
        int32_t current_tile_size = (i == this->loop_times - 1) ? this->tail_elements : tiling.tile_element_num;
        CopyIn(i, current_tile_size);
        Compute(i, current_tile_size);
        CopyOut(i, current_tile_size);
    }
}

} // namespace NsIm2Col
#endif // IM_2_COL_H