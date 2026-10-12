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
 * \file col2_im.h
 * \brief
 */
#ifndef COL_2_IM_H
#define COL_2_IM_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "col2_im_tiling_data.h"
#include "col2_im_tiling_key.h"

namespace NsCol2Im {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Col2Im {
public:
    __aicore__ inline Col2Im(){};

    __aicore__ inline void Init(GM_ADDR col, GM_ADDR x, const Col2ImTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalculateCoreRange();
    __aicore__ inline void ComputeVectorized(int32_t progress, int32_t current_tile_size);
    __aicore__ inline void BatchLoadColData(
        AscendC::LocalTensor<T>& colLocal, int32_t pixelStart, int32_t tile_size, int32_t kh, int32_t kw,
        int32_t k_offset);
    __aicore__ inline int32_t CalculateColIndexForPixel(int32_t pixelIdx, int32_t kh, int32_t kw);
    __aicore__ inline void CopyOut(int32_t progress, int32_t current_tile_size);
    __aicore__ inline void PixelIndexToNCHW(int32_t pixelIdx, int32_t& n, int32_t& c, int32_t& h, int32_t& w);
    __aicore__ inline bool CalculateOutputPosition(
        int32_t h, int32_t w, int32_t kh, int32_t kw, int32_t& h_out, int32_t& w_out);
    __aicore__ inline int32_t CalculateColIndex(
        int32_t n, int32_t c, int32_t kh, int32_t kw, int32_t h_out, int32_t w_out);

private:
    TPipe pipe;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueX;
    TBuf<QuePosition::VECCALC> accumBuffer;
    TBuf<QuePosition::VECCALC> colDataBuffer;

    GlobalTensor<T> colGm;
    GlobalTensor<T> xGm;

    int32_t core_id;
    int32_t aligned_tile_size;

    // 在kernel中计算的核间划分参数
    int32_t pixel_start;
    int32_t pixel_count;
    int32_t loop_count;
    int32_t tail_pixels;

    Col2ImTilingData tiling;
};

template <typename T>
__aicore__ inline void Col2Im<T>::Init(GM_ADDR col, GM_ADDR x, const Col2ImTilingData& tilingData)
{
    this->tiling = tilingData;

    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    this->core_id = AscendC::GetBlockIdx();

    // 在kernel中计算当前core的像素范围
    CalculateCoreRange();

    if (this->pixel_count <= 0) {
        return;
    }

    colGm.SetGlobalBuffer((__gm__ T*)col, tiling.input_elements);
    xGm.SetGlobalBuffer((__gm__ T*)x, tiling.output_elements);

    this->aligned_tile_size = tiling.aligned_tile_size;

    // 初始化缓冲区
    pipe.InitBuffer(outQueueX, BUFFER_NUM, this->aligned_tile_size * sizeof(T));
    pipe.InitBuffer(accumBuffer, this->aligned_tile_size * sizeof(T));
    pipe.InitBuffer(colDataBuffer, tiling.kernel_h * tiling.kernel_w * this->aligned_tile_size * sizeof(T));
}

template <typename T>
__aicore__ inline void Col2Im<T>::CalculateCoreRange()
{
    int32_t pixels_per_core =
        (core_id < tiling.big_core_num) ? (tiling.base_pixels_per_core + 1) : tiling.base_pixels_per_core;

    // 计算起始位置
    if (core_id < tiling.big_core_num) {
        this->pixel_start = core_id * (tiling.base_pixels_per_core + 1);
    } else {
        this->pixel_start = tiling.big_core_num * (tiling.base_pixels_per_core + 1) +
                            (core_id - tiling.big_core_num) * tiling.base_pixels_per_core;
    }

    this->pixel_count = pixels_per_core;

    // 计算循环次数
    if (pixels_per_core > 0) {
        this->loop_count = (pixels_per_core + tiling.tile_pixel_num - 1) / tiling.tile_pixel_num;
        this->tail_pixels = pixels_per_core % tiling.tile_pixel_num;
        if (this->tail_pixels == 0) {
            this->tail_pixels = tiling.tile_pixel_num;
        }
    } else {
        this->loop_count = 0;
        this->tail_pixels = 0;
    }
}

template <typename T>
__aicore__ inline void Col2Im<T>::ComputeVectorized(int32_t progress, int32_t current_tile_size)
{
    AscendC::LocalTensor<T> xLocal = outQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> accumLocal = accumBuffer.Get<T>();
    AscendC::LocalTensor<T> colLocal = colDataBuffer.Get<T>();

    // 向量化初始化累加器为0
    T zero_val = static_cast<T>(0);
    AscendC::Duplicate(accumLocal, zero_val, this->aligned_tile_size);

    int32_t pixelStart = this->pixel_start + progress * tiling.tile_pixel_num;

    // 优化策略：批量预取col数据，然后向量化累加
    for (int32_t kh = 0; kh < tiling.kernel_h; kh++) {
        for (int32_t kw = 0; kw < tiling.kernel_w; kw++) {
            int32_t k_offset = kh * tiling.kernel_w + kw;

            // 批量加载当前kernel位置的col数据
            BatchLoadColData(colLocal, pixelStart, current_tile_size, kh, kw, k_offset);

            // 向量化累加：accumLocal += colLocal[k_offset的部分]
            int32_t col_offset = k_offset * this->aligned_tile_size;
            AscendC::LocalTensor<T> colSlice = colLocal[col_offset];

            if constexpr (std::is_same_v<T, T>) {
                // 类型相同，直接向量加法
                AscendC::Add(accumLocal, accumLocal, colSlice, this->aligned_tile_size);
            } else {
                // 类型转换后累加
                AscendC::LocalTensor<T> colConverted;
                AscendC::Cast(colConverted, colSlice, AscendC::RoundMode::CAST_NONE, this->aligned_tile_size);
                AscendC::Add(accumLocal, accumLocal, colConverted, this->aligned_tile_size);
            }
        }
    }

    // 向量化复制结果
    AscendC::DataCopy(xLocal, accumLocal, this->aligned_tile_size);

    colDataBuffer.FreeTensor(colLocal);
    accumBuffer.FreeTensor(accumLocal);
    outQueueX.EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void Col2Im<T>::BatchLoadColData(
    AscendC::LocalTensor<T>& colLocal, int32_t pixelStart, int32_t tile_size, int32_t kh, int32_t kw, int32_t k_offset)
{
    int32_t col_base_offset = k_offset * this->aligned_tile_size;

    // 向量化处理：按向量长度批量加载
    int32_t vec_len = tiling.vec_elements;
    int32_t full_vectors = tile_size / vec_len;
    int32_t remainder = tile_size % vec_len;

    // 处理完整向量
    for (int32_t vec_idx = 0; vec_idx < full_vectors; vec_idx++) {
        int32_t local_offset = vec_idx * vec_len;

        // 批量计算该向量内所有像素的col索引
        for (int32_t i = 0; i < vec_len; i++) {
            int32_t pixelIdx = pixelStart + local_offset + i;

            if (pixelIdx < tiling.total_output_pixels) {
                int32_t colIdx = CalculateColIndexForPixel(pixelIdx, kh, kw);

                if (colIdx >= 0 && colIdx < tiling.input_elements) {
                    T value = colGm.GetValue(colIdx);
                    colLocal.SetValue(col_base_offset + local_offset + i, value);
                } else {
                    colLocal.SetValue(col_base_offset + local_offset + i, static_cast<T>(0));
                }
            } else {
                colLocal.SetValue(col_base_offset + local_offset + i, static_cast<T>(0));
            }
        }
    }

    // 处理剩余元素
    if (remainder > 0) {
        int32_t local_offset = full_vectors * vec_len;
        for (int32_t i = 0; i < remainder; i++) {
            int32_t pixelIdx = pixelStart + local_offset + i;

            if (pixelIdx < tiling.total_output_pixels) {
                int32_t colIdx = CalculateColIndexForPixel(pixelIdx, kh, kw);

                if (colIdx >= 0 && colIdx < tiling.input_elements) {
                    T value = colGm.GetValue(colIdx);
                    colLocal.SetValue(col_base_offset + local_offset + i, value);
                } else {
                    colLocal.SetValue(col_base_offset + local_offset + i, static_cast<T>(0));
                }
            } else {
                colLocal.SetValue(col_base_offset + local_offset + i, static_cast<T>(0));
            }
        }
    }
}

template <typename T>
__aicore__ inline int32_t Col2Im<T>::CalculateColIndexForPixel(int32_t pixelIdx, int32_t kh, int32_t kw)
{
    // 解析像素坐标
    int32_t n, c, h, w;
    PixelIndexToNCHW(pixelIdx, n, c, h, w);

    // 计算输出位置
    int32_t h_out, w_out;
    if (!CalculateOutputPosition(h, w, kh, kw, h_out, w_out)) {
        return -1;
    }

    // 计算col索引
    return CalculateColIndex(n, c, kh, kw, h_out, w_out);
}

template <typename T>
__aicore__ inline void Col2Im<T>::PixelIndexToNCHW(int32_t pixelIdx, int32_t& n, int32_t& c, int32_t& h, int32_t& w)
{
    int32_t HW = tiling.H * tiling.W;
    int32_t CHW = tiling.C * HW;

    n = pixelIdx / CHW;
    int32_t remaining = pixelIdx % CHW;
    c = remaining / HW;
    remaining = remaining % HW;
    h = remaining / tiling.W;
    w = remaining % tiling.W;
}

template <typename T>
__aicore__ inline bool Col2Im<T>::CalculateOutputPosition(
    int32_t h, int32_t w, int32_t kh, int32_t kw, int32_t& h_out, int32_t& w_out)
{
    int32_t kh_dilated = kh * tiling.dilation_val;
    int32_t kw_dilated = kw * tiling.dilation_val;

    int32_t h_numerator = h + tiling.padding_val - kh_dilated;
    int32_t w_numerator = w + tiling.padding_val - kw_dilated;

    if (h_numerator % tiling.stride_val != 0 || w_numerator % tiling.stride_val != 0) {
        return false;
    }

    h_out = h_numerator / tiling.stride_val;
    w_out = w_numerator / tiling.stride_val;

    return (h_out >= 0 && h_out < tiling.out_H && w_out >= 0 && w_out < tiling.out_W);
}

template <typename T>
__aicore__ inline int32_t Col2Im<T>::CalculateColIndex(
    int32_t n, int32_t c, int32_t kh, int32_t kw, int32_t h_out, int32_t w_out)
{
    int32_t channel_block = c * (tiling.kernel_h * tiling.kernel_w) + kh * tiling.kernel_w + kw;
    int32_t pos_idx = h_out * tiling.out_W + w_out;
    return n * (tiling.input_channels * tiling.L) + channel_block * tiling.L + pos_idx;
}

template <typename T>
__aicore__ inline void Col2Im<T>::CopyOut(int32_t progress, int32_t current_tile_size)
{
    AscendC::LocalTensor<T> xLocal = outQueueX.DeQue<T>();

    int32_t pixelOffset = this->pixel_start + progress * tiling.tile_pixel_num;

    if (current_tile_size > 0) {
        // 使用向量化拷贝
        uint32_t copy_elements =
            ((current_tile_size + tiling.vec_elements - 1) / tiling.vec_elements) * tiling.vec_elements;
        copy_elements = (copy_elements > this->aligned_tile_size) ? this->aligned_tile_size : copy_elements;

        uint32_t bytes_to_copy = copy_elements * sizeof(T);

        if (bytes_to_copy % 32 == 0) {
            AscendC::DataCopy(xGm[pixelOffset], xLocal, copy_elements);
        } else {
            AscendC::DataCopyExtParams copyParams{1, bytes_to_copy, 0, 0, 0};
            AscendC::DataCopyPad(xGm[pixelOffset], xLocal, copyParams);
        }
    }

    outQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Col2Im<T>::Process()
{
    for (int32_t i = 0; i < this->loop_count; i++) {
        int32_t current_tile_size = (i == this->loop_count - 1) ? this->tail_pixels : tiling.tile_pixel_num;

        ComputeVectorized(i, current_tile_size);
        CopyOut(i, current_tile_size);
    }
}

} // namespace NsCol2Im
#endif // COL_2_IM_H