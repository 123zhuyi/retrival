/**

Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
/*!

\file grouped_bias_add_grad_struct.h
\brief tiling data struct for grouped_bias_add_grad arch35
*/
#ifndef GROUPED_BIAS_ADD_GRAD_STRUCT_H
#define GROUPED_BIAS_ADD_GRAD_STRUCT_H

#include "atvoss/reduce/reduce_tiling_data.h"

/**

@brief TilingData struct for H-axis split (Template 2)
模版2：核切H轴
使用场景：核数 > 32，对H轴进行128Byte切分，按切分块数进行分核
UB空间分配：
groupedIdxSize: group_idx[G] 空间，32B对齐
outputSize: output[G, H] 空间，32B对齐
useUbSize: (可用UB - groupedIdxSize - outputSize - 256B预留) / 2 (double buffer)
*/
struct GroupedBiasAddGradCutHTilingData {
    int64_t blockFactor;     // 主核分配的大小，多少块H
    int64_t blockTailFactor; // 尾核分配的大小，多少块H
    int64_t groupIdxDim;     // G的大小
    int64_t inputShape[2];   // 输入shape [GB, H]
    int64_t hTailFactor;     // 尾部H的长度（最后一块的实际元素数）
    int64_t useUbSize;       // 可用UB空间大小（每个buffer）
    int64_t groupedIdxSize;  // group_idx空间大小[G]，32B对齐
    int64_t outputSize;      // output空间大小[G,H]，32B对齐
    int64_t maxOutputElements;
    int64_t useTempBuf;
    bool groupIdxType;       // 是否需要cumsum（由groupIdxType属性决定）
};
/**

@brief TilingData struct for G-axis split (Template 3)
模版3：核切G*H块
使用场景：核数 <= 32，按 G * (H轴128B切分块数) 进行均匀分核
分核策略：
优先使尾核处理块数相等（例如 3322 优于 3331）
申请workspace用于核间同步，大小为 group_idx[G] * 核数
UB空间分配与模版2相同
*/
struct GroupedBiasAddGradCutGTilingData {
    int64_t cutGDim;             // G的总数（核切分G的维度大小）
    int64_t cutHDim;             // H轴128B切分的块数
    int64_t blockFactor;         // 主核分配的块数
    int64_t blockTailFactor;     // 尾核分配的块数
    int64_t blockTailStartIndex; // 尾核的起始索引
    int64_t inputShape[2];       // 输入shape [GB, H]
    int64_t ubHTailFactor;       // UB尾部H长度（最后一块的实际元素数）
    int64_t useUbSize;           // 可用UB空间大小（每个buffer）
    int64_t groupedIdxSize;      // group_idx空间大小，32B对齐
    int64_t outputSize;          // output空间大小，32B对齐
    int64_t maxOutputElements;
    int64_t useTempBuf;
    bool groupIdxType;           // 是否需要cumsum（由groupIdxType属性决定）
};

struct GroupedBiasAddGradEmptyTensorTilingData {
    int64_t blockDim = 0;
};

struct GroupedBiasAddGradARATilingData {
    Ops::Base::ReduceOpTilingData reduceTiling;
};
/**

@brief Tiling mode enum for arch35
决定使用哪种tiling模式：
REDUCE_SUM_3D: 3D输入场景，使用reduceSum模版
CUT_H_MODE: 2D输入，核数>32，切分H轴（模版2）
CUT_G_MODE: 2D输入，核数<=32，切分G*H块（模版3）
*/
enum class GroupedBiasAddGradTilingModeArch35 : uint32_t
{
    IS_REDUCE_T = 0, // 3D input, use reduceSum template (not implemented yet)
    CUT_H_MODE = 1,  // 2D input, coreNum > 32, split H axis
    CUT_G_MODE = 2,  // 2D input, coreNum <= 32, split G * H blocks
    EMPTY_TENSOR = 3,
};
#endif // GROUPED_BIAS_ADD_GRAD_STRUCT_H