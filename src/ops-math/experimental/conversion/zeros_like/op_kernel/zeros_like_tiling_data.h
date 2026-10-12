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
 * \file zeros_like_tiling_data.h
 * \brief ascend910b (DAV_2201) 标准 AscendC kernel 与 host tiling 共享的 TilingData 结构。
 *        纯写 0 的退化 Elementwise：按总字节数多核切分，无 CopyIn，单块零缓冲循环写出。
 */
#ifndef ZEROS_LIKE_TILING_DATA_H
#define ZEROS_LIKE_TILING_DATA_H

#include <cstdint>

namespace ZerosLikeNs {

// host tiling 与 kernel 共享的 32B 对齐基本块字节数（多核切分 / UB 切分的对齐粒度）。
// 单一定义点：host (zeros_like_tiling.cpp) 与 kernel (zeros_like.cpp) 均包含本头文件，
// 避免两侧各自重复定义导致取值不一致。host 侧与 uint64 混合运算时会按需提升，安全。
constexpr uint32_t ZL_BLOCK_BYTES = 32;

// host / kernel 共享的普通 C++ struct（不使用 BEGIN_TILING_DATA_DEF 宏）。
// 所有切分以「字节」为单位描述，dtype 仅决定 Duplicate 视图与每元素字节数。
struct ZerosLikeTilingData {
    uint64_t totalBytes;   // 输出总字节数（= 元素总数 × bytesPerElem）
    uint64_t perCoreBytes; // 每核基础字节数（按 32B 对齐块均分，前 tailCoreNum 核多 1 块）
    uint64_t tailCoreNum;  // 前 tailCoreNum 个核各多 1 个 32B 块
    uint64_t tileBytes;    // 单次 UB 处理字节数（32B 对齐）
    uint32_t usedCoreNum;  // 实际使用核数 = SetBlockDim
    uint32_t bytesPerElem; // 归一字节宽度 1/2/4/8
};

} // namespace ZerosLikeNs

#endif // ZEROS_LIKE_TILING_DATA_H
