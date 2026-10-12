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
 * \file mod_tiling_data.h
 * \brief tiling data struct
 */
#ifndef MOD_TILING_DATA_H
#define MOD_TILING_DATA_H

#include <cstdint>

namespace ModNs {

struct ModTilingData {
    uint32_t usableUbSize;
    uint32_t needCoreNum;
    uint64_t totalDataCount;
    uint64_t perCoreDataCount;
    uint64_t tailDataCoreNum;
    uint64_t lastCoreDataCount;
    bool isInput2Scalar;
    bool isInput2SameShape;
    uint32_t dimNum;
    uint64_t input1Shape[8];
    uint64_t input2Shape[8];
    uint64_t input2Stride[8];
    // K1 自适应路由阈值 (host 下发，默认 256，env FMOD_NAIVE_THRESH sweep 可覆盖)。
    float naiveThresh;
    // 融合广播 tiling 字段：仅被 arch22 kernel 的融合广播分支消费；所有非融合/非 arch22 路 bcastFusedMode=0
    //   (通用 ProcessBroadcast 不受影响)。融合字段追加在尾部 (sizeof 248 -> 288，含 0811 修复新增的 bcIpad)，
    //   对五种 ubDivider 的 usableUbSize 均不变 -> 非广播路 maxDataCount/tile 边界不变。
    uint32_t bcastFusedMode; // 0=off(通用), 1=OUTER 行广播(other=[1,INNER]), 2=INNER 列广播(other=[OUTER,1])
    uint64_t bcOuter;        // collapse 后 OUTER 维 (行数)
    uint64_t bcInner;        // collapse 后 INNER 维 (列数, 原始几何, 可非 32B 对齐)
    uint64_t bcUbFormer;     // 每 UB tile 的 OUTER 行数
    uint64_t bcBlockFactor;  // 每核的 OUTER 行数 (按 needCoreNum 切分, SetBlockDim 与通用路一致)
    // 0811 tile 塌陷修复：padding 行布局的行步长 = ceil(bcInner*sizeof(dtype)/32)*32/sizeof(dtype)
    //   (fp32 8-elem / 2B 16-elem 单位，与 DataCopyPad blockCount 模式自动 padding 落块规则一致)。
    //   bcIpad==bcInner 时退化 1D 平铺 (原对齐 case 行为不变)。UB 内 self/out/otherF32 三 buffer 统一按
    //   [rows, bcIpad] 排布；pad 车道消毒见 mod_bcast_impl.h (x1 priming 置 0 / x2 预填 1.0)。
    uint64_t bcIpad;
};

} // namespace ModNs

#endif // MOD_TILING_DATA_H
