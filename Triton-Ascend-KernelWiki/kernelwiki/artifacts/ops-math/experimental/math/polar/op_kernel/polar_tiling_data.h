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
 * \file polar_tiling_data.h
 * \brief Polar tiling data struct（host 填写 / kernel 读取共用同一 struct）。
 *        两条路径：bcastMode=0 同 shape elementwise；bcastMode=1 angle inner-broadcast
 *        （angle 周期 K << input，trailing-dim 重复）—— 详见 op_kernel/polar.h。
 *        非 inner-bcast 的广播（中间轴/input 侧）仍由 aclnn L2 层 BroadcastTo + Contiguous 完成。
 */
#ifndef POLAR_TILING_DATA_H_
#define POLAR_TILING_DATA_H_

struct PolarTilingData {
    uint32_t totalLen;      // 输出元素总数
    uint32_t tileLen;       // 每 tile 元素数
    uint32_t bigCoreNum;    // 大核数量（前 bigCoreNum 个核各多处理 1 元素）
    uint32_t bigCoreLen;    // 大核分块长度
    uint32_t smallCoreLen;  // 小核分块长度
    uint32_t coreNum;       // 实际启用核数（偶数）
    uint32_t tmpBufferSize; // Sin/Cos 显式 sharedTmpBuffer 字节数（由 Get{Sin,Cos}MaxMinTmpSize 算）
    uint32_t inN;           // input.numel（bcastMode=1 时 = totalLen）
    uint32_t anN;           // angle.numel（bcastMode=1 时 = K，周期长度）
    uint32_t bcastMode;     // 0: same-shape；1: angle inner-broadcast
};

#endif // POLAR_TILING_DATA_H_
