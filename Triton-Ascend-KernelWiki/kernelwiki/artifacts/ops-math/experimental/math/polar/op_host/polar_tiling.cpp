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
 * \file polar_tiling.cpp
 * \brief Polar Tiling —— 同 shape elementwise：仅做 totalLen 计算 + 分核（小 case 少核、强制偶数核）。
 *        广播由 op_api 层 BroadcastTo + Contiguous 完成；kernel 收到的 input/angle 已是同 shape。
 */
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "lib/math/sin_tiling.h" // GetSinMaxMinTmpSize
#include "lib/math/cos_tiling.h" // GetCosMaxMinTmpSize
#include "../op_kernel/polar_tiling_data.h"
#include "../op_kernel/polar_tiling_key.h"

namespace optiling {

static constexpr uint32_t POLAR_TILE_LEN = 2048; // 每 tile fp32 元素数
static constexpr uint32_t MIN_PER_CORE = 2048u; // 每核最少元素数（小 case 收敛到少核，降 launch 开销）

static uint64_t Numel(const gert::Shape& s)
{
    uint64_t n = 1;
    uint32_t r = static_cast<uint32_t>(s.GetDimNum());
    for (uint32_t i = 0; i < r; i++) {
        n *= (uint64_t)s.GetDim(i);
    }
    return n == 0 ? 1 : n;
}

// angle inner-broadcast 判定（与 op_api/aclnn_polar.cpp 的判定保持一致）：
//   inN==totalLen, anN<inN, inN%anN==0, anN<=tileLen, anN%8==0（32B 对齐）
static bool IsAngleInnerBcast(uint64_t inN, uint64_t anN, uint64_t totalLen)
{
    return inN == totalLen && anN > 0 && anN < inN && (inN % anN) == 0 && anN <= POLAR_TILE_LEN && (anN % 8u) == 0u;
}

// 核数收敛：小 case 少核（want 上限）+ 强制偶数（>2 时下取偶）
static uint32_t ClampWantEven(uint32_t coreNum, uint32_t want)
{
    if (want < 1u)
        want = 1u;
    if (coreNum > want)
        coreNum = want;
    if (coreNum > 2u && (coreNum & 1u))
        coreNum -= 1u;
    return coreNum;
}

// bcastMode=1 K-block 分核：每核负载是 K 整数倍 → coreStart % K == 0，tile 起点 period 对齐
static void SplitCoresBcast(PolarTilingData* tiling, uint64_t totalLen, uint64_t anN, uint32_t& coreNum)
{
    uint32_t K = (uint32_t)anN;
    uint64_t totalBlocks = totalLen / K;
    coreNum = ClampWantEven(coreNum, (uint32_t)totalBlocks);
    if (coreNum == 0)
        coreNum = 1;
    uint64_t perBlk = totalBlocks / coreNum;
    uint32_t remBlk = (uint32_t)(totalBlocks % coreNum);
    tiling->bigCoreNum = remBlk;
    tiling->bigCoreLen = (uint32_t)((perBlk + 1) * (uint64_t)K);
    tiling->smallCoreLen = (uint32_t)(perBlk * (uint64_t)K);
}

// 同 shape elementwise 分核：核数 ① 平台 AIV ② 小 case 少核 ③ 强制偶数，且不超过 totalLen
static void SplitCoresElementwise(PolarTilingData* tiling, uint64_t totalLen, uint32_t& coreNum)
{
    coreNum = ClampWantEven(coreNum, (uint32_t)(totalLen / MIN_PER_CORE));
    if ((uint64_t)coreNum > totalLen)
        coreNum = totalLen > 0 ? (uint32_t)totalLen : 1u;
    if (coreNum == 0)
        coreNum = 1;
    uint64_t per = totalLen / coreNum;
    uint32_t rem = (uint32_t)(totalLen % coreNum);
    tiling->bigCoreNum = rem;
    tiling->bigCoreLen = (uint32_t)(per + 1);
    tiling->smallCoreLen = (uint32_t)per;
}

// Sin/Cos 显式 sharedTmpBuffer 大小：按 tileLen 元素一次性处理（fp32 noreuse，需 3×N×typeSize 字节）。
// ★ 取 maxValue（充足）而非 minValue：minValue 仅"勉强够用"，Sin/Cos 会按小 buffer 分块多趟计算，
//   大 same-shape case 性能暴跌（实测 16M same-shape: cosMin 5800μs → cosMax ~650μs，~9x）。
//   UB 余量充足（24KB << 192KB），用 maxValue 让 Sin/Cos 一次算完整 tile；再 clamp 到 UB 上限防越界。
static uint32_t CalcSinCosTmpSize(platform_ascendc::PlatformAscendC& plat)
{
    ge::Shape tileShape({(int64_t)POLAR_TILE_LEN});
    uint32_t cosMax = 0, cosMin = 0;
    AscendC::GetCosMaxMinTmpSize(tileShape, 4u, /*isReuseSource=*/false, cosMax, cosMin);
    uint64_t ubSize = 0;
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t tmpSize = cosMax;
    if ((uint64_t)tmpSize > ubSize)
        tmpSize = (uint32_t)ubSize;
    return tmpSize;
}

static ge::graphStatus PolarTilingFunc(gert::TilingContext* context)
{
    PolarTilingData* tiling = context->GetTilingData<PolarTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(context, "tiling is null"), return ge::GRAPH_FAILED);

    // input/out 同 shape（aclnn 层把 input BroadcastTo out）；angle inner-bcast 时保留原 shape [K]
    auto inShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShapePtr);
    auto anShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, anShapePtr);
    uint64_t inN = Numel(inShapePtr->GetStorageShape());
    uint64_t anN = Numel(anShapePtr->GetStorageShape());
    uint64_t totalLen = inN > anN ? inN : anN;
    uint32_t bcastMode = IsAngleInnerBcast(inN, anN, totalLen) ? 1u : 0u;

    auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = plat.GetCoreNumAiv();
    if (coreNum == 0)
        coreNum = 8;

    if (bcastMode == 1) {
        SplitCoresBcast(tiling, totalLen, anN, coreNum);
    } else {
        SplitCoresElementwise(tiling, totalLen, coreNum);
    }

    tiling->totalLen = (uint32_t)totalLen;
    tiling->tileLen = POLAR_TILE_LEN;
    tiling->coreNum = coreNum;
    tiling->inN = (uint32_t)inN;
    tiling->anN = (uint32_t)anN;
    tiling->bcastMode = bcastMode;
    tiling->tmpBufferSize = CalcSinCosTmpSize(plat);

    context->SetBlockDim(coreNum);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

struct PolarCompileInfo {};

static ge::graphStatus TilingParseForPolar([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Polar).Tiling(PolarTilingFunc).TilingParse<PolarCompileInfo>(TilingParseForPolar);
} // namespace optiling
