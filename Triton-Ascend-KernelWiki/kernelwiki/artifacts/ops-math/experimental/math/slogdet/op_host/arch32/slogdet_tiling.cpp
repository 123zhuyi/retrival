/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file slogdet_tiling.cpp
 * \brief Slogdet Tiling 实现（ascend910b / DAV_2201）。
 *
 * 现代 registry-invoke 规范：
 *   - 普通 C++ struct SlogdetTilingData，通过 context->GetTilingData<>() 写入；
 *   - ASCENDC_TPL_SEL_PARAM(context, dtypeVal, memStrategy) 选择模板参数（不使用 SetTilingKey 整型常量）。
 *
 * 支持 FULL 路径（MEM_STRATEGY=0）和 BLOCKED 路径（MEM_STRATEGY=1）。
 * 平台核数 / UB 容量通过 platform API 运行时获取（GetCoreNumAiv / GetCoreMemSize），禁止写死。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/slogdet_tiling_data.h"
#include "../../op_kernel/slogdet_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 32U;            // DataBlock 字节数（32B 对齐基本单位）
constexpr uint64_t WS_SYS_SIZE = 16ULL * 1024U * 1024U;  // 16MB 系统 workspace（沿用 cholesky 线代算子约定）
constexpr uint32_t UB_RESERVE = 8U * 1024U;     // UB 系统保留（约 8KB）
constexpr uint32_t MEM_STRATEGY_FULL = 0U;      // 全驻留
constexpr uint32_t MEM_STRATEGY_BLOCKED = 1U;   // 核内分块（large-n，U 常驻 GM workspace）
constexpr uint32_t FULL_RESIDENT_FALLBACK = 256U;  // ResolveResidentMax 兜底上界（UB 信息异常时）
constexpr uint32_t COL_BLOCK = 64U;             // BLOCKED 列块宽度（写入 tiling blockSize 占位字段，kernel 未消费）
// BLOCKED 列 gather 的 DataCopyPad blockCount = 子列长度 m=n-k（最大 ≈ n），uint16 取值范围 [1,4095]
//   （asc-devkit DataCopyPad(ISASI).md 实测约束）。故 BLOCKED 路径安全 n 上界 = 4095。
//   超出则 blockCount 静默越界（读错列长度）→ 必须 host 拒绝（MED-1）。文档 N 上界同步收窄至此值。
constexpr uint32_t SLOGDET_MAX_N = 4095U;
// ResolveResidentMax UB 预留裕量系数（LOW-3）：在容量约束上再乘 0.95，避免 residentMax 边界 FULL 路径
//   UB 余量过薄（原边界仅 ~864B），降低后续新增 buffer/UB_RESERVE 估计偏差导致的临界溢出风险。
constexpr double UB_MARGIN_RATIO = 0.95;

// 32B 对齐后的字节数
static inline uint64_t Align32(uint64_t bytes)
{
    return ((bytes + BLOCK_SIZE - 1U) / BLOCK_SIZE) * BLOCK_SIZE;
}

// 由运行时 UB 容量推导单矩阵全驻留可行的最大 n（解 n*align(n*4,32) + 常量 buffer ≤ 可用 UB * 裕量系数）。
// FULL 路径 buffer 估算（fp32）：uWork[n*align(n*4,32)] + col + absCol + diag(≈align(n*4,32) 三块)
//   + sharedTmp(32B) + 结果(64B)。保守按 uWork + 4*align(n*4,32) + 96B 计。
static uint32_t ResolveResidentMax(uint64_t ubSize)
{
    if (ubSize <= UB_RESERVE) {
        return FULL_RESIDENT_FALLBACK;
    }
    // 预留裕量（LOW-3）：实际可用按 (ubSize-UB_RESERVE)*0.95 计，避免边界 n 处 UB 余量过薄。
    const uint64_t avail = static_cast<uint64_t>(
        static_cast<double>(ubSize - UB_RESERVE) * UB_MARGIN_RATIO);
    // 线性扫描求满足容量约束的最大 n（n 通常 <= 数百，开销可忽略）
    uint32_t best = 1U;
    for (uint32_t n = 1U; n <= 4096U; ++n) {
        const uint64_t rowBytes = Align32(static_cast<uint64_t>(n) * sizeof(float));
        const uint64_t total = static_cast<uint64_t>(n) * rowBytes  // uWork
                               + 4U * rowBytes                       // col / absCol / diag / 备用
                               + BLOCK_SIZE                          // sharedTmp
                               + 2U * BLOCK_SIZE;                    // 结果标量 buffer
        if (total <= avail) {
            best = n;
        } else {
            break;
        }
    }
    return best;
}

// 奇异判定阈值（host 侧）：相对阈值 + 绝对下限策略，对齐 torch fp32 oracle。
//   kernel 计算 eps = max(n·FLT_EPSILON·maxPiv, epsFloor)，`|piv| <= eps` 判奇异。
//   maxPiv = LU 过程已见主元 |U_ii| 运行最大值（最大对角 U 元素，LAPACK getrf 风格相对尺度；仅依赖主元值，
//   不做全矩阵标量扫描）。host 仅下发 epsFloor。
//   纯绝对阈值 1e-30 会对 dup-col 精确奇异结构漏判；n·FLT_EPS·maxPiv 相对阈值可将 dup-col 判奇异，
//   同时对 ill[64,64] / rand256 等有限矩阵保留 7 个数量级以上裕量。
//   epsFloor：取极小绝对值（1e-30）作下限，仅在 maxPiv 极小时兜底，相对项为主驱动。
static float ComputeEps([[maybe_unused]] uint32_t matSizeN)
{
    constexpr float SLOGDET_EPS_FLOOR = 1e-30f;  // 绝对下限 floor；相对阈值 n·FLT_EPS·maxPiv 由 kernel 主导
    return SLOGDET_EPS_FLOOR;
}

struct SlogdetPlatformInfo {
    int64_t coreNum;
    uint64_t ubSize;
};

struct SlogdetShapeInfo {
    uint32_t matSizeN;
    uint64_t matrixNumCount;
};

struct SlogdetStrategyInfo {
    uint32_t memStrategy;
    uint32_t blockSize;
    uint32_t blockNum;
    uint32_t needCoreNum;
};

static ge::graphStatus GetSlogdetPlatformInfo(gert::TilingContext* context, SlogdetPlatformInfo& info)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    info.coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(info.coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    info.ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, info.ubSize);
    OP_CHECK_IF(info.ubSize == 0U, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetSlogdetShapeInfo(gert::TilingContext* context, SlogdetShapeInfo& info)
{
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const auto& shape = inputShapePtr->GetStorageShape();
    const size_t rank = shape.GetDimNum();
    OP_CHECK_IF(rank < 2U, OP_LOGE(context, "self rank must be >= 2, got %zu", rank), return ge::GRAPH_FAILED);

    info.matSizeN = static_cast<uint32_t>(shape.GetDim(rank - 1));
    OP_CHECK_IF(info.matSizeN == 0U, OP_LOGE(context, "matSizeN is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.matSizeN > SLOGDET_MAX_N,
                OP_LOGE(context, "matSizeN=%u exceeds supported upper bound %u (BLOCKED DataCopyPad "
                        "blockCount uint16 limit); split or upgrade gather to support larger n.",
                        info.matSizeN, SLOGDET_MAX_N),
                return ge::GRAPH_FAILED);

    info.matrixNumCount = 1U;
    for (size_t i = 0; i + 2U < rank; ++i) {
        info.matrixNumCount *= static_cast<uint64_t>(shape.GetDim(i));
    }
    OP_CHECK_IF(info.matrixNumCount == 0U, OP_LOGE(context, "matrixNumCount is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static SlogdetStrategyInfo ResolveSlogdetStrategy(const SlogdetPlatformInfo& platform, const SlogdetShapeInfo& shape,
                                                  gert::TilingContext* context)
{
    SlogdetStrategyInfo info{MEM_STRATEGY_FULL, shape.matSizeN, 1U, 1U};
    const uint32_t residentMax = ResolveResidentMax(platform.ubSize);
    if (shape.matSizeN > residentMax) {
        info.memStrategy = MEM_STRATEGY_BLOCKED;
        info.blockSize = COL_BLOCK;
        OP_LOGI(context, "Slogdet: n=%u > residentMax=%u → BLOCKED (U-resident GM workspace).",
                shape.matSizeN, residentMax);
    }
    info.needCoreNum = (static_cast<uint64_t>(platform.coreNum) < shape.matrixNumCount)
                           ? static_cast<uint32_t>(platform.coreNum)
                           : static_cast<uint32_t>(shape.matrixNumCount);
    return info;
}

static ge::graphStatus FillSlogdetTilingData(gert::TilingContext* context, const SlogdetShapeInfo& shape,
                                             const SlogdetStrategyInfo& strategy)
{
    OP_CHECK_IF(strategy.needCoreNum == 0U, OP_LOGE(context, "needCoreNum is 0"), return ge::GRAPH_FAILED);
    SlogdetTilingData* tiling = context->GetTilingData<SlogdetTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(SlogdetTilingData), 0, sizeof(SlogdetTilingData)) != EOK,
                OP_LOGE(context, "memset tiling data failed"), return ge::GRAPH_FAILED);
    tiling->matSizeN = shape.matSizeN;
    tiling->matrixNumCount = shape.matrixNumCount;
    tiling->blockSize = strategy.blockSize;
    tiling->blockNum = strategy.blockNum;
    tiling->epsSingular = ComputeEps(shape.matSizeN);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetSlogdetWorkspace(gert::TilingContext* context, const SlogdetShapeInfo& shape,
                                           const SlogdetStrategyInfo& strategy)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    uint64_t wsBytes = WS_SYS_SIZE;
    if (strategy.memStrategy == MEM_STRATEGY_BLOCKED) {
        uint64_t blockedSlots = static_cast<uint64_t>(strategy.needCoreNum) * static_cast<uint64_t>(shape.matSizeN) *
                                static_cast<uint64_t>(shape.matSizeN) * sizeof(float);
        if (blockedSlots > wsBytes) {
            wsBytes = blockedSlots;
        }
    }
    currentWorkspace[0] = static_cast<size_t>(wsBytes);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SelectSlogdetTemplate(gert::TilingContext* context, uint32_t memStrategy)
{
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    uint32_t dtypeVal = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, dtypeVal, memStrategy);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SlogdetTilingFunc(gert::TilingContext* context)
{
    SlogdetPlatformInfo platform{};
    auto status = GetSlogdetPlatformInfo(context, platform);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    SlogdetShapeInfo shape{};
    status = GetSlogdetShapeInfo(context, shape);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    SlogdetStrategyInfo strategy = ResolveSlogdetStrategy(platform, shape, context);
    status = FillSlogdetTilingData(context, shape, strategy);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = SetSlogdetWorkspace(context, shape, strategy);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    context->SetBlockDim(strategy.needCoreNum);

    OP_LOGI(context, "Slogdet Tiling: n=%u matrixNumCount=%lu memStrategy=%u blockSize=%u blockNum=%u needCore=%u",
            shape.matSizeN, shape.matrixNumCount, strategy.memStrategy, strategy.blockSize, strategy.blockNum,
            strategy.needCoreNum);

    return SelectSlogdetTemplate(context, strategy.memStrategy);
}

static ge::graphStatus TilingParseForSlogdet([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SlogdetCompileInfo {};

IMPL_OP_OPTILING(Slogdet)
    .Tiling(SlogdetTilingFunc)
    .TilingParse<SlogdetCompileInfo>(TilingParseForSlogdet);

} // namespace optiling
