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
 * \file logdet_tiling.cpp
 * \brief Logdet Tiling 实现（ascend910b / DAV_2201）。
 *
 * registry-invoke 规范：
 *   - 普通 C++ struct LogdetTilingData，通过 context->GetTilingData<>() 写入；
 *   - ASCENDC_TPL_SEL_PARAM(context, dtypeVal, memStrategy) 选择模板参数（不使用 SetTilingKey 整型常量）。
 *
 * 平台核数 / UB 容量通过 platform API 运行时获取（GetCoreNumAiv / GetCoreMemSize），禁止写死。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/logdet_tiling_data.h"
#include "../op_kernel/logdet_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 32U;            // DataBlock 字节数（32B 对齐基本单位）
constexpr uint64_t WS_SYS_SIZE = 16ULL * 1024U * 1024U;  // 16MB 系统 workspace（沿用 cholesky 线代算子约定）
constexpr uint32_t UB_RESERVE = 8U * 1024U;     // UB 系统保留（约 8KB）
constexpr uint32_t MEM_STRATEGY_FULL = 0U;      // 全驻留
constexpr uint32_t MEM_STRATEGY_BLOCKED = 1U;   // 核内分块（large-n，U 常驻 GM workspace）
constexpr uint32_t FULL_RESIDENT_FALLBACK = 256U;  // ResolveResidentMax 兜底上界（UB 信息异常时）
constexpr uint32_t COL_BLOCK = 64U;             // BLOCKED 列块宽度（写入 tiling blockSize 占位字段，kernel 未消费）
// BLOCKED 列 gather 的 DataCopyPad blockCount = 子列长度 m=n-k（最大 ≈ n），uint16 取值范围 [1,4095]。
//   故 BLOCKED 路径安全 n 上界 = 4095。超出则 blockCount 静默越界（读错列长度）→ 必须 host 拒绝。
constexpr uint32_t LOGDET_MAX_N = 4095U;
// ResolveResidentMax UB 预留裕量系数：在容量约束上再乘 0.95，避免 residentMax 边界 FULL 路径
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
    // 预留裕量：实际可用按 (ubSize-UB_RESERVE)*0.95 计，避免边界 n 处 UB 余量过薄。
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

// ST golden uses abs(pivot) < 1e-38 as singular criterion in LU decomposition.
static float ComputeEps(uint32_t)
{
    constexpr float LOGDET_EPS_FLOOR = 1e-38f;
    return LOGDET_EPS_FLOOR;
}

// 平台信息解析（运行时取核数 / UB 容量，禁止写死）。出参 coreNum / ubSize。
static ge::graphStatus ParsePlatform(gert::TilingContext* context, int64_t& coreNum, uint64_t& ubSize)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0U, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// shape 解析 + 校验：出参 n（末维方阵边长）与 matrixNumCount（batch 乘积）。
static ge::graphStatus ParseShape(gert::TilingContext* context, uint32_t& matSizeN, uint64_t& matrixNumCount)
{
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const auto& shape = inputShapePtr->GetStorageShape();
    const size_t rank = shape.GetDimNum();
    OP_CHECK_IF(rank < 2U, OP_LOGE(context, "self rank must be >= 2, got %zu", rank), return ge::GRAPH_FAILED);

    matSizeN = static_cast<uint32_t>(shape.GetDim(rank - 1));
    OP_CHECK_IF(matSizeN == 0U, OP_LOGE(context, "matSizeN is 0"), return ge::GRAPH_FAILED);
    // BLOCKED 列 gather 的 DataCopyPad blockCount(uint16, ≤4095) = n-k ≤ n；n>4095 静默越界 →
    //   显式拒绝，不静默错误（uint16 DataCopyPad 限制）。
    OP_CHECK_IF(matSizeN > LOGDET_MAX_N,
                OP_LOGE(context, "matSizeN=%u exceeds supported upper bound %u (BLOCKED DataCopyPad "
                        "blockCount uint16 limit); split or upgrade gather to support larger n.",
                        matSizeN, LOGDET_MAX_N),
                return ge::GRAPH_FAILED);
    matrixNumCount = 1U;
    for (size_t i = 0; i + 2U < rank; ++i) {
        matrixNumCount *= static_cast<uint64_t>(shape.GetDim(i));
    }
    OP_CHECK_IF(matrixNumCount == 0U, OP_LOGE(context, "matrixNumCount is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 由 UB 容量推导分块策略：出参 memStrategy / blockSize / blockNum（语义同原内联逻辑）。
static void SelectStrategy(gert::TilingContext* context, uint32_t matSizeN, uint64_t ubSize,
                          uint32_t& memStrategy, uint32_t& blockSize, uint32_t& blockNum)
{
    const uint32_t residentMax = ResolveResidentMax(ubSize);
    blockNum = 1U;
    if (matSizeN <= residentMax) {
        // FULL：单矩阵 n×n 整块可驻留 UB
        memStrategy = MEM_STRATEGY_FULL;
        blockSize = matSizeN;
    } else {
        // BLOCKED：n > N_RESIDENT_MAX，U 工作区常驻 GM workspace，UB 仅持 O(n) 行/列向量
        memStrategy = MEM_STRATEGY_BLOCKED;
        blockSize = COL_BLOCK;
        OP_LOGI(context, "Logdet: n=%u > residentMax=%u → BLOCKED (U-resident GM workspace).",
                matSizeN, residentMax);
    }
}

static inline uint64_t AlignElem8(uint64_t elems)
{
    return ((elems + 7U) / 8U) * 8U;
}

// workspace 字节：FULL 仅系统预留（16MB，沿用 cholesky 约定）；BLOCKED 每核独占
//   n*align8(n) fp32 slot，保证 workspace 每行/每 slot 起址 32B 对齐。
static uint64_t ComputeWorkspaceBytes(uint32_t memStrategy, uint32_t needCoreNum, uint32_t matSizeN)
{
    uint64_t wsBytes = WS_SYS_SIZE;
    if (memStrategy == MEM_STRATEGY_BLOCKED) {
        const uint64_t rowStride = AlignElem8(matSizeN);
        uint64_t blockedSlots = static_cast<uint64_t>(needCoreNum) *
                                static_cast<uint64_t>(matSizeN) * rowStride * sizeof(float);
        if (blockedSlots > wsBytes) {
            wsBytes = blockedSlots;
        }
    }
    return wsBytes;
}

static ge::graphStatus LogdetTilingFunc(gert::TilingContext* context)
{
    // 1) 平台信息（运行时取，禁止写死）
    int64_t coreNum = 0;
    uint64_t ubSize = 0U;
    if (ParsePlatform(context, coreNum, ubSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2) shape：n + matrixNumCount
    uint32_t matSizeN = 0U;
    uint64_t matrixNumCount = 0U;
    if (ParseShape(context, matSizeN, matrixNumCount) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 3) 由运行时 UB 容量推导 N_RESIDENT_MAX 与分块策略 → MEM_STRATEGY
    uint32_t memStrategy = MEM_STRATEGY_FULL;
    uint32_t blockSize = matSizeN;
    uint32_t blockNum = 1U;
    SelectStrategy(context, matSizeN, ubSize, memStrategy, blockSize, blockNum);

    // batch 按核切分
    uint32_t needCoreNum = (static_cast<uint64_t>(coreNum) < matrixNumCount)
                               ? static_cast<uint32_t>(coreNum)
                               : static_cast<uint32_t>(matrixNumCount);
    // BLOCKED 路径的大矩阵 workspace-resident LU 当前使用 GM workspace 作为可变 U 副本。
    // 多核 batch 在高 blockIdx 上会触发 MTE DDR out-of-range（device 日志 blk=4/5/6）。
    // 先以单核串行保证正确性；小矩阵 FULL 路径仍保留多核 batch 并行。
    if (memStrategy == MEM_STRATEGY_BLOCKED) {
        needCoreNum = 1U;
    }
    OP_CHECK_IF(needCoreNum == 0U, OP_LOGE(context, "needCoreNum is 0"), return ge::GRAPH_FAILED);

    // 4) 写 TilingData（普通 struct 指针 + memset_s 清零）
    LogdetTilingData* tiling = context->GetTilingData<LogdetTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(LogdetTilingData), 0, sizeof(LogdetTilingData)) != EOK,
                OP_LOGE(context, "memset tiling data failed"), return ge::GRAPH_FAILED);
    tiling->matSizeN = matSizeN;
    tiling->matrixNumCount = matrixNumCount;
    tiling->blockSize = blockSize;
    tiling->blockNum = blockNum;
    tiling->epsSingular = ComputeEps(matSizeN);

    // workspace（见 ComputeWorkspaceBytes）
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(ComputeWorkspaceBytes(memStrategy, needCoreNum, matSizeN));

    context->SetBlockDim(needCoreNum);

    OP_LOGI(context, "Logdet Tiling: n=%u matrixNumCount=%lu memStrategy=%u blockSize=%u blockNum=%u needCore=%u",
            matSizeN, matrixNumCount, memStrategy, blockSize, blockNum, needCoreNum);

    // 5) 选择模板参数（dtype 固定 fp32；MEM_STRATEGY 决定路径）→ 映射 kernel 入口模板参数
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    uint32_t dtypeVal = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, dtypeVal, memStrategy);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLogdet(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

struct LogdetCompileInfo {};

IMPL_OP_OPTILING(Logdet)
    .Tiling(LogdetTilingFunc)
    .TilingParse<LogdetCompileInfo>(TilingParseForLogdet);

} // namespace optiling
