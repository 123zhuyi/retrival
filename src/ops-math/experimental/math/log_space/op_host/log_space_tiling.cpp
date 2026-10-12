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
 * \file log_space_tiling.cpp
 * \brief LogSpace Tiling 实现（Ascend950 / Atlas A2/A3：ascend910b / ascend910_93）
 *
 * 覆盖 14 个 TilingKey（7 种输出 dtype × NORMAL/SINGLE）。
 * Tiling 按输出 dtype 计算 stepF/logBase/分核策略，TilingKey 通过 (D_T_Y, MODE) 二元组下发。
 */

#include <cmath>
#include <climits>
#include <limits>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/log_space_tiling_data.h"
#include "../op_kernel/log_space_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_PER_CORE = 64;     // 每个核最少处理元素数，小于该值则缩减核数
constexpr int64_t UB_CHUNK_ELEMS = 2048; // 单次搬运粒度（fp32 元素数）

// 属性索引常量（与 op_host/log_space_def.cpp / op_host/log_space_infershape.cpp 对齐）
constexpr int ATTR_IDX_START = 0;
constexpr int ATTR_IDX_END = 1;
constexpr int ATTR_IDX_STEPS = 2;
constexpr int ATTR_IDX_BASE = 3;

// MODE 枚举（必须与 log_space_tiling_key.h 一致）
constexpr uint32_t MODE_NORMAL = 0;
constexpr uint32_t MODE_SINGLE = 1;

// 平台信息 + 工作空间一次性初始化：取平台句柄与 workspace 指针 → 置 workspace（本算子无需额外
// workspace，置 0）→ 从平台对象读 UB 容量与 AIV 核数并一并校验。把两处上下文查询合并，避免在主
// Tiling 流程里散落多次 context 访问，核数/UB 的取用也集中在一处。
static ge::graphStatus PrepareContext(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    size_t* wsSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, wsSizes);
    wsSizes[0] = WS_SYS_SIZE;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(ubSize == 0 || coreNum == 0,
                OP_LOGE(context, "invalid platform: ubSize=%lu coreNum=%ld", ubSize, coreNum), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus LogSpaceTilingFunc(gert::TilingContext* context)
{
    // 1. 平台信息 + 工作空间
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(PrepareContext(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "PrepareContext error"),
                return ge::GRAPH_FAILED);

    // 2. 属性
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* startPtr = attrs->GetAttrPointer<float>(ATTR_IDX_START);
    const float* endPtr = attrs->GetAttrPointer<float>(ATTR_IDX_END);
    const int64_t* stepsPtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_STEPS);
    const float* basePtr = attrs->GetAttrPointer<float>(ATTR_IDX_BASE);
    OP_CHECK_NULL_WITH_CONTEXT(context, startPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, endPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, stepsPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, basePtr);

    const float startF = *startPtr;
    const float endF = *endPtr;
    const int64_t steps = *stepsPtr;
    const float baseF = *basePtr;

    OP_CHECK_IF(steps < 0, OP_LOGE(context, "steps must be >= 0, got %ld", steps), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        steps > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        OP_LOGE(context, "steps must be <= UINT32_MAX (%u), got %ld", std::numeric_limits<uint32_t>::max(), steps),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(baseF <= 0.0f, OP_LOGE(context, "base must be > 0, got %f", baseF), return ge::GRAPH_FAILED);

    // 3. 输出 dtype
    auto outDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDesc);
    ge::DataType dtype = outDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16 && dtype != ge::DT_INT8 &&
                    dtype != ge::DT_INT16 && dtype != ge::DT_INT32 && dtype != ge::DT_UINT8,
                OP_LOGE(context, "unsupported dtype %d", static_cast<int>(dtype)), return ge::GRAPH_FAILED);

    // 4. 填充 TilingData
    LogSpaceTilingData* tiling = context->GetTilingData<LogSpaceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(LogSpaceTilingData), 0, sizeof(LogSpaceTilingData)) != EOK,
                OP_LOGE(context, "memset tiling data failed"), return ge::GRAPH_FAILED);

    tiling->totalLen = static_cast<uint64_t>(steps);
    tiling->startF = startF;
    tiling->logBase = std::log(baseF);
    // 整型路径 arg 空间 double-float 系数：argStart = start*ln(base)（double 计算后拆 hi/lo 两个 fp32）。
    const double lnBaseD = std::log(static_cast<double>(baseF));
    const double argStartD = static_cast<double>(startF) * lnBaseD;
    tiling->argStartHi = static_cast<float>(argStartD);
    tiling->argStartLo = static_cast<float>(argStartD - static_cast<double>(tiling->argStartHi));
    // 端点值用 double std::pow 精确算好（与验收 golden 的 numpy base**linspace 同口径，整数幂精确），
    // 避免设备 fp32 exp(k·ln base) 对整数幂的不稳定上/下溢。kernel 端点直接落型这两个值。
    tiling->startValF = static_cast<float>(std::pow(static_cast<double>(baseF), static_cast<double>(startF)));
    tiling->endValF = static_cast<float>(std::pow(static_cast<double>(baseF), static_cast<double>(endF)));
    tiling->ubChunk = static_cast<uint32_t>(UB_CHUNK_ELEMS);

    uint32_t mode = MODE_NORMAL;
    int64_t usedCoreNum = 1;
    if (steps <= 0) {
        // steps==0：空 tensor，Host 侧已短路（理论上不会走到这里），保险起见设置单核空跑
        tiling->coreNum = 1;
        tiling->tileLen = 0;
        tiling->tailTileLen = 0;
        tiling->tailCoreIdx = 0;
        tiling->stepF = 0.0f;
        mode = MODE_SINGLE;
    } else if (steps == 1) {
        tiling->coreNum = 1;
        tiling->tileLen = 1;
        tiling->tailTileLen = 1;
        tiling->tailCoreIdx = 0;
        tiling->stepF = 0.0f;
        mode = MODE_SINGLE;
        usedCoreNum = 1;
    } else {
        // steps >= 2: 常规多步生成
        int64_t maxCores = CeilDiv(steps, MIN_PER_CORE);
        int64_t useCores = (maxCores < coreNum) ? maxCores : coreNum;
        if (useCores < 1)
            useCores = 1;
        int64_t tileLen = steps / useCores;
        int64_t tailLen = steps - tileLen * (useCores - 1);

        tiling->coreNum = static_cast<uint32_t>(useCores);
        tiling->tileLen = static_cast<uint32_t>(tileLen);
        tiling->tailTileLen = static_cast<uint32_t>(tailLen);
        tiling->tailCoreIdx = static_cast<uint32_t>(useCores - 1);
        tiling->stepF = (endF - startF) / static_cast<float>(steps - 1);
        // 整型 arg 空间步长（double 算后拆 hi/lo）：stepLn = step*ln(base)
        const double stepD = (static_cast<double>(endF) - static_cast<double>(startF)) / static_cast<double>(steps - 1);
        const double stepLnD = stepD * lnBaseD;
        tiling->stepLnHi = static_cast<float>(stepLnD);
        tiling->stepLnLo = static_cast<float>(stepLnD - static_cast<double>(tiling->stepLnHi));
        // x 空间 step 的 double-float 低位（kernel 用 df 算准 chunk 基址 xBase，定位整数指数网格点）
        tiling->stepLoX = static_cast<float>(stepD - static_cast<double>(tiling->stepF));
        // 整数幂精确覆写表（仅整型输出需要：fp 路径 CAST_RINT 对 9.9999 会就近舍入到 10，不受下溢影响）。
        // m ∈ [floor(min(start,end)), ceil(max(start,end))]，baseN[k]=double pow(base, nmin+k)。
        const bool isIntOut = (dtype == ge::DT_INT8 || dtype == ge::DT_INT16 || dtype == ge::DT_INT32 ||
                               dtype == ge::DT_UINT8);
        if (isIntOut) {
            const double lo = (startF < endF) ? static_cast<double>(startF) : static_cast<double>(endF);
            const double hi = (startF < endF) ? static_cast<double>(endF) : static_cast<double>(startF);
            const int64_t nmin = static_cast<int64_t>(std::floor(lo));
            const int64_t nmax = static_cast<int64_t>(std::ceil(hi));
            int64_t cnt = nmax - nmin + 1;
            if (cnt >= 1 && cnt <= 96) { // 范围在 96 内才启用，超出则禁用（极端大动态范围，整数幂多溢出）
                tiling->nmin = static_cast<int32_t>(nmin);
                tiling->nCount = static_cast<int32_t>(cnt);
                for (int64_t k = 0; k < cnt; ++k) {
                    tiling->baseN[k] = static_cast<float>(
                        std::pow(static_cast<double>(baseF), static_cast<double>(nmin + k)));
                }
            }
            // [df-V 门控] int8/uint8 值 ∈ (2^24, 2^32)：fp32 算 V 不足以让 mod256 准，用向量 df-exp。
            if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) {
                const double vS = std::pow(static_cast<double>(baseF), static_cast<double>(startF));
                const double vE = std::pow(static_cast<double>(baseF), static_cast<double>(endF));
                const double mx = (std::fabs(vS) > std::fabs(vE)) ? std::fabs(vS) : std::fabs(vE);
                // 值 > 2^24 即启用：[2^24,2^31] 是敏感区(mod256≠0,需精确)；值 ≥2^31 fp32 是 256 倍数 → mod=0(kernel
                // 直接置 0)。
                if (mx > 16777216.0) {
                    tiling->useDfV = 1;
                }
            }
            // df-V 的向量 df-exp 是每 chunk 固定开销 → 放大 chunk 减少调用次数（UB: 4×8192×4B + 队列 ≈ 145KB<192KB）
            if (tiling->useDfV) {
                tiling->ubChunk = 4096; // chunk 跨度越小 → 每元素 ebH·s 的 fp32 舍入误差越小（int8 严阈值需要）
                // 递推等比常数 const = base^(ubChunk·step) = exp(ubChunk·stepLn)，df-exp 整核只算 1 次
                const double cD = std::exp(static_cast<double>(tiling->ubChunk) * stepLnD);
                tiling->constHi = static_cast<float>(cD);
                tiling->constLo = static_cast<float>(cD - static_cast<double>(tiling->constHi));
            }
            // 1/k! 的 double-float 常数（向量泰勒 exp 用，host 精确算）
            double fact = 1.0;
            for (int k = 0; k < 12; ++k) {
                if (k >= 2) {
                    fact *= static_cast<double>(k);
                }
                const double rf = 1.0 / fact;
                tiling->rfHi[k] = static_cast<float>(rf);
                tiling->rfLo[k] = static_cast<float>(rf - static_cast<double>(tiling->rfHi[k]));
            }
        }
        mode = MODE_NORMAL;
        usedCoreNum = useCores;
    }

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 5. TilingKey 选择：D_T_Y × MODE
    uint32_t dTypeY = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, dTypeY, mode);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLogSpace([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct LogSpaceCompileInfo {};

IMPL_OP_OPTILING(LogSpace).Tiling(LogSpaceTilingFunc).TilingParse<LogSpaceCompileInfo>(TilingParseForLogSpace);

} // namespace optiling
