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
 * \file acos_grad_v2_tiling.cpp
 * \brief AcosGradV2 Tiling — 多核 + UB 两级切分（arch22 / Ascend910B）
 *
 * 切分策略：
 *   1) 多核（blockDim）：按 ELEM_ALIGN(512) 对齐切 totalLength → blockFormer × blockNum
 *   2) 核内 UB：按 dtype 的 bytesPerElem / alignFactor 从动态获取的 UB size 推出 ubFormer
 *   3) 核内 loop/tail 由 kernel 侧按 blockLength_ 自行推导，tiling 只下发 blockFormer/ubFormer
 */

#include "register/op_def_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "../../op_kernel/arch22/acos_grad_v2_tiling_data.h"
#include "../../op_kernel/arch22/acos_grad_v2_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint32_t ELEM_ALIGN = 512U;
constexpr uint32_t FP32_BYTES_PER_ELEM = 32U;
constexpr uint32_t FP32_ALIGN = 64U;
constexpr uint32_t LOWPREC_BYTES_PER_ELEM = 28U;
constexpr uint32_t LOWPREC_ALIGN = 128U;

static const gert::Shape g_scalar_to_vec1 = {1};

// ---- 核内 UB 切分计算：按 dtype 选 bytesPerElem / alignFactor，推出 ubFormer ----
// ubSize 由平台信息动态获取（不同款型 UB size 未必相同），不再写死 184KB
static void CalcAcosGradV2UbTiling(uint64_t totalLength, uint32_t availCoreNum, uint64_t ubSize, ge::DataType dataType,
                                   AcosGradV2TilingData* tiling)
{
    // 多核切分：每核至少 ELEM_ALIGN 个元素，向上对齐
    uint32_t coreNum = static_cast<uint32_t>(
        CeilDiv(static_cast<int64_t>(totalLength), static_cast<int64_t>(ELEM_ALIGN)));
    coreNum = std::min(coreNum, availCoreNum);
    coreNum = std::max(coreNum, 1U);

    uint32_t blockFormerRaw = static_cast<uint32_t>(
        CeilDiv(static_cast<int64_t>(totalLength), static_cast<int64_t>(coreNum)));
    uint32_t blockFormer = static_cast<uint32_t>(
        CeilDiv(static_cast<int64_t>(blockFormerRaw), static_cast<int64_t>(ELEM_ALIGN)) * ELEM_ALIGN);
    blockFormer = std::max(blockFormer, ELEM_ALIGN);

    uint32_t blockNum = static_cast<uint32_t>(
        CeilDiv(static_cast<int64_t>(totalLength), static_cast<int64_t>(blockFormer)));
    blockNum = std::max(blockNum, 1U);

    // UB 切分：按 dtype 确定每元素占用字节数和对齐因子
    uint32_t bytesPerElem = (dataType == ge::DT_FLOAT) ? FP32_BYTES_PER_ELEM : LOWPREC_BYTES_PER_ELEM;
    uint32_t alignFactor = (dataType == ge::DT_FLOAT) ? FP32_ALIGN : LOWPREC_ALIGN;

    uint32_t ubFormerRaw = static_cast<uint32_t>(ubSize / bytesPerElem);
    uint32_t ubFormer = static_cast<uint32_t>(
        FloorDiv(static_cast<int64_t>(ubFormerRaw), static_cast<int64_t>(alignFactor)) * alignFactor);
    ubFormer = std::max(ubFormer, alignFactor);
    ubFormer = std::min(ubFormer, blockFormer);

    // 核内 loop/tail 由 kernel 侧按 blockLength_ 自行推导，无需在此预计算
    tiling->totalLength = totalLength;
    tiling->blockFormer = blockFormer;
    tiling->blockNum = blockNum;
    tiling->ubFormer = ubFormer;
}

// 本算子硬件能力：AIV 核数 + UB 字节数合并查询（区别于模板的逐项 out-param 写法）
struct AcosGradV2HwCap {
    uint32_t aivCoreNum = 0;
    uint64_t ubBytes = 0;
};

static ge::graphStatus QueryAcosGradV2HwCap(gert::TilingContext* context, AcosGradV2HwCap& cap)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    platform_ascendc::PlatformAscendC plat(platformInfoPtr);

    int64_t aiv = plat.GetCoreNumAiv();
    OP_CHECK_IF(aiv <= 0, OP_LOGE(context, "AcosGradV2: invalid AIV core count %ld", aiv), return ge::GRAPH_FAILED);
    cap.aivCoreNum = static_cast<uint32_t>(aiv);

    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, cap.ubBytes);
    OP_CHECK_IF(cap.ubBytes == 0, OP_LOGE(context, "AcosGradV2: UB size unavailable on this SoC"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 把存储 shape 中的标量(0维)规整为 {1}，便于按一维长度统一切分
static gert::Shape AsVecIfScalar(const gert::Shape& storageShape)
{
    return (storageShape.GetDimNum() == 0) ? g_scalar_to_vec1 : storageShape;
}

// ---- tiling 入口 ----
static ge::graphStatus AcosGradV2TilingFunc(gert::TilingContext* context)
{
    OP_LOGI(context->GetNodeName(), "Enter AcosGradV2TilingFunc");

    // 1) shape：标量视作 {1}，并校验 y/dy/z 三者元素数一致
    auto inputY = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputY);
    auto inputDy = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDy);
    auto outputZ = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputZ);

    gert::Shape yShape = AsVecIfScalar(inputY->GetStorageShape());
    gert::Shape dyShape = AsVecIfScalar(inputDy->GetStorageShape());
    gert::Shape zShape = AsVecIfScalar(outputZ->GetStorageShape());
    int64_t yElemCnt = yShape.GetShapeSize();
    OP_CHECK_IF(yElemCnt != dyShape.GetShapeSize() || yElemCnt != zShape.GetShapeSize(),
                OP_LOGE(context, "AcosGradV2: shape size mismatch: y=%ld, dy=%ld, z=%ld", yElemCnt,
                        dyShape.GetShapeSize(), zShape.GetShapeSize()),
                return ge::GRAPH_FAILED);
    uint64_t totalLength = static_cast<uint64_t>(yElemCnt);

    // 2) dtype 仅支持 fp16/fp32/bf16
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    OP_CHECK_IF(dataType != ge::DT_FLOAT16 && dataType != ge::DT_FLOAT && dataType != ge::DT_BF16,
                OP_LOGE(context, "AcosGradV2: unsupported dtype %d", static_cast<int>(dataType)),
                return ge::GRAPH_FAILED);

    // 3) 取 tiling 缓冲并清零
    AcosGradV2TilingData* tiling = context->GetTilingData<AcosGradV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(AcosGradV2TilingData), 0, sizeof(AcosGradV2TilingData)) != EOK,
                OP_LOGE(context, "AcosGradV2: memset_s tiling data error"), return ge::GRAPH_FAILED);

    // 4) 空 tensor 无需查询硬件，单核直接返回
    if (totalLength == 0UL) {
        context->SetBlockDim(1U);
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType));
        return ge::GRAPH_SUCCESS;
    }

    // 5) 查询硬件能力(AIV 核数 + UB)，再做多核 + UB 两级切分
    AcosGradV2HwCap cap;
    OP_CHECK_IF(QueryAcosGradV2HwCap(context, cap) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "AcosGradV2: query hardware capability failed"), return ge::GRAPH_FAILED);
    CalcAcosGradV2UbTiling(totalLength, cap.aivCoreNum, cap.ubBytes, dataType, tiling);
    context->SetBlockDim(tiling->blockNum);

    // 6) workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    OP_LOGI(context, "[AcosGradV2 Tiling] totalLength=%lu, blockFormer=%u, blockNum=%u, ubFormer=%u",
            tiling->totalLength, tiling->blockFormer, tiling->blockNum, tiling->ubFormer);

    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAcosGradV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AcosGradV2CompileInfo {};

IMPL_OP_OPTILING(AcosGradV2).Tiling(AcosGradV2TilingFunc).TilingParse<AcosGradV2CompileInfo>(TilingParseForAcosGradV2);

} // namespace optiling
