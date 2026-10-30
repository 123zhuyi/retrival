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
 * \file zeros_like_tiling.cpp
 * \brief experimental 自包含 ascend910b (DAV_2201) 标准 AscendC tiling：IMPL_OP_OPTILING(ZerosLike)。
 *        退化 Elementwise（仅写 0）：按总字节数以 32B 对齐块多核均衡切分，
 *        单块零缓冲 UB 切分，按字节宽度（1/2/4/8）选 TilingKey；0 元素保护；workspace=0。
 *        标准顶层 tiling（非 arch32 子目录）：experimental zeros_like 仅 ascend910b，
 *        无 950 arch35 入口，故不存在重复注册 IMPL_OP_OPTILING(ZerosLike) 风险。
 */
#include <map>
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "zeros_like_tiling.h"
#include "experimental/conversion/zeros_like/op_kernel/zeros_like_tiling_data.h"
#include "experimental/conversion/zeros_like/op_kernel/zeros_like_tiling_key.h"

using namespace ZerosLikeNs;

namespace optiling {

static constexpr uint64_t ZL_WORKSPACE = 0;
// ZL_BLOCK_BYTES（32B 对齐基本块）单一定义于共享头 zeros_like_tiling_data.h，
// 经 using namespace ZerosLikeNs 引入；与 uint64 运算时按需提升，避免与 kernel 重复定义。
static constexpr uint64_t ZL_RESERVED_UB = 8 * 1024;       // 系统/tiling 保留
static constexpr uint64_t ZL_TILE_BYTES_LIMIT = 64 * 1024; // 单块零缓冲上限（纯写出，过大无收益）

// outputDtype -> 归一字节宽度（与 zeros_like_tiling_key.h 的 ZL_KEY_* 一致）
static const std::map<ge::DataType, uint32_t> g_zlByteWidth = {
    {ge::DT_INT8, 1}, {ge::DT_UINT8, 1}, {ge::DT_BOOL, 1},  {ge::DT_FLOAT16, 2},
    {ge::DT_BF16, 2}, {ge::DT_FLOAT, 4}, {ge::DT_INT32, 4}, {ge::DT_INT64, 8},
};

static uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    return (b == 0) ? 0 : ((a + b - 1) / b);
}

// 解析输出 dtype（归一字节宽度）并做 dtype/shape 一致性校验，
// 输出 bytesPerElem 与元素总数 outElem。
static ge::graphStatus ParseZerosLikeOutput(
    gert::TilingContext* tilingContext, uint32_t& bytesPerElem, uint64_t& outElem)
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();
    ge::DataType outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        outputDtype != inputDtype, OP_LOGE(tilingContext, "ZerosLike output y dtype not same as input x"),
        return ge::GRAPH_FAILED);

    // dtype -> 字节宽度桶
    auto itByte = g_zlByteWidth.find(outputDtype);
    OP_CHECK_IF(
        itByte == g_zlByteWidth.end(),
        OP_LOGE(tilingContext, "ZerosLike unsupported output dtype %d", static_cast<int32_t>(outputDtype)),
        return ge::GRAPH_FAILED);
    bytesPerElem = itByte->second;

    // shape 一致性校验 + 元素总数
    auto inputShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputShape);
    auto outputShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputShape);
    uint64_t inElem = inputShape->GetStorageShape().GetShapeSize();
    outElem = outputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(
        inElem != outElem, OP_LOGE(tilingContext, "ZerosLike input/output shape size mismatch"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 计算单块零缓冲字节数 tileBytes（32B 对齐，且不超过上限/可用 UB）。
static uint64_t ComputeZerosLikeTileBytes(const ZerosLikeCompileInfo* compileInfo)
{
    uint64_t usableUb = static_cast<uint64_t>(compileInfo->ubSize) - ZL_RESERVED_UB - sizeof(ZerosLikeTilingData);
    uint64_t tileBytes = usableUb / ZL_BLOCK_BYTES * ZL_BLOCK_BYTES;
    if (tileBytes > ZL_TILE_BYTES_LIMIT) {
        tileBytes = ZL_TILE_BYTES_LIMIT;
    }
    if (tileBytes == 0) {
        tileBytes = ZL_BLOCK_BYTES;
    }
    return tileBytes;
}

// 按总字节数在核间以 32B 对齐块均衡切分，填充 TilingData 各字段并设置 BlockDim。
static void ComputeZerosLikeSplit(
    gert::TilingContext* tilingContext, const ZerosLikeCompileInfo* compileInfo, ZerosLikeTilingData* tilingData,
    uint32_t bytesPerElem, uint64_t totalBytes)
{
    uint64_t tileBytes = ComputeZerosLikeTileBytes(compileInfo);

    if (totalBytes == 0) {
        // 0 元素保护：blockDim=1，各 count=0，kernel 入口直接 return
        tilingData->totalBytes = 0;
        tilingData->perCoreBytes = 0;
        tilingData->tailCoreNum = 0;
        tilingData->tileBytes = tileBytes;
        tilingData->usedCoreNum = 1;
        tilingData->bytesPerElem = bytesPerElem;
        tilingContext->SetBlockDim(1);
        return;
    }

    // 按 32B 对齐块在核间均衡切分；前 tailCoreNum 核各多 1 块
    uint64_t totalBlock = CeilDiv(totalBytes, ZL_BLOCK_BYTES);
    uint32_t usedCore = static_cast<uint32_t>(compileInfo->totalCoreNum);
    if (totalBlock < usedCore) {
        usedCore = static_cast<uint32_t>(totalBlock);
    }
    uint64_t blockPerCore = totalBlock / usedCore;
    uint64_t tailCoreNum = totalBlock % usedCore;
    uint64_t perCoreBytes = blockPerCore * ZL_BLOCK_BYTES;

    // perCoreBytes/tailCoreNum 以「整 32B 块」描述；末尾不足 32B 的零头
    // 由最后一个有效核覆盖：kernel 侧按 totalBytes 对各核字节范围 clamp，
    // 非对齐尾部用 DataCopyPad 精确写出。
    tilingData->totalBytes = totalBytes;
    tilingData->perCoreBytes = perCoreBytes;
    tilingData->tailCoreNum = tailCoreNum;
    tilingData->tileBytes = tileBytes;
    tilingData->usedCoreNum = usedCore;
    tilingData->bytesPerElem = bytesPerElem;
    tilingContext->SetBlockDim(usedCore);
}

static ge::graphStatus TilingPrepare4ZerosLike(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ZerosLikeCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0 || compileInfo->ubSize <= 0),
        OP_LOGE(
            context, "ZerosLike get hardware info failed, coreNum:%d, ubSize:%ld.", compileInfo->totalCoreNum,
            compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(context, "ZerosLike prepare: totalCoreNum=%d, ubSize=%ld", compileInfo->totalCoreNum, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ZerosLike(gert::TilingContext* tilingContext)
{
    OP_CHECK_IF(tilingContext == nullptr, OP_LOGE("ZerosLike", "tiling context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext, "Entering Tiling4ZerosLike");

    auto compileInfo = reinterpret_cast<const ZerosLikeCompileInfo*>(tilingContext->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);

    // dtype/shape 校验 + 归一字节宽度与元素总数解析（失败路径日志已在 Parse 内部打印）
    uint32_t bytesPerElem = 0;
    uint64_t outElem = 0;
    if (ParseZerosLikeOutput(tilingContext, bytesPerElem, outElem) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    ZerosLikeTilingData* tilingData = tilingContext->GetTilingData<ZerosLikeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tilingData);

    uint64_t totalBytes = outElem * bytesPerElem;

    // 按总字节数多核切分并填充 TilingData（含 0 元素保护与 BlockDim 设置）
    ComputeZerosLikeSplit(tilingContext, compileInfo, tilingData, bytesPerElem, totalBytes);

    // TilingKey = 字节宽度（1/2/4/8），与 kernel template<int BYTE_KEY> + if constexpr 一一映射
    ASCENDC_TPL_SEL_PARAM(tilingContext, static_cast<uint64_t>(bytesPerElem));

    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, workspaces);
    workspaces[0] = ZL_WORKSPACE;

    OP_LOGD(
        tilingContext,
        "ZerosLike tiling: totalBytes=%lu, usedCore=%u, perCoreBytes=%lu, tailCoreNum=%lu, "
        "tileBytes=%lu, bytesPerElem=%u",
        static_cast<unsigned long>(tilingData->totalBytes), tilingData->usedCoreNum,
        static_cast<unsigned long>(tilingData->perCoreBytes), static_cast<unsigned long>(tilingData->tailCoreNum),
        static_cast<unsigned long>(tilingData->tileBytes), tilingData->bytesPerElem);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ZerosLike).Tiling(Tiling4ZerosLike).TilingParse<ZerosLikeCompileInfo>(TilingPrepare4ZerosLike);

} // namespace optiling
