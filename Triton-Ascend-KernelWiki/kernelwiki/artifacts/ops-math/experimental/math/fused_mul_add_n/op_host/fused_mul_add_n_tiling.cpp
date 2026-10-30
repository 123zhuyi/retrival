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
 * \file fused_mul_add_n_tiling.cpp
 * \brief A2 (DAV_2201 / ascend910b) host tiling for FusedMulAddN.
 *        y = x1 * x3[0] + x2 (elementwise). 5 TilingKeys by dtype:
 *        0=fp32(direct) / 1=fp16(cast->fp32) / 2=int32(direct) / 3=int16(direct) / 4=bf16(cast->fp32).
 */

#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/fused_mul_add_n_tiling_data.h"

using namespace ge;

namespace optiling {

constexpr int64_t INDEX_X1 = 0;
constexpr int64_t INDEX_X2 = 1;
constexpr int64_t INDEX_X3 = 2;
constexpr int64_t INDEX_Y = 0;

// TilingKey 按 dtype 唯一确定
constexpr int64_t TILING_KEY_FP32 = 0;
constexpr int64_t TILING_KEY_FP16 = 1;
constexpr int64_t TILING_KEY_INT32 = 2;
constexpr int64_t TILING_KEY_INT16 = 3;
constexpr int64_t TILING_KEY_BF16 = 4;
constexpr int64_t TILING_KEY_INVALID = -1;

constexpr int64_t BLOCK_BYTES = 32;           // 32B 对齐粒度
constexpr int64_t WORKSPACE_SIZE = 32;        // 逐元素无需大 workspace
// 保守系统保留 UB（参考 mul_addn REMAINED_UB），从实测可用 UB 中扣减
constexpr int64_t REMAINED_UB = 16 * 1024;
// 核数 cap（min-bytes-per-core）：每核至少处理 48KB（3 路 IO：x1 读 + x2 读 + y 写），
// 否则小 shape 把琐碎工作摊到全部核，固定开销（scalar / 跨核同步 / 启动 / 低效 DMA）摊不薄。
// 大/中 shape 每核负载远超阈值 → 仍用满核（饱和点不退化）。
constexpr int64_t MIN_BYTES_PER_CORE = 48 * 1024;
constexpr int64_t IO_PATHS_PER_ELEM = 3;      // x1 读 + x2 读 + y 写

// UB buffer 系数：直算 3 块 T × DB(2) = 6×sizeof(T)；Cast 域算因路径不同分两档。
//   直算：6 × sizeof(T)
//   bf16 Cast 域算（5 op，2 块 fp32 scratch）：3×2×2 + 2×4×1 = 12 + 8 = 20，即每元素 20 字节
//   fp16 Cast 域算（Axpy 融合，3 op，仅 1 块 fp32 scratch）：3×2×2 + 1×4×1 = 12 + 4 = 16
constexpr int64_t BUFFER_COEF_DIRECT = 6;     // ×sizeof(T)
constexpr int64_t BUFFER_BYTES_CAST_BF16 = 20; // bf16：2 块 fp32 scratch
constexpr int64_t BUFFER_BYTES_CAST_FP16 = 16; // fp16：Axpy 融合后仅 1 块 fp32 scratch

static int64_t GetTilingKeyByDtype(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return TILING_KEY_FP32;
        case ge::DT_FLOAT16:
            return TILING_KEY_FP16;
        case ge::DT_INT32:
            return TILING_KEY_INT32;
        case ge::DT_INT16:
            return TILING_KEY_INT16;
        case ge::DT_BF16:
            return TILING_KEY_BF16;
        default:
            return TILING_KEY_INVALID;
    }
}

// 每元素的 UB 占用字节数（含 double buffer + cast 中间块）
static int64_t GetBytesPerElem(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:  // fp32 直算
        case ge::DT_INT32:  // int32 直算
            return BUFFER_COEF_DIRECT * static_cast<int64_t>(sizeof(float)); // 24
        case ge::DT_INT16:  // int16 直算
            return BUFFER_COEF_DIRECT * static_cast<int64_t>(sizeof(int16_t)); // 12
        case ge::DT_FLOAT16: // fp16 cast 域算（Axpy 融合，1 块 fp32 scratch）
            return BUFFER_BYTES_CAST_FP16; // 16
        case ge::DT_BF16:    // bf16 cast 域算（5 op，2 块 fp32 scratch）
            return BUFFER_BYTES_CAST_BF16; // 20
        default:
            return BUFFER_COEF_DIRECT * static_cast<int64_t>(sizeof(float));
    }
}

static int64_t GetElemSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
            return 2;
        default:
            return 4;
    }
}

// ---- 平台信息（动态获取核数与 UB，不写死）----
// coreNum 取得后立即兜底为 >=1，使所有后续除法点可证非零。
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t& coreNum, int64_t& ubSize)
{
    coreNum = 0;
    ubSize = 0;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = static_cast<int64_t>(ubSizePlatform);
    } else {
        auto compileInfo = reinterpret_cast<const FusedMulAddNCompileInfo*>(context->GetCompileInfo());
        OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
        coreNum = compileInfo->coreNum;
        ubSize = compileInfo->ubSize;
    }
    OP_CHECK_IF(
        (coreNum <= 0 || ubSize <= 0),
        OP_LOGE(context, "FusedMulAddN get platform failed, coreNum:%ld, ubSize:%ld", coreNum, ubSize),
        return ge::GRAPH_FAILED);
    // 兜底：保证 coreNum >= 1，消除除零数据流告警（上面校验已确保 >0，此处显式收敛）
    if (coreNum <= 0) {
        coreNum = 1;
    }
    return ge::GRAPH_SUCCESS;
}

// ---- dtype / shape 校验（5 元同 dtype；x1==x2；x3 单元素）----
static ge::graphStatus CheckDtypeAndShape(
    gert::TilingContext* context, ge::DataType& dtypeX1, int64_t& tilingKey, int64_t& totalNum)
{
    auto inputX1 = context->GetInputDesc(INDEX_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX1);
    auto inputX2 = context->GetInputDesc(INDEX_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX2);
    auto inputX3 = context->GetInputDesc(INDEX_X3);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX3);
    auto outputY = context->GetOutputDesc(INDEX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputY);

    dtypeX1 = inputX1->GetDataType();
    auto dtypeX2 = inputX2->GetDataType();
    auto dtypeX3 = inputX3->GetDataType();
    auto dtypeY = outputY->GetDataType();
    OP_CHECK_IF(
        (dtypeX1 != dtypeX2 || dtypeX1 != dtypeX3 || dtypeX1 != dtypeY),
        OP_LOGE(context, "FusedMulAddN: x1/x2/x3/y dtype must be the same"),
        return ge::GRAPH_FAILED);

    tilingKey = GetTilingKeyByDtype(dtypeX1);
    OP_CHECK_IF(
        (tilingKey == TILING_KEY_INVALID),
        OP_LOGE(context, "FusedMulAddN: unsupported dtype %d (expect FLOAT/FLOAT16/INT32/INT16/BF16)",
                static_cast<int>(dtypeX1)),
        return ge::GRAPH_FAILED);

    auto inputX1Shape = context->GetInputShape(INDEX_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX1Shape);
    auto inputX2Shape = context->GetInputShape(INDEX_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX2Shape);
    auto inputX3Shape = context->GetInputShape(INDEX_X3);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX3Shape);
    auto shapeX1 = inputX1Shape->GetStorageShape();
    auto shapeX2 = inputX2Shape->GetStorageShape();
    auto shapeX3 = inputX3Shape->GetStorageShape();
    OP_CHECK_IF(
        (shapeX1 != shapeX2),
        OP_LOGE(context, "FusedMulAddN: shapes of x1 and x2 must be the same"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (shapeX3.GetShapeSize() != 1),
        OP_LOGE(context, "FusedMulAddN: x3 must be a single-element scalar, got shapeSize %ld",
                shapeX3.GetShapeSize()),
        return ge::GRAPH_FAILED);

    totalNum = shapeX1.GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

// ---- 有效核数封顶（min-bytes-per-core cap）----
// 每核至少分到 MIN_BYTES_PER_CORE 字节（含 3 路 IO）才值得开核：
//   minElemsPerCore = MIN_BYTES_PER_CORE / (IO_PATHS_PER_ELEM * elemByteSize)
//   effCoreNum      = clamp(totalNum / minElemsPerCore, 1, coreNum)   // 向下取整，至少 1、至多 coreNum
// elemByteSize 为该 dtype 的元素真实字节数（fp32/int32=4，fp16/bf16/int16=2），
// 非 tiling 含 cast scratch 的 bytesPerElem（24/20/12）。
// 前置：coreNum >= 1、elemByteSize > 0、totalNum > 0。
static int64_t CalcEffectiveCoreNum(int64_t totalNum, int64_t coreNum, int64_t elemByteSize)
{
    const int64_t elemByteSizeNz = (elemByteSize > 0) ? elemByteSize : 1;
    const int64_t coreNumNz = (coreNum > 0) ? coreNum : 1;
    int64_t minElemsPerCore = MIN_BYTES_PER_CORE / (IO_PATHS_PER_ELEM * elemByteSizeNz);
    if (minElemsPerCore < 1) {
        minElemsPerCore = 1;
    }
    int64_t effCoreNum = totalNum / minElemsPerCore; // 向下取整
    if (effCoreNum < 1) {
        effCoreNum = 1;
    }
    if (effCoreNum > coreNumNz) {
        effCoreNum = coreNumNz;
    }
    return effCoreNum;
}

// ---- 多核切分（former/tail 两段，按 32B 对齐粒度）----
// 前置：coreNum >= 1（由 GetPlatformInfo 兜底）、elemPerBlockAlign > 0、totalNum > 0、elemByteSize > 0。
// 先按 min-bytes-per-core cap 封顶有效核数（小 shape 降核），再用 effCoreNum 参与 perCore/blockFormer 计算。
static void CalcBlockTiling(int64_t totalNum, int64_t coreNum, int64_t elemPerBlockAlign, int64_t elemByteSize,
    int64_t& blockNum, int64_t& blockFormer, int64_t& blockTail)
{
    // 用 min-bytes-per-core cap 取代「用满 coreNum」，得到参与切分的有效核数。
    const int64_t effCoreNum = CalcEffectiveCoreNum(totalNum, coreNum, elemByteSize);
    // 除数表达式本身在语法上可证非零（三目守卫），与运行时恒 >0 等价，不改变计算结果。
    const int64_t coreNumNz = (effCoreNum > 0) ? effCoreNum : 1;
    const int64_t elemPerBlockAlignNz = (elemPerBlockAlign > 0) ? elemPerBlockAlign : 1;
    // 每核分到的元素数，向上对齐到 elemPerBlockAlign
    int64_t perCore = (totalNum + coreNumNz - 1) / coreNumNz;
    blockFormer = (perCore + elemPerBlockAlignNz - 1) / elemPerBlockAlignNz * elemPerBlockAlignNz;
    if (blockFormer < elemPerBlockAlignNz) {
        blockFormer = elemPerBlockAlignNz;
    }
    const int64_t blockFormerNz = (blockFormer > 0) ? blockFormer : 1;
    blockNum = (totalNum + blockFormerNz - 1) / blockFormerNz; // 实际用核数 <= coreNum
    if (blockNum < 1) {
        blockNum = 1;
    }
    blockTail = totalNum - blockFormer * (blockNum - 1); // 尾核元素数 (1 .. blockFormer)
}

// ---- UB 切分：单 tile 元素数由可用 UB 反推，并对齐 ----
// 前置：bytesPerElem > 0、elemPerBlockAlign > 0（均由 dtype 派生，调用方已校验）。
static void CalcUbTiling(int64_t blockFormer, int64_t blockTail, int64_t ubSize, int64_t bytesPerElem,
    int64_t elemPerBlockAlign, int64_t& ubFormer, int64_t& ubLoopOfFormerBlock, int64_t& ubTailOfFormerBlock,
    int64_t& ubLoopOfTailBlock, int64_t& ubTailOfTailBlock)
{
    // 除数表达式本身在语法上可证非零（三目守卫），与运行时恒 >0 等价，不改变计算结果。
    const int64_t bytesPerElemNz = (bytesPerElem > 0) ? bytesPerElem : 1;
    const int64_t elemPerBlockAlignNz = (elemPerBlockAlign > 0) ? elemPerBlockAlign : 1;
    int64_t usableUb = ubSize - REMAINED_UB;
    if (usableUb < 0) {
        usableUb = 0;
    }
    int64_t ubFormerMax = usableUb / bytesPerElemNz;          // 满足 buffer 预算
    ubFormerMax = ubFormerMax / elemPerBlockAlignNz * elemPerBlockAlignNz; // 32B 对齐
    if (ubFormerMax < elemPerBlockAlignNz) {
        ubFormerMax = elemPerBlockAlignNz;
    }
    // 单 tile 不超过单核 former 任务量
    ubFormer = (blockFormer < ubFormerMax) ? blockFormer : ubFormerMax;
    if (ubFormer < elemPerBlockAlignNz) {
        ubFormer = elemPerBlockAlignNz;
    }
    const int64_t ubFormerNz = (ubFormer > 0) ? ubFormer : 1;

    // former 核
    ubLoopOfFormerBlock = (blockFormer + ubFormerNz - 1) / ubFormerNz;
    ubTailOfFormerBlock = blockFormer - ubFormer * (ubLoopOfFormerBlock - 1);
    // 尾核
    ubLoopOfTailBlock = (blockTail + ubFormerNz - 1) / ubFormerNz;
    ubTailOfTailBlock = blockTail - ubFormer * (ubLoopOfTailBlock - 1);
}

// 计算并填充 TilingData 的全部字段（多核 + UB 两段切分；totalNum==0 走空 tensor 分支）
// 前置：coreNum >= 1、elemPerBlockAlign > 0、bytesPerElem > 0、elemByteSize > 0（调用方已校验）。
//   bytesPerElem：UB 预算字节/元素（含 cast scratch + DB），用于 UB 切分；
//   elemByteSize：该 dtype 元素真实字节数（4/2），用于核数 cap。
// 直接写入 plain-struct TilingData（实验态约定：host 通过 context->GetTilingData<T>() 取得后赋值）。
static void ComputeTiling(int64_t totalNum, int64_t coreNum, int64_t ubSize, int64_t bytesPerElem,
    int64_t elemByteSize, int64_t elemPerBlockAlign, FusedMulAddNTilingData& tilingData)
{
    int64_t blockNum = 1;
    int64_t blockFormer = 0;
    int64_t blockTail = 0;
    int64_t ubFormer = elemPerBlockAlign;
    int64_t ubLoopOfFormerBlock = 0;
    int64_t ubLoopOfTailBlock = 0;
    int64_t ubTailOfFormerBlock = 0;
    int64_t ubTailOfTailBlock = 0;

    if (totalNum > 0) {
        CalcBlockTiling(totalNum, coreNum, elemPerBlockAlign, elemByteSize, blockNum, blockFormer, blockTail);
        CalcUbTiling(blockFormer, blockTail, ubSize, bytesPerElem, elemPerBlockAlign, ubFormer,
            ubLoopOfFormerBlock, ubTailOfFormerBlock, ubLoopOfTailBlock, ubTailOfTailBlock);
    }
    // else 空 tensor / 0 元素：保持上面的默认值（1 核、0 循环、ubFormer=elemPerBlockAlign），kernel 直接返回

    tilingData.totalNum = totalNum;
    tilingData.blockNum = blockNum;
    tilingData.blockFormer = blockFormer;
    tilingData.blockTail = blockTail;
    tilingData.ubFormer = ubFormer;
    tilingData.ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    tilingData.ubLoopOfTailBlock = ubLoopOfTailBlock;
    tilingData.ubTailOfFormerBlock = ubTailOfFormerBlock;
    tilingData.ubTailOfTailBlock = ubTailOfTailBlock;
}

// 填充 TilingData 并写回 context（含 blockDim / tilingKey / workspace / 日志）
static ge::graphStatus FillAndSaveTiling(gert::TilingContext* context, const FusedMulAddNTilingData& tilingData,
    int64_t tilingKey, int64_t totalNum)
{
    context->SetBlockDim(static_cast<uint32_t>(tilingData.blockNum));
    context->SetTilingKey(static_cast<uint64_t>(tilingKey));

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WORKSPACE_SIZE;

    OP_LOGI(context,
            "FusedMulAddN tiling: key=%ld totalNum=%ld blockNum=%ld blockFormer=%ld blockTail=%ld "
            "ubFormer=%ld ubLoopFormer=%ld ubLoopTail=%ld ubTailFormer=%ld ubTailTail=%ld",
            tilingKey, totalNum, tilingData.blockNum, tilingData.blockFormer,
            tilingData.blockTail, tilingData.ubFormer, tilingData.ubLoopOfFormerBlock,
            tilingData.ubLoopOfTailBlock, tilingData.ubTailOfFormerBlock,
            tilingData.ubTailOfTailBlock);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FusedMulAddNTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    int64_t coreNum = 0;
    int64_t ubSize = 0;
    OP_CHECK_IF(
        (GetPlatformInfo(context, coreNum, ubSize) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "FusedMulAddN: get platform info failed"),
        return ge::GRAPH_FAILED);

    ge::DataType dtypeX1 = ge::DT_FLOAT;
    int64_t tilingKey = TILING_KEY_INVALID;
    int64_t totalNum = 0;
    OP_CHECK_IF(
        (CheckDtypeAndShape(context, dtypeX1, tilingKey, totalNum) != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "FusedMulAddN: dtype/shape check failed"),
        return ge::GRAPH_FAILED);

    // elemSize / bytesPerElem 均由 dtype 派生，dtype 已校验为受支持类型，理论恒 > 0；
    // 显式校验既是防御，也消除除零静态告警（守卫支配后续除法语句）。
    int64_t elemSize = GetElemSize(dtypeX1);
    OP_CHECK_IF(
        (elemSize <= 0),
        OP_LOGE(context, "FusedMulAddN: invalid elemSize %ld", elemSize),
        return ge::GRAPH_FAILED);
    int64_t bytesPerElem = GetBytesPerElem(dtypeX1);
    OP_CHECK_IF(
        (bytesPerElem <= 0),
        OP_LOGE(context, "FusedMulAddN: invalid bytesPerElem %ld", bytesPerElem),
        return ge::GRAPH_FAILED);
    // 除数表达式本身在语法上可证非零（三目守卫），elemSize 已校验 >0，三目为 no-op，不改变计算结果。
    const int64_t elemSizeNz = (elemSize > 0) ? elemSize : 1;
    int64_t elemPerBlockAlign = BLOCK_BYTES / elemSizeNz; // fp32/int32=8, fp16/bf16/int16=16

    // 实验态约定：通过 context->GetTilingData<T>() 取得 plain-struct 缓冲并直接写字段。
    FusedMulAddNTilingData* tilingData = context->GetTilingData<FusedMulAddNTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    OP_CHECK_IF(
        (memset_s(tilingData, sizeof(FusedMulAddNTilingData), 0, sizeof(FusedMulAddNTilingData)) != EOK),
        OP_LOGE(context, "FusedMulAddN: memset tiling data failed"),
        return ge::GRAPH_FAILED);

    // elemSize 即 dtype 元素真实字节数（4/2），用于核数 cap；bytesPerElem（24/20/16/12）用于 UB 切分。
    ComputeTiling(totalNum, coreNum, ubSize, bytesPerElem, elemSize, elemPerBlockAlign, *tilingData);

    return FillAndSaveTiling(context, *tilingData, tilingKey, totalNum);
}

static ge::graphStatus FusedMulAddNTilingPrepare(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<FusedMulAddNCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatform = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatform);
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0 || compileInfo->ubSize <= 0),
        OP_LOGE(context, "FusedMulAddN prepare failed, coreNum:%ld, ubSize:%ld", compileInfo->coreNum,
                compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedMulAddN)
    .Tiling(FusedMulAddNTilingFunc)
    .TilingParse<FusedMulAddNCompileInfo>(FusedMulAddNTilingPrepare);

} // namespace optiling
