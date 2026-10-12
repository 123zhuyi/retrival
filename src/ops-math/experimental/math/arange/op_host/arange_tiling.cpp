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
 * \file arange_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/arange_tiling_data.h"
#include "../op_kernel/arange_tiling_key.h"
#define DIVIDE_AND_ALIGN(size, split, align) ((((size) / (split)) + ((align) - 1)) & ~((align) - 1))

namespace optiling {

const uint32_t DTYPE_SIZE1 = 1; // int8 / uint8
const uint32_t DTYPE_SIZE2 = 2;
const uint32_t DTYPE_SIZE4 = 4;
const uint32_t DTYPE_SIZE8 = 8;

struct ArangeCompileInfo {};

// 计算某一段（former 或 tail）的 UB 子循环：unitLoops + 尾块元素数
// segLength 为 int64（接 formerLength/tailLength），中间乘积用 int64 防溢出；
//   unitLoops（每核循环数 <2³²）、tailNum（<unitNum）仍是小量 uint32，赋值处显式收窄。
static void CalcUnitLoops(int64_t segLength, uint32_t unitNum, uint32_t& unitLoops, uint32_t& tailNum)
{
    if (segLength == 0 || unitNum == 0) {
        unitLoops = 0;
        tailNum = 0;
        return;
    }
    int64_t loops = segLength / unitNum;
    int64_t tail = segLength - static_cast<int64_t>(unitNum) * loops; // 中间乘积用 int64 防溢出
    unitLoops = static_cast<uint32_t>(loops);                         // 每核循环数 <2³²
    tailNum = static_cast<uint32_t>(tail);                            // <unitNum
    if (tailNum > 0) {
        unitLoops += 1;
    }
}

// ArangeTilingFunc 按三段边界拆为 DecideDtypeSizeAndTilingKey / CalcUnitNum /
// CalcCoreSplitAndFillTiling 三个 static 子函数，ArangeTilingFunc 仅做编排。

// 决定 dtype 字节数 + TilingKey（默认 MODE_0 Cast 路径；仅 DT_FLOAT 走 MODE_1 纯 FP32 直算）。
static uint32_t DecideDtypeSizeAndTilingKey(ge::DataType dtype_out, uint64_t& tilingkey)
{
    uint32_t dtype_size = DTYPE_SIZE2;
    tilingkey = 0; // 默认 MODE_0（Cast 路径）；仅 DT_FLOAT 走 MODE_1 纯 FP32 直算
    switch (dtype_out) {
        // —— 窄整型：int8/uint8(1B)、int16(2B)，均走 MODE_0(Cast 路径) ——
        case ge::DataType::DT_INT8:
        case ge::DataType::DT_UINT8:
            dtype_size = DTYPE_SIZE1;
            break;
        case ge::DataType::DT_INT16:
            dtype_size = DTYPE_SIZE2;
            break;
        case ge::DataType::DT_FLOAT16:
        case ge::DataType::DT_BF16:
            dtype_size = DTYPE_SIZE2;
            break;
        case ge::DataType::DT_FLOAT:
            dtype_size = DTYPE_SIZE4;
            tilingkey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
            break;
        case ge::DataType::DT_INT32:
            dtype_size = DTYPE_SIZE4;
            break;
        case ge::DataType::DT_INT64:
            dtype_size = DTYPE_SIZE8;
            break;
        default:
            dtype_size = DTYPE_SIZE2;
            break;
    }
    return dtype_size;
}

// 单 UB 块元素数（全核共用，按 FP32 字节统一切）。
static uint32_t CalcUnitNum(uint64_t ub_size, uint32_t dtype_size, uint32_t blockSize)
{
    /*单次api计算大小：将ub 10等份后并按32B块对齐*/
    uint64_t ub_unit_size = DIVIDE_AND_ALIGN(ub_size, 10, blockSize);
    // unitNum 必须按 FP32 字节统一切，而非随 1B/2B dtype 放大。
    // 原因：Cast 路径有 4 份 FP32 中间 buffer（calc_init/step/temp/out，各 unitNum*sizeof(float)）。
    // 若用 unitNum=ub_unit_size/dtype_size，int8(1B) 下 unitNum 放大 4 倍 → 4 份 FP32 中间约 354KB > 184KB 爆 UB。
    // 用 max(dtype_size, sizeof(float)) 后：4*unitNum*4 + 2*unitNum*dtype_size ≤ 6*ub_unit_size ≈ 0.6*ub_size ≤
    // 184KB（全 dtype 安全）。
    uint32_t fp32_size = static_cast<uint32_t>(sizeof(float));                    // 4
    uint32_t unit_dtype_size = (dtype_size > fp32_size) ? dtype_size : fp32_size; // max(dtype_size, 4)
    uint32_t unitNum = ub_unit_size / unit_dtype_size; // 单 UB 块元素数（全核共用，按 FP32 字节统一切）
    return unitNum;
}

// 多核 former/tail 切分，并把 formerNum/formerLength/tailLength/
// formerUnitLoops/formerTailNum/tailUnitLoops/tailTailNum 七个字段直接写入 tiling；返回 coreNum。
static uint32_t CalcCoreSplitAndFillTiling(int64_t totalNum, uint32_t dtype_size, int64_t maxCoreNum, uint32_t unitNum,
                                           uint32_t blockSize, ArangeTilingData* tiling)
{
    // —— 多核动态切分（former/tail 模型）——
    // 除零护栏：dtype_size 恒为 DTYPE_SIZE{1,2,4,8}（switch 各分支与 default 均非 0），
    //   此处仅为消除跨函数边界后分析器无法追踪该不变量的告警，不改变任何实际取值。
    uint32_t safeDtypeSize = (dtype_size == 0) ? 1 : dtype_size;
    uint32_t alignNum = blockSize / safeDtypeSize; // 32B 内元素数
    if (alignNum == 0) {
        alignNum = 1;
    }
    // 按 32B 对齐的块总数（向上取整）
    // 大 N（int8 下块数可超 2³²）→ totalBlocks 放宽 int64
    int64_t totalBlocks = (totalNum + alignNum - 1) / alignNum;
    if (totalBlocks == 0) {
        totalBlocks = 1; // N=0 兜底（理论上 caller 不会传 0）
    }
    // 小 shape 保护：块数 < 核数时只开块数个核；至少 1 核
    uint32_t coreNum = static_cast<uint32_t>(maxCoreNum);
    if (totalBlocks < coreNum) {
        coreNum = static_cast<uint32_t>(totalBlocks); // 此分支 totalBlocks<coreNum(≤核数)，显式收窄安全
    }
    if (coreNum == 0) {
        coreNum = 1;
    }

    // former/tail 均衡：前 formerNum 个核各多 1 个 32B 块
    // formerBlocks/tailBlocks 放宽 int64，使 *alignNum 乘积在 64 位算（formerLength/tailLength 已 int64）
    uint32_t formerNum = static_cast<uint32_t>(totalBlocks % coreNum);
    int64_t formerBlocks = (totalBlocks + coreNum - 1) / coreNum; // ceil
    int64_t tailBlocks = totalBlocks / coreNum;                   // floor
    int64_t formerLength = (formerNum != 0) ? (formerBlocks * alignNum) : 0;
    int64_t tailLength = tailBlocks * alignNum;

    // 每核内 UB 子循环：former 段与 tail 段两组核负载不同，故各算一套（非冗余调用）。
    //   former 核（前 formerNum 个）每核处理 formerLength 元素，比 tail 核多 1 个 32B 块；
    //   tail 核每核处理 tailLength 元素。formerLength≠tailLength 时，两组的 UB 子循环次数
    //   (unitLoops) 与最后一个 UB 块的元素数 (tailNum) 不同，必须对两段分别计算，结果写入
    //   独立字段（formerUnitLoops/formerTailNum vs tailUnitLoops/tailTailNum），kernel 侧
    //   ParseCoreParams 再按 GetBlockIdx() 把每个核分派到对应段的参数。
    //   formerNum==0（totalBlocks 被 coreNum 整除）时全核等长、former 段长为 0，仅需 tail 一组。
    uint32_t formerUnitLoops = 0, formerTailNum = 0;
    uint32_t tailUnitLoops = 0, tailTailNum = 0;
    if (formerNum != 0) {
        CalcUnitLoops(formerLength, unitNum, formerUnitLoops, formerTailNum);
    }
    CalcUnitLoops(tailLength, unitNum, tailUnitLoops, tailTailNum);

    tiling->formerNum = formerNum;
    tiling->formerLength = formerLength;
    tiling->tailLength = tailLength;
    tiling->formerUnitLoops = formerUnitLoops;
    tiling->formerTailNum = formerTailNum;
    tiling->tailUnitLoops = tailUnitLoops;
    tiling->tailTailNum = tailTailNum;

    return coreNum;
}

// tiling 分发入口
static ge::graphStatus ArangeTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("Arange", "ArangeTilingFunc: context is nullptr"), return ge::GRAPH_FAILED);
    ArangeTilingData* tiling = context->GetTilingData<ArangeTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE("Arange", "ArangeTilingFunc: tiling data is nullptr"),
                return ge::GRAPH_FAILED);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_IF(outShape == nullptr, OP_LOGE("Arange", "ArangeTilingFunc: output shape is nullptr"),
                return ge::GRAPH_FAILED);
    // GetShapeSize() 返回 int64，直接以 int64 承接消截断（N>2³² 不丢高位）
    int64_t totalLength = outShape->GetOriginShape().GetShapeSize();
    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_IF(outputDesc == nullptr, OP_LOGE("Arange", "ArangeTilingFunc: output desc is nullptr"),
                return ge::GRAPH_FAILED);
    ge::DataType dtype_out = outputDesc->GetDataType();
    uint64_t tilingkey = 0;
    uint32_t dtype_size = DecideDtypeSizeAndTilingKey(dtype_out, tilingkey);
    context->SetTilingKey(tilingkey);

    uint64_t ub_size;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    const uint32_t blockSize = Ops::Base::GetUbBlockSize(context); // UB 对齐块大小（32B）
    int64_t maxCoreNum = ascendcPlatform.GetCoreNum();             // 平台可用核数（禁止写死）
    if (maxCoreNum <= 0) {
        maxCoreNum = 1;
    }

    int64_t totalNum = totalLength; // 元素计数链放宽 int64
    uint32_t unitNum = CalcUnitNum(ub_size, dtype_size, blockSize);
    uint32_t coreNum = CalcCoreSplitAndFillTiling(totalNum, dtype_size, maxCoreNum, unitNum, blockSize, tiling);

    tiling->dtypeSize = dtype_size;
    tiling->totalNum = totalNum;
    tiling->unitNum = unitNum;
    tiling->coreNum = coreNum;

    context->SetBlockDim(coreNum);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForArange([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Arange).Tiling(ArangeTilingFunc).TilingParse<ArangeCompileInfo>(TilingParseForArange);
} // namespace optiling
