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
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file asin_with_agent_tiling.cpp
 * \brief AsinWithAgent Host Tiling 实现（arch32: Ascend910B）
 *
 * 迭代二：实现全部 TilingKey（0-8）的 tileLength 和 tmpBufferSize 计算
 * 性能优化：Group A (TK0/1/2) 改用手动泰勒展开，tmpBuffer 从 Asin API 大小改为 4 个 float 工作区
 *
 * TilingKey 说明：
 *   0: FLOAT (fp32)  -> Group A: 手动泰勒展开，4 个 float 工作 buffer
 *   1: FLOAT16       -> Group A: 手动泰勒展开（升 float 计算），4 个 float 工作 buffer
 *   2: DOUBLE        -> Group B: op_api 层 Host 端 fp64->fp32 转换，Kernel 接收 fp32
 *                                -> Kernel 侧与 fp32 相同（手动泰勒展开）
 *   3: INT8          -> Group C: int8(1B)->half(2B)->float32(4B)->Asin
 *   4: INT16         -> Group C: int16(2B)->float32(4B)->Asin
 *   5: INT32         -> Group C: int32(4B)->float32(4B)->Asin
 *   6: INT64         -> Group C: int64(8B)->int32(4B)->float32(4B)->Asin
 *   7: UINT8         -> Group C: uint8(1B)->half(2B)->float32(4B)->Asin
 *   8: BOOL          -> Group C: bool=uint8(1B)->half(2B)->float32(4B)->Asin
 *
 * UB 占用分析（含双缓冲，per tile element）：
 *   TK0 (fp32):   srcBuf×2+dstBuf×2+tmpBuf×2 = 2×4+2×4+2×16 = 48B/elem
 *                 (tmpBuf = 4 个 float 工作区 = 4×4=16B/elem)
 *   TK1 (fp16):   srcBuf×2+dstBuf×2+tmpBuf×2 = 2×2+2×2+2×16 = 40B/elem
 *   TK2 (double): Kernel 接收 fp32 -> 与 TK0 相同，48B/elem
 *   TK3 (int8):   srcBuf=1B, halfBuf=2B, dstBuf=4B => 双缓冲: 2×(1+2+4)=14B/elem
 *   TK4 (int16):  srcBuf=2B, dstBuf=4B => 双缓冲: 2×(2+4)=12B/elem
 *   TK5 (int32):  srcBuf=4B, dstBuf=4B => 双缓冲: 2×(4+4)=16B/elem
 *   TK6 (int64):  srcBuf=8B, i32Buf=4B, dstBuf=4B => 双缓冲: 2×(8+4+4)=32B/elem
 *   TK7 (uint8):  srcBuf=1B, halfBuf=2B, dstBuf=4B => 双缓冲: 2×(1+2+4)=14B/elem
 *   TK8 (bool):   同 TK7 = 14B/elem
 */

#include <algorithm>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/asin_with_agent_tiling_data.h"
#include "../op_kernel/asin_with_agent_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

// 系统 workspace 大小（当前算子不需要额外 workspace）
constexpr uint32_t WS_SYS_SIZE = 0U;

// UB 总可用容量（预留 12KB 给系统开销）
// Ascend910B: ~192KB = 196608 bytes，预留后约 184320 bytes
constexpr uint64_t UB_RESERVED_BYTES = 12 * 1024U;

// 32字节对齐基本单位
constexpr uint32_t ALIGNMENT_BYTES = 32U;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 计算 TilingKey（由 dtype 映射）
static uint32_t ComputeTilingKey(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return 0;
        case ge::DT_FLOAT16:
            return 1;
        case ge::DT_DOUBLE:
            return 2;
        case ge::DT_INT8:
            return 3;
        case ge::DT_INT16:
            return 4;
        case ge::DT_INT32:
            return 5;
        case ge::DT_INT64:
            return 6;
        case ge::DT_UINT8:
            return 7;
        case ge::DT_BOOL:
            return 8;
        default:
            return UINT32_MAX;
    }
}

// 通用 tileLength 计算：
//   perElemBytes：每元素占用的 UB 字节数（双缓冲已算入）
//   alignElem：对齐单位（元素数）
//   tmpTotalUB：tmpBuf Ping + Pong 总字节数
static uint32_t ComputeTileLength(uint64_t ubSize, uint32_t perElemBytes, uint32_t alignElem, uint32_t tmpTotalUB)
{
    uint64_t availUB = (ubSize > UB_RESERVED_BYTES) ? (ubSize - UB_RESERVED_BYTES) : 0;

    if (availUB <= (uint64_t)tmpTotalUB || perElemBytes == 0) {
        return alignElem;
    }

    uint64_t rawTileLen = (availUB - tmpTotalUB) / perElemBytes;
    uint32_t tileLen =
        static_cast<uint32_t>(FloorAlign(static_cast<int64_t>(rawTileLen), static_cast<int64_t>(alignElem)));
    if (tileLen == 0) {
        tileLen = alignElem;
    }
    return tileLen;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// Tiling 分发入口
static ge::graphStatus AsinWithAgentTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 获取输入形状和 dtype
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto storageShape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalLength = storageShape.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    uint32_t tilingKey = ComputeTilingKey(dataType);
    OP_CHECK_IF(
        tilingKey == UINT32_MAX, OP_LOGE(context, "Unsupported dtype: %d", static_cast<int>(dataType)),
        return ge::GRAPH_FAILED);

    // 3. 设置 workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. 获取 TilingData 指针
    AsinWithAgentTilingData* tiling = context->GetTilingData<AsinWithAgentTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AsinWithAgentTilingData), 0, sizeof(AsinWithAgentTilingData)) != EOK,
        OP_LOGE(context, "memset tiling data error"), return ge::GRAPH_FAILED);

    // 上限校验：totalLength 超过 UINT32_MAX 时截断，拒绝处理
    if (totalLength > static_cast<int64_t>(UINT32_MAX)) {
        OP_LOGE(context, "totalLength exceeds UINT32_MAX");
        return ge::GRAPH_FAILED;
    }

    // 空 tensor 处理
    if (totalLength == 0) {
        tiling->totalLength = 0;
        tiling->tileLength = 8;
        tiling->loopCount = 0;
        tiling->tailTileLength = 0;
        tiling->usedCoreNum = 1;
        tiling->tmpBufferSize = 0;
        tiling->tilingKey = tilingKey;
        tiling->midBufferSize = 0;
        context->SetBlockDim(1);
        uint32_t dTypeVal = static_cast<uint32_t>(dataType);
        ASCENDC_TPL_SEL_PARAM(context, dTypeVal);
        return ge::GRAPH_SUCCESS;
    }

    // 5. 多核切分
    uint32_t usedCoreNum = static_cast<uint32_t>(std::min(coreNum, totalLength));
    uint32_t perCoreLength = static_cast<uint32_t>(totalLength / usedCoreNum);

    // 6. 按 dtype 分支计算 tmpBufferSize / tileLength / midBufferSize
    uint32_t tmpBufferSize = 0;
    uint32_t tileLength = 8;
    uint32_t midBufferSize = 0;

    // Group A 改用手动泰勒展开，不再调用 Asin 高阶 API，tmpBufferSize 按 tileLength 固定计算

    switch (tilingKey) {
        case 0: {
            // Group A fp32：手动泰勒展开，tmpBuf = 5 个 float 工作区
            // UB 占用：srcBuf×2(8B) + dstBuf×2(8B) + tmpBuf×2(5×4×2=40B) = 56B/elem
            tileLength = ComputeTileLength(ubSize, 56, ALIGNMENT_BYTES / 4, 0);
            tmpBufferSize = tileLength * 5 * static_cast<uint32_t>(sizeof(float));
            midBufferSize = 0;
            break;
        }
        case 1: {
            // Group A fp16：手动泰勒展开（升 float 精度），tmpBuf = 5 float
            // UB 占用：srcBuf×2(4B) + dstBuf×2(4B) + tmpBuf×2(40B) = 48B/elem
            tileLength = ComputeTileLength(ubSize, 48, ALIGNMENT_BYTES / 2, 0);
            tmpBufferSize = tileLength * 5 * static_cast<uint32_t>(sizeof(float));
            midBufferSize = 0;
            break;
        }
        case 2: {
            // Group B DOUBLE: op_api 层转换，Kernel 接收 fp32，走 Group A fp32 路径
            tileLength = ComputeTileLength(ubSize, 56, ALIGNMENT_BYTES / 4, 0);
            tmpBufferSize = tileLength * 5 * static_cast<uint32_t>(sizeof(float));
            midBufferSize = 0;
            break;
        }
        case 3: {
            // Group C INT8: srcBuf=1B + halfBuf=2B + floatCastBuf=4B + dstBuf=4B，双缓冲 = 2×11=22B/elem
            // alignElem = 32/1 = 32（以 srcBuf 输入对齐，最大对齐要求）
            // midBuf 包含 halfBuf(2B/elem) + floatCastBuf(4B/elem)：共 6B/elem
            // 性能优化：使用最小 seed（alignElem=32）以最大化 tileLength
            tmpBufferSize = 0;
            uint32_t perElem = 2 * (1 + 2 + 4 + 4); // 22B/elem
            tileLength = ComputeTileLength(ubSize, perElem, ALIGNMENT_BYTES / 1, 0);
            midBufferSize = tileLength * (sizeof(uint16_t) + sizeof(float)); // half(2B) + float(4B) = 6B/elem
            break;
        }
        case 4: {
            // Group C INT16: srcBuf=2B + castBuf(float)=4B + dstBuf=4B，双缓冲 = 2×10=20B/elem
            // alignElem = 32/2 = 16
            tmpBufferSize = 0;
            uint32_t perElem = 2 * (2 + 4 + 4); // 20B/elem（src=2B + castBuf=4B + dst=4B）
            tileLength = ComputeTileLength(ubSize, perElem, ALIGNMENT_BYTES / 2, 0);
            midBufferSize = tileLength * sizeof(float); // float castBuf，用于 Cast 中间结果
            break;
        }
        case 5: {
            // Group C INT32: srcBuf=4B + castBuf(float)=4B + dstBuf=4B，双缓冲 = 2×12=24B/elem
            // alignElem = 32/4 = 8
            tmpBufferSize = 0;
            uint32_t perElem = 2 * (4 + 4 + 4); // 24B/elem（src=4B + castBuf=4B + dst=4B）
            tileLength = ComputeTileLength(ubSize, perElem, ALIGNMENT_BYTES / 4, 0);
            midBufferSize = tileLength * sizeof(float); // float castBuf，避免 Asin src==dst
            break;
        }
        case 6: {
            // Group C INT64: srcBuf=8B + i32Buf=4B + dstBuf=4B，双缓冲 = 2×16=32B/elem
            // alignElem = 32/8 = 4（以 srcBuf int64 对齐）
            tmpBufferSize = 0;
            uint32_t perElem = 2 * (8 + 4 + 4); // 32B/elem
            tileLength = ComputeTileLength(ubSize, perElem, ALIGNMENT_BYTES / 8, 0);
            // 确保 tileLength >= 4（int64 对齐最小值）
            if (tileLength < 4)
                tileLength = 4;
            midBufferSize = tileLength * sizeof(int32_t); // int32 中间 buffer
            break;
        }
        case 7:
        case 8: {
            // Group C UINT8/BOOL: srcBuf=1B + halfBuf=2B + floatCastBuf=4B + dstBuf=4B，双缓冲 = 22B/elem
            // midBuf 包含 halfBuf(2B/elem) + floatCastBuf(4B/elem)：共 6B/elem
            tmpBufferSize = 0;
            uint32_t perElem = 2 * (1 + 2 + 4 + 4); // 22B/elem
            tileLength = ComputeTileLength(ubSize, perElem, ALIGNMENT_BYTES / 1, 0);
            midBufferSize = tileLength * (sizeof(uint16_t) + sizeof(float)); // half(2B) + float(4B) = 6B/elem
            break;
        }
        default:
            OP_LOGE(context, "Invalid tilingKey: %u", tilingKey);
            return ge::GRAPH_FAILED;
    }

    // 7. 计算 per-core 的 loopCount 和 tailTileLength
    uint32_t loopCount = (perCoreLength > 0) ? (perCoreLength / tileLength) : 0;
    uint32_t tailTileLength = (perCoreLength > 0) ? (perCoreLength % tileLength) : 0;

    // 8. 填写 TilingData
    tiling->totalLength = static_cast<uint32_t>(totalLength);
    tiling->tileLength = tileLength;
    tiling->loopCount = loopCount;
    tiling->tailTileLength = tailTileLength;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tmpBufferSize = tmpBufferSize;
    tiling->tilingKey = tilingKey;
    tiling->midBufferSize = midBufferSize;

    context->SetBlockDim(usedCoreNum);

    // 9. 设置 TilingKey（模板选择）
    uint32_t dTypeVal = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeVal);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAsinWithAgent([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AsinWithAgentCompileInfo {}; // 必须定义，入图场景依赖

IMPL_OP_OPTILING(AsinWithAgent)
    .Tiling(AsinWithAgentTilingFunc)
    .TilingParse<AsinWithAgentCompileInfo>(TilingParseForAsinWithAgent);

} // namespace optiling
