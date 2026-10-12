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
 * \file truncate_div_tiling.cpp
 * \brief TruncateDiv tiling（element-wise 二元算子，无 workspace）。
 *
 * 依据 (x1, x2) dtype 组合选择 schMode（tiling key），并按元素个数做 core / tile 切分。
 * kernel 使用 DataCopyPad，故切分仅需 64 元素对齐（保证各类型 DMA 起始 >=32B 对齐），
 * 尾部按精确元素数搬运，天然支持混合 dtype。int64 走标量 GM 路径，仅用 coreLength。
 */

#include <string>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/truncate_div_tiling_data.h"
#include "../op_kernel/truncate_div_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;

constexpr uint64_t VEC_ALIGN = 64u;            // 元素对齐粒度（256B / sizeof(float)）
constexpr uint64_t MIN_ELEMS_PER_CORE = 1024u; // 每核最小工作量，避免小 shape 占满全部核
constexpr uint64_t UB_RESERVE = 8192u;         // 预留给 tiling 结构 / 栈

constexpr uint32_t INPUT_X1_IDX = 0u;
constexpr uint32_t INPUT_X2_IDX = 1u;

static uint64_t AlignUp(uint64_t v, uint64_t a)
{
    if (a == 0u) {
        return v;
    }
    return (v == 0u) ? a : ((v + a - 1u) / a) * a;
}

// 依据 (x1, x2) dtype 组合确定 schMode 与各张量字节数（索引与 def / kernel 一致）。
static bool SelectCombo(ge::DataType x1Dtype, ge::DataType x2Dtype, uint64_t& schMode, uint32_t& x1Size,
                        uint32_t& x2Size, uint32_t& ySize)
{
    struct Combo {
        ge::DataType x1;
        ge::DataType x2;
        uint64_t mode;
        uint32_t s1;
        uint32_t s2;
        uint32_t sy;
    };
    static const Combo kCombos[] = {
        {ge::DT_BF16, ge::DT_BF16, TRUNCATEDIV_TPL_SCH_MODE_0, 2u, 2u, 2u},
        {ge::DT_FLOAT16, ge::DT_FLOAT16, TRUNCATEDIV_TPL_SCH_MODE_1, 2u, 2u, 2u},
        {ge::DT_FLOAT16, ge::DT_FLOAT, TRUNCATEDIV_TPL_SCH_MODE_2, 2u, 4u, 4u},
        {ge::DT_FLOAT, ge::DT_FLOAT16, TRUNCATEDIV_TPL_SCH_MODE_3, 4u, 2u, 4u},
        {ge::DT_FLOAT, ge::DT_FLOAT, TRUNCATEDIV_TPL_SCH_MODE_4, 4u, 4u, 4u},
        {ge::DT_FLOAT, ge::DT_INT32, TRUNCATEDIV_TPL_SCH_MODE_5, 4u, 4u, 4u},
        {ge::DT_INT32, ge::DT_INT32, TRUNCATEDIV_TPL_SCH_MODE_6, 4u, 4u, 4u},
        {ge::DT_INT32, ge::DT_FLOAT, TRUNCATEDIV_TPL_SCH_MODE_7, 4u, 4u, 4u},
        {ge::DT_UINT8, ge::DT_UINT8, TRUNCATEDIV_TPL_SCH_MODE_8, 1u, 1u, 1u},
        {ge::DT_INT8, ge::DT_INT8, TRUNCATEDIV_TPL_SCH_MODE_9, 1u, 1u, 1u},
        {ge::DT_INT64, ge::DT_INT64, TRUNCATEDIV_TPL_SCH_MODE_10, 8u, 8u, 8u},
        {ge::DT_INT16, ge::DT_INT16, TRUNCATEDIV_TPL_SCH_MODE_11, 2u, 2u, 2u},
    };
    for (const auto& c : kCombos) {
        if (c.x1 == x1Dtype && c.x2 == x2Dtype) {
            schMode = c.mode;
            x1Size = c.s1;
            x2Size = c.s2;
            ySize = c.sy;
            return true;
        }
    }
    return false;
}

// 依据总元素数、核数、UB 预算计算 core / tile 切分（纯计算，无副作用）。
static void ComputeSplit(uint64_t totalLength, uint64_t aivCoreNum, uint64_t ubSize, uint32_t x1Size, uint32_t x2Size,
                         uint32_t ySize, uint64_t& coreNum, uint64_t& coreLength, uint64_t& tileLength)
{
    // core split：每核至少 MIN_ELEMS_PER_CORE，coreLength 向上取整到 64 元素。
    coreNum = CeilDiv(totalLength, MIN_ELEMS_PER_CORE);
    if (coreNum > aivCoreNum) {
        coreNum = aivCoreNum;
    }
    if (coreNum == 0u) {
        coreNum = 1u;
    }
    coreLength = AlignUp(CeilDiv(totalLength, coreNum), VEC_ALIGN);
    coreNum = CeilDiv(totalLength, coreLength);
    if (coreNum == 0u) {
        coreNum = 1u;
    }

    // tile split：UB 预算按 (x1 + x2 + y) 队列 + 3 个 float 计算缓冲 + 1 个 half 中转缓冲。
    uint64_t perElem = static_cast<uint64_t>(x1Size) + x2Size + ySize + 3u * sizeof(float) + sizeof(uint16_t);
    uint64_t usable = (ubSize > UB_RESERVE) ? (ubSize - UB_RESERVE) : ubSize;
    uint64_t maxElems = (perElem == 0u) ? 0u : (usable / perElem);
    tileLength = (maxElems / VEC_ALIGN) * VEC_ALIGN;
    if (tileLength == 0u) {
        tileLength = VEC_ALIGN;
    }
    if (tileLength > coreLength) {
        tileLength = coreLength; // 单核工作量小于一个 tile 时无需过大缓冲
    }
}

static ge::graphStatus TruncateDivTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "TruncateDiv tiling starts.");

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint64_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivCoreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0u;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    auto x1Desc = context->GetInputDesc(INPUT_X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    auto x2Desc = context->GetInputDesc(INPUT_X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);

    uint64_t schMode = 0u;
    uint32_t x1Size = 0u;
    uint32_t x2Size = 0u;
    uint32_t ySize = 0u;
    OP_CHECK_IF(!SelectCombo(x1Desc->GetDataType(), x2Desc->GetDataType(), schMode, x1Size, x2Size, ySize),
                OP_LOGE(context, "unsupported (x1, x2) dtype combination for TruncateDiv."), return ge::GRAPH_FAILED);

    auto x1Shape = context->GetInputShape(INPUT_X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    uint64_t totalLength = static_cast<uint64_t>(x1Shape->GetStorageShape().GetShapeSize());
    OP_CHECK_IF(totalLength == 0u, OP_LOGE(context, "input shape size must not be 0."), return ge::GRAPH_FAILED);

    uint64_t coreNum = 0u;
    uint64_t coreLength = 0u;
    uint64_t tileLength = 0u;
    ComputeSplit(totalLength, aivCoreNum, ubSize, x1Size, x2Size, ySize, coreNum, coreLength, tileLength);

    TruncateDivTilingData* tiling = context->GetTilingData<TruncateDivTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(TruncateDivTilingData), 0, sizeof(TruncateDivTilingData)) != EOK,
                OP_LOGE(context, "memset tiling data error"), return ge::GRAPH_FAILED);
    tiling->coreNum = coreNum;
    tiling->totalLength = totalLength;
    tiling->coreLength = coreLength;
    tiling->tileLength = tileLength;

    context->SetBlockDim(coreNum);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0u;

    context->SetTilingKey(schMode);
    OP_LOGD(context, "TruncateDiv tiling: key=%lu coreNum=%lu totalLen=%lu coreLength=%lu tileLength=%lu", schMode,
            coreNum, totalLength, coreLength, tileLength);
    return ge::GRAPH_SUCCESS;
}

struct TruncateDivCompileInfo {};

static ge::graphStatus TilingParseForTruncateDiv([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TruncateDiv)
    .Tiling(TruncateDivTilingFunc)
    .TilingParse<TruncateDivCompileInfo>(TilingParseForTruncateDiv);

} // namespace optiling
